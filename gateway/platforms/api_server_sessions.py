"""Session CRUD, listing, and message history mixin for APIServerAdapter."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from contextlib import suppress
from functools import partial, wraps
from pathlib import Path
from typing import Any, Optional

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]


def _api_mod():
    mod = sys.modules.get("gateway.platforms.api_server")
    if mod is None:
        import gateway.platforms.api_server as mod
    return mod


def _require_auth(handler):
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        return await handler(self, request, *args, **kwargs)
    return _wrapped


class _DualMethod:
    """Descriptor allowing a method to be called both on an instance and on the class."""

    def __init__(self, func):
        self.func = func
        wraps(func)(self)

    def __get__(self, instance, owner=None):
        return partial(self.func, instance)


def _etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    if not if_none_match:
        return False
    raw = if_none_match.strip()
    if raw == "*":
        return True
    norm_target = etag.strip()
    if norm_target.startswith("W/"):
        norm_target = norm_target[2:].strip()
    for part in raw.split(","):
        token = part.strip()
        if token.startswith("W/"):
            token = token[2:].strip()
        if token == norm_target or token.strip('"') == norm_target.strip('"'):
            return True
    return False


class _SessionsMixin:
    """Session management and message history HTTP handlers."""

    @staticmethod
    def _parse_nonnegative_int(value: Any, default: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return default if parsed < 0 else min(parsed, maximum)

    @staticmethod
    def _session_db_unavailable() -> web.Response:
        return _api_mod()._error_response("Session database unavailable", 503, code="session_db_unavailable")

    def _get_active_subagents_for_session(self, session_id: str) -> list[dict[str, Any]]:
        if not session_id:
            return []
        active = []
        try:
            from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
            with _active_subagents_lock:
                for sid, rec in _active_subagents.items():
                    owner_sid = str(rec.get("owner_agent_session_id") or rec.get("owner_session_id") or "")
                    parent_sid = str(rec.get("parent_session_id") or "")
                    gw_key = str(rec.get("gateway_session_key") or "")
                    is_match = (
                        owner_sid == session_id
                        or parent_sid == session_id
                        or gw_key == session_id
                    )
                    if is_match:
                        started = rec.get("started_at")
                        active.append({
                            "subagent_id": rec.get("subagent_id", sid),
                            "parent_id": rec.get("parent_id"),
                            "depth": rec.get("depth", 0),
                            "goal": rec.get("goal", ""),
                            "model": rec.get("model", ""),
                            "status": rec.get("status", "running"),
                            "last_tool": rec.get("last_tool"),
                            "tool_count": rec.get("tool_count", 0),
                            "started_at": started if isinstance(started, (int, float)) and not isinstance(started, bool) else None,
                            "running_seconds": round(_api_mod().time.time() - started, 1) if isinstance(started, (int, float)) and not isinstance(started, bool) else None,
                        })
        except Exception as e:
            _api_mod().logger.debug("Failed to query live subagents for session %s: %s", session_id, e)
        return active

    def _get_recent_subagents_for_session(
        self,
        session_id: str,
        *,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        if not session_id:
            return []
        try:
            from tools.delegate_tool_registry import get_recent_subagents_for_session
            records = get_recent_subagents_for_session(session_id, exclude_ids=exclude_ids)
        except Exception as e:
            _api_mod().logger.debug("Failed to query recent subagents for session %s: %s", session_id, e)
            return []
        recent: list[dict[str, Any]] = []
        for rec in records:
            started = rec.get("started_at")
            ended = rec.get("ended_at") or rec.get("completed_at")
            duration = rec.get("duration_seconds")
            recent.append({
                "subagent_id": rec.get("subagent_id"),
                "parent_id": rec.get("parent_id"),
                "depth": rec.get("depth", 0),
                "goal": rec.get("goal", ""),
                "model": rec.get("model", ""),
                "status": rec.get("status", "completed"),
                "last_tool": rec.get("last_tool"),
                "tool_count": rec.get("tool_count", 0),
                "started_at": started if isinstance(started, (int, float)) and not isinstance(started, bool) else None,
                "ended_at": ended if isinstance(ended, (int, float)) and not isinstance(ended, bool) else None,
                "completed_at": ended if isinstance(ended, (int, float)) and not isinstance(ended, bool) else None,
                "duration_seconds": float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
                "failure_reason": rec.get("failure_reason"),
                "error": rec.get("error"),
            })
        return recent

    def _session_response(
        self,
        session: dict[str, Any],
        *,
        last_msg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return a stable, client-safe session representation."""
        safe_keys = (
            "id", "source", "user_id", "model", "title", "started_at", "ended_at", "end_reason",
            "message_count", "tool_call_count", "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
            "actual_cost_usd", "api_call_count", "parent_session_id", "last_active", "preview",
            "_lineage_root_id", "pinned", "archived", "hidden")
        payload = {key: session.get(key) for key in safe_keys if key in session}
        # SQLite stores the flags as 0/1.
        payload.update(
            {f: bool(payload[f]) for f in ("pinned", "archived", "hidden") if f in payload})
        # Full system prompts / model_config never cross the client API; only their presence.
        payload["has_system_prompt"] = bool(session.get("system_prompt"))
        payload["has_model_config"] = bool(session.get("model_config"))
        raw_model_config = session.get("model_config")
        try:
            model_config = (
                json.loads(raw_model_config)
                if isinstance(raw_model_config, str)
                else raw_model_config
            )
        except (TypeError, json.JSONDecodeError):
            model_config = None
        # Exact-id consumers may inspect/resume delegate children even though
        # list endpoints intentionally omit them. Project only the provenance
        # bit the client needs so it cannot accidentally promote such a row
        # into an ordinary session list; never expose the model snapshot.
        payload["is_internal_child"] = bool(
            isinstance(model_config, dict)
            and model_config.get("_delegate_from") is not None
        )

        session_id = str(session.get("id") or "")
        active_subagents = self._get_active_subagents_for_session(session_id)
        running_subagents = [
            s for s in active_subagents
            if str(s.get("status") or "running").lower() == "running"
        ]
        active_ids = {str(s.get("subagent_id")) for s in active_subagents if s.get("subagent_id")}
        recent_subagents = self._get_recent_subagents_for_session(session_id, exclude_ids=active_ids)
        has_live_async = False
        has_pending_delivery = False
        try:
            from tools.async_delegation import has_live_for_session, has_pending_delivery_for_session
            has_live_async = has_live_for_session(
                session_key=session_id, origin_ui_session_id=session_id, parent_session_id=session_id,
            )
            has_pending_delivery = has_pending_delivery_for_session(
                session_key=session_id, origin_ui_session_id=session_id, parent_session_id=session_id,
            )
        except Exception as e:
            _api_mod().logger.debug("Failed to check async delegations for session %s: %s", session_id, e)
        has_active_subagents = bool(running_subagents or has_live_async)

        active_run = None
        if session_id and hasattr(self, "_run_statuses"):
            for r_id, r_info in list(self._run_statuses.items()):
                if str(r_info.get("session_id") or "") == session_id and r_info.get("status") in {"queued", "running", "waiting_for_approval"}:
                    active_run = (r_id, r_info)
                    break
        waiting_approval = False
        if active_run:
            r_id, r_info = active_run
            run_status = str(r_info.get("status") or "")
            if run_status == "waiting_for_approval":
                waiting_approval = True
                payload["is_generating"] = False
                payload["active_run_id"] = r_id
                payload["tool_status"] = r_info.get("tool_status") or "Ожидает подтверждения..."
                payload["current_tool"] = r_info.get("current_tool")
            else:
                payload["is_generating"] = True
                payload["active_run_id"] = r_id
                payload["tool_status"] = r_info.get("tool_status") or "Ассистент думает над задачей..."
                payload["current_tool"] = r_info.get("current_tool")
        else:
            is_gen = False
            tool_status = None
            current_tool = None
            runner = getattr(self, "gateway_runner", None)
            if runner:
                adapters_to_check = list(getattr(runner, "adapters", {}).values())
                prof_adapters = getattr(runner, "_profile_adapters", {})
                if isinstance(prof_adapters, dict):
                    for p_dict in prof_adapters.values():
                        if isinstance(p_dict, dict):
                            adapters_to_check.extend(p_dict.values())
                chat_id_val = str(session.get("chat_id") or "")
                for ad in adapters_to_check:
                    active_sess = getattr(ad, "_active_sessions", {})
                    if not isinstance(active_sess, dict):
                        continue
                    for sk, guard in active_sess.items():
                        sk_str = str(sk)
                        is_match = False
                        if session_id and (sk_str == session_id or sk_str.endswith(f":{session_id}")):
                            is_match = True
                        elif chat_id_val and (sk_str == chat_id_val or sk_str.endswith(f":{chat_id_val}")):
                            is_match = True
                        elif session.get("gateway_session_key") and sk_str == str(session.get("gateway_session_key")):
                            is_match = True

                        if is_match:
                            is_gen = True
                            tool_status = (getattr(ad, "_last_status", {}) or {}).get(sk) or "Ассистент думает над задачей..."
                            break
                    if is_gen:
                        break
            if not is_gen and has_active_subagents:
                is_gen = True
                if running_subagents and running_subagents[0].get("last_tool"):
                    last_tool = str(running_subagents[0]["last_tool"])
                    verb = None
                    try:
                        from agent.display import get_tool_verb
                        verb = get_tool_verb(last_tool)
                    except Exception:
                        verb = None
                    if verb:
                        tool_status = f"Подзадача: {verb}..."
                    else:
                        tool_status = f"Подзадача выполняет {last_tool}..."
                elif running_subagents:
                    tool_status = f"Выполняются подзадачи ({len(running_subagents)} в работе)..."
                else:
                    tool_status = "Выполняется фоновая подзадача..."
                current_tool = "delegate_task"

            payload["is_generating"] = is_gen
            payload["active_run_id"] = None
            payload["tool_status"] = tool_status
            payload["current_tool"] = current_tool

        payload["has_active_subagents"] = has_active_subagents
        payload["has_pending_delegation_delivery"] = has_pending_delivery
        active_subagents_count = len(running_subagents)
        payload["active_subagents_count"] = active_subagents_count
        payload["active_subagents"] = active_subagents
        payload["subagents"] = active_subagents
        payload["recent_subagents"] = recent_subagents

        last_role = last_msg.get("role") if isinstance(last_msg, dict) else None
        last_content = str(last_msg.get("content") or "") if isinstance(last_msg, dict) else ""
        last_has_tool_calls = bool(last_msg.get("has_tool_calls")) if isinstance(last_msg, dict) else False
        last_finish_reason = str(last_msg.get("finish_reason") or "") if isinstance(last_msg, dict) else ""

        last_read_at = session.get("last_read_at")
        if last_read_at is not None:
            last_active_ts = (
                session.get("last_active")
                or (last_msg.get("timestamp") if isinstance(last_msg, dict) else None)
                or session.get("started_at")
            )
            session_unread = float(last_active_ts or 0) > float(last_read_at)
        else:
            session_unread = False

        is_generating = bool(payload.get("is_generating", False))
        is_gen = bool(is_generating or has_active_subagents or active_subagents_count > 0)
        waiting_reason: Optional[str] = None
        is_unread_completion = False

        if is_gen:
            status = "running"
        elif waiting_approval:
            status = "waiting_input"
            waiting_reason = "approval"
        elif last_role == "assistant" and not last_has_tool_calls and last_finish_reason != "tool_calls":
            if _api_mod()._assistant_expects_user_reply(last_content):
                status = "waiting_input"
                waiting_reason = "question"
            elif session_unread:
                status = "completed"
                is_unread_completion = True
            else:
                status = "idle"
        elif session.get("ended_at") or session.get("end_reason"):
            if session_unread:
                status = "completed"
                is_unread_completion = True
            else:
                status = "idle"
        else:
            status = "idle"

        payload["status"] = status
        payload["waiting_reason"] = waiting_reason
        payload["last_role"] = last_role
        payload["is_unread_completion"] = is_unread_completion
        return payload

    def _resolve_message_media(self, text: str, *, api_key: Optional[str] = None) -> str:
        resolved_key = api_key if api_key is not None else self._expected_api_key()
        return _api_mod()._resolve_message_media_text(
            text,
            api_key=resolved_key,
            ref_secret=getattr(self, "_media_ref_secret", _api_mod()._PROCESS_MEDIA_REF_SECRET),
        )

    @_DualMethod
    def _message_response(self, message: dict[str, Any]) -> dict[str, Any]:
        message = _api_mod()._project_client_message(message)
        safe_keys = (
            "id", "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name",
            "timestamp", "token_count", "finish_reason", "reasoning", "reasoning_content",
            "display_kind")
        res = {key: message.get(key) for key in safe_keys if key in message}
        if res.get("role") in ("assistant", "user") and isinstance(res.get("content"), str):
            if self is not None:
                res["content"] = self._resolve_message_media(res["content"])
            else:
                res["content"] = _api_mod()._resolve_message_media_text(res["content"])
        return res

    async def _read_json_body(self, request: web.Request) -> tuple[dict[str, Any], Optional[web.Response]]:
        try:
            body = await request.json()
        except Exception:
            return {}, _api_mod()._error_response("Invalid JSON in request body", 400)
        if not isinstance(body, dict):
            return {}, _api_mod()._error_response("Request body must be a JSON object", 400)
        return body, None

    async def _get_existing_session_or_404(self, session_id: str) -> tuple[Optional[dict[str, Any]], Optional[web.Response]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return None, self._session_db_unavailable()
        session = await _api_mod().asyncio.to_thread(db.get_session, session_id)
        if not session:
            return None, _api_mod()._error_response(f"Session not found: {session_id}", 404, code="session_not_found")
        return session, None

    async def _conversation_history_for_session(self, session_id: str) -> list[dict[str, Any]]:
        db = await self._ensure_session_db_async()
        if db is None:
            return []
        try:
            return await _api_mod().asyncio.to_thread(db.get_messages_as_conversation, session_id)
        except Exception as exc:
            _api_mod().logger.warning("Failed to load session history for %s: %s", session_id, exc)
            return []

    async def run_internal_session_turn(self, *, session_id: str, text: str, profile: str,
                                        notification_category: str = "result",
                                        delegation_id: str = "",
                                        delivery_notice: str = "") -> None:
        """Run one background wake turn against a raw session id IN-PROCESS (no HTTP, no API key);
        see ``api_server_runs.run_internal_session_turn``."""
        await _api_mod()._api_runs.run_internal_session_turn(
            self, session_id=session_id, text=text, profile=profile,
            notification_category=notification_category,
            delegation_id=delegation_id, delivery_notice=delivery_notice,
            _api_server=_api_mod())

    @_require_auth
    async def _handle_list_sessions(self, request: web.Request) -> web.Response:
        """GET /api/sessions — list persisted Hermes sessions."""
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        limit = self._parse_nonnegative_int(request.query.get("limit"), default=50, maximum=200)
        offset = self._parse_nonnegative_int(request.query.get("offset"), default=0, maximum=1_000_000)
        source = request.query.get("source") or None
        exclude_sources_raw = request.query.get("exclude_sources")
        exclude_sources = [s.strip() for s in exclude_sources_raw.split(",") if s.strip()] if exclude_sources_raw else None
        include_children = _api_mod()._coerce_request_bool(request.query.get("include_children"), default=False)
        # Exact-title lookup (`hermes peer dm` -> canonical "Bot Chat"). include_hidden is honored
        # ONLY with a title filter: a blanket hidden listing stays off this client surface.
        title_filter = (request.query.get("title") or "").strip() or None
        include_hidden = bool(title_filter) and _api_mod()._coerce_request_bool(
            request.query.get("include_hidden"), default=False)

        async def _list() -> list:
            # include_pinned back-fills pins past the recency window; search_query pushes the
            # title needle into SQL (substring) so a hidden/old row is found, exact match below.
            rows = await _api_mod().asyncio.to_thread(
                db.list_sessions_rich, source=source, exclude_sources=exclude_sources, limit=limit, offset=offset,
                include_children=include_children, order_by_last_active=True, include_pinned=True,
                search_query=title_filter, include_hidden=include_hidden)
            if title_filter:
                rows = [s for s in rows if (s.get("title") or "").strip() == title_filter]
            return rows

        sessions = await _list() or (await _api_mod().reset_local_title_owner(db, title_filter, include_hidden, source))
        if title_filter and not sessions:
            # A canonical Bot Chat auto-archived by the orphan reaper would make `hermes peer dm`
            # mint transient sessions: resurrect and re-list; deliberate archives stay put.
            try:
                # Recoverable-archive resurrection (#92687): a canonical Bot Chat archived by the ws-orphan
                # reaper / older agent cleanup is invisible to list_sessions_rich (include_archived=False),
                # which would fail `hermes peer dm` resolution and mint transient sessions — same accident
                # the tui_gateway lookups heal.
                from tools.bot_mode_probe import BOT_CHAT_TITLE

                def _resurrect() -> bool:
                    # Lookup + unarchive (a WRITE with the full write patience) as one worker-thread
                    # hop: a contended lock parks this thread, never the event loop (#113772).
                    stale = db.get_session_by_title(title_filter)
                    return bool(stale and stale.get("archived") and db.unarchive_recoverable_session(stale["id"]))

                if title_filter == BOT_CHAT_TITLE and await _api_mod().asyncio.to_thread(_resurrect):
                    sessions = await _list()
            except Exception:
                pass  # resolution degrades to today's no-row behavior
        session_ids = [str(s.get("id") or "") for s in sessions if s.get("id")]
        last_msgs = (
            await _api_mod().asyncio.to_thread(db.session_last_messages, session_ids)
            if session_ids and hasattr(db, "session_last_messages")
            else {}
        )
        data = [self._session_response(s, last_msg=last_msgs.get(str(s.get("id") or ""))) for s in sessions]
        etag_rows = [
            (
                item.get("id"),
                item.get("last_active"),
                item.get("status"),
                item.get("title"),
                item.get("is_unread_completion"),
                item.get("unread"),
                item.get("active_subagents_count"),
                (last_msgs.get(str(item.get("id") or "")) or {}).get("id"),
            )
            for item in data
        ]
        etag_basis = (limit, offset, source, exclude_sources, include_children, title_filter, etag_rows)
        etag_digest = hashlib.sha256(
            json.dumps(etag_basis, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        etag = f'"{etag_digest}"'
        if _etag_matches(request.headers.get("If-None-Match"), etag):
            return web.Response(status=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        # Back-filled pins arrive PAST the limit: counting them would report a page that doesn't exist.
        windowed = sum(1 for s in sessions if not s.get("pinned"))
        return web.json_response(
            {
                "object": "list",
                "data": data,
                "limit": limit,
                "offset": offset,
                "has_more": windowed >= limit,
            },
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    @_require_auth
    async def _handle_create_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions -- create an empty Hermes session row. Existence check, insert and
        title handling run as ONE off-loop write so concurrent same-id creates can't both 201."""
        body, err = await self._read_json_body(request)
        if err:
            return err
        from gateway.platforms.api_server_bot_chat import create_peer_bot_chat
        if (bot_chat := await create_peer_bot_chat(self, body)) is not None:
            return bot_chat  # the owner minted the canonical Bot Chat (#139370)
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        raw_id = body.get("id") or body.get("session_id")
        session_id = str(raw_id).strip() if raw_id else f"api_{int(_api_mod().time.time())}_{uuid.uuid4().hex[:8]}"
        from gateway.session import _is_path_unsafe
        if not session_id or re.search(r'[\r\n\x00]', session_id) or _is_path_unsafe(session_id):
            return _api_mod()._error_response("Invalid session ID", 400, code="invalid_session_id")
        if len(session_id) > self._MAX_SESSION_HEADER_LEN:
            return _api_mod()._error_response("Session ID too long", 400, code="invalid_session_id")
        system_prompt = body.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return _api_mod()._error_response("system_prompt must be a string", 400, code="invalid_system_prompt")
        source = self._normalize_session_source(body.get("source") or "api_server")
        runtime_request = self._session_runtime_request_from_body(body)
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        requested = runtime_request.get("requested") or {}
        # The normalized requested["model"] (prefix split, virtual alias nulled) — the raw body
        # would persist "hermes-agent" and later send it to the provider literally.
        model_name = self._clean_runtime_id(requested.get("model")) or None
        model_config = None
        if requested.get("model") or requested.get("provider"):
            model_config = {"browser_model_lock": {
                "provider": requested.get("provider") or "", "model": requested.get("model") or "",
                "model_options": runtime_request.get("model_options") or {},
                "route_source": runtime_request.get("route_source") or "",
                "confirmed": bool(runtime_request.get("require_model_lock")),
                "updated_at": _api_mod().time.time()}}
        raw_source = str(body.get("source") or "").strip().lower()
        if raw_source == "webui" or source == "webui":
            model_config = dict(model_config) if isinstance(model_config, dict) else {}
            model_config["auto_resume_on_delegation"] = True
        title = body.get("title")

        def _atomic(conn):
            from hermes_state_mutation_retirement import create_conflict
            if conflict := create_conflict(conn, session_id):  # one BEGIN IMMEDIATE write: a racing create sees the row
                return None, conflict
            conn.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt, started_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, source, model_name, json.dumps(model_config) if model_config else None,
                 system_prompt, _api_mod().time.time()))
            if title is not None:
                clean_title = db.sanitize_title(str(title))
                if clean_title:
                    try:
                        db._resolve_title_conflict(conn, session_id, clean_title)
                    except ValueError as exc:  # the DB's uniqueness rule; undo the INSERT
                        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                        return None, f"title:{exc}"
                conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean_title, session_id))
            session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return (dict(session_row) if session_row else {
                "id": session_id, "source": source, "model": model_name, "title": title}), None
        session, err = await _api_mod().asyncio.to_thread(db._execute_write, _atomic)
        if isinstance(err, tuple):  # a live row, or a deleted id's fence: refused now, not on every turn
            return _api_mod()._error_response(f"{err[1]}: {session_id}", 409, code=err[0])
        if err and err.startswith("title:"):
            return _api_mod()._error_response(err[len("title:"):], 400, code="invalid_title")
        return web.json_response({"object": "hermes.session", "session": self._session_response(session)}, status=201)

    async def _last_message_for_session(self, session_id: str) -> Optional[dict[str, Any]]:
        if not session_id:
            return None
        db = await self._ensure_session_db_async()
        if db is None or not hasattr(db, "session_last_messages"):
            return None
        try:
            last_msgs = await _api_mod().asyncio.to_thread(db.session_last_messages, [session_id])
            return last_msgs.get(session_id)
        except Exception:
            return None

    @_require_auth
    async def _handle_get_session(self, request: web.Request) -> web.Response:
        """GET /api/sessions/{session_id}."""
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        last_msg = await self._last_message_for_session(str(session.get("id") or session_id))
        return web.json_response({"object": "hermes.session", "session": self._session_response(session, last_msg=last_msg)})

    @_require_auth
    async def _handle_patch_session(self, request: web.Request) -> web.Response:
        """PATCH /api/sessions/{session_id} — update client-safe session metadata."""
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        # pinned/archived/unread are durable desktop-sidebar flags.
        unknown = sorted(set(body) - {"title", "end_reason", "pinned", "archived", "hidden", "unread"})
        if unknown:
            return _api_mod()._error_response(
                f"Unsupported session fields: {', '.join(unknown)}", 400, code="unsupported_session_field")
        for flag in ("pinned", "archived", "hidden", "unread"):
            if flag in body and not isinstance(body[flag], bool):
                return _api_mod()._error_response(f"'{flag}' must be a boolean", 400, code="invalid_session_field")
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        prev_title = session.get("title")
        if "title" in body:
            try:
                await _api_mod().asyncio.to_thread(
                    db.set_session_title, session_id, "" if body["title"] is None else str(body["title"]))
            except ValueError as exc:
                return _api_mod()._error_response(str(exc), 400, code="invalid_title")
        # Pinned last: set_session_pinned clears hidden, so a pin in the same request
        # wins over an explicit hidden (same order as the dashboard's _RENAME_FLAG_SETTERS).
        for flag, setter in (("archived", db.set_session_archived), ("hidden", db.set_session_hidden),
                             ("pinned", db.set_session_pinned)):
            if flag in body:
                await _api_mod().asyncio.to_thread(setter, session_id, body[flag])
        if "unread" in body:
            await _api_mod().asyncio.to_thread(db.set_session_read, session_id, read=not body["unread"])
        if body.get("end_reason"):
            await _api_mod().asyncio.to_thread(db.end_session, session_id, str(body["end_reason"]))
        session = await _api_mod().asyncio.to_thread(db.get_session, session_id) or session
        if "title" in body:
            updated_title = str(session.get("title") or "").strip()
            if updated_title and updated_title != prev_title:
                sync_fn = getattr(self, "_sync_telegram_topic_title", None)
                if callable(sync_fn):
                    await sync_fn(session_id, updated_title)
        last_msg = await self._last_message_for_session(str(session.get("id") or session_id))
        return web.json_response({"object": "hermes.session", "session": self._session_response(session, last_msg=last_msg)})

    @_require_auth
    async def _handle_read_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/read — mark session read up to now."""
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        await _api_mod().asyncio.to_thread(db.set_session_read, session_id, True)
        session = await _api_mod().asyncio.to_thread(db.get_session, session_id) or session
        last_msg = await self._last_message_for_session(str(session.get("id") or session_id))
        return web.json_response({"object": "hermes.session", "session": self._session_response(session, last_msg=last_msg)})

    @_require_auth
    async def _handle_delete_session(self, request: web.Request) -> web.Response:
        """DELETE /api/sessions/{session_id}."""
        session_id = request.match_info["session_id"]
        _session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        # Same profile home the DB was resolved from (the profile middleware scopes
        # get_hermes_home() for this request) — without it the transcript/dump scrub is skipped.
        sessions_dir = None
        try:
            from hermes_constants import get_hermes_home
            sessions_dir = Path(get_hermes_home()) / "sessions"
        except Exception:
            _api_mod().logger.debug("sessions dir unavailable for delete of %s", session_id, exc_info=True)
        try:
            deleted = await _api_mod().asyncio.to_thread(
                db.delete_session, session_id, sessions_dir=sessions_dir, exclude_active_write_guards=True)
        except _api_mod().SessionActiveWriteGuardError as exc:
            return _api_mod()._error_response(str(exc), 409, code="session_active_turn")
        if deleted:
            # A hard delete must also drop the gateway's durable channel→session routing entries
            # for the id, or the next inbound message resolves the SAME id and run_agent's
            # INSERT OR IGNORE resurrects the deleted row (#42422).
            runner = self.gateway_runner or request.app.get("gateway_runner")
            store = getattr(runner, "session_store", None)
            if store is not None:
                await _api_mod().asyncio.to_thread(store.remove_by_session_id, session_id)
            # Retirement erased the chat's image references; bytes nothing else holds go too. Off the
            # loop like every other media release: its holder check and unlink are one write txn.
            from gateway.session_ingress_media import collect_unheld_api_images
            await _api_mod().asyncio.to_thread(collect_unheld_api_images, db)
            coord = getattr(self, "_session_coordinator", None)
            if coord is not None:
                coord.cleanup_session(session_id)
            else:
                locks = getattr(self, "_telegram_sync_locks", None)
                if isinstance(locks, dict):
                    lock = locks.get(session_id)
                    if lock is not None and not lock.locked():
                        locks.pop(session_id, None)
        return web.json_response({"object": "hermes.session.deleted", "id": session_id, "deleted": bool(deleted)})

    @_require_auth
    async def _handle_session_messages(self, request: web.Request) -> web.Response:
        """GET /api/sessions/{session_id}/messages."""
        api = _api_mod()
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        resolved_id = await api.asyncio.to_thread(db.resolve_resume_session_id, session_id)
        raw_after_id = request.query.get("after_id")
        after_id: Optional[int] = None
        if raw_after_id is not None and raw_after_id != "":
            try:
                after_id = int(raw_after_id)
            except (TypeError, ValueError):
                after_id = -1
            if after_id < 0:
                return api._error_response("after_id must be a non-negative integer", 400, code="invalid_pagination")
        raw_limit, raw_offset = request.query.get("limit"), request.query.get("offset", "0")
        order = request.query.get("order")
        if order not in (None, "oldest", "latest"):
            return api._error_response("order must be one of: oldest, latest", 400, code="invalid_pagination")
        try:
            offset = int(raw_offset)
            requested_limit = None if raw_limit is None else int(raw_limit)
        except (TypeError, ValueError):
            offset = requested_limit = -1
        if offset < 0 or (requested_limit is not None and requested_limit < 0):
            return api._error_response("limit and offset must be non-negative integers", 400, code="invalid_pagination")
        default_page = requested_limit is None
        latest_page = order == "latest" or (order is None and default_page)
        limit = 500 if default_page else min(requested_limit, 500)
        include_compacted = api._coerce_request_bool(request.query.get("include_compacted"), default=False)
        if after_id is not None:
            if offset > 0 or include_compacted or order == "latest":
                return api._error_response(
                    "after_id cannot be combined with offset > 0, include_compacted=true, or order=latest",
                    400,
                    code="invalid_pagination",
                )
            if resolved_id != session_id:
                return web.json_response(
                    {
                        "object": "list",
                        "session_id": resolved_id,
                        "sync_reset": True,
                        "full_reload_required": True,
                        "data": [],
                    },
                    headers={"X-Hermes-Sync-Reset": "1", "Cache-Control": "no-cache"},
                )
            messages = await api.asyncio.to_thread(
                db.get_messages,
                resolved_id,
                after_id=after_id,
                limit=limit,
                offset=0,
                latest=False,
                include_compacted=False,
                include_ancestors=False,
            )
        else:
            # Compression lineage: return root→tip messages, matching the REST router (#51058).
            messages = await api.asyncio.to_thread(
                db.get_messages,
                resolved_id,
                limit=limit,
                offset=offset,
                latest=latest_page,
                include_compacted=include_compacted,
                include_ancestors=True,
            )
        last_msg = messages[-1] if messages else None
        last_msg_id = last_msg.get("id") if isinstance(last_msg, dict) else None
        last_msg_hash = (
            hashlib.sha256(
                json.dumps(
                    {
                        "id": last_msg.get("id"),
                        "role": last_msg.get("role"),
                        "content": last_msg.get("content"),
                        "timestamp": last_msg.get("timestamp"),
                        "tool_calls": last_msg.get("tool_calls"),
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(last_msg, dict)
            else ""
        )
        etag_basis = (
            resolved_id,
            after_id,
            include_compacted,
            order,
            limit,
            offset,
            len(messages),
            last_msg_id,
            last_msg_hash,
        )
        etag_digest = hashlib.sha256(
            json.dumps(etag_basis, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        etag = f'"{etag_digest}"'
        if _etag_matches(request.headers.get("If-None-Match"), etag):
            return web.Response(status=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        pagination: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "order": ("oldest" if after_id is not None else (order or ("latest" if default_page else "oldest"))),
            "returned": len(messages),
        }
        if after_id is not None:
            pagination["last_id"] = last_msg_id if isinstance(last_msg_id, int) else after_id
        elif isinstance(last_msg_id, int):
            pagination["last_id"] = last_msg_id
        return web.json_response(
            {
                "object": "list",
                "session_id": resolved_id,
                "data": [self._message_response(m) for m in messages],
                "pagination": pagination,
            },
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    @_require_auth
    async def _handle_fork_session(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/fork — branch via current SessionDB primitives."""
        source_id = request.match_info["session_id"]
        source, err = await self._get_existing_session_or_404(source_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        db = await self._ensure_session_db_async()
        fork_id = str(body.get("id") or body.get("session_id") or f"api_{int(_api_mod().time.time())}_{uuid.uuid4().hex[:8]}").strip()
        if not fork_id or re.search(r'[\r\n\x00]', fork_id):
            return _api_mod()._error_response("Invalid session ID", 400, code="invalid_session_id")
        if await _api_mod().asyncio.to_thread(db.get_session, fork_id):
            return _api_mod()._error_response(f"Session already exists: {fork_id}", 409, code="session_exists")

        # CLI /branch semantics: create the child, then end the original as branched (child first, so
        # a failed create never leaves the source ended with no fork, #11030).
        # ``_branched_from`` is the durable branch marker (same as CLI /branch): with the child created
        # first, the timestamp fallback in _BRANCH_CHILD_SQL (child.started_at >= parent.ended_at) no
        # longer holds, and an unmarked child would vanish from default session listings.
        await _api_mod().asyncio.to_thread(
            db.create_session, fork_id, "api_server", model=source.get("model"),
            system_prompt=source.get("system_prompt"), parent_session_id=source_id,
            model_config={"_branched_from": source_id})
        await _api_mod().asyncio.to_thread(db.end_session, source_id, "branched")
        messages = await _api_mod().asyncio.to_thread(db.get_messages, source_id)
        await _api_mod().asyncio.to_thread(db.replace_messages, fork_id, messages)
        title = body.get("title")
        if title is None:
            base = source.get("title") or "fork"
            title = f"{base} fork"
            with suppress(Exception):
                title = await _api_mod().asyncio.to_thread(db.get_next_title_in_lineage, base)
        try:
            await _api_mod().asyncio.to_thread(db.set_session_title, fork_id, str(title))
        except ValueError as exc:
            return _api_mod()._error_response(str(exc), 400, code="invalid_title")
        fork = await _api_mod().asyncio.to_thread(db.get_session, fork_id) or {"id": fork_id, "parent_session_id": source_id}
        return web.json_response({"object": "hermes.session", "session": self._session_response(fork)}, status=201)
