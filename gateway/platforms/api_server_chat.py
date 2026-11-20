"""Session chat, SSE streaming, Telegram mirror, and model lock mixin."""

from __future__ import annotations

import asyncio
import sys
import threading
import uuid
from contextlib import suppress
from functools import wraps
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


def _admit_api_agent_request(handler):
    @wraps(handler)
    async def _wrapped(self, request, *args, **kwargs):
        api = _api_mod()
        auth_err = (
            self._check_run_auth(request, permission="dispatch")
            if api._api_runs._uses_room_run_auth(self, request)
            else self._check_auth(request)
        )
        if auth_err:
            return auth_err
        draining = self._draining_response()
        if draining is not None:
            return draining
        reservation = {"active": True}
        token = api._api_agent_request_reservation.set(reservation)
        self._pending_agent_requests += 1
        try:
            return await handler(self, request, *args, **kwargs)
        finally:
            api._release_pending_api_work(self, reservation)
            api._api_agent_request_reservation.reset(token)
    return _wrapped


class _SessionEventQueue:
    """Ordered SSE event queue for one /api/sessions/{id}/chat/stream run. ``payload`` stamps
    session_id/run_id/seq/ts; ``enqueue`` is executor-thread safe (hops onto the owning loop)."""

    def __init__(self, session_id: str, run_id: str):
        self.loop = _api_mod().asyncio.get_running_loop()
        self.queue: _api_mod().asyncio.Queue[Optional[tuple[str, dict[str, Any]]]] = _api_mod().asyncio.Queue(maxsize=500)
        self.session_id = session_id
        self.run_id = run_id
        self.seq = 0
        # Cross-thread puts handed to the loop and not yet applied. The loop does not order them
        # against the worker future's completion, so the turn's terminal frames wait on flush().
        self._inflight = 0
        self._inflight_lock = threading.Lock()

    def payload(self, name: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        self.seq += 1
        payload.setdefault("session_id", self.session_id)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("seq", self.seq)
        payload.setdefault("ts", _api_mod().time.time())
        return name, payload

    def enqueue(self, name: str, payload: dict[str, Any]) -> None:
        event = self.payload(name, payload)
        try:
            running_loop = _api_mod().asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self.loop:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            self.queue.put_nowait(event)
            return
        with self._inflight_lock:
            self._inflight += 1

        def _apply() -> None:
            try:
                self.queue.put_nowait(event)
            finally:
                self._settle_one()
        try:
            self.loop.call_soon_threadsafe(_apply)
        except RuntimeError:  # loop closed: the stream is gone, nothing left to deliver to
            self._settle_one()

    def _settle_one(self) -> None:
        with self._inflight_lock:
            self._inflight -= 1

    async def flush(self) -> None:
        """Wait until every event a worker thread enqueued has reached the queue. Call it before
        the terminal frames: a commentary/delta/approval handed over before the worker returned
        otherwise lands after ``done``/``None`` and the reader never sees it."""
        while True:
            with self._inflight_lock:
                if not self._inflight:
                    return
            await _api_mod().asyncio.sleep(0)


class _SessionChatMixin:
    """Session chat, SSE streaming, and model lock HTTP handlers."""

    async def _prepare_session_chat(self, request: web.Request) -> tuple:
        """Shared prelude for /api/sessions/{id}/chat[/stream]: header/body validation, then
        runtime selection — a Browser model lock (body ``require_model_lock`` or a confirmed
        persisted lock) wins; else the session-persisted model routes via model_routes when an
        alias or threads through as ``session_model`` when raw, then body values.
        Returns ``(ctx, None)`` or ``(None, error)``; ``ctx["run_kwargs"]`` feeds ``_run_agent``."""
        gateway_session_key, key_err = self._parse_session_key_header(request)
        if key_err is not None:
            return None, key_err
        session_id = request.match_info["session_id"]
        session, err = await self._get_existing_session_or_404(session_id)
        if err:
            return None, err
        body, err = await self._read_json_body(request)
        if err:
            return None, err
        user_message, err = _api_mod()._session_chat_user_message(body)
        if err is not None:
            return None, err
        try:
            turn_author = _api_mod()._request_turn_author(body)
        except ValueError as exc:
            return None, _api_mod()._error_response(str(exc), 400, code="invalid_author")
        system_prompt = body.get("system_message") or body.get("instructions")
        if system_prompt is not None and not isinstance(system_prompt, str):
            return None, _api_mod()._error_response("system_message must be a string", 400, code="invalid_system_message")
        runtime_request = self._effective_session_runtime_request(session=session, body=body)
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return None, lock_error
        if not await _api_mod().asyncio.to_thread(self._persist_session_runtime_lock, session_id, runtime_request):
            return None, _api_mod()._error_response(
                "Could not persist the requested session model lock", 500, code="model_lock_persistence_failed")
        lock_active = bool(runtime_request.get("require_model_lock"))
        if lock_active:
            route = runtime_request.get("route")
            session_model = None
            requested = runtime_request.get("requested") or {}
            agent_overrides: dict[str, Any] = {}
            for src_key, dst_key in (("model", "requested_model"), ("provider", "requested_provider")):
                if requested.get(src_key):
                    agent_overrides[dst_key] = requested[src_key]
            if runtime_request.get("model_options"):
                agent_overrides["model_options"] = runtime_request["model_options"]
        else:
            stored_model = self._stored_session_model(session)
            stored_route = self._resolve_route(stored_model)
            route = stored_route or self._resolve_route(body.get("model"))
            session_model = stored_model if (stored_model and stored_route is None) else None
            agent_overrides = _api_mod()._request_agent_overrides(body, virtual_model=self._model_name)
            selection_error = self._request_route_conflict_error(
                session_id=session_id, gateway_session_key=gateway_session_key,
                requested_model=agent_overrides.get("requested_model"),
                requested_provider=agent_overrides.get("requested_provider"), route=route)
            if selection_error:
                return None, _api_mod()._error_response(selection_error, 400)
        run_kwargs = dict(
            user_message=user_message, ephemeral_system_prompt=system_prompt, session_id=session_id,
            gateway_session_key=gateway_session_key, route=route, session_model=session_model,
            requested_runtime=runtime_request.get("requested") or {},
            route_source=runtime_request.get("route_source") or "global",
            confirmed_runtime_lock=lock_active, turn_author=turn_author,
            # #98619: the client addresses this session by construction — the id is in the
            # request path (/api/sessions/{session_id}/chat) — so a wake self-post lands where
            # the client will read it. The audited native-session opt-in.
            session_history_delivery="1", **agent_overrides)
        return {
            "gateway_session_key": gateway_session_key, "session_id": session_id, "body": body,
            "user_message": user_message, "runtime_request": runtime_request,
            "lock_active": lock_active, "run_kwargs": run_kwargs}, None

    @staticmethod
    def _session_headers(session_id: str, gateway_session_key: Optional[str]) -> dict[str, str]:
        """``X-Hermes-Session-Id`` (+ ``X-Hermes-Session-Key`` when declared) response headers."""
        headers = {"X-Hermes-Session-Id": session_id}
        if gateway_session_key:
            headers["X-Hermes-Session-Key"] = gateway_session_key
        return headers

    def _effective_turn_runtime(self, runtime_request: dict[str, Any], result: Any, usage: Any) -> dict[str, Any]:
        """Sanitized runtime metadata for a finished session-chat turn."""
        runtime = self._result_runtime(result, usage)
        return self._sanitize_runtime_metadata(
            # Same shared ladder /api/status uses. Before this was unified, the two endpoints disagreed on
            # the same page load — the sidebar strip read "running" (it probed GATEWAY_HEALTH_URL and scoped
            # to the requested profile) while the Channels page rendered "The gateway is not running" (it
            # did neither). Cross-container, profile-scoped, and launch-service-managed deployments each hit
            # that split. profile_home is passed when the request was scoped to a named profile:
            # gateway/status readers resolve process-level paths and do NOT follow the HERMES_HOME
            # contextvar override (#56986 / #69143), so the profile's directory has to be handed over
            # explicitly or messaging silently reports another profile's gateway (#71211).
            runtime=runtime,
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=self._model_lock_state(runtime_request, runtime))

    @staticmethod
    def _result_runtime(result: Any, usage: Any) -> dict[str, Any]:
        """Runtime metadata from the result dict, falling back to the usage dict."""
        runtime = (result.get("runtime") or {}) if isinstance(result, dict) else {}
        return runtime or ((usage.get("runtime") or {}) if isinstance(usage, dict) else {})

    @staticmethod
    def _model_lock_state(runtime_request: dict[str, Any], runtime: Any) -> str:
        """``confirmed`` once a runtime was observed under a lock, ``accepted`` before, else ``""``."""
        if not runtime_request.get("require_model_lock"):
            return ""
        return "confirmed" if runtime else "accepted"

    async def _admit_to_live_bot_chat(
        self, session_id: str, message: Any, author: Optional[dict[str, Any]],
    ) -> Optional[tuple[Path, dict[str, Any]]]:
        """Admit a turn aimed at the canonical Bot Chat to the Desktop session that holds it live.

        ``(profile home, mailbox record)`` when a live owner took it; None when this process should
        run the turn itself — the session is not the canonical Bot Chat's own lineage, or nobody
        holds that chat. Both peer transports (``/api/sessions/{id}/chat`` for ``peer dm``,
        ``/v1/runs`` for ``peer run``) go through here, so the two lanes cannot drift.
        """
        if not isinstance(message, str):
            return None
        # Under session authority the Bot Chat is a LOCAL session whose only writer is the
        # authority FIFO (a Desktop showing it is a viewer, not a lease holder): the turn enters
        # through the same door as ``bot_relay.deliver`` and the receipt is its admission's.
        if getattr(self.gateway_runner, "session_authority", None) is not None:
            from gateway.platforms.api_server_bot_chat import admit_peer_turn
            try:
                return await admit_peer_turn(self.gateway_runner, session_id, message, author)
            except (ValueError, Exception) as exc:
                _api_mod().logger.debug("Failed to admit peer turn to session authority: %s", exc)
                return None
        db = await self._ensure_session_db_async()
        if db is None:
            return None
        home = Path(db.db_path).parent
        from tools.bot_live_delivery import deliver_to_live_owner, find_canonical_live_owner

        def _admit() -> Optional[dict[str, Any]]:
            try:
                owner = find_canonical_live_owner(home)
                # Only the canonical Bot Chat's own lineage: a peer turn into any other session runs here.
                if owner is None or db.get_compression_tip(session_id) != owner["session_id"]:
                    return None
                return deliver_to_live_owner(home, owner, message, author=author)
            except (ValueError, Exception) as exc:
                # No reachable profile authority to hand the chat to: this process runs the turn.
                _api_mod().logger.debug("Live bot chat admission skipped: %s", exc)
                return None

        try:
            record = await _api_mod().asyncio.to_thread(_admit)
        except (ValueError, Exception) as exc:
            _api_mod().logger.debug("Live bot chat admission thread error: %s", exc)
            return None
        return None if record is None else (home, record)

    async def _answer_through_live_bot_chat(self, ctx: dict[str, Any]) -> Optional[web.Response]:
        """Hand a turn aimed at a canonical Bot Chat that a Desktop holds live to that owner.

        This is the ``hermes peer dm`` transport. Running the turn here would make this process a
        second writer beside the lease holder: the open chat never shows the message or the reply,
        its live context never learns of them, and the two transcripts interleave in state.db.
        Local and relayed DMs already hand such a message to the owner's mailbox
        (``tools/bot_mode_dm.py``, ``tui_gateway/methods_bot_relay.py``). This waits for the owner's
        receipt on the same budget as the local path, so the peer still gets the reply on this call.
        """
        try:
            admitted = await self._admit_to_live_bot_chat(ctx["session_id"], ctx["user_message"], ctx["run_kwargs"].get("turn_author"))
        except (ValueError, Exception) as exc:
            _api_mod().logger.warning("Error admitting to live bot chat: %s, falling back to local run", exc)
            return None
        if admitted is None:
            return None
        session_id = admitted[1].get("session_id") or ctx["session_id"]  # a moved-aside API row: the local chat (MSG-21)
        record = await self._await_live_bot_chat_receipt(*admitted)
        delivery_id = record["delivery_id"]
        headers = self._session_headers(session_id, ctx["gateway_session_key"])
        if record["status"] == "settled":
            return web.json_response(
                {"object": "hermes.session.chat.completion", "session_id": session_id,
                 "message": {"role": "assistant", "content": record.get("reply") or ""},
                 "usage": {}, "runtime": {}, "delivery_id": delivery_id}, headers=headers)
        if record["status"] in ("queued", "claimed"):
            return web.json_response(
                {"object": "hermes.session.chat.queued", "session_id": session_id,
                 "status": record["status"], "delivery_id": delivery_id}, status=202, headers=headers)
        return _api_mod()._error_response(record.get("error") or f"Bot Chat delivery {record['status']}", 502,
                               code=record.get("reason") or record["status"], headers=headers)

    async def _await_live_bot_chat_receipt(self, home: Path, record: dict[str, Any], *, keepalive=None) -> dict[str, Any]:
        """Wait on the live Bot Chat's receipt (the authority admission's waiter, or the legacy owner
        mailbox through ``await_delivery_async``) until it settles or the local DM budget runs out;
        ``keepalive`` (async) is called every SSE keepalive interval so a streaming caller's proxy
        keeps the socket."""
        from gateway.platforms.api_server_bot_chat import await_live_delivery
        from tools.bot_mode_dm import _LIVE_WAIT_SECONDS
        deadline = _api_mod().time.monotonic() + _LIVE_WAIT_SECONDS
        while record["status"] in ("queued", "claimed"):
            remaining = deadline - _api_mod().time.monotonic()
            if remaining <= 0:
                break
            budget = remaining if keepalive is None else min(remaining, _api_mod().CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
            record = await await_live_delivery(self, home, record, budget)
            if keepalive is not None and record["status"] in ("queued", "claimed"):
                await keepalive()
        return record

    async def _stream_through_live_bot_chat(self, request: web.Request, ctx: dict[str, Any]) -> Optional[web.StreamResponse]:
        """``_answer_through_live_bot_chat`` for the SSE sibling route: the owner's settled receipt is
        the run's single ``assistant.completed`` event; a receipt still open at the budget is a
        ``run.queued`` event (the 202 shape), a failed one an ``error`` event carrying the reason."""
        try:
            admitted = await self._admit_to_live_bot_chat(ctx["session_id"], ctx["user_message"], ctx["run_kwargs"].get("turn_author"))
        except (ValueError, Exception) as exc:
            _api_mod().logger.warning("Error admitting to live bot chat in stream: %s, falling back to local run", exc)
            return None
        if admitted is None:
            return None
        session_id = admitted[1].get("session_id") or ctx["session_id"]  # a moved-aside API row: the local chat (MSG-21)
        events = _SessionEventQueue(session_id, f"run_{uuid.uuid4().hex}")
        response = await self._prepare_sse_response(request, session_id, ctx["gateway_session_key"])

        async def _write(name: str, payload: dict[str, Any]) -> None:
            name, payload = events.payload(name, payload)
            await response.write(_api_mod()._sse_frame(payload, event=name, ensure_ascii=False))

        async def _keepalive() -> None:
            await response.write(b": keepalive\n\n")

        try:
            await _write("run.started", {"user_message": {"role": "user", "content": ctx["user_message"]}, "runtime": {}})
            record = await self._await_live_bot_chat_receipt(*admitted, keepalive=_keepalive)
            delivery_id = record["delivery_id"]
            if record["status"] == "settled":
                message_id = f"msg_{uuid.uuid4().hex}"
                await _write("message.started", {"message": {"id": message_id, "role": "assistant"}})
                await _write("assistant.completed", {
                    "message_id": message_id, "content": record.get("reply") or "", "delivery_id": delivery_id, "runtime": {}})
                await _write("run.completed", {"message_id": message_id, "delivery_id": delivery_id, "usage": {}, "runtime": {}})
            elif record["status"] in ("queued", "claimed"):
                await _write("run.queued", {"status": record["status"], "delivery_id": delivery_id})
            else:
                await _write("error", {"message": record.get("error") or f"Bot Chat delivery {record['status']}",
                                       "code": record.get("reason") or record["status"], "delivery_id": delivery_id})
            await _write("done", {})
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            _api_mod().logger.info("Session SSE client disconnected while a live Bot Chat held the turn")
        return response

    class _TelegramMirrorCoordinator:
        """Per-turn coordinator for mirroring WebUI commentary and final replies to Telegram."""

        def __init__(
            self,
            adapter: "APIServerAdapter",
            *,
            session_id: str,
            loop: _api_mod().asyncio.AbstractEventLoop,
            enabled: bool,
        ) -> None:
            self._adapter = adapter
            self.session_id = session_id
            self.loop = loop
            self.enabled = bool(enabled)
            self.mirror_lock = _api_mod().asyncio.Lock()
            self.pending_mirror_tasks: list[_api_mod().asyncio.Task] = []
            self.sent_mirror_texts: set[str] = set()

        @staticmethod
        def _norm(text: str) -> str:
            return " ".join(text.split()).strip()

        async def _send_telegram_mirror(self, text: str) -> None:
            async with self.mirror_lock:
                try:
                    _runner, telegram = self._adapter._telegram_runner_and_adapter()
                    if not telegram:
                        return
                    db = self._adapter._ensure_session_db()
                    if not db:
                        return
                    if hasattr(db, "apply_telegram_topic_migration"):
                        db.apply_telegram_topic_migration()
                    binding = db.get_telegram_topic_binding_by_session(session_id=self.session_id)
                    if not binding or not binding.get("delivery_enabled", True):
                        return
                    chat_id = binding.get("chat_id")
                    if not chat_id:
                        return
                    thread_id = binding.get("thread_id") or binding.get("message_thread_id")
                    metadata = {"thread_id": str(thread_id)} if thread_id else {}
                    metadata["origin"] = "webui_mirror"
                    try:
                        await telegram.send(chat_id=str(chat_id), content=text, metadata=metadata)
                    except TypeError:
                        await telegram.send(chat_id=str(chat_id), text=text, metadata=metadata)
                except Exception:
                    _api_mod().logger.warning(
                        "Failed to mirror WebUI message to Telegram for session %s (non-fatal)",
                        self.session_id,
                        exc_info=True,
                    )

        def _schedule_telegram_mirror(self, text: str) -> None:
            if not self.enabled or not (isinstance(text, str) and text.strip()):
                return
            norm = self._norm(text)
            if not norm or norm in self.sent_mirror_texts:
                return
            self.sent_mirror_texts.add(norm)
            task = self.loop.create_task(self._send_telegram_mirror(text))
            self._adapter._track_background_task(task, tolerate_missing=True)
            self.pending_mirror_tasks.append(task)

        def enqueue_commentary(self, text: str) -> None:
            if not self.enabled or not (isinstance(text, str) and text.strip()):
                return
            try:
                running_loop = _api_mod().asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is self.loop:
                self._schedule_telegram_mirror(text)
            else:
                try:
                    self.loop.call_soon_threadsafe(self._schedule_telegram_mirror, text)
                except RuntimeError:
                    _api_mod().logger.debug(
                        "Event loop closed while scheduling Telegram commentary mirror for session %s",
                        self.session_id,
                        exc_info=True,
                    )

        async def drain(self) -> None:
            await _api_mod().asyncio.sleep(0)
            while self.pending_mirror_tasks:
                tasks = list(self.pending_mirror_tasks)
                self.pending_mirror_tasks.clear()
                await _api_mod().asyncio.gather(*tasks, return_exceptions=True)

        async def send_final(self, final_response: str) -> None:
            await self.drain()
            if not self.enabled or not (isinstance(final_response, str) and final_response.strip()):
                return
            norm = self._norm(final_response)
            if not norm or norm in self.sent_mirror_texts:
                return
            self.sent_mirror_texts.add(norm)
            await self._send_telegram_mirror(final_response)

    def _new_telegram_mirror(
        self,
        *,
        session_id: str,
        loop: _api_mod().asyncio.AbstractEventLoop,
        enabled: bool,
    ) -> "_TelegramMirrorCoordinator":
        return self._TelegramMirrorCoordinator(
            self,
            session_id=session_id,
            loop=loop,
            enabled=enabled,
        )

    @_admit_api_agent_request
    async def _handle_session_chat(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/chat — one synchronous agent turn (plus the delivery lanes'
        one bounded re-run of a transient failure; ``hermes peer dm`` is the client)."""
        from tools.bot_failure_reasons import RETRY_NONE, result_retry_action
        # This turn runs through _run_agent, so it already COUNTS toward the cap (#7483).
        # Spending the budget without checking it refused every other caller while never
        # refusing this route — and a fleet's cross-machine DMs all arrive here.
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited
        ctx, err = await self._prepare_session_chat(request)
        if err is not None:
            return err
        from gateway.platforms.api_server_openai_routes import _response_status
        handed_off = await self._answer_through_live_bot_chat(ctx)
        if handed_off is not None:
            return handed_off
        gateway_session_key = ctx["gateway_session_key"]
        session_id = ctx["session_id"]
        from gateway.wake import reset_consecutive_auto_wakes
        reset_consecutive_auto_wakes(session_id)
        await _api_mod().asyncio.to_thread(self._mark_session_auto_resume, session_id)
        db_turn = await self._ensure_session_db_async()
        if db_turn is not None and hasattr(db_turn, "set_session_read"):
            with suppress(Exception):
                await _api_mod().asyncio.to_thread(db_turn.set_session_read, session_id, True)
        ctx["run_kwargs"]["auto_resume_on_delegation"] = True
        duplicate_to_telegram = bool(
            ctx["body"].get("duplicate_to_telegram")
            if "duplicate_to_telegram" in ctx["body"]
            else self.is_telegram_bound(session_id)
        )
        loop = _api_mod().asyncio.get_running_loop()
        self._pulse_loop = loop
        self._pulse_prefs()[session_id] = duplicate_to_telegram
        mirror = self._new_telegram_mirror(session_id=session_id, loop=loop, enabled=duplicate_to_telegram)

        def _commentary(text: str, *, already_streamed: bool = False) -> None:
            mirror.enqueue_commentary(text)

        try:
            history = await self._conversation_history_for_session(session_id)
            result, usage = await self._run_agent(
                conversation_history=history,
                interim_assistant_callback=_commentary,
                **ctx["run_kwargs"],
            )
            # One policy-gated re-run of a transiently failed turn — the peer-DM transport's half of the
            # retry the local (``tools.bot_mode_dm``) and relayed (``tui_gateway.methods_bot_relay``)
            # delivery lanes already apply (#93091 item 5, #115325). Same policy, same gate: transient
            # classes (429 / 5xx) re-run the SAME session once, a context overflow lets the re-run's
            # pre-API compaction shrink the transcript first, and auth/quota/config/model never re-run. The
            # store is read again first: the failed attempt's turn-start persist left the DM as the
            # transcript's unanswered tail row, and the re-run resumes that row instead of appending a
            # second copy of it. A turn that fails again reaches the peer client exactly as before.
            if result_retry_action(result) != RETRY_NONE:
                history = await self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    conversation_history=history,
                    resume_unanswered_turn=True,
                    interim_assistant_callback=_commentary,
                    **ctx["run_kwargs"],
                )
        finally:
            try:
                await mirror.drain()
            except Exception:
                _api_mod().logger.debug("Telegram mirror drain failed in _handle_session_chat (non-fatal)", exc_info=True)
        is_dict = isinstance(result, dict)
        effective_session_id = result.get("session_id") if is_dict else session_id
        raw_final_response = result.get("final_response", "") if is_dict else ""
        final_response = self._resolve_message_media(raw_final_response)
        await mirror.send_final(raw_final_response)
        headers = self._session_headers(effective_session_id or session_id, gateway_session_key)
        return web.json_response(
            {"object": "hermes.session.chat.completion", "status": _response_status(result),
             "session_id": effective_session_id or session_id,
             "message": {"role": "assistant", "content": final_response}, "usage": usage,
             "runtime": self._effective_turn_runtime(ctx["runtime_request"], result, usage)},
            headers=headers)

    @_admit_api_agent_request
    async def _handle_session_chat_stream(self, request: web.Request) -> web.StreamResponse:
        """POST /api/sessions/{session_id}/chat/stream — SSE wrapper over _run_agent."""
        limited = self._concurrency_limited_response()
        if limited is not None:
            return limited
        ctx, err = await self._prepare_session_chat(request)
        if err is not None:
            return err
        handed_off = await self._stream_through_live_bot_chat(request, ctx)
        if handed_off is not None:
            return handed_off
        gateway_session_key, session_id = ctx["gateway_session_key"], ctx["session_id"]
        from gateway.wake import reset_consecutive_auto_wakes
        reset_consecutive_auto_wakes(session_id)
        await _api_mod().asyncio.to_thread(self._mark_session_auto_resume, session_id)
        db_turn = await self._ensure_session_db_async()
        if db_turn is not None and hasattr(db_turn, "set_session_read"):
            with suppress(Exception):
                await _api_mod().asyncio.to_thread(db_turn.set_session_read, session_id, True)
        ctx["run_kwargs"]["auto_resume_on_delegation"] = True
        user_message, runtime_request = ctx["user_message"], ctx["runtime_request"]
        runtime_meta = self._sanitize_runtime_metadata(
            requested_runtime=runtime_request.get("requested"),
            route_source=runtime_request.get("route_source") or "global",
            model_lock=("accepted" if ctx["lock_active"] else ""))
        message_id = f"msg_{uuid.uuid4().hex}"
        run_id = f"run_{uuid.uuid4().hex}"
        events = _SessionEventQueue(session_id, run_id)
        queue, _event_payload = events.queue, events.payload
        duplicate_to_telegram = bool(
            ctx["body"].get("duplicate_to_telegram") if "duplicate_to_telegram" in ctx["body"] else self.is_telegram_bound(session_id)
        )
        preserve_on_disconnect = bool(
            duplicate_to_telegram or self.is_telegram_bound(session_id) or ctx["body"].get("preserve_on_disconnect")
        )
        loop = _api_mod().asyncio.get_running_loop()
        self._pulse_loop = loop
        self._pulse_prefs()[session_id] = duplicate_to_telegram
        mirror = self._new_telegram_mirror(session_id=session_id, loop=loop, enabled=duplicate_to_telegram)

        async def _finalize_pulse_ticker(fallback_status: str) -> None:
            entry = self._pulse_registry().get(session_id)
            if entry is None:
                return
            stop_task = entry.get("stop_task")
            if stop_task is not None:
                try:
                    await stop_task
                except Exception:
                    _api_mod().logger.debug("Pulse ticker stop_task failed (non-fatal)", exc_info=True)
                return
            if entry.get("stopped"):
                return
            ticker = entry["ticker"]
            start_task = entry.get("start_task")
            if not ticker.active and not entry.get("starting") and (start_task is None or start_task.done()):
                return
            has_running = any(t.get("status") == "running" for t in ticker.tasks.values())
            is_draining = bool(getattr(self, "_draining", False))
            if fallback_status == "cancelled" or is_draining:
                final_status = "interrupted" if is_draining and fallback_status != "cancelled" else "cancelled"
                final_summary = entry.get("terminal_summary") or ""
                entry["stopped"] = True
                stop_task = self._spawn_tracked_task(
                    self._stop_pulse_entry(session_id, entry, final_status, final_summary)
                )
                entry["stop_task"] = stop_task
                try:
                    await stop_task
                except Exception:
                    _api_mod().logger.debug("Pulse ticker stop failed (non-fatal)", exc_info=True)
                return
            if has_running and fallback_status in ("completed", "failed"):
                return
            entry["stopped"] = True
            final_status = entry.get("terminal_status") or fallback_status
            final_summary = entry.get("terminal_summary") or ""
            stop_task = self._spawn_tracked_task(
                self._stop_pulse_entry(session_id, entry, final_status, final_summary)
            )
            entry["stop_task"] = stop_task
            try:
                await stop_task
            except Exception:
                _api_mod().logger.debug("Pulse ticker stop failed (non-fatal)", exc_info=True)

        # Claim ownership inside the request's profile scope before any run-keyed state
        # exists, so /v1/runs/{id}* control is confined to the starting profile.
        # See #93689.
        self._run_owners[run_id] = self._run_idempotency_scope(request)
        self._set_run_status(
            run_id, "queued", session_id=session_id, model=ctx["body"].get("model", self._model_name))

        def _delta(delta: str) -> None:
            if delta:
                events.enqueue("assistant.delta", {"message_id": message_id, "delta": delta})

        def _tool_progress(event_type: str, tool_name: str | None = None, preview: str | None = None, args=None, **kwargs) -> None:
            if event_type == "reasoning.available":
                events.enqueue("tool.progress", {"message_id": message_id, "tool_name": tool_name or "_thinking", "delta": preview or ""})
            elif event_type in {"tool.started", "tool.completed", "tool.failed"}:
                event_name = (
                    "tool.failed"
                    if event_type == "tool.completed" and kwargs.get("is_error")
                    else event_type
                )
                events.enqueue(event_name, {"message_id": message_id, "tool_name": tool_name, "preview": preview, "args": args})
            elif event_type in _api_mod()._SUBAGENT_PULSE_EVENTS or event_type in ("subagent.tool", "subagent_tool"):
                def _json_safe(val: Any) -> Any:
                    if val is None or isinstance(val, (str, int, float, bool)):
                        return val
                    if isinstance(val, dict):
                        return {str(k): _json_safe(v) for k, v in val.items()}
                    if isinstance(val, (list, tuple)):
                        return [_json_safe(v) for v in val]
                    return str(val)

                _subagent_event_map = {
                    "subagent_start": "subagent.start",
                    "subagent.start": "subagent.start",
                    "subagent_progress": "subagent.progress",
                    "subagent.progress": "subagent.progress",
                    "subagent_tool": "subagent.tool",
                    "subagent.tool": "subagent.tool",
                    "subagent_complete": "subagent.complete",
                    "subagent.complete": "subagent.complete",
                }
                normalized_event_name = _subagent_event_map.get(event_type, event_type)
                args_dict = args if isinstance(args, dict) else {}
                subagent_id = (
                    kwargs.get("subagent_id")
                    or args_dict.get("subagent_id")
                    or (f"task_{kwargs['task_index']}" if kwargs.get("task_index") is not None else None)
                )
                started_at = kwargs.get("started_at")
                if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
                    started_at = args_dict.get("started_at")
                if not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
                    started_at = None
                    try:
                        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
                        task_idx = kwargs.get("task_index")
                        with _active_subagents_lock:
                            rec = _active_subagents.get(subagent_id) if subagent_id else None
                            if rec is None:
                                for cand in _active_subagents.values():
                                    owner_sid = str(cand.get("owner_agent_session_id") or cand.get("owner_session_id") or "")
                                    parent_sid = str(cand.get("parent_session_id") or "")
                                    gw_key = str(cand.get("gateway_session_key") or "")
                                    sid_match = bool(session_id) and (
                                        owner_sid == session_id
                                        or parent_sid == session_id
                                        or gw_key == session_id
                                    )
                                    if not sid_match:
                                        continue
                                    if subagent_id and cand.get("subagent_id") == subagent_id:
                                        rec = cand
                                        break
                                    if task_idx is not None and cand.get("task_index") == task_idx:
                                        rec = cand
                                        break
                            if rec is not None:
                                cand_started = rec.get("started_at")
                                if isinstance(cand_started, (int, float)) and not isinstance(cand_started, bool):
                                    started_at = cand_started
                    except Exception:
                        pass
                now_ts = _api_mod().time.time()
                default_status = "completed" if normalized_event_name == "subagent.complete" else "running"
                subagent_payload: dict[str, Any] = {
                    "message_id": message_id,
                    "event_type": normalized_event_name,
                    "subagent_id": _json_safe(subagent_id),
                    "goal": _json_safe(
                        kwargs.get("goal")
                        or args_dict.get("goal")
                        or (preview if normalized_event_name == "subagent.start" else None)
                    ),
                    "tool_name": _json_safe(tool_name or kwargs.get("tool_name") or kwargs.get("action")),
                    "preview": _json_safe(preview or kwargs.get("summary") or kwargs.get("action")),
                    "args": _json_safe(args),
                    "status": _json_safe(kwargs.get("status") or default_status),
                    "timestamp": now_ts,
                }
                if isinstance(started_at, (int, float)) and not isinstance(started_at, bool):
                    subagent_payload["started_at"] = float(started_at)
                    subagent_payload["running_seconds"] = round(max(0.0, now_ts - float(started_at)), 1)
                elif isinstance(kwargs.get("running_seconds"), (int, float)) and not isinstance(kwargs.get("running_seconds"), bool):
                    subagent_payload["running_seconds"] = round(max(0.0, float(kwargs["running_seconds"])), 1)
                if "task_index" in kwargs:
                    subagent_payload["task_index"] = _json_safe(kwargs.get("task_index"))
                events.enqueue(normalized_event_name, subagent_payload)
                if event_type in _api_mod()._SUBAGENT_PULSE_EVENTS and getattr(self.gateway_runner, "session_authority", None) is None:
                    self.handle_session_subagent_pulse(
                        session_id,
                        event_type,
                        tool_name,
                        preview,
                        args,
                        **kwargs,
                    )

        def _commentary(text: str, *, already_streamed: bool = False) -> None:
            # Mid-turn assistant commentary (Codex ``phase="commentary"``, text beside tool calls)
            # as its own typed event — never folded into ``assistant.completed`` (#67580).
            if isinstance(text, str) and text.strip():
                events.enqueue("assistant.commentary", {
                    "message_id": message_id,
                    "text": self._resolve_message_media(text),
                    "already_streamed": bool(already_streamed),
                })
                mirror.enqueue_commentary(text)

        approval_notify = self._register_session_stream_approval(run_id, events, message_id)

        async def _run_and_signal() -> None:
            turn_cancelled = False
            turn_failed = False
            try:
                await queue.put(_event_payload("run.started", {
                    "user_message": {"role": "user", "content": user_message},
                    "runtime": runtime_meta,
                    "timestamp": _api_mod().time.time()}))
                self._set_run_status(run_id, "running", last_event="run.started")
                await queue.put(_event_payload("message.started", {
                    "message": {"id": message_id, "role": "assistant", "timestamp": _api_mod().time.time()},
                    "timestamp": _api_mod().time.time()}))
                history = await self._conversation_history_for_session(session_id)
                result, usage = await self._run_agent(
                    conversation_history=history, stream_delta_callback=_delta,
                    tool_progress_callback=_tool_progress, interim_assistant_callback=_commentary,
                    active_run_id=run_id, approval_notify_callback=approval_notify,
                    approval_session_key=run_id, **ctx["run_kwargs"])
                await events.flush()
                is_dict = isinstance(result, dict)
                raw_final_response = result.get("final_response", "") if is_dict else ""
                final_response = self._resolve_message_media(raw_final_response)
                effective_session_id = result.get("session_id", session_id) if is_dict else session_id
                turn_messages = self._turn_transcript_messages(history, user_message, result) if is_dict else []
                effective_runtime = self._effective_turn_runtime(runtime_request, result, usage)
                # Terminal status and flags come from the result (interrupted -> cancelled,
                # unfinished -> failed); a late steer rides along as ``pending_steer`` for replay.
                status, fields = _api_mod()._api_runs.terminal_run_status(result if is_dict else {})
                await mirror.drain()
                await _finalize_pulse_ticker(status or "completed")
                await mirror.send_final(raw_final_response)
                await queue.put(_event_payload("assistant.completed", {
                    "session_id": effective_session_id, "message_id": message_id,
                    "content": final_response, **fields, "runtime": effective_runtime,
                    "timestamp": _api_mod().time.time()}))
                await queue.put(_event_payload(f"run.{status}", {
                    "session_id": effective_session_id, "message_id": message_id, **fields,
                    "messages": turn_messages, "usage": usage, "runtime": effective_runtime}))
                self._set_run_status(
                    run_id, status, session_id=effective_session_id,
                    # The reply text, so a caller whose stream died can still read it from
                    # GET /v1/runs/{run_id}; POST /v1/runs already records output in `_finish`.
                    output=final_response, usage=usage,
                    last_event=f"run.{status}", **fields)
            except asyncio.CancelledError:
                turn_cancelled = True
                self._set_run_status(run_id, "cancelled", last_event="run.cancelled")
                raise
            except Exception as exc:
                turn_failed = True
                _api_mod().logger.exception("[api_server] session chat stream failed")
                self._set_run_status(
                    run_id, "failed", error=_api_mod()._redact_api_error_text(exc), last_event="run.failed")
                await queue.put(_event_payload("error", {"message": _api_mod()._redact_api_error_text(exc)}))
            finally:
                try:
                    await mirror.drain()
                except Exception:
                    _api_mod().logger.debug("Telegram mirror drain failed in finally (non-fatal)", exc_info=True)
                await _finalize_pulse_ticker("cancelled" if turn_cancelled else "failed" if turn_failed else "completed")
                self._unregister_active_agent(run_id=run_id, session_id=session_id)
                self._run_approval_sessions.pop(run_id, None)
                self._release_run_owner_if_forgotten(run_id)
                await events.flush()
                await queue.put(_event_payload("done", {}))
                await queue.put(None)

        # NOT in _active_run_tasks: _run_agent already counts this turn for the shutdown drain.
        task = self._spawn_tracked_task(_run_and_signal())
        response = await self._prepare_sse_response(request, session_id, gateway_session_key)
        try:
            while True:
                try:
                    item = await _api_mod().asyncio.wait_for(queue.get(), timeout=_api_mod().CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS)
                except TimeoutError:
                    await response.write(b": keepalive\n\n")
                    continue
                if item is None:
                    break
                name, payload = item
                await response.write(_api_mod()._sse_frame(payload, event=name, ensure_ascii=False))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            if not self._is_preserved(session_id, preserve_on_disconnect):
                await self._drain_session_stream_task_on_disconnect(
                    run_id, task, interrupt_message="SSE client disconnected", shield_wait=False)
                _api_mod().logger.info("Session SSE client disconnected; interrupted live run %s", run_id)
            else:
                _api_mod().logger.info("Session SSE client disconnected; preserved live run %s", run_id)
        except asyncio.CancelledError:
            if not self._is_preserved(session_id, preserve_on_disconnect):
                await self._drain_session_stream_task_on_disconnect(
                    run_id, task, interrupt_message="SSE task cancelled", shield_wait=True)
                _api_mod().logger.info("Session SSE task cancelled; drained live run %s", run_id)
            else:
                _api_mod().logger.info("Session SSE task cancelled; preserved live run %s", run_id)
            raise
        except Exception as exc:
            _api_mod().logger.debug("[api_server] session SSE stream error: %s", exc)
        return response

    def _register_session_stream_approval(self, run_id: str, events: Any, message_id: str):
        """Route a session-stream turn's dangerous-command approvals to its SSE queue and to
        ``POST /v1/runs/{run_id}/approval`` (#58856). Keyed by the run id (never the shared
        session key) so concurrent turns on one session can't cross-resolve."""
        self._run_approval_sessions[run_id] = run_id

        def _approval_notify(approval_data: dict[str, Any]) -> None:
            event = _api_mod()._approval_request_event(run_id, approval_data, message_id=message_id)
            self._set_run_status(run_id, "waiting_for_approval", last_event="approval.request", approval=event)
            events.enqueue("approval.request", event)  # executor thread -> loop hop inside
        return _approval_notify

    async def _drain_session_stream_task_on_disconnect(
        self, run_id: str, task: _api_mod().asyncio.Task, *, interrupt_message: str, shield_wait: bool
    ) -> None:
        """Preserve live run control refs until the executor-backed turn actually exits."""
        agent = self._active_run_agents.get(run_id)
        if agent is None:
            if not task.done():
                task.cancel()
                with suppress(Exception):
                    await task
            return
        with suppress(Exception):
            agent.interrupt(interrupt_message)
        if not task.done():
            with suppress(Exception):
                await (_api_mod().asyncio.shield(task) if shield_wait else task)

    @_require_auth
    async def _handle_session_model_lock(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/model — backend-ack a Browser model lock."""
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        body, err = await self._read_json_body(request)
        if err:
            return err
        runtime_request = self._session_runtime_request_from_body(body)
        runtime_request["require_model_lock"] = True
        lock_error = self._runtime_lock_error(runtime_request)
        if lock_error is not None:
            return lock_error
        if not await _api_mod().asyncio.to_thread(self._persist_session_runtime_lock, session_id, runtime_request):
            return _api_mod()._error_response(
                "Could not persist the requested session model lock", 500, code="model_lock_persistence_failed")
        requested = runtime_request.get("requested") or {}
        route = runtime_request.get("route") or {}
        runtime = self._sanitize_runtime_metadata(
            runtime={
                "provider": route.get("provider") or requested.get("provider") or "",
                "model": route.get("model") or requested.get("model") or "",
                "route_source": runtime_request.get("route_source") or "raw_request"},
            requested_runtime=requested,
            route_source=runtime_request.get("route_source") or "raw_request",
            model_lock="accepted")
        return web.json_response(
            {"object": "hermes.session.model_lock", "session_id": session_id, "runtime": runtime})
