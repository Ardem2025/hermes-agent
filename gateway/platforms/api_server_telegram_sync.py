"""Telegram topic binding, history sync, pulse ticker, and per-session coordinator."""

from __future__ import annotations

import asyncio
import sys
from collections import OrderedDict
from functools import wraps
from typing import Any, Dict, Optional

from gateway.response_filters import (
    is_intentional_silence_response,
    is_machinery_display_kind,
    strip_trailing_silence_marker,
)

_TELEGRAM_SYNC_MACHINERY_MARKERS = (
    "[ASYNC DELEGATION BATCH COMPLETE",
    "[ASYNC DELEGATION COMPLETE",
    "[INTERNAL NOTIFICATION",
    "[CONTEXT COMPACTION",
)


def _internal_notification_footer() -> str:
    from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER

    return INTERNAL_NOTIFICATION_FOOTER


def _message_display_kind(message: Any) -> Optional[str]:
    if not isinstance(message, dict):
        return None
    display_kind = message.get("display_kind")
    if display_kind:
        return str(display_kind)
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        meta_kind = metadata.get("display_kind")
        if meta_kind:
            return str(meta_kind)
    return None


def _should_skip_telegram_sync_message(message: Any, text: Optional[str]) -> bool:
    if not isinstance(message, dict):
        return True
    role = str(message.get("role") or "").strip().lower()
    if role not in {"user", "assistant"}:
        return True
    if is_machinery_display_kind(_message_display_kind(message)):
        return True
    if text is None or not str(text).strip():
        return True
    text_str = str(text)
    if any(marker in text_str for marker in _TELEGRAM_SYNC_MACHINERY_MARKERS):
        return True
    footer = _internal_notification_footer()
    if footer and footer in text_str:
        return True
    if role == "assistant" and is_intentional_silence_response(text_str):
        return True
    return False

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


class _SessionCoordinator:
    """Unified per-session coordinator for active turns, agent refs, and Telegram sync locks."""

    def __init__(self, adapter: Any, *, max_sync_locks: int = 1024) -> None:
        self._adapter = adapter
        self._max_sync_locks = max_sync_locks
        self.telegram_sync_locks: OrderedDict[str, _api_mod().asyncio.Lock] = OrderedDict()
        self.active_agents_by_id: dict[int, Any] = {}
        self.session_runs: dict[str, set[str]] = {}
        self.run_sessions: dict[str, str] = {}

    def get_telegram_sync_lock(self, session_id: str) -> _api_mod().asyncio.Lock:
        locks = getattr(self._adapter, "_telegram_sync_locks", self.telegram_sync_locks)
        lock = locks.get(session_id)
        if lock is not None:
            if isinstance(locks, OrderedDict):
                locks.move_to_end(session_id)
            else:
                locks.pop(session_id, None)
                locks[session_id] = lock
            return lock
        api = _api_mod()
        lock = api.asyncio.Lock()
        locks[session_id] = lock
        max_locks = getattr(api, "MAX_TELEGRAM_SYNC_LOCKS", self._max_sync_locks)
        if len(locks) > max_locks:
            for stale_sid in list(locks.keys()):
                if len(locks) <= max_locks:
                    break
                if stale_sid == session_id:
                    continue
                candidate = locks.get(stale_sid)
                if candidate is not None and not candidate.locked():
                    locks.pop(stale_sid, None)
        return lock

    def register_active_agent(
        self, agent: Any, run_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> None:
        if agent is None:
            return
        registry = getattr(self._adapter, "_active_agents_by_id", None)
        if not isinstance(registry, dict):
            registry = self.active_agents_by_id
            self._adapter._active_agents_by_id = registry
        registry[id(agent)] = agent
        shutdown_reg = getattr(self._adapter, "_shutdown_interruptible_agents", None)
        if isinstance(shutdown_reg, dict) and shutdown_reg is not registry:
            shutdown_reg[id(agent)] = agent
        if run_id:
            self._adapter._active_run_agents[run_id] = agent
            if session_id:
                self.run_sessions[run_id] = session_id
                self.session_runs.setdefault(session_id, set()).add(run_id)

    def unregister_active_agent(
        self, agent: Any = None, run_id: Optional[str] = None, session_id: Optional[str] = None
    ) -> None:
        if run_id:
            popped = self._adapter._active_run_agents.pop(run_id, None)
            if agent is None:
                agent = popped
            sid = session_id or self.run_sessions.pop(run_id, None)
            if sid and sid in self.session_runs:
                self.session_runs[sid].discard(run_id)
                if not self.session_runs[sid]:
                    self.session_runs.pop(sid, None)
        if agent is not None:
            still_in_run = any(a is agent for a in self._adapter._active_run_agents.values())
            if not still_in_run:
                registry = getattr(self._adapter, "_active_agents_by_id", None)
                if isinstance(registry, dict):
                    registry.pop(id(agent), None)
                shutdown_reg = getattr(self._adapter, "_shutdown_interruptible_agents", None)
                if isinstance(shutdown_reg, dict) and shutdown_reg is not registry:
                    shutdown_reg.pop(id(agent), None)

    def has_active_turn(self, session_id: str) -> bool:
        return bool(self.session_runs.get(session_id))

    def cleanup_session(self, session_id: str) -> None:
        locks = getattr(self._adapter, "_telegram_sync_locks", self.telegram_sync_locks)
        if isinstance(locks, dict):
            lock = locks.get(session_id)
            if lock is not None and not lock.locked():
                locks.pop(session_id, None)
        if not self.session_runs.get(session_id):
            self.session_runs.pop(session_id, None)


class _TelegramSyncMixin:
    """Telegram topic binding, history sync, and subagent pulse handlers."""

    def _get_telegram_sync_lock(self, session_id: str) -> _api_mod().asyncio.Lock:
        coord = getattr(self, "_session_coordinator", None)
        if coord is not None:
            return coord.get_telegram_sync_lock(session_id)
        locks = self._telegram_sync_locks
        lock = locks.get(session_id)
        if lock is not None:
            if isinstance(locks, OrderedDict):
                locks.move_to_end(session_id)
            else:
                locks.pop(session_id, None)
                locks[session_id] = lock
            return lock
        lock = _api_mod().asyncio.Lock()
        locks[session_id] = lock
        max_locks = _api_mod().MAX_TELEGRAM_SYNC_LOCKS
        if len(locks) > max_locks:
            for stale_sid in list(locks.keys()):
                if len(locks) <= max_locks:
                    break
                if stale_sid == session_id:
                    continue
                candidate = locks.get(stale_sid)
                if candidate is not None and not candidate.locked():
                    locks.pop(stale_sid, None)
        return lock

    def _telegram_delivery_enabled(self, session_id: str) -> bool:
        """Return the binding's durable delivery preference (unbound is off)."""
        db = self._ensure_session_db()
        if not db:
            return False
        try:
            if hasattr(db, "apply_telegram_topic_migration"):
                db.apply_telegram_topic_migration()
            binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
        except Exception:
            _api_mod().logger.debug("Failed to get Telegram delivery state for session %s", session_id, exc_info=True)
            return False
        return bool(binding and binding.get("delivery_enabled", True))

    def is_telegram_bound(self, session_id: str) -> bool:
        """Return True if session has an active Telegram topic binding with delivery enabled."""
        return self._telegram_delivery_enabled(session_id)

    def _pulse_registry(self) -> dict[str, dict[str, Any]]:
        reg = getattr(self, "_session_pulse_tickers", None)
        if reg is None:
            reg = {}
            setattr(self, "_session_pulse_tickers", reg)
        return reg

    def _pulse_locks(self) -> dict[str, _api_mod().asyncio.Lock]:
        locks = getattr(self, "_session_pulse_locks", None)
        if locks is None:
            locks = {}
            setattr(self, "_session_pulse_locks", locks)
        return locks

    def _pulse_lock_for(self, session_id: str) -> _api_mod().asyncio.Lock:
        locks = self._pulse_locks()
        lock = locks.get(session_id)
        if lock is None:
            lock = _api_mod().asyncio.Lock()
            locks[session_id] = lock
        return lock

    def _pulse_prefs(self) -> dict[str, bool]:
        prefs = getattr(self, "_session_pulse_prefs", None)
        if prefs is None:
            prefs = {}
            setattr(self, "_session_pulse_prefs", prefs)
        return prefs

    def _ensure_pulse_loop(self) -> Optional[_api_mod().asyncio.AbstractEventLoop]:
        loop = getattr(self, "_pulse_loop", None)
        if loop is not None and not loop.is_closed():
            return loop
        try:
            loop = _api_mod().asyncio.get_running_loop()
        except RuntimeError:
            return None
        setattr(self, "_pulse_loop", loop)
        return loop

    def _resolve_telegram_user_config(self) -> dict[str, Any]:
        runner = getattr(self, "gateway_runner", None)
        user_config = getattr(runner, "user_config", None) if runner is not None else None
        if isinstance(user_config, dict):
            return user_config
        try:
            from gateway.run import _load_gateway_config
            loaded = _load_gateway_config()
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            _api_mod().logger.debug("Failed to load gateway config for telegram display settings", exc_info=True)
        return {}

    def _resolve_telegram_cleanup_progress(self, telegram_adapter: Any) -> bool:
        if telegram_adapter is None:
            return False
        has_real_delete = getattr(type(telegram_adapter), "delete_message", None) not in (
            None,
            _api_mod().BasePlatformAdapter.delete_message,
        )
        if not has_real_delete:
            return False
        user_config = self._resolve_telegram_user_config()
        return bool(_api_mod().resolve_display_setting(user_config, "telegram", "cleanup_progress"))

    def _resolve_telegram_display_settings(self, telegram_adapter: Any) -> dict[str, Any]:
        user_config = self._resolve_telegram_user_config()
        resolve_setting = getattr(_api_mod(), "resolve_display_setting", None)
        if not callable(resolve_setting):
            from gateway.display_config import resolve_display_setting as resolve_setting
        resolve_tp = getattr(_api_mod(), "resolve_tool_progress", None)
        if not callable(resolve_tp):
            from gateway.display_config import resolve_tool_progress as resolve_tp
        tool_progress_mode, _ = resolve_tp(user_config, "telegram")
        return {
            "tool_progress": str(tool_progress_mode or "all"),
            "tool_progress_details": bool(resolve_setting(user_config, "telegram", "tool_progress_details", True)),
            "friendly_tool_labels": bool(resolve_setting(user_config, "telegram", "friendly_tool_labels", True)),
            "tool_preview_length": int(resolve_setting(user_config, "telegram", "tool_preview_length", 0) or 0),
            "tool_progress_grouping": str(resolve_setting(user_config, "telegram", "tool_progress_grouping", "accumulate") or "accumulate"),
            "cleanup_progress": self._resolve_telegram_cleanup_progress(telegram_adapter),
        }

    def _prune_stale_pulse_entries(self, now: Optional[float] = None) -> None:
        tickers = self._pulse_registry()
        locks = self._pulse_locks()
        prefs = self._pulse_prefs()
        ts = _api_mod().time.monotonic() if now is None else now
        stale_sids: list[str] = []
        for sid, entry in list(tickers.items()):
            last_at = float(entry.get("last_event_at") or 0.0)
            if ts - last_at <= _api_mod()._SESSION_PULSE_TTL_SECONDS:
                continue
            ticker = entry.get("ticker")
            is_active = bool(getattr(ticker, "active", False))
            if not is_active or (ts - last_at > _api_mod()._SESSION_PULSE_TTL_SECONDS):
                stale_sids.append(sid)
        for sid in stale_sids:
            entry = tickers.pop(sid, None)
            if entry is not None:
                start_task = entry.get("start_task")
                if start_task is not None and not start_task.done():
                    start_task.cancel()
                ticker = entry.get("ticker")
                if ticker is not None and getattr(ticker, "active", False):
                    try:
                        self._spawn_tracked_task(ticker.stop(status="interrupted"), tolerate_missing=True)
                    except Exception:
                        pass
            lock = locks.get(sid)
            if lock is not None and not lock.locked():
                locks.pop(sid, None)
            prefs.pop(sid, None)
        for sid in list(locks.keys()):
            if sid not in tickers and not locks[sid].locked():
                locks.pop(sid, None)
        for sid in list(prefs.keys()):
            if sid not in tickers:
                # Keep explicit prefs only while session has an active entry or recent turn;
                # stale prefs without entries are safe to prune when locks are gone.
                if sid not in locks:
                    pass

    async def _start_pulse_entry(
        self,
        session_id: str,
        target_entry: dict[str, Any],
        prev_stop_task: Optional[_api_mod().asyncio.Task],
    ) -> None:
        try:
            async with self._pulse_lock_for(session_id):
                if prev_stop_task is not None and not prev_stop_task.done():
                    try:
                        await prev_stop_task
                    except Exception:
                        _api_mod().logger.debug("Previous pulse ticker stop_task failed (non-fatal)", exc_info=True)
                await target_entry["ticker"].start()
        except Exception:
            _api_mod().logger.debug("Failed to start SubagentPulseTicker for session %s (non-fatal)", session_id, exc_info=True)
        finally:
            target_entry["starting"] = False

    async def _stop_pulse_entry(
        self,
        session_id: str,
        target_entry: dict[str, Any],
        status: str,
        summary: str,
        *,
        force: bool = False,
    ) -> None:
        try:
            async with self._pulse_lock_for(session_id):
                start_task = target_entry.get("start_task")
                if start_task is not None and not start_task.done():
                    try:
                        await start_task
                    except Exception:
                        _api_mod().logger.debug("Pulse ticker start_task failed before stop (non-fatal)", exc_info=True)
                ticker = target_entry["ticker"]
                if ticker.active or ticker.ticker_task is not None:
                    if force:
                        await ticker.stop(status=status, summary=summary, force=True)
                    else:
                        await ticker.stop(status=status, summary=summary)
        except Exception:
            _api_mod().logger.debug("Failed to stop SubagentPulseTicker for session %s (non-fatal)", session_id, exc_info=True)
        finally:
            tickers = self._pulse_registry()
            if tickers.get(session_id) is target_entry:
                tickers.pop(session_id, None)

    def _apply_pulse_event_on_loop(
        self,
        session_id: str,
        event_type: str,
        tool_name: str | None,
        preview: str | None,
        args: Any,
        kwargs: dict[str, Any],
    ) -> None:
        if not session_id or getattr(self, "_disconnected", False):
            return
        source_platform = kwargs.get("source_platform")
        if kwargs.get("telegram_pulse_owned") or source_platform in (
            _api_mod().Platform.TELEGRAM,
            _api_mod().Platform.TELEGRAM.value,
            "telegram",
        ):
            return
        prefs = self._pulse_prefs()
        if session_id in prefs:
            allowed = bool(prefs[session_id])
        else:
            allowed = bool(self.is_telegram_bound(session_id))
        if not allowed:
            return

        now = _api_mod().time.monotonic()
        self._prune_stale_pulse_entries(now)
        tickers = self._pulse_registry()
        task_idx = kwargs.get("task_index", 0)
        delegation_id = kwargs.get("delegation_id")
        subagent_id = kwargs.get("subagent_id")

        if event_type == "subagent.start":
            entry = tickers.get(session_id)
            need_new_ticker = (
                entry is None
                or entry["stopped"]
                or (not entry["ticker"].active and not entry["starting"])
            )
            if need_new_ticker:
                try:
                    _runner, telegram = self._telegram_runner_and_adapter()
                    if not telegram:
                        return
                    display_settings = self._resolve_telegram_display_settings(telegram)
                    if display_settings["tool_progress"] in ("off", "log"):
                        return
                    db = self._ensure_session_db()
                    if not db:
                        return
                    if hasattr(db, "apply_telegram_topic_migration"):
                        db.apply_telegram_topic_migration()
                    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
                    if not binding or not binding.get("delivery_enabled", True):
                        return
                    chat_id = binding.get("chat_id")
                    if not chat_id:
                        return
                    thread_id = binding.get("thread_id") or binding.get("message_thread_id")
                    metadata = {"thread_id": str(thread_id)} if thread_id else {}
                    cleanup_progress = display_settings["cleanup_progress"]
                    from gateway.subagent_pulse_ticker import SubagentPulseTicker
                    ticker = SubagentPulseTicker(
                        adapter=telegram,
                        chat_id=chat_id,
                        thread_id=thread_id,
                        metadata=metadata,
                        cleanup_progress=cleanup_progress,
                        tool_progress_details=display_settings["tool_progress_details"],
                        friendly_tool_labels=display_settings["friendly_tool_labels"],
                        tool_preview_length=display_settings["tool_preview_length"],
                        tool_progress_grouping=display_settings["tool_progress_grouping"],
                    )
                except Exception:
                    _api_mod().logger.debug("Failed to initialize SubagentPulseTicker for session %s", session_id, exc_info=True)
                    return
                prev_stop_task = entry.get("stop_task") if entry is not None else None
                entry = {
                    "ticker": ticker,
                    "starting": True,
                    "stopped": False,
                    "start_task": None,
                    "stop_task": None,
                    "terminal_status": None,
                    "terminal_summary": "",
                    "last_event_at": now,
                    "cleanup_progress": cleanup_progress,
                }
                tickers[session_id] = entry
                entry["start_task"] = self._spawn_tracked_task(
                    self._start_pulse_entry(session_id, entry, prev_stop_task),
                    tolerate_missing=True,
                )
            else:
                entry["last_event_at"] = now
            goal = kwargs.get("goal") or preview or ""
            transcript = kwargs.get("live_transcript") or kwargs.get("transcript_path")
            entry["ticker"].on_subagent_start(
                task_idx,
                goal,
                transcript,
                delegation_id=delegation_id,
                subagent_id=subagent_id,
            )
        elif event_type in ("subagent.progress", "subagent_progress"):
            entry = tickers.get(session_id)
            if entry is None or entry["stopped"]:
                return
            entry["last_event_at"] = now
            action = kwargs.get("action") or preview or tool_name or ""
            entry["ticker"].on_subagent_progress(
                task_idx,
                action,
                delegation_id=delegation_id,
                subagent_id=subagent_id,
            )
        elif event_type == "subagent.complete":
            entry = tickers.get(session_id)
            if entry is None or entry["stopped"]:
                return
            entry["last_event_at"] = now
            status = kwargs.get("status") or "completed"
            summary = kwargs.get("summary") or preview
            ticker = entry["ticker"]
            ticker.on_subagent_complete(
                task_idx,
                status,
                summary,
                delegation_id=delegation_id,
                subagent_id=subagent_id,
            )
            active_left = any(t.get("status") == "running" for t in ticker.tasks.values())
            if not active_left:
                failed_task = next(
                    (t for t in ticker.tasks.values() if t.get("status") in _api_mod()._SUBAGENT_FAILURE_STATUSES),
                    None,
                )
                if failed_task is not None:
                    terminal_status = str(failed_task.get("status"))
                    terminal_summary = str(failed_task.get("summary") or summary or "")
                else:
                    terminal_status = status if status in _api_mod()._SUBAGENT_FAILURE_STATUSES else "completed"
                    terminal_summary = str(summary or "")
                entry["terminal_status"] = terminal_status
                entry["terminal_summary"] = terminal_summary
                entry["stopped"] = True
                if (ticker.active or entry["starting"]) and entry["stop_task"] is None:
                    entry["stop_task"] = self._spawn_tracked_task(
                        self._stop_pulse_entry(session_id, entry, terminal_status, terminal_summary),
                        tolerate_missing=True,
                    )

    def handle_session_subagent_pulse(
        self,
        session_id: str,
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        args: Any = None,
        **kwargs,
    ) -> None:
        """Thread-safe session-level sink for subagent.* pulse events."""
        if not session_id or event_type not in _api_mod()._SUBAGENT_PULSE_EVENTS:
            return
        if getattr(self, "_disconnected", False):
            return
        loop = self._ensure_pulse_loop()
        if loop is None or loop.is_closed():
            return
        try:
            running_loop = _api_mod().asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        payload_kwargs = dict(kwargs)
        if running_loop is loop:
            self._apply_pulse_event_on_loop(session_id, event_type, tool_name, preview, args, payload_kwargs)
        else:
            try:
                loop.call_soon_threadsafe(
                    self._apply_pulse_event_on_loop,
                    session_id,
                    event_type,
                    tool_name,
                    preview,
                    args,
                    payload_kwargs,
                )
            except RuntimeError:
                _api_mod().logger.debug("Pulse loop closed before scheduling event for %s", session_id, exc_info=True)

    def _is_preserved(self, session_id: str, initial_flag: bool) -> bool:
        """Dynamic disconnect preservation: initial flag or dynamic Telegram binding check."""
        if initial_flag:
            return True
        try:
            return bool(self.is_telegram_bound(session_id))
        except Exception:
            return initial_flag

    def _user_facing_message_text(self, message: Any) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            return "\n".join(parts)
        return str(content) if content is not None else None

    def _telegram_sync_text(self, message: Any, text: Optional[str]) -> str:
        res = text or ""
        if _message_display_kind(message) == "failed_turn":
            return f"⚠️ {res}"
        return res

    async def _forward_to_telegram(self, session_id: str, role: str, content: Any) -> Dict[str, Any]:
        """Forward a web turn to its bound Telegram topic."""
        db = self._ensure_session_db()
        if not db:
            return {"status": "unavailable", "error": "session_db_unavailable"}
        try:
            binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
        except Exception as e:
            _api_mod().logger.debug("Failed to get Telegram topic binding for session %s: %s", session_id, e)
            return {"status": "failed", "error": "binding_lookup_failed"}

        if not binding:
            return {"status": "unbound"}

        chat_id = binding.get("chat_id")
        thread_id = binding.get("thread_id")
        if not chat_id or not thread_id:
            return {"status": "invalid_binding", "error": "missing_chat_or_thread"}

        _, telegram_adapter = self._telegram_runner_and_adapter()
        if not telegram_adapter:
            return {"status": "unavailable", "error": "telegram_not_connected"}

        if not content:
            return {"status": "skipped", "reason": "empty_content"}

        if role == "user":
            formatted_content = f"👤 **User:**\n{content}"
        else:
            formatted_content = content

        try:
            result = await telegram_adapter.send(
                chat_id=str(chat_id),
                content=formatted_content,
                metadata={"thread_id": str(thread_id)}
            )
            if getattr(result, "success", True) is False:
                return {"status": "failed", "error": str(getattr(result, "error", "telegram_send_failed"))}
            return {"status": "sent", "chat_id": str(chat_id), "thread_id": str(thread_id)}
        except Exception as e:
            _api_mod().logger.exception("Failed to send forwarded message to Telegram chat %s, thread %s: %s", chat_id, thread_id, e)
            return {"status": "failed", "error": str(e)[:500]}

    async def _sync_telegram_history(self, session_id: str) -> Dict[str, Any]:
        """Incrementally deliver a bound transcript and durably checkpoint rows."""
        lock = self._get_telegram_sync_lock(session_id)
        async with lock:
            delivery = {"mode": "incremental", "status": "completed", "sent": 0, "failed": 0, "skipped": 0}
            db = self._ensure_session_db()
            if not db:
                delivery.update(status="not_requested")
                return delivery
            binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
            if not binding or not binding.get("delivery_enabled", True):
                delivery.update(status="not_requested")
                return delivery
            checkpoint = binding.get("last_synced_message_id")
            resolved_id = db.resolve_resume_session_id(session_id) if hasattr(db, "resolve_resume_session_id") else session_id
            raw_messages = db.get_messages(resolved_id) if hasattr(db, "get_messages") else []

            if checkpoint is None:
                max_id = max((m.get("id") for m in raw_messages if isinstance(m.get("id"), int)), default=None)
                if max_id is not None and hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=max_id)
                delivery["skipped"] = len(raw_messages)
                return delivery

            for message in raw_messages:
                message_id = message.get("id")
                if not isinstance(message_id, int) or (checkpoint is not None and message_id <= checkpoint):
                    continue
                if message.get("platform_message_id"):
                    delivery["skipped"] += 1
                    if hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                        db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=message_id)
                    continue

                text = self._user_facing_message_text(message)
                if _should_skip_telegram_sync_message(message, text):
                    delivery["skipped"] += 1
                    if hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                        db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=message_id)
                    continue

                role = str(message.get("role") or "").strip().lower()
                if role == "assistant":
                    text = strip_trailing_silence_marker(text)
                    if not str(text or "").strip():
                        delivery["skipped"] += 1
                        if hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                            db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=message_id)
                        continue

                result = await self._forward_to_telegram(
                    session_id, role, self._telegram_sync_text(message, text)
                )
                if result.get("status") in {"sent", "skipped"}:
                    delivery["sent" if result.get("status") == "sent" else "skipped"] += 1
                    if hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                        db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=message_id)
                    continue
                delivery["failed"] += 1
                delivery["status"] = "partial" if delivery["sent"] or delivery["skipped"] else "failed"
                break
            return delivery

    def _telegram_runner_and_adapter(self) -> tuple[Optional[Any], Optional[Any]]:
        """Return the live gateway runner and Telegram adapter, if available."""
        try:
            from gateway.run import _gateway_runner_ref
            runner = _gateway_runner_ref()
            telegram = runner.adapters.get(_api_mod().Platform.TELEGRAM) if runner and hasattr(runner, "adapters") else None
            return runner, telegram
        except Exception:
            _api_mod().logger.debug("Telegram gateway lookup failed", exc_info=True)
            return None, None

    async def _sync_telegram_topic_title(self, session_id: str, title: str) -> None:
        """Best-effort sync of a manual WebUI session rename to its bound Telegram DM topic."""
        clean_title = str(title or "").strip()
        if not clean_title:
            return
        db = await self._ensure_session_db_async()
        if db is None:
            return
        get_binding = getattr(db, "get_telegram_topic_binding_by_session", None)
        if not callable(get_binding):
            return
        try:
            apply_migration = getattr(db, "apply_telegram_topic_migration", None)
            if callable(apply_migration):
                await _api_mod().asyncio.to_thread(apply_migration)
            binding = await _api_mod().asyncio.to_thread(get_binding, session_id=session_id)
        except Exception:
            _api_mod().logger.debug("Telegram topic binding lookup failed for session %s", session_id, exc_info=True)
            return
        if not isinstance(binding, dict):
            return
        chat_id = str(binding.get("chat_id") or "").strip()
        thread_id = str(binding.get("thread_id") or "").strip()
        if not chat_id or not thread_id or thread_id == "1":
            return
        _, telegram = self._telegram_runner_and_adapter()
        if telegram is None:
            return
        sanitize_fn = getattr(telegram, "_sanitize_topic_title_for_db", None)
        sanitized = sanitize_fn(clean_title) if callable(sanitize_fn) else clean_title
        topic_name = sanitized if isinstance(sanitized, str) else clean_title
        if not topic_name:
            return
        try:
            rename_fn = getattr(telegram, "rename_dm_topic", None)
            if callable(rename_fn):
                await rename_fn(chat_id=chat_id, thread_id=thread_id, name=topic_name)
                return
            bot = getattr(telegram, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) or getattr(bot, "editForumTopic", None)
            if not callable(edit_forum_topic):
                return
            try:
                chat_id_arg: Any = int(chat_id)
            except (TypeError, ValueError):
                chat_id_arg = chat_id
            try:
                thread_id_arg: Any = int(thread_id)
            except (TypeError, ValueError):
                thread_id_arg = thread_id
            await edit_forum_topic(chat_id=chat_id_arg, message_thread_id=thread_id_arg, name=topic_name)
            record_echo = getattr(telegram, "_record_bot_topic_rename", None)
            if callable(record_echo):
                record_echo(chat_id, thread_id, topic_name)
        except Exception as exc:
            _api_mod().logger.warning(
                "Failed to sync WebUI rename to Telegram topic (session=%s, chat_id=%s, thread_id=%s): %s",
                session_id, chat_id, thread_id, exc,
            )

    @staticmethod
    def _telegram_topic_link(chat_id: str, thread_id: str) -> Optional[str]:
        """Return Telegram's stable forum-topic URL when the chat id permits it."""
        chat = str(chat_id)
        if chat.startswith("-100") and len(chat) > 4:
            return f"https://t.me/c/{chat[4:]}/{thread_id}"
        return None

    @staticmethod
    def _telegram_binding_response(
        session_id: str,
        binding: Optional[dict[str, Any]],
        *,
        delivery: Optional[dict[str, Any]] = None,
        ended: bool = False,
        end_reason: Optional[str] = None,
        rebound_to_session_id: Optional[str] = None,
        fallback_thread_id: Optional[str] = None,
        fallback_chat_id: Optional[str] = None,
    ) -> dict[str, Any]:
        bound = bool(binding)
        chat_id = str(binding.get("chat_id")) if bound else (str(fallback_chat_id) if fallback_chat_id else None)
        thread_id = str(binding.get("thread_id")) if bound else (str(fallback_thread_id) if fallback_thread_id else None)
        return {
            "object": "hermes.telegram_binding",
            "bound": bound,
            "ended": ended,
            "end_reason": end_reason,
            "rebound_to_session_id": rebound_to_session_id,
            "session_id": session_id,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "link": _api_mod().APIServerAdapter._telegram_topic_link(chat_id, thread_id) if chat_id and thread_id else None,
            "topic_url": _api_mod().APIServerAdapter._telegram_topic_link(chat_id, thread_id) if chat_id and thread_id else None,
            "delivery_enabled": bool(binding.get("delivery_enabled", True)) if bound else False,
            "last_synced_message_id": binding.get("last_synced_message_id") if bound else None,
            "future_duplication": {
                "persistent": True,
                "enabled": bool(binding.get("delivery_enabled", True)) if bound else False,
                "default": "binding_delivery_enabled",
                "request_field": "duplicate_to_telegram",
                "request_override": True,
            },
            "backfill": delivery or {"mode": "incremental", "status": "not_requested", "sent": 0, "failed": 0, "skipped": 0},
        }

    @_require_auth
    async def _handle_get_telegram_binding(self, request: web.Request) -> web.Response:
        """GET /api/sessions/{session_id}/telegram-binding — current topic binding."""
        session_id = request.match_info["session_id"]
        existing_sess, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        try:
            binding = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
            sess_dict = existing_sess if isinstance(existing_sess, dict) else {}
            is_ended = bool(sess_dict.get("ended_at") or sess_dict.get("end_reason"))
            end_reason = sess_dict.get("end_reason")
            sess_thread = str(sess_dict.get("thread_id") or "").strip()
            if not sess_thread:
                origin_raw = sess_dict.get("origin_json") or ""
                if isinstance(origin_raw, str) and origin_raw.startswith("{"):
                    try:
                        import json as _json
                        origin_data = _json.loads(origin_raw)
                        sess_thread = str(origin_data.get("thread_id") or "").strip()
                    except Exception:
                        pass
            chat_id = str(sess_dict.get("user_id") or sess_dict.get("chat_id") or getattr(self, "home_chat_id", "") or "").strip()

            if is_ended:
                rebound_to_sid = None
                if chat_id and sess_thread and hasattr(db, "get_telegram_topic_binding"):
                    try:
                        curr_b = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding, chat_id=chat_id, thread_id=sess_thread)
                        if curr_b and curr_b.get("session_id") != session_id:
                            rebound_to_sid = curr_b.get("session_id")
                    except Exception:
                        pass
                return web.json_response(self._telegram_binding_response(
                    session_id,
                    binding,
                    ended=True,
                    end_reason=end_reason,
                    rebound_to_session_id=rebound_to_sid,
                    fallback_thread_id=sess_thread or None,
                    fallback_chat_id=chat_id or None,
                ))

            if not binding:
                if chat_id and sess_thread and hasattr(db, "bind_telegram_topic"):
                    try:
                        uid = chat_id
                        runner, _ = self._telegram_runner_and_adapter()
                        from gateway.session import SessionSource, build_session_key
                        source = SessionSource(_api_mod().Platform.TELEGRAM, chat_id, chat_type="dm", user_id=uid, thread_id=sess_thread)
                        session_key = runner._session_key_for_source(source) if runner and hasattr(runner, "_session_key_for_source") else build_session_key(source)
                        await _api_mod().asyncio.to_thread(
                            db.bind_telegram_topic,
                            chat_id=chat_id,
                            thread_id=sess_thread,
                            user_id=uid,
                            session_key=session_key,
                            session_id=session_id,
                            managed_mode="auto",
                            delivery_enabled=True,
                        )
                        resolved_id = db.resolve_resume_session_id(session_id) if hasattr(db, "resolve_resume_session_id") else session_id
                        raw_messages = await _api_mod().asyncio.to_thread(db.get_messages, resolved_id) if hasattr(db, "get_messages") else []
                        max_id = max((m.get("id") for m in raw_messages if isinstance(m.get("id"), int)), default=None)
                        if max_id is not None and hasattr(db, "advance_telegram_topic_sync_checkpoint"):
                            await _api_mod().asyncio.to_thread(db.advance_telegram_topic_sync_checkpoint, session_id=session_id, message_id=max_id)
                        binding = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
                        _api_mod().logger.info("[WebUI] Auto-recovered topic binding for session %s to thread %s (checkpoint=%s)", session_id, sess_thread, max_id)
                    except Exception as recover_err:
                        _api_mod().logger.warning("[WebUI] Auto-recovery of binding failed for %s: %s", session_id, recover_err)
        except Exception:
            _api_mod().logger.exception("Failed to read Telegram binding for %s", session_id)
            return web.json_response(_api_mod()._openai_error("Telegram binding lookup failed", code="telegram_binding_failed"), status=503)
        return web.json_response(self._telegram_binding_response(session_id, binding, ended=is_ended, end_reason=end_reason))

    @_require_auth
    async def _handle_post_telegram_binding(self, request: web.Request) -> web.Response:
        """POST /api/sessions/{session_id}/telegram-binding."""
        try:
            session_id = request.match_info["session_id"]
            session, err = await self._get_existing_session_or_404(session_id)
            if err:
                return err
            body, err = await self._read_json_body(request)
            if err:
                return err
            chat_id = str(body.get("chat_id") or body.get("telegram_chat_id") or session.get("user_id") or session.get("chat_id") or getattr(self, "home_chat_id", "") or "").strip()
            thread_id = str(body.get("thread_id") or "").strip()
            topic_name = str(body.get("topic_name") or session.get("title") or f"Hermes {session_id[:24]}").strip()
            delivery_enabled = bool(body.get("delivery_enabled", True))
            db = await self._ensure_session_db_async()
            if db is None:
                return self._session_db_unavailable()
            if hasattr(db, "apply_telegram_topic_migration"):
                await _api_mod().asyncio.to_thread(db.apply_telegram_topic_migration)
            existing = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
            if existing:
                if chat_id and chat_id != str(existing.get("chat_id")):
                    return web.json_response(_api_mod()._openai_error("Session is already bound to another Telegram chat", code="telegram_binding_conflict"), status=409)
                if thread_id and thread_id != str(existing.get("thread_id")):
                    return web.json_response(_api_mod()._openai_error("Session is already bound to another Telegram topic", code="telegram_binding_conflict"), status=409)
                if "delivery_enabled" in body and hasattr(db, "set_telegram_topic_delivery_enabled"):
                    await _api_mod().asyncio.to_thread(db.set_telegram_topic_delivery_enabled, session_id=session_id, delivery_enabled=delivery_enabled)
                    existing = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
                return web.json_response(self._telegram_binding_response(session_id, existing))
            if not chat_id:
                return web.json_response(_api_mod()._openai_error("chat_id is required", code="missing_chat_id"), status=400)
            runner, telegram = self._telegram_runner_and_adapter()
            if not thread_id:
                sess_thread = str(session.get("thread_id") or "").strip()
                if sess_thread:
                    thread_id = sess_thread
                elif telegram:
                    create_topic = getattr(telegram, "create_handoff_thread", None)
                    if callable(create_topic):
                        try:
                            thread_id = str(await create_topic(chat_id, topic_name) or "")
                        except Exception as exc:
                            _api_mod().logger.exception("Failed creating Telegram topic for session %s", session_id)
                            return web.json_response(_api_mod()._openai_error(f"Failed to create Telegram topic: {exc}", code="telegram_topic_create_failed"), status=502)
            if not thread_id:
                return web.json_response(_api_mod()._openai_error("Telegram did not create a topic", code="telegram_topic_create_failed"), status=502)
            from gateway.session import SessionSource, build_session_key
            source = SessionSource(_api_mod().Platform.TELEGRAM, chat_id, chat_type="dm", user_id=str(body.get("user_id") or chat_id), thread_id=thread_id)
            session_key = runner._session_key_for_source(source) if runner and hasattr(runner, "_session_key_for_source") else build_session_key(source)
            if hasattr(db, "bind_telegram_topic"):
                await _api_mod().asyncio.to_thread(
                    db.bind_telegram_topic,
                    chat_id=chat_id, thread_id=thread_id, user_id=source.user_id or "",
                    session_key=session_key, session_id=session_id, managed_mode="api", delivery_enabled=delivery_enabled
                )
            binding = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
            return web.json_response(self._telegram_binding_response(session_id, binding), status=201)
        except Exception as e:
            _api_mod().logger.exception("Error in telegram binding: %s", e)
            return web.json_response(_api_mod()._openai_error(f"Error in telegram binding: {e}", code="telegram_binding_failed"), status=500)

    @_require_auth
    async def _handle_delete_telegram_binding(self, request: web.Request) -> web.Response:
        """DELETE /api/sessions/{session_id}/telegram-binding."""
        session_id = request.match_info["session_id"]
        _, err = await self._get_existing_session_or_404(session_id)
        if err:
            return err
        db = await self._ensure_session_db_async()
        if db is None:
            return self._session_db_unavailable()
        binding = await _api_mod().asyncio.to_thread(db.get_telegram_topic_binding_by_session, session_id=session_id)
        if binding and hasattr(db, "unbind_telegram_topic"):
            try:
                await _api_mod().asyncio.to_thread(db.unbind_telegram_topic, chat_id=str(binding["chat_id"]), thread_id=str(binding["thread_id"]))
            except Exception:
                _api_mod().logger.exception("Failed to remove Telegram binding for %s", session_id)
                return web.json_response(_api_mod()._openai_error("Failed to remove Telegram binding", code="telegram_binding_failed"), status=503)
        payload = self._telegram_binding_response(session_id, None)
        payload["deleted"] = bool(binding)
        return web.json_response(payload)
