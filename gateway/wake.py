"""Wake an existing agent session from a background completion event. Push-capable adapters
(``supports_async_delivery``) get a synthetic ``MessageEvent(internal=True)`` via handle_message;
stateless adapters (API server) would run that under a ``build_session_key()`` key that never
matches the raw ``X-Hermes-Session-Id`` real turns use (invisible parallel session), so we self-POST
``/v1/chat/completions`` with the raw id header to resume the REAL session. Exception:
async-delegation completions: the CLIENT owns the next turn, so they are never self-POSTed as a
new ``role=user`` prompt (could cross a pending human-confirmation gate); instead
``persist_delegation_delivery`` writes a durable DELIVERY row (``display_kind=
"async_delegation_complete"``, read by TUI/desktop pollers). Failures RAISE (after bounded retries
on transient errors) so callers can rewind cursors / retry instead of silently losing the event."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# A wake self-post runs the whole agent turn synchronously (stream=false); generous ceiling so long
# tool-using turns aren't killed mid-flight.
WAKE_TURN_TIMEOUT_SECONDS = 600.0

# Backoff between retries on transient failures. The API server has no per-session lock (concurrent
# turns are last-writer-wins) but DOES enforce a global max_concurrent_runs cap via HTTP 429, which
# is worth waiting out.
_RETRY_DELAYS_SECONDS = (2.0, 5.0, 10.0)


def adapter_supports_push(adapter: Any) -> bool:
    """Whether this adapter can push a message to the user after a turn ends. Reads
    ``supports_async_delivery`` off the adapter class rather than the request-scoped contextvar
    (background watchers run outside any bound session context). Adapters that don't declare
    the flag are push-capable."""
    return bool(getattr(adapter, "supports_async_delivery", True))


class WakeNotAccepted(RuntimeError):
    """No adapter admission: retry without treating a healthy chat as dead."""


def session_owned_by_profile(config: Any, profile: Optional[str], session_id: Any) -> bool:
    """True when a stateless (``api_server``) destination's raw session id is canonically owned by
    served *profile*'s own session store.

    A shared-listener mirror platform has no chat/thread/guild anchor a ``profile_routes`` entry
    could match, so the session store itself is the ownership proof: the row must exist in that
    profile's ``state.db`` under its own home and carry that profile's stamp (a NULL legacy stamp
    belongs to the store's own profile — the same rule the dashboard's session routes apply). An
    unserved profile, a missing row, a row stamped for another profile, or an unreadable store all
    fail closed. Shared by the Kanban notifier and the background-process wake path.
    """
    import contextlib
    from pathlib import Path
    if not session_id or not profile:
        return False
    profile = str(profile)
    try:
        from gateway.run import _multiplex_profile_homes
        home = dict(_multiplex_profile_homes(config)).get(profile)
        if home is None:
            return False
        from hermes_state import SessionDB
        db = SessionDB(Path(home) / "state.db", read_only=True)
    except Exception as exc:
        logger.debug("wake: session ownership check unavailable for %s/%s: %s", profile, session_id, exc)
        return False
    try:
        row = db.get_session(str(session_id))
    except Exception as exc:
        logger.debug("wake: session ownership lookup failed for %s/%s: %s", profile, session_id, exc)
        return False
    finally:
        with contextlib.suppress(Exception):
            db.close()
    return bool(row) and (row.get("profile_name") or profile) == profile


async def admit_internal_event(adapter: Any, event: Any) -> None:
    """Require a concrete adapter admission, not merely a handler returning None.

    The public handler return stays unchanged. This receipt means scheduled/queued,
    not model execution, authorization of a later turn, or successful outbound delivery.
    """
    event._gateway_accepted = False
    await adapter.handle_message(event)
    if event._gateway_accepted is not True:
        raise WakeNotAccepted("internal wake not accepted by adapter")


def is_session_turn_busy(session_id: str, adapter: Optional[Any] = None) -> bool:
    """Check if session_id currently holds an active, unexpired turn lease."""
    if not session_id:
        return False
    sid = str(session_id)
    now = time.time()
    try:
        db_inst = None
        if adapter is not None:
            ensure = getattr(adapter, "_ensure_session_db", None)
            if callable(ensure):
                db_inst = ensure()
        if db_inst is None:
            from hermes_state import SessionDB
            db_inst = SessionDB()

        # Check direct sid match in active turn leases
        row = db_inst._read_one(
            "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ? AND expires_at > ?",
            (sid, now),
        )
        if row:
            return True

        # Check through compression conversation lineage
        key_fn = getattr(db_inst, "_session_turn_lease_key", None)
        if callable(key_fn):
            conv_id = key_fn(sid)
            if conv_id and conv_id != sid:
                row = db_inst._read_one(
                    "SELECT expires_at FROM session_turn_leases WHERE conversation_id = ? AND expires_at > ?",
                    (conv_id, now),
                )
                if row:
                    return True
    except Exception as exc:
        logger.debug("is_session_turn_busy check failed for %s: %s", sid, exc)
    return False


async def deliver_wake(adapter: Any, *, text: str, session_id: str = "", source: Any = None,
                       notification_category: str = "result", profile: Optional[str] = None) -> None:
    """Deliver a wake turn to the session behind ``adapter``. ``session_id`` is the RAW session id
    (``X-Hermes-Session-Id`` / state.db key) — required for non-push adapters. ``source`` is the
    ``SessionSource`` for the synthetic event — required for push-capable adapters. ``profile``
    names the served profile that canonically owns a non-push destination; a non-default value is
    delivered in-process under the caller's profile scope (see ``_self_post_chat_completion``).
    Raises on failure so the caller can rewind/retry."""
    if session_id and is_session_turn_busy(session_id, adapter):
        logger.info("Session %s holds an active turn lease — skipping wake self-post (in-flight turn will consume delivery)", session_id)
        return
    if adapter_supports_push(adapter):
        if source is None:
            raise ValueError("deliver_wake: push-capable adapter requires a SessionSource")
        from gateway.platforms.event import MessageEvent, MessageType
        synth_event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=True,
                                   metadata={"notification_category": notification_category})
        await admit_internal_event(adapter, synth_event)
        return
    if not session_id:
        raise ValueError("deliver_wake: non-push adapter (supports_async_delivery=False) "
                         "requires the raw session id to self-post the wake turn")
    extra: dict = {}
    if profile:
        extra["profile"] = profile
    if notification_category == "diagnostic":
        extra["notification_category"] = notification_category
    await _self_post_chat_completion(adapter, text=text, session_id=session_id, **extra)


def _delegation_display_metadata(evt: dict) -> dict:
    """Display-only metadata for a persisted delegation delivery row. Mirrors
    ``tui_gateway.server._async_delegation_display_metadata`` (same ``display_kind`` consumer
    contract) without importing the TUI stack."""
    raw_results = evt.get("results")
    results = [r for r in raw_results if isinstance(r, dict)] if isinstance(raw_results, list) else []
    task_count = len(results) or 1
    completed_count = sum(1 for r in results if r.get("status") in {"completed", "success"})
    failed_count = sum(1 for r in results if r.get("status") in {"failed", "error"})
    metadata = {"delegation_id": str(evt.get("delegation_id") or ""), "task_count": task_count,
                "completed_count": completed_count or task_count - failed_count,
                "failed_count": failed_count}
    if evt.get("task_failure_notice"):
        metadata["delivery_notice"] = f"task_failure:{results[0].get('task_index', '') if results else ''}"
        metadata["notification_category"] = "diagnostic"
        from gateway.warning_notifications import warning_notifications_enabled
        if not warning_notifications_enabled("api_server"):
            metadata["presentation_suppressed"] = True
    duration = evt.get("total_duration_seconds") or evt.get("duration_seconds")
    if isinstance(duration, (int, float)):
        metadata["duration_seconds"] = duration
    return metadata


async def persist_delegation_delivery(adapter: Any, *, text: str, session_id: str, evt: Optional[dict] = None) -> None:
    """Persist an async-delegation completion as a durable DELIVERY row (see module docstring)
    WITHOUT running any agent turn. Raises on failure so the caller can release the durable claim
    and retry.

    85957: on stateless api_server sessions the client owns the turn after ``event.complete`` — a completion
    must never become a new ``role=user`` prompt via the self-post (that starts an unauthorized agent turn
    and can cross a pending human-confirmation gate). Instead, append the completion to the session
    transcript as a timeline bookkeeping row (``display_kind="async_delegation_complete"`` + display
    metadata — the exact shape the TUI/desktop delivery path persists), WITHOUT running any agent turn.
    Clients polling ``GET /api/sessions/{id}/messages`` see it immediately; the pre-request repair belt
    folds it into the next real client turn as context. See #85957.
    """
    if not session_id:
        raise ValueError("persist_delegation_delivery: raw session id required to persist "
                         "the completion on the api_server session transcript")
    ensure = getattr(adapter, "_ensure_session_db", None)
    db: Any = await asyncio.to_thread(ensure) if callable(ensure) else None
    if db is None:
        raise RuntimeError("persist_delegation_delivery: api_server SessionDB unavailable — "
                           f"cannot persist completion for session {session_id}")
    # #98619: the parent run may have compressed/rotated between dispatch and this detached
    # completion — the captured origin id is then a closed parent and the append below is
    # rejected with CompressionSessionClosedError forever (the watcher retries the same stale
    # id). Adopt the live continuation tip first, the same canonical resolution
    # /api/sessions/{id}/messages reads use, so the delivery row lands where the next run and
    # the messages endpoint both resolve. Fails open to the original id.
    resolver = getattr(db, "resolve_resume_session_id", None)
    if callable(resolver):
        try:
            resolved = await asyncio.to_thread(resolver, session_id)
            if resolved:
                session_id = str(resolved)
        except Exception:
            logger.debug("delegation delivery continuation resolve failed for %s", session_id, exc_info=True)
    await asyncio.to_thread(
        db.append_delegation_delivery, session_id, text, _delegation_display_metadata(evt or {}),
    )
    logger.info(
        "async delegation completion persisted as delivery row for api_server session %s (no wake turn)", session_id
    )


async def _self_post_chat_completion(adapter: Any, *, text: str, session_id: str,
                                      notification_category: str = "result",
                                      profile: Optional[str] = None) -> None:
    """POST the wake text to the in-pod API server as a normal session turn, using the adapter's
    own bind host/port/key. Session continuation via ``X-Hermes-Session-Id`` is 403-gated on
    ``API_SERVER_KEY``, so a missing key is a hard error rather than a wake in a fresh session
    nobody watches.

    ``profile`` (a served secondary under ``gateway.multiplex_profiles``) takes the IN-PROCESS
    route instead of the HTTP one: the shared listener's ``/p/<profile>/`` mirror authenticates
    with that profile's own ``API_SERVER_KEY`` — which a route-only profile legitimately does not
    have — and an unprefixed self-post would resume the session in the DEFAULT profile's store.
    The caller already holds the owner profile's runtime scope, so the turn lands in that profile's
    own session. A non-default profile whose adapter cannot run in-process fails closed.
    """
    if profile and str(profile) != "default":
        in_process: Any = getattr(adapter, "run_internal_session_turn", None)
        if not callable(in_process):
            raise RuntimeError(
                f"wake self-post for served profile {profile!r} requires in-process session "
                "delivery; refusing to self-post as the default profile")
        await in_process(session_id=session_id, text=text, profile=str(profile),
                         notification_category=notification_category)
        sync_tg = getattr(adapter, "_sync_telegram_history", None)
        if callable(sync_tg):
            try:
                await asyncio.shield(sync_tg(session_id))
            except Exception as sync_exc:
                logger.warning("In-process post-wake Telegram sync failed for session %s: %s", session_id, sync_exc)
        return
    import aiohttp
    host = str(getattr(adapter, "_host", "") or "127.0.0.1")
    if host in ("0.0.0.0", "::", "*"):
        host = "127.0.0.1"  # wildcard bind — connect over loopback
    port = int(getattr(adapter, "_port", 0) or 8642)
    api_key = str(getattr(adapter, "_api_key", "") or "")
    if not api_key:
        raise RuntimeError("wake self-post requires API_SERVER_KEY: session continuation via "
                           "X-Hermes-Session-Id is rejected (403) on an unauthenticated API "
                           "server, so the wake cannot reach the target session")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"  # bare IPv6 literal
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "X-Hermes-Session-Id": session_id}
    payload = {"model": str(getattr(adapter, "_model_name", "") or "hermes-agent"),
               "messages": [{"role": "user", "content": text}], "stream": False}
    if notification_category == "diagnostic":
        payload["hermes_notification_category"] = "diagnostic"
    last_err: Optional[BaseException] = None
    attempts = 1 + len(_RETRY_DELAYS_SECONDS)
    for attempt in range(attempts):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt - 1])
        try:
            timeout = aiohttp.ClientTimeout(total=WAKE_TURN_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as http:
                async with http.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 429:  # concurrency cap — transient; back off and retry
                        last_err = RuntimeError(
                            f"wake self-post got HTTP 429 (concurrency cap) for session {session_id}"
                        )
                        logger.warning("%s; attempt %d/%d", last_err, attempt + 1, attempts)
                        continue
                    if resp.status >= 400:  # non-transient (auth/validation): fail immediately
                        body = (await resp.text())[:300]
                        raise RuntimeError(
                            f"wake self-post failed for session {session_id}: HTTP {resp.status}: {body}"
                        )
                    await resp.read()
                    logger.info("wake self-post delivered for session %s (attempt %d)", session_id, attempt + 1)
                    # Trigger instant Telegram sync for bound sessions (SpDD task-gateway-wake-telegram-sync-20261112)
                    sync_tg = getattr(adapter, "_sync_telegram_history", None)
                    if callable(sync_tg):
                        try:
                            await asyncio.shield(sync_tg(session_id))
                        except Exception as sync_exc:
                            logger.warning("Post-wake Telegram sync failed for session %s: %s", session_id, sync_exc)
                    return
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            last_err = exc
            logger.warning("wake self-post transient failure for session %s (attempt %d/%d): %s",
                           session_id, attempt + 1, attempts, exc)
    raise RuntimeError(
        f"wake self-post gave up for session {session_id} after {attempts} attempts: {last_err}"
    ) from last_err


_CONSECUTIVE_AUTO_WAKES: dict[str, int] = {}
MAX_CONSECUTIVE_AUTO_WAKES: int = 10


def get_consecutive_auto_wakes(session_id: str) -> int:
    return _CONSECUTIVE_AUTO_WAKES.get(str(session_id), 0)


def increment_consecutive_auto_wakes(session_id: str) -> int:
    sid = str(session_id)
    _CONSECUTIVE_AUTO_WAKES[sid] = _CONSECUTIVE_AUTO_WAKES.get(sid, 0) + 1
    return _CONSECUTIVE_AUTO_WAKES[sid]


def reset_consecutive_auto_wakes(session_id: str) -> None:
    _CONSECUTIVE_AUTO_WAKES.pop(str(session_id), None)


def check_coordinator_wake_eligibility(session_id: str, adapter: Any, evt: Optional[dict] = None) -> bool:
    """Check if an async_delegation completion event should autonomously wake a coordinator session.

    Conditions for auto-wake:
    1. Pending human gates (clarifications/choice prompts) MUST be clear.
    2. Bounded loop protection: consecutive_auto_wakes must be strictly less than MAX_CONSECUTIVE_AUTO_WAKES (10).
    3. The session must be an active SpDD coordinator (active passport under ~/.hermes/tasks/*/state.json
       matching session_id or in_progress, or auto_resume_on_delegation=True in session/event metadata).
    """
    if not session_id:
        return False

    sid = str(session_id)

    # 1. Check pending clarify
    try:
        from tools import clarify_gateway
        if clarify_gateway.get_pending_for_session(sid, include_choice_prompts=True) is not None:
            logger.info("coordinator auto-wake suppressed: pending clarify/choice gate for session %s", sid)
            return False
    except Exception as exc:
        logger.debug("clarify_gateway check error for session %s: %s", sid, exc)

    # 2. Check loop bounds
    if get_consecutive_auto_wakes(sid) >= MAX_CONSECUTIVE_AUTO_WAKES:
        logger.warning(
            "coordinator auto-wake suppressed: reached ceiling of %d consecutive auto-wakes for session %s",
            MAX_CONSECUTIVE_AUTO_WAKES, sid
        )
        return False

    # 3. Check active turn lease (busy coordinator)
    if is_session_turn_busy(sid, adapter):
        logger.info("coordinator auto-wake suppressed: session %s holds active turn lease (in-flight turn will consume delivery)", sid)
        return False

    # 4. Check if SpDD coordinator intent
    # 3a. Event metadata
    if evt:
        if evt.get("auto_resume_on_delegation") is True or evt.get("auto_resume") is True:
            return True
        meta = evt.get("metadata")
        if isinstance(meta, dict) and (meta.get("auto_resume_on_delegation") or meta.get("auto_resume")):
            return True

    # 3b. Session metadata via adapter / SessionDB
    if adapter is not None:
        ensure = getattr(adapter, "_ensure_session_db", None)
        if callable(ensure):
            try:
                db_inst: Any = ensure()
                get_sess = getattr(db_inst, "get_session", None)
                if callable(get_sess):
                    row = get_sess(sid)
                    if row:
                        smeta = row.get("metadata") or {}
                        if isinstance(smeta, str):
                            import json
                            try:
                                smeta = json.loads(smeta)
                            except Exception:
                                smeta = {}
                        if smeta.get("auto_resume_on_delegation") or smeta.get("coordinator_role"):
                            return True
            except Exception as exc:
                logger.debug("session metadata check error for session %s: %s", sid, exc)

    # 3c. Active SpDD task passport in ~/.hermes/tasks/*/state.json
    try:
        from pathlib import Path
        import json
        tasks_dir = Path.home() / ".hermes" / "tasks"
        if tasks_dir.is_dir():
            for passport in sorted(tasks_dir.glob("*/state.json")):
                try:
                    data = json.loads(passport.read_text(encoding="utf-8"))
                    p_status = data.get("status")
                    p_sid = data.get("session_id")
                    if p_sid == sid:
                        if p_status in ("in_progress", "active"):
                            return True
                    elif p_status in ("in_progress", "active"):
                        # If active task passport has no explicit session_id bound, or matches coordinator
                        return True
                except Exception:
                    continue
    except Exception as exc:
        logger.debug("task passport search error for session %s: %s", sid, exc)

    return False

