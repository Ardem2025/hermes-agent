"""Tests for telegram-flood-control-hardening (BDD-1..BDD-5)."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_progress import (
    _ACTIVE_PROGRESS_LOOPS,
    GatewayTurnProgressMixin,
    _progress_chat_key,
)
from gateway.turn_context import TurnContext
from plugins.platforms.telegram import telegram_flood_window, telegram_typing_gate
from plugins.platforms.telegram.adapter import (
    _MAX_FLOOD_COOLDOWN_SECS,
    _TELEGRAM_CHAT_INTERIM_SOFT_LIMIT,
    _TELEGRAM_CHAT_TYPING_SOFT_LIMIT,
    TelegramAdapter,
)


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    adapter._bot.edit_message_text = AsyncMock(return_value=MagicMock())
    adapter._bot.send_chat_action = AsyncMock(return_value=True)
    adapter._bot.send_message_draft = AsyncMock(return_value=True)
    adapter._bot.delete_message = AsyncMock(return_value=True)
    return adapter


class _FakeRetryAfter(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Flood control exceeded. Retry in {retry_after} seconds")


# ---------------------------------------------------------------------------
# BDD-1: Honest retry_after (24h cap), monotonicity, typing 429 sync
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdd1_long_retry_after_preserved_and_monotonic():
    adapter = _make_adapter()
    res = adapter._record_send_flood_cooldown("12345", 43910.0)
    assert res.success is False
    assert res.retry_after == 43910.0
    rem = adapter._send_flood_cooldown_remaining("12345")
    assert rem is not None and rem > 43000.0

    # Shorter retry_after must NOT shrink an active longer deadline
    adapter._record_send_flood_cooldown("12345", 30.0)
    rem_after = adapter._send_flood_cooldown_remaining("12345")
    assert rem_after is not None and rem_after > 43000.0

    # Values above 24h are clamped to _MAX_FLOOD_COOLDOWN_SECS (86400s)
    adapter._record_send_flood_cooldown("99999", 200000.0)
    rem_capped = adapter._send_flood_cooldown_remaining("99999")
    assert rem_capped is not None and 86390.0 <= rem_capped <= _MAX_FLOOD_COOLDOWN_SECS


@pytest.mark.asyncio
async def test_bdd1_typing_429_syncs_send_cooldown_and_oserror_does_not():
    adapter = _make_adapter()
    adapter._record_typing_cooldown("12345", _FakeRetryAfter(43910.0))

    rem = adapter._send_flood_cooldown_remaining("12345")
    assert rem is not None and rem > 43000.0

    # send(), edit_message(), delete_message(), send_draft() all fail closed / defer without API calls
    send_res = await adapter.send("12345", "hello")
    assert send_res.success is False and "flood_control:" in (send_res.error or "")
    assert adapter._bot.send_message.await_count == 0

    edit_res = await adapter.edit_message("12345", "10", "updated", finalize=True)
    assert edit_res.success is False and "flood_control:" in (edit_res.error or "")
    assert adapter._bot.edit_message_text.await_count == 0

    draft_res = await telegram_flood_window.send_draft(adapter, "12345", 1, "draft", None)
    assert draft_res.success is True and (draft_res.raw_response or {}).get("skipped") is True
    assert adapter._bot.send_message_draft.await_count == 0

    with patch.object(telegram_flood_window, "_defer", return_value=False) as defer_mock:
        del_res = await adapter.delete_message("12345", "10")
        assert del_res is False
        assert defer_mock.call_count == 1
        assert adapter._bot.delete_message.await_count == 0

    # OSError without retry_after must NOT touch _telegram_send_cooldown_until
    adapter._record_typing_cooldown("77777", OSError("network reset"))
    assert adapter._send_flood_cooldown_remaining("77777") is None
    assert str("77777") in adapter._telegram_typing_cooldown_until


# ---------------------------------------------------------------------------
# BDD-2: tool_progress finalize=False, skipped handling, FloodedEditAdapter compat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdd2_progress_finalize_flag_and_skipped_retry():
    calls: list[tuple[str, bool]] = []

    class _FinalizeAdapter(BasePlatformAdapter):
        REQUIRES_EDIT_FINALIZE = True

        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
            self.skip_next = False

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            return SendResult(success=True, message_id="m1")

        async def edit_message(self, chat_id, message_id, content, *, finalize: bool = False, metadata=None) -> SendResult:
            calls.append((content, finalize))
            if self.skip_next:
                return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None) -> None:
            return None

        async def get_chat_info(self, chat_id: str):
            return {"id": chat_id}

    adapter = _FinalizeAdapter()
    ctx = TurnContext(
        session_key="tg:123",
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="123", thread_id=None),
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        _run_still_current=lambda: True,
    )
    helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    helper._runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter, _is_draining=lambda: False)
    helper._ctx = ctx
    helper._agent_interrupted = lambda: False

    st = helper._progress_edit_state(adapter)
    st.progress_msg_id = "m1"
    st.progress_lines = ["line 1"]
    adapter.skip_next = True
    ok = await helper._progress_send_or_edit(st, "line 1")
    assert ok is False
    assert st.flood_pending is True
    assert calls[-1] == ("line 1", False)

    # Next idle tick delivers the pending interim edit with finalize=False
    adapter.skip_next = False
    delivered = await helper._deliver_flood_pending_progress(st)
    assert delivered is True
    assert st.flood_pending is False
    assert calls[-1] == ("line 1", False)

    # Closed owed_bubbles use finalize=True and stay queued when skipped
    st.owed_bubbles.append(("m0", "closed bubble"))
    adapter.skip_next = True
    assert await helper._deliver_flood_pending_progress(st) is True
    assert len(st.owed_bubbles) == 1
    assert calls[-1] == ("closed bubble", True)
    adapter.skip_next = False
    assert await helper._deliver_flood_pending_progress(st) is True
    assert st.owed_bubbles == []

    # Final flush uses finalize=True
    await helper._flush_progress_edit(st)
    assert calls[-1] == ("line 1", True)


@pytest.mark.asyncio
async def test_bdd2_legacy_adapter_without_requires_edit_finalize_receives_no_finalize_kwarg():
    seen_kwargs: list[dict] = []

    class _LegacyAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            return True

        async def disconnect(self) -> None:
            return None

        async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
            return SendResult(success=True, message_id="m1")

        async def edit_message(self, chat_id, message_id, content) -> SendResult:
            seen_kwargs.append({"chat_id": chat_id, "message_id": message_id, "content": content})
            return SendResult(success=True, message_id=message_id)

        async def send_typing(self, chat_id, metadata=None) -> None:
            return None

        async def get_chat_info(self, chat_id: str):
            return {"id": chat_id}

    adapter = _LegacyAdapter()
    ctx = TurnContext(
        session_key="tg:123",
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="123", thread_id=None),
        _run_still_current=lambda: True,
    )
    helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    helper._runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter)
    helper._ctx = ctx
    st = helper._progress_edit_state(adapter)
    st.progress_msg_id = "m1"
    st.progress_lines = ["line 1"]
    res = await helper._edit_progress_message(st, "m1", "line 1", finalize=True)
    assert res.success is True
    assert len(seen_kwargs) == 1


# ---------------------------------------------------------------------------
# BDD-3: _send_chunks, _resume_partial_send, _send_chunk_with_retries slot pacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdd3_send_chunks_and_resume_and_retry_slot_pacing(monkeypatch):
    adapter = _make_adapter()
    adapter._telegram_chat_outbound_slot_secs = 0.5
    sleeps: list[float] = []

    async def _fake_sleep(delay: float):
        sleeps.append(delay)
        # Advance slot expiry so next check sees 0
        adapter._telegram_chat_outbound_slot_until.clear()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    # Single-chunk send via _send_text_locked holds slot once (no double sleep)
    res1 = await adapter.send("c1", "short message")
    assert res1.success is True
    assert sleeps == []

    # Multi-chunk _send_chunks (3 chunks) with slot_held_for_first=True sleeps before chunks 2 and 3
    sleeps.clear()
    adapter._hold_chat_outbound_slot("c2")
    error_types = adapter._telegram_error_types()
    res3 = await adapter._send_chunks("c2", ["a", "b", "c"], [], None, None, error_types, slot_held_for_first=True)
    assert res3.success is True
    assert len(sleeps) == 2

    # _resume_partial_send waits for slot before first resumed chunk
    sleeps.clear()
    adapter._hold_chat_outbound_slot("c3")
    partial = SendResult(
        success=False,
        error="flood_control:1",
        raw_response={"undelivered_chunks": ("tail1",), "delivered_message_ids": ("10",)},
    )
    resumed = await adapter._resume_partial_send("c3", partial, reply_to=None, metadata=None)
    assert resumed is not None and resumed.success is True
    assert len(sleeps) == 1

    # _send_chunk_with_retries waits for slot after retry sleep if another caller held the slot
    sleeps.clear()
    attempts = 0

    async def _sleep_with_concurrent_slot(delay: float):
        sleeps.append(delay)
        if delay == 1.0:
            # Simulate another coroutine holding the chat slot while we slept on RetryAfter
            adapter._hold_chat_outbound_slot("c4")
        else:
            adapter._telegram_chat_outbound_slot_until.clear()

    monkeypatch.setattr(asyncio, "sleep", _sleep_with_concurrent_slot)

    async def _flaky_send(chunk, send_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _FakeRetryAfter(1.0)
        return SimpleNamespace(message_id=555)

    monkeypatch.setattr(adapter, "_send_chunk_markdown_or_plain", _flaky_send)
    msg, _ = await adapter._send_chunk_with_retries("c4", "hi", 0, None, None, None, False, error_types)
    assert msg.message_id == 555
    assert len(sleeps) == 2 and sleeps[0] == 1.0 and sleeps[1] > 0.0


# ---------------------------------------------------------------------------
# BDD-4: telegram_flood_window.delete_message slot pacing & _defer on re-opened long window
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdd4_delete_message_waits_for_slot_and_defers_on_long_reopened_window(monkeypatch):
    sleeps: list[float] = []
    deleted = False

    async def _fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(telegram_flood_window.asyncio, "sleep", _fake_sleep)

    slot_calls = 0
    held = []
    recorded = []

    class _SlotAdapter:
        name = "telegram"
        _bot = SimpleNamespace(delete_message=AsyncMock(return_value=True))

        def _send_flood_cooldown_remaining(self, chat_id):
            return None

        def _chat_outbound_slot_remaining(self, chat_id):
            nonlocal slot_calls
            slot_calls += 1
            return 0.4 if slot_calls == 1 else 0.0

        def _hold_chat_outbound_slot(self, chat_id):
            held.append(chat_id)

        def _record_chat_outbound_call(self, chat_id):
            recorded.append(chat_id)

    ok = await telegram_flood_window.delete_message(_SlotAdapter(), "12345", "99")
    assert ok is True
    assert _SlotAdapter._bot.delete_message.await_count == 1
    assert sleeps == [0.4]
    assert held == ["12345"]
    assert recorded == ["12345"]

    # Re-opened long flood window during slot wait -> _defer called when inline=True
    sleeps.clear()
    deferred_calls = []

    class _ReopenedFloodAdapter:
        name = "telegram"
        _bot = SimpleNamespace(delete_message=AsyncMock(return_value=True))

        def __init__(self):
            self.checks = 0

        def _send_flood_cooldown_remaining(self, chat_id):
            self.checks += 1
            return None if self.checks == 1 else 120.0

        def _chat_outbound_slot_remaining(self, chat_id):
            return 0.5

    monkeypatch.setattr(
        telegram_flood_window,
        "_defer",
        lambda adapter, chat_id, delete, remaining: deferred_calls.append((chat_id, remaining)) or False,
    )
    res = await telegram_flood_window.delete_message(_ReopenedFloodAdapter(), "67890", "100")
    assert res is False
    assert deferred_calls == [("67890", 120.0)]


# ---------------------------------------------------------------------------
# BDD-5: Adaptive EDIT_INTERVAL + 60s sliding-window rate governor + saturated preview
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bdd5_rate_governor_typing_interim_draft_and_saturated_preview():
    adapter = _make_adapter()
    # Clear slot so only 60s window count governs
    adapter._telegram_chat_outbound_slot_until = {}

    for _ in range(_TELEGRAM_CHAT_TYPING_SOFT_LIMIT):
        adapter._record_chat_outbound_call("c1")

    assert adapter._chat_outbound_calls_in_window("c1") == _TELEGRAM_CHAT_TYPING_SOFT_LIMIT
    assert telegram_typing_gate.typing_may_fire(adapter, "c1") is False

    # Interim edit still allowed at 15 (< 18)
    adapter._telegram_chat_outbound_slot_until.clear()
    res_15 = await adapter.edit_message("c1", "10", "ok", finalize=False)
    assert res_15.success is True and not (res_15.raw_response or {}).get("skipped")

    # Push to >= 18 calls in 60s window
    while adapter._chat_outbound_calls_in_window("c1") < _TELEGRAM_CHAT_INTERIM_SOFT_LIMIT:
        adapter._record_chat_outbound_call("c1")
    adapter._telegram_chat_outbound_slot_until.clear()

    skipped_edit = await adapter.edit_message("c1", "10", "interim", finalize=False)
    assert skipped_edit.success is True and (skipped_edit.raw_response or {}).get("skipped") is True

    skipped_draft = await telegram_flood_window.send_draft(adapter, "c1", 1, "draft", None)
    assert skipped_draft.success is True and (skipped_draft.raw_response or {}).get("skipped") is True

    # Tier 1 send() and finalize=True edit_message() are NOT blocked by soft limit
    final_edit = await adapter.edit_message("c1", "10", "final", finalize=True)
    assert final_edit.success is True and not (final_edit.raw_response or {}).get("skipped")
    tier1_send = await adapter.send("c1", "final answer")
    assert tier1_send.success is True

    # Saturated preview edit records outbound call
    adapter2 = _make_adapter()
    oversized = "x" * (adapter2.MAX_MESSAGE_LENGTH + 50)
    await adapter2.edit_message("c_sat", "20", oversized, finalize=False)
    assert adapter2._chat_outbound_calls_in_window("c_sat") == 1


@pytest.mark.asyncio
async def test_bdd5_adaptive_edit_interval_scales_and_recovers(monkeypatch):
    adapter = _make_adapter()
    key = _progress_chat_key(adapter, "chat-multi")
    _ACTIVE_PROGRESS_LOOPS.pop(key, None)

    observed_sleeps: list[float] = []

    async def _run_one(idx: int):
        ctx = TurnContext(
            session_key=f"tg:chat-multi:{idx}",
            source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-multi", thread_id=None),
            tool_progress_enabled=True,
            progress_queue=queue.Queue(),
            _run_still_current=lambda: True,
        )
        ctx.progress_queue.put(f"step {idx}")
        helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
        helper._runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter)
        helper._ctx = ctx
        helper._agent_interrupted = lambda: False
        helper._progress_send_or_edit = AsyncMock(return_value=True)
        helper._progress_restore_typing = AsyncMock()

        # Stop after first sleep or queue empty
        async def _fake_sleep(delay: float):
            observed_sleeps.append(delay)
            if delay == 0.3:
                raise asyncio.CancelledError()

        with patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await helper.send_progress_messages()

    # Simulate 4 other active loops already running in the same chat
    _ACTIVE_PROGRESS_LOOPS[key] = 4
    try:
        await _run_one(5)
        # With 4 + 1 = 5 active loops, effective_interval = min(10.0, 1.5 * 5) = 7.5s
        # First iteration has last_edit_ts=0.0 so it sends immediately, then hits queue.Empty -> 0.3s cancel.
        # Verify active count was incremented to 5 during run and restored to 4 in finally:
        assert _ACTIVE_PROGRESS_LOOPS[key] == 4
    finally:
        _ACTIVE_PROGRESS_LOOPS.pop(key, None)

    # Now verify dynamic recalculation inside the loop when remaining > 0
    _ACTIVE_PROGRESS_LOOPS[key] = 4
    try:
        ctx = TurnContext(
            session_key="tg:chat-multi:dyn",
            source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="chat-multi", thread_id=None),
            tool_progress_enabled=True,
            progress_queue=queue.Queue(),
            _run_still_current=lambda: True,
        )
        ctx.progress_queue.put("first")
        ctx.progress_queue.put("second")
        ctx.progress_queue.put("third")
        ctx.progress_queue.put("fourth")
        helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
        helper._runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter)
        helper._ctx = ctx
        helper._agent_interrupted = lambda: False
        helper._progress_send_or_edit = AsyncMock(return_value=True)
        helper._progress_restore_typing = AsyncMock()

        throttle_sleeps: list[float] = []
        now_val = 100.0

        async def _dyn_sleep(delay: float):
            nonlocal now_val
            if delay == 0.3:
                raise asyncio.CancelledError()
            throttle_sleeps.append(delay)
            # Simulate 4 sibling loops finishing while we slept on the first throttle
            _ACTIVE_PROGRESS_LOOPS[key] = 1
            now_val += delay

        with (
            patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now_val),
            patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_dyn_sleep),
        ):
            with pytest.raises(asyncio.CancelledError):
                await helper.send_progress_messages()

        # First throttle sleep saw 5 active loops -> 7.5s; after siblings finished, active=1 -> 1.5s
        assert throttle_sleeps == [pytest.approx(7.5, abs=0.05), pytest.approx(1.5, abs=0.05)]
    finally:
        _ACTIVE_PROGRESS_LOOPS.pop(key, None)


@pytest.mark.asyncio
async def test_bdd6_subagent_pulse_ticker_skipped_and_delete_slot_compat(monkeypatch):
    from gateway.subagent_pulse_ticker import SubagentPulseTicker

    adapter = _make_adapter()
    adapter._bot.delete_message = AsyncMock(return_value=True)
    ticker = SubagentPulseTicker(adapter, "12345", cleanup_progress=True)
    ticker.current_message_id = "777"
    ticker._last_content = "initial"
    ticker.tasks = {0: {"goal": "g", "status": "completed", "started_at": 0.0}}

    # 1. When send_or_update_status returns skipped=True, _last_content is NOT overwritten
    with patch(
        "plugins.platforms.telegram.telegram_status.send_or_update_status",
        new=AsyncMock(return_value=SendResult(success=True, message_id="777", raw_response={"skipped": True})),
    ):
        await ticker._send_or_edit("interim update", finalize=False)
    assert ticker._last_content == "initial"

    # 2. stop() with cleanup_progress=True and all tasks completed deletes via telegram_flood_window slot pacing
    sleeps = []

    async def _fake_sleep(delay: float):
        sleeps.append(delay)
        adapter._telegram_chat_outbound_slot_until.clear()

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    adapter._hold_chat_outbound_slot("12345")

    with patch(
        "plugins.platforms.telegram.telegram_status.send_or_update_status",
        new=AsyncMock(return_value=SendResult(success=True, message_id="777")),
    ):
        await ticker.stop(status="completed")

    assert adapter._bot.delete_message.await_count == 1
    assert len(sleeps) == 1 and sleeps[0] > 0.0



@pytest.mark.asyncio
async def test_bdd5_media_send_slot_records_outbound_call():
    adapter = _make_adapter()
    async with adapter._media_send_slot("c_media"):
        pass
    assert adapter._chat_outbound_calls_in_window("c_media") == 1
