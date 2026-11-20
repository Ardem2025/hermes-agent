"""Tests for throttled progress delivery and CancelledError drain in gateway/run_turn_progress.py."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_progress import (
    _ACTIVE_PROGRESS_LOOPS,
    GatewayTurnProgressMixin,
    _progress_chat_key,
)
from gateway.turn_context import TurnContext


class _FakeProgressAdapter(BasePlatformAdapter):
    REQUIRES_EDIT_FINALIZE = True

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.sent: list[str] = []
        self.edits: list[tuple[str, str, bool]] = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(content)
        return SendResult(success=True, message_id=f"m{len(self.sent)}")

    async def edit_message(self, chat_id, message_id, content, finalize: bool = False) -> SendResult:
        self.edits.append((message_id, content, finalize))
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_helper(
    chat_id: str,
    *,
    grouping: str = "accumulate",
    still_current_fn=None,
    interrupted_fn=None,
):
    adapter = _FakeProgressAdapter()
    ctx = TurnContext(
        session_key=f"tg:{chat_id}",
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id=chat_id, thread_id=None),
        progress_grouping=grouping,
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        _run_still_current=still_current_fn or (lambda: True),
    )
    helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    helper._runner = SimpleNamespace(_delivery_adapter_for=lambda _s: adapter)
    helper._ctx = ctx
    helper._agent_interrupted = interrupted_fn or (lambda: False)
    helper._progress_restore_typing = AsyncMock()

    captured_states = []
    orig_state_fn = GatewayTurnProgressMixin._progress_edit_state

    def _capture_state(adp):
        st = orig_state_fn(helper, adp)
        captured_states.append(st)
        return st

    helper._progress_edit_state = _capture_state
    key = _progress_chat_key(adapter, chat_id)
    _ACTIVE_PROGRESS_LOOPS.pop(key, None)
    return helper, ctx, adapter, captured_states, key


@pytest.mark.asyncio
async def test_scenario1_accumulate_delivers_throttled_update_on_queue_empty():
    still_current = True
    helper, ctx, adapter, states, key = _make_helper(
        "chat-s1", grouping="accumulate", still_current_fn=lambda: still_current
    )
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    now = 100.0
    throttle_sleeps: list[float] = []

    async def _fake_sleep(delay: float):
        nonlocal now, still_current
        if delay == 0.3:
            # By the time queue.Empty reaches sleep(0.3), the throttled update must already be delivered.
            if adapter.edits:
                still_current = False
            return
        throttle_sleeps.append(delay)
        assert states[0].flood_pending is True
        now += delay

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await helper.send_progress_messages()

    assert throttle_sleeps == [pytest.approx(1.5, abs=0.01)]
    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", False)]
    assert states[0].flood_pending is False
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario2_replace_delivers_latest_throttled_update():
    still_current = True
    helper, ctx, adapter, states, key = _make_helper(
        "chat-s2", grouping="replace", still_current_fn=lambda: still_current
    )
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    now = 100.0
    injected = False

    async def _fake_sleep(delay: float):
        nonlocal now, still_current, injected
        if delay == 0.3:
            still_current = False
            return
        if not injected:
            injected = True
            ctx.progress_queue.put("step 3")
            ctx.progress_queue.put("step 4")
            now += 0.2
            return
        if not ctx.progress_queue.empty():
            now += 0.2
            return
        now += delay

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 4", False)]
    assert states[0].flood_pending is False
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario3a_flood_window_defers_delivery_until_window_closes():
    still_current = True
    helper, ctx, adapter, states, key = _make_helper(
        "chat-s3a", grouping="accumulate", still_current_fn=lambda: still_current
    )
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    now = 100.0
    empty_ticks = 0

    async def _fake_sleep(delay: float):
        nonlocal now, still_current, empty_ticks
        if delay == 0.3:
            empty_ticks += 1
            if empty_ticks == 1:
                # First queue.Empty saw open flood window -> nothing edited yet, flood_pending stays True.
                assert adapter.edits == []
                assert states[0].flood_pending is True
                states[0].flood_until = 0.0
            else:
                still_current = False
            return
        # Open flood window while sleeping on throttle.
        states[0].flood_until = now + 10.0
        now += delay

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", False)]
    assert states[0].flood_pending is False
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario3b_reset_during_throttle_sleep_queues_and_delivers_owed_bubble():
    still_current = True
    helper, ctx, adapter, states, key = _make_helper(
        "chat-s3b", grouping="accumulate", still_current_fn=lambda: still_current
    )
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    now = 100.0
    reset_sent = False

    async def _fake_sleep(delay: float):
        nonlocal now, still_current, reset_sent
        if delay == 0.3:
            still_current = False
            return
        if not reset_sent:
            reset_sent = True
            ctx.progress_queue.put(("__reset__",))
        now += delay

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", True)]
    assert states[0].owed_bubbles == []
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario4_not_current_during_throttle_sleep_drains_without_sending():
    still_current = True
    helper, ctx, adapter, _states, key = _make_helper(
        "chat-s4", grouping="accumulate", still_current_fn=lambda: still_current
    )
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")
    ctx.progress_queue.put("step 3")

    now = 100.0

    async def _fake_sleep(delay: float):
        nonlocal now, still_current
        if delay > 0.3:
            still_current = False
        now += delay

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == []
    assert ctx.progress_queue.empty()
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario5_cancel_during_idle_sleep_drains_and_finalizes():
    helper, ctx, adapter, _states, key = _make_helper("chat-s5", grouping="accumulate")
    ctx.progress_queue.put("step 1")

    now = 100.0

    async def _fake_sleep(delay: float):
        if delay == 0.3:
            ctx.progress_queue.put("step 2")
            raise asyncio.CancelledError()

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", True)]
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario6a_cancel_during_throttle_sleep_drains_and_finalizes():
    helper, ctx, adapter, _states, key = _make_helper("chat-s6a", grouping="accumulate")
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    now = 100.0

    async def _fake_sleep(delay: float):
        if delay > 0.3:
            raise asyncio.CancelledError()

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", True)]
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
async def test_scenario6b_cancel_during_error_backoff_sleep_drains_and_finalizes():
    helper, ctx, adapter, _states, key = _make_helper("chat-s6b", grouping="accumulate")
    ctx.progress_queue.put("step 1")
    ctx.progress_queue.put("step 2")

    async def _boom(_st):
        raise RuntimeError("typing failed")

    helper._progress_restore_typing = _boom
    now = 100.0

    async def _fake_sleep(delay: float):
        if delay == 1:
            raise asyncio.CancelledError()

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == [("m1", "step 1\nstep 2", True)]
    assert key not in _ACTIVE_PROGRESS_LOOPS


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["interrupted", "not_current"])
async def test_edge_cancel_when_interrupted_or_not_current_drains_without_editing(reason: str):
    still_current = True
    interrupted = False
    helper, ctx, adapter, _states, key = _make_helper(
        f"chat-edge-{reason}",
        grouping="accumulate",
        still_current_fn=lambda: still_current,
        interrupted_fn=lambda: interrupted,
    )
    ctx.progress_queue.put("step 1")

    now = 100.0

    async def _fake_sleep(delay: float):
        nonlocal still_current, interrupted
        if delay == 0.3:
            ctx.progress_queue.put("step 2")
            if reason == "interrupted":
                interrupted = True
            else:
                still_current = False
            raise asyncio.CancelledError()

    with (
        patch("gateway.run_turn_progress.time.monotonic", side_effect=lambda: now),
        patch("gateway.run_turn_progress.asyncio.sleep", side_effect=_fake_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await helper.send_progress_messages()

    assert adapter.sent == ["step 1"]
    assert adapter.edits == []
    assert ctx.progress_queue.empty()
    assert key not in _ACTIVE_PROGRESS_LOOPS
