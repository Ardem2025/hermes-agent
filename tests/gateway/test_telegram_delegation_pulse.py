"""Unit tests for Telegram delegation card persistence, pulse ticker lifecycle, and i18n (BDD 1.1-1.4)."""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.subagent_pulse_ticker import SubagentPulseTicker
from gateway.turn_context import TurnContext


def _make_telegram_source(chat_id: str = "12345", thread_id: str | None = None):
    return SimpleNamespace(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        thread_id=thread_id,
        user_id="u1",
    )


def _make_adapter(message_id: str = "msg-pulse-1"):
    adapter = MagicMock()
    adapter._send_flood_cooldown_remaining = MagicMock(return_value=0.0)
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id=message_id))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True, message_id=message_id))
    return adapter


def _make_progress_helper(runner, ctx):
    helper = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    helper._runner = runner
    helper._ctx = ctx
    return helper


def _make_turn_runner(draining: bool = False):
    runner = GatewayTurnMixin.__new__(GatewayTurnMixin)
    runner._draining = draining
    runner._active_sessions = set()
    runner._active_session_tasks = {}
    runner._session_locks = {}
    runner._session_lock_refs = {}
    runner._release_session_lock = MagicMock()
    runner._release_running_agent_state = MagicMock()
    runner._update_runtime_status = MagicMock()
    return runner


@pytest.mark.asyncio
async def test_pulse_ticker_does_not_register_message_in_cleanup_msg_ids():
    """BDD 1.1: SubagentPulseTicker card must NOT be added to _cleanup_msg_ids even when _cleanup_progress=True."""
    adapter = _make_adapter("tg-card-42")
    turn_ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        _cleanup_progress=True,
        _cleanup_msg_ids=[],
    )
    ticker = SubagentPulseTicker(
        adapter=adapter,
        chat_id="12345",
        turn_context=turn_ctx,
    )
    ticker.on_subagent_start(0, "Audit gateway")
    await ticker._send_or_edit(ticker._render_card())

    assert ticker.current_message_id == "tg-card-42"
    assert turn_ctx._cleanup_msg_ids == []


def test_progress_callback_suppresses_delegate_task_tool_started_on_telegram():
    """BDD 1.1: On Telegram, tool.started for delegate_task is suppressed so no temporary bubble flickers."""
    adapter = _make_adapter()
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda _src: adapter,
        _is_draining=lambda: False,
    )
    ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        _status_adapter=adapter,
        _run_still_current=lambda: True,
    )
    helper = _make_progress_helper(runner, ctx)
    emitted: list[str] = []
    helper._progress_emit = emitted.append

    helper.progress_callback("tool.started", "delegate_task", "goal", {"tasks": [{"goal": "x"}]})
    assert emitted == []

    # Other tools still emit progress normally
    helper.progress_callback("tool.started", "terminal", "ls -la", {"command": "ls -la"})
    assert len(emitted) == 1


def test_progress_callback_keeps_delegate_task_tool_started_on_non_telegram():
    """Non-Telegram platforms without SubagentPulseTicker still emit delegate_task tool.started."""
    adapter = _make_adapter()
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda _src: adapter,
        _is_draining=lambda: False,
    )
    ctx = TurnContext(
        session_key="dc:999",
        source=SimpleNamespace(platform=Platform.DISCORD, chat_id="999", thread_id=None),
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        _run_still_current=lambda: True,
    )
    helper = _make_progress_helper(runner, ctx)
    emitted: list[str] = []
    helper._progress_emit = emitted.append

    helper.progress_callback("tool.started", "delegate_task", "goal", {"tasks": [{"goal": "x"}]})
    assert len(emitted) == 1


@pytest.mark.asyncio
async def test_cleanup_turn_tasks_keeps_pulse_ticker_alive_on_normal_background_exit():
    """BDD 1.2: Normal parent turn completion with running background subagents must NOT stop ticker as interrupted."""
    adapter = _make_adapter()
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
    ticker.active = True
    ticker.on_subagent_start(0, "Background research")
    ticker.stop = AsyncMock()

    turn_ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        _pulse_ticker=ticker,
        agent_holder=[SimpleNamespace(is_interrupted=False)],
        result_holder=[{"final_response": "Started in background"}],
    )
    runner = _make_turn_runner(draining=False)

    async def _noop():
        return None

    t1 = asyncio.create_task(_noop())
    t2 = asyncio.create_task(_noop())
    t3 = asyncio.create_task(_noop())
    t4 = asyncio.create_task(_noop())
    await asyncio.gather(t1, t2, t3, t4)

    await runner._run_agent_cleanup_turn_tasks(
        turn_ctx,
        progress_task=None,
        log_task=None,
        interrupt_monitor=t1,
        _notify_task=t2,
        tracking_task=t3,
        stream_task=t4,
        interrupted=False,
    )

    ticker.stop.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interrupted_arg,agent_interrupted,result_interrupted,draining",
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
async def test_cleanup_turn_tasks_stops_pulse_ticker_as_interrupted_on_real_interrupt(
    interrupted_arg, agent_interrupted, result_interrupted, draining
):
    """BDD 1.2: Real interrupt/cancel/drain stops active SubagentPulseTicker with status='interrupted'."""
    adapter = _make_adapter()
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
    ticker.active = True
    ticker.on_subagent_start(0, "Background research")
    ticker.stop = AsyncMock()

    turn_ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        _pulse_ticker=ticker,
        agent_holder=[SimpleNamespace(is_interrupted=agent_interrupted)],
        result_holder=[{"interrupted": result_interrupted}],
    )
    runner = _make_turn_runner(draining=draining)

    async def _noop():
        return None

    t1 = asyncio.create_task(_noop())
    t2 = asyncio.create_task(_noop())
    t3 = asyncio.create_task(_noop())
    t4 = asyncio.create_task(_noop())
    await asyncio.gather(t1, t2, t3, t4)

    await runner._run_agent_cleanup_turn_tasks(
        turn_ctx,
        progress_task=None,
        log_task=None,
        interrupt_monitor=t1,
        _notify_task=t2,
        tracking_task=t3,
        stream_task=t4,
        interrupted=interrupted_arg,
    )

    ticker.stop.assert_awaited_once_with(status="interrupted")


@pytest.mark.asyncio
async def test_pulse_ticker_stop_does_not_render_cancelled_card_on_task_cancel():
    """Stopping ticker cleanly sets active=False before cancelling loop so CancelledError won't overwrite final card."""
    adapter = _make_adapter("msg-1")
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345", interval_seconds=10.0)
    ticker.on_subagent_start(0, "Task A")
    await ticker.start()
    await asyncio.sleep(0.02)

    ticker.on_subagent_complete(0, status="completed", summary="Done")
    await ticker.stop(status="completed", summary="Done")

    sent_texts = [call.kwargs.get("content") or call.args[1] for call in adapter.send.await_args_list]
    edited_texts = [call.kwargs.get("content") or call.args[2] for call in adapter.edit_message.await_args_list]
    all_texts = sent_texts + edited_texts
    assert any("✅" in txt for txt in all_texts)
    assert not any("⏹️" in txt for txt in all_texts)


@pytest.mark.asyncio
async def test_handle_subagent_pulse_resolves_status_adapter_and_completes_after_parent_turn():
    """BDD 1.3: _handle_subagent_pulse works with _status_adapter and processes subagent.complete after parent turn."""
    adapter = _make_adapter("msg-bg-1")
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda _src: adapter,
        _is_draining=lambda: False,
    )
    ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        tool_progress_enabled=True,
        _status_adapter=adapter,
        _cleanup_progress=True,
        _cleanup_msg_ids=[],
        _loop_for_step=asyncio.get_running_loop(),
        _run_still_current=lambda: False,
    )
    helper = _make_progress_helper(runner, ctx)

    helper.progress_callback("subagent.start", None, "Analyze logs", None, task_index=0, goal="Analyze logs")
    await asyncio.sleep(0.05)
    assert ctx._pulse_ticker is not None
    assert ctx._pulse_ticker.active is True
    assert ctx._cleanup_msg_ids == []

    helper.progress_callback("subagent.progress", "terminal", "ls -la", None, task_index=0)
    helper.progress_callback(
        "subagent.complete",
        None,
        "All clear",
        None,
        task_index=0,
        status="completed",
        summary="All clear",
    )
    await asyncio.sleep(0.05)

    assert ctx._pulse_ticker.active is False
    edited_texts = [call.kwargs.get("content") or call.args[2] for call in adapter.edit_message.await_args_list]
    assert any("✅" in txt and "All clear" in txt for txt in edited_texts)


def test_pulse_ticker_respects_get_language_ru_and_en(tmp_path):
    """BDD 1.4: SubagentPulseTicker localizes cards and transcript actions for both 'ru' and 'en'."""
    transcript = tmp_path / "subagent.log"
    transcript.write_text("2026-11-19 tool -> terminal(pytest -q)\n", encoding="utf-8")

    adapter = _make_adapter()

    with patch("agent.i18n.get_language", return_value="ru"):
        ticker_ru = SubagentPulseTicker(adapter=adapter, chat_id="12345")
        assert "подготовка" in ticker_ru._render_card()
        ticker_ru.on_subagent_start(0, "", transcript_path=transcript)
        running_ru = ticker_ru._render_card()
        assert "Субагент 1/1" in running_ru
        assert "Выполнение команды" in running_ru
        ticker_ru.on_subagent_complete(0, status="completed", summary="Готово")
        done_ru = ticker_ru._render_card(final_status="completed")
        assert "Субагент 1/1 завершен: Готово" in done_ru
        assert "отменено пользователем" in ticker_ru._render_card(cancelled=True)

    with patch("agent.i18n.get_language", return_value="en"):
        ticker_en = SubagentPulseTicker(adapter=adapter, chat_id="12345")
        assert "preparing" in ticker_en._render_card()
        ticker_en.on_subagent_start(0, "", transcript_path=transcript)
        running_en = ticker_en._render_card()
        assert "Subagent 1/1" in running_en
        assert "Running (pytest -q)" in running_en
        assert "Выполнение" not in running_en
        ticker_en.on_subagent_complete(0, status="completed", summary="Done")
        done_en = ticker_en._render_card(final_status="completed")
        assert "Subagent 1/1 completed: Done" in done_en
        assert "cancelled by user" in ticker_en._render_card(cancelled=True)
