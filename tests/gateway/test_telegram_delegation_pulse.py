"""Unit tests for Telegram delegation card persistence, pulse ticker lifecycle, and i18n (BDD 1.1-1.4)."""
from __future__ import annotations

import asyncio
import queue
import re
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
    adapter._status_locks = {}
    adapter._status_message_ids = {}
    adapter._STATUS_MESSAGE_IDS_MAX = 256
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


@pytest.mark.asyncio
async def test_send_or_update_status_finalize_passthrough_and_default():
    """BDD 1 & 5: send_or_update_status defaults finalize=True and forwards explicit finalize=False."""
    from gateway.platforms.base import SendResult
    from plugins.platforms.telegram.telegram_status import send_or_update_status

    adapter = MagicMock()
    adapter._status_locks = {}
    adapter._status_message_ids = {("12345", "pulse"): "msg-99"}
    adapter._STATUS_MESSAGE_IDS_MAX = 256
    adapter.edit_message = AsyncMock(return_value=SendResult(success=True, message_id="msg-99"))

    # Backward-compatible call without finalize -> finalize=True
    await send_or_update_status(adapter, "12345", "pulse", "final text")
    assert adapter.edit_message.await_args.kwargs["finalize"] is True

    # Interim call with finalize=False -> finalize=False
    await send_or_update_status(adapter, "12345", "pulse", "interim text", finalize=False)
    assert adapter.edit_message.await_args.kwargs["finalize"] is False


def test_pulse_ticker_interval_defaults_and_minimum():
    """BDD 1: Default interval is 10.0s and minimum is clamped to 5.0s."""
    adapter = _make_adapter()
    assert SubagentPulseTicker(adapter=adapter, chat_id="12345").interval == 10.0
    assert SubagentPulseTicker(adapter=adapter, chat_id="12345", interval_seconds=1.0).interval == 5.0
    assert SubagentPulseTicker(adapter=adapter, chat_id="12345", interval_seconds=12.5).interval == 12.5


@pytest.mark.asyncio
async def test_pulse_ticker_interim_timer_quantization_and_terminal_exact_time():
    """BDD 1-4: _format_time helper remains intact while _render_card omits [MM:SS] and avoids redundant edits."""
    timer_re = re.compile(r"\[\d{2}:\d{2}\]")
    adapter = _make_adapter("msg-q")
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
    assert ticker._format_time(45, quantize_secs=30) == "00:30"
    assert ticker._format_time(65, quantize_secs=30) == "01:00"
    assert ticker._format_time(45) == "00:45"

    with patch("agent.i18n.get_language", return_value="ru"):
        ticker.started_at = 1000.0
        ticker.on_subagent_start(0, "Task A")
        ticker.tasks[0]["started_at"] = 1000.0

        with patch("gateway.subagent_pulse_ticker.time.time", return_value=1010.0):
            card_10 = ticker._render_card()
            await ticker._send_or_edit(card_10, finalize=False)
        with patch("gateway.subagent_pulse_ticker.time.time", return_value=1025.0):
            card_25 = ticker._render_card()
            await ticker._send_or_edit(card_25, finalize=False)
        with patch("gateway.subagent_pulse_ticker.time.time", return_value=1065.0):
            card_65 = ticker._render_card()
            await ticker._send_or_edit(card_65, finalize=False)

        assert not timer_re.search(card_10)
        assert card_10.startswith("⏳ 🔀 Субагент 1/1: Task A")
        assert card_10 == card_25 == card_65
        assert adapter.send.await_count == 1
        assert adapter.edit_message.await_count == 0

        # Batch mode interim & terminal cards omit [MM:SS]
        ticker.on_subagent_start(1, "Task B")
        ticker.tasks[1]["started_at"] = 1000.0
        with patch("gateway.subagent_pulse_ticker.time.time", return_value=1065.0):
            batch_interim = ticker._render_card()
            assert not timer_re.search(batch_interim)
            assert batch_interim.startswith("⏳ 🔀 Субагенты (2 активны):")
            assert " • #1 " in batch_interim
            assert " • #2 " in batch_interim

        ticker.on_subagent_complete(0, status="completed", summary="ok")
        ticker.on_subagent_complete(1, status="completed", summary="ok")
        with patch("gateway.subagent_pulse_ticker.time.time", return_value=1067.0):
            batch_final = ticker._render_card(final_status="completed")
            cancelled_card = ticker._render_card(cancelled=True)
            assert not timer_re.search(batch_final)
            assert not timer_re.search(cancelled_card)
            assert batch_final == "✅ 🔀 Все субагенты (2) завершены"
            assert cancelled_card == "⏹️ 🔀 Выполнение отменено пользователем"


@pytest.mark.asyncio
async def test_pulse_ticker_hides_timer_across_all_states_ru_and_en():
    """BDD 1-5: All _render_card states (RU/EN, batch error, single, prep, timeout, cancel) omit [MM:SS] and [15:00]."""
    timer_re = re.compile(r"\[\d{2}:\d{2}\]")

    for lang in ("ru", "en"):
        with patch("agent.i18n.get_language", return_value=lang):
            adapter = _make_adapter(f"msg-{lang}")
            ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
            ticker.started_at = 1000.0

            # BDD-2: Preparing (total_tasks == 0) at 45s
            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1045.0):
                prep_card = ticker._render_card()
                assert not timer_re.search(prep_card)
                assert prep_card == ("⏳ 🔀 Субагент: подготовка..." if lang == "ru" else "⏳ 🔀 Subagent: preparing...")

            # BDD-2 & BDD-3: Single subagent running, completed, error
            ticker.on_subagent_start(0, "Audit logs")
            ticker.tasks[0]["started_at"] = 1000.0
            ticker.on_subagent_progress(0, "⚙️ Reading")
            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1045.0):
                single_run = ticker._render_card()
                assert not timer_re.search(single_run)
                expected_prefix = "⏳ 🔀 Субагент 1/1: Audit logs" if lang == "ru" else "⏳ 🔀 Subagent 1/1: Audit logs"
                assert single_run == f"{expected_prefix}\n↳ ⚙️ Reading"

            ticker.on_subagent_complete(0, status="completed", summary="All good")
            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1050.0):
                single_done = ticker._render_card(final_status="completed")
                assert not timer_re.search(single_done)
                assert single_done == (
                    "✅ 🔀 Субагент 1/1 завершен: All good"
                    if lang == "ru"
                    else "✅ 🔀 Subagent 1/1 completed: All good"
                )

            ticker.on_subagent_complete(0, status="failed", summary="Boom")
            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1050.0):
                single_err = ticker._render_card(final_status="failed")
                assert not timer_re.search(single_err)
                assert single_err == (
                    "❌ 🔀 Субагент 1/1 ошибка (failed): Boom"
                    if lang == "ru"
                    else "❌ 🔀 Subagent 1/1 error (failed): Boom"
                )

            # BDD-1: Batch mode with running, error, and completed tasks at 65s
            batch_ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
            batch_ticker.started_at = 1000.0
            batch_ticker.on_subagent_start(0, "Task 1")
            batch_ticker.on_subagent_start(1, "Task 2")
            batch_ticker.on_subagent_start(2, "Task 3")
            batch_ticker.on_subagent_progress(0, "⚙️ Обработка: | execute_code ok 0.1s")
            batch_ticker.on_subagent_complete(1, status="error", summary="Failed step")
            batch_ticker.on_subagent_complete(2, status="completed", summary="Done")

            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1065.0):
                batch_card = batch_ticker._render_card()
                assert not timer_re.search(batch_card)
                lines = batch_card.splitlines()
                assert lines[0] == ("⏳ 🔀 Субагенты (1 активны):" if lang == "ru" else "⏳ 🔀 Subagents (1 active):")
                assert lines[1] == " • #1 ⚙️ Обработка: | execute_code ok 0.1s"
                assert lines[2] == (" • #2 ❌ Ошибка" if lang == "ru" else " • #2 ❌ Error")
                assert lines[3] == (" • #3 ✅ Завершен" if lang == "ru" else " • #3 ✅ Completed")

                # Batch stopped/error terminal card
                batch_stopped = batch_ticker._render_card(final_status="failed")
                assert not timer_re.search(batch_stopped)
                assert batch_stopped == (
                    "❌ 🔀 Субагенты остановлены (failed)"
                    if lang == "ru"
                    else "❌ 🔀 Subagents stopped (failed)"
                )

                # Batch completed terminal card
                batch_ticker.on_subagent_complete(0, status="completed", summary="Done 1")
                batch_done = batch_ticker._render_card(final_status="completed")
                assert not timer_re.search(batch_done)
                assert batch_done == (
                    "✅ 🔀 Все субагенты (3) завершены"
                    if lang == "ru"
                    else "✅ 🔀 All subagents (3) completed"
                )

            # BDD-3: Cancelled and Timeout (>= 900s)
            with patch("gateway.subagent_pulse_ticker.time.time", return_value=1950.0):
                timeout_card = batch_ticker._render_card()
                cancelled_card = batch_ticker._render_card(cancelled=True)
                assert not timer_re.search(timeout_card)
                assert "[15:00]" not in timeout_card
                assert timeout_card == (
                    "⏱️ 🔀 Превышен таймаут выполнения (900s)"
                    if lang == "ru"
                    else "⏱️ 🔀 Execution timeout exceeded (900s)"
                )
                assert not timer_re.search(cancelled_card)
                assert cancelled_card == (
                    "⏹️ 🔀 Выполнение отменено пользователем"
                    if lang == "ru"
                    else "⏹️ 🔀 Execution cancelled by user"
                )


@pytest.mark.asyncio
async def test_pulse_ticker_finalize_flags_on_interim_timeout_and_cancel():
    """BDD 1: Interim ticks pass finalize=False; timeout (>=900s), CancelledError, and stop() pass finalize=True."""
    adapter = _make_adapter("msg-fin")
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
    ticker.on_subagent_start(0, "Long task")

    calls: list[bool] = []

    async def spy_send_or_edit(text: str, *, finalize: bool = False) -> None:
        calls.append(finalize)

    ticker._send_or_edit = spy_send_or_edit  # type: ignore[method-assign]

    # 1. Interim tick -> finalize=False, then timeout tick -> finalize=True
    ticker.active = True
    ticker.started_at = 1000.0
    sleep_calls = 0

    async def fake_sleep(_secs: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        ticker.started_at = 0.0  # force timeout on next iteration

    with (
        patch("gateway.subagent_pulse_ticker.time.time", return_value=1010.0),
        patch("gateway.subagent_pulse_ticker.asyncio.sleep", side_effect=fake_sleep),
    ):
        await ticker.run_loop()
    assert calls == [False, True]

    # 2. CancelledError while active -> finalize=True
    calls.clear()
    ticker.active = True
    ticker.started_at = 1000.0

    async def cancel_sleep(_secs: float) -> None:
        raise asyncio.CancelledError()

    with (
        patch("gateway.subagent_pulse_ticker.time.time", return_value=1010.0),
        patch("gateway.subagent_pulse_ticker.asyncio.sleep", side_effect=cancel_sleep),
    ):
        with pytest.raises(asyncio.CancelledError):
            await ticker.run_loop()
    assert calls == [False, True]


@pytest.mark.asyncio
async def test_pulse_ticker_cooldown_slot_budget_skipped_and_magicmock_guard():
    """BDD 1: Cooldown/slot budget skips interim edits without updating _last_content; skipped=True preserves _last_content; MagicMock safe."""
    from gateway.platforms.base import SendResult

    # 1. MagicMock adapter where _chat_outbound_slot_remaining returns a MagicMock (not float)
    mock_adapter = _make_adapter("msg-mm")
    ticker_mm = SubagentPulseTicker(adapter=mock_adapter, chat_id="12345")
    ticker_mm.current_message_id = "msg-mm"
    await ticker_mm._send_or_edit("hello", finalize=False)
    assert ticker_mm._last_content == "hello"

    # 2. Flood cooldown > 0 skips interim edit and leaves _last_content untouched
    adapter = _make_adapter("msg-cd")
    adapter._send_flood_cooldown_remaining = MagicMock(return_value=3.0)
    adapter._chat_outbound_slot_remaining = MagicMock(return_value=0.0)
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345")
    with patch("plugins.platforms.telegram.telegram_status.send_or_update_status", new_callable=AsyncMock) as mock_send:
        await ticker._send_or_edit("interim 1", finalize=False)
        mock_send.assert_not_awaited()
        assert ticker._last_content == ""

        # 3. Outbound slot > 0 skips interim edit when current_message_id is set
        adapter._send_flood_cooldown_remaining.return_value = 0.0
        adapter._chat_outbound_slot_remaining.return_value = 0.7
        ticker.current_message_id = "msg-cd"
        await ticker._send_or_edit("interim 2", finalize=False)
        mock_send.assert_not_awaited()
        assert ticker._last_content == ""

        # 4. finalize=True bypasses cooldown and slot checks
        mock_send.return_value = SendResult(success=True, message_id="msg-cd")
        await ticker._send_or_edit("final card", finalize=True)
        mock_send.assert_awaited_once()
        assert mock_send.await_args.kwargs["finalize"] is True
        assert ticker._last_content == "final card"

        # 5. raw_response={"skipped": True} does NOT update _last_content
        adapter._chat_outbound_slot_remaining.return_value = 0.0
        mock_send.reset_mock()
        mock_send.return_value = SendResult(success=True, message_id="msg-cd", raw_response={"skipped": True})
        await ticker._send_or_edit("skipped text", finalize=False)
        assert ticker._last_content == "final card"


@pytest.mark.asyncio
async def test_pulse_ticker_stop_cleanup_progress_behavior():
    """BDD 3: stop(status='completed') deletes card when cleanup_progress=True, keeps it on error or cleanup_progress=False."""
    # Case A: completed + cleanup_progress=True (via turn_context fallback) -> deletes card & pops _status_message_ids
    adapter = _make_adapter("msg-del-1")
    adapter.delete_message = AsyncMock(return_value=True)
    turn_ctx = TurnContext(session_key="tg:12345", source=_make_telegram_source(), _cleanup_progress=True)
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345", turn_context=turn_ctx)
    adapter._status_message_ids = {("12345", ticker.status_key): "msg-del-1"}
    ticker.current_message_id = "msg-del-1"
    ticker.on_subagent_start(0, "Task")
    ticker.on_subagent_complete(0, status="completed", summary="Done")

    await ticker.stop(status="completed", summary="Done")
    assert adapter.edit_message.await_args.kwargs["finalize"] is True
    adapter.delete_message.assert_awaited_once_with("12345", "msg-del-1")
    assert ("12345", ticker.status_key) not in adapter._status_message_ids

    # Case B: non-awaitable delete_message & non-dict _status_message_ids do not crash
    adapter_sync = _make_adapter("msg-del-2")
    adapter_sync.delete_message = MagicMock(return_value=None)
    adapter_sync._status_message_ids = MagicMock()
    ticker_sync = SubagentPulseTicker(adapter=adapter_sync, chat_id="12345", cleanup_progress=True)
    ticker_sync.current_message_id = "msg-del-2"
    ticker_sync.on_subagent_start(0, "Task")
    ticker_sync.on_subagent_complete(0, status="completed", summary="Done")
    await ticker_sync.stop(status="completed", summary="Done")
    adapter_sync.delete_message.assert_called_once_with("12345", "msg-del-2")

    # Case C: status != 'completed' -> card stays in chat
    adapter_err = _make_adapter("msg-err")
    adapter_err.delete_message = AsyncMock()
    adapter_err._status_message_ids = {}
    ticker_err = SubagentPulseTicker(adapter=adapter_err, chat_id="12345", cleanup_progress=True)
    ticker_err.current_message_id = "msg-err"
    await ticker_err.stop(status="failed", summary="Boom")
    adapter_err.delete_message.assert_not_awaited()

    # Case D: cleanup_progress=False -> card stays in chat
    adapter_keep = _make_adapter("msg-keep")
    adapter_keep.delete_message = AsyncMock()
    ticker_keep = SubagentPulseTicker(adapter=adapter_keep, chat_id="12345", cleanup_progress=False)
    ticker_keep.current_message_id = "msg-keep"
    await ticker_keep.stop(status="completed", summary="Done")
    adapter_keep.delete_message.assert_not_awaited()

    # Case E: cleanup_progress=True with empty tasks or non-completed task does NOT delete card
    adapter_empty = _make_adapter("msg-empty")
    adapter_empty.delete_message = AsyncMock()
    ticker_empty = SubagentPulseTicker(adapter=adapter_empty, chat_id="12345", cleanup_progress=True)
    ticker_empty.current_message_id = "msg-empty"
    await ticker_empty.stop(status="completed", summary="Done")
    adapter_empty.delete_message.assert_not_awaited()

    for bad_status in ("failed", "error", "timeout", "cancelled"):
        adapter_bad = _make_adapter(f"msg-{bad_status}")
        adapter_bad.delete_message = AsyncMock()
        ticker_bad = SubagentPulseTicker(adapter=adapter_bad, chat_id="12345", cleanup_progress=True)
        ticker_bad.current_message_id = f"msg-{bad_status}"
        ticker_bad.on_subagent_start(0, "Task")
        ticker_bad.on_subagent_complete(0, status=bad_status, summary="Err")
        await ticker_bad.stop(status="completed", summary="Done")
        adapter_bad.delete_message.assert_not_awaited()

    adapter_running = _make_adapter("msg-running")
    adapter_running.delete_message = AsyncMock()
    ticker_running = SubagentPulseTicker(adapter=adapter_running, chat_id="12345", cleanup_progress=True)
    ticker_running.current_message_id = "msg-running"
    ticker_running.on_subagent_start(0, "Still running")
    await ticker_running.stop(status="completed", summary="Done")
    adapter_running.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_transcript_passed_in_subagent_start_and_ticker_path(tmp_path):
    """BDD 4: wrap_progress_callback and _run_single_child pass live_transcript; ticker stores tasks[idx]['path']."""
    from tools.delegation_live_log import LiveTranscriptWriter, wrap_progress_callback

    log_path = tmp_path / "deleg-test" / "task-0.log"
    writer = LiveTranscriptWriter("deleg-test", task_index=0, goal="Inspect logs", root=tmp_path)
    assert writer.path == log_path
    captured: list[tuple[tuple, dict]] = []
    wrapped = wrap_progress_callback(lambda *a, **kw: captured.append((a, kw)), writer)
    wrapped("subagent.start", None, "Inspect logs", None, task_index=0)
    assert captured[0][1]["live_transcript"] == str(log_path)

    # Verify _handle_subagent_pulse populates ticker.tasks[0]["path"]
    adapter = _make_adapter("msg-lt")
    runner = SimpleNamespace(_delivery_adapter_for=lambda _src: adapter, _is_draining=lambda: False)
    ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        tool_progress_enabled=True,
        _status_adapter=adapter,
        _loop_for_step=asyncio.get_running_loop(),
        _run_still_current=lambda: True,
    )
    helper = _make_progress_helper(runner, ctx)
    helper.progress_callback(
        "subagent.start", None, "Inspect logs", None, task_index=0, goal="Inspect logs", live_transcript=str(log_path)
    )
    await asyncio.sleep(0.03)
    assert ctx._pulse_ticker is not None
    assert ctx._pulse_ticker.tasks[0]["path"] == log_path
    await ctx._pulse_ticker.stop(status="completed", summary="ok")


def test_child_progress_relay_localization_grouping_and_threadpool_override(tmp_path):
    """BDD 4 & 5: _ChildProgressRelay localizes & groups ×N across ThreadPoolExecutor, supporting ru/en/raw."""
    from concurrent.futures import ThreadPoolExecutor
    from hermes_constants import get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override
    from tools.delegate_tool_progress import _ChildProgressRelay

    parent_cb = MagicMock()
    token = set_hermes_home_override(tmp_path)
    try:
        relay = _ChildProgressRelay(0, "goal", None, parent_cb, 4, "sub-1", "parent-1", 1, None, None, {})
    finally:
        reset_hermes_home_override(token)

    assert relay._hermes_home_override == str(tmp_path)

    for _ in range(4):
        relay("tool.started", "read_file", "a.py", {})
    relay("tool.started", "terminal", "pytest -q", {})
    # 5th tool triggers auto-flush; run another batch inside ThreadPoolExecutor where ContextVar is None
    parent_cb.reset_mock()
    for _ in range(4):
        relay("tool.started", "read_file", "b.py", {})
    relay("tool.started", "terminal", "ls", {})

    seen_override_inside_flush: list[object] = []

    def _spy_get_verb(name: str):
        seen_override_inside_flush.append(get_hermes_home_override())
        return {"read_file": "Чтение файла", "terminal": "Выполнение команды"}.get(name)

    with (
        patch("agent.display.get_tool_verb", side_effect=_spy_get_verb),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        # Force a manual flush in worker thread where ContextVar is unset
        relay.batch = ["read_file", "read_file", "read_file", "read_file", "terminal"]
        pool.submit(relay._flush).result()

    assert seen_override_inside_flush and all(v == str(tmp_path) for v in seen_override_inside_flush)
    preview_ru = parent_cb.call_args_list[-1].kwargs.get("preview") or parent_cb.call_args_list[-1].args[2]
    assert preview_ru == "🔀 [1/4] Чтение файла ×4, Выполнение команды"

    # English locale
    parent_cb.reset_mock()
    relay.batch = ["read_file", "read_file", "terminal"]
    with (
        patch("agent.display._friendly_tool_labels", True),
        patch("agent.i18n.get_language", return_value="en"),
    ):
        relay._flush()
    preview_en = parent_cb.call_args_list[-1].kwargs.get("preview") or parent_cb.call_args_list[-1].args[2]
    assert preview_en == "🔀 [1/4] Reading ×2, Running"

    # friendly_tool_labels=False -> raw names with ×N grouping
    parent_cb.reset_mock()
    relay.batch = ["read_file", "read_file", "terminal"]
    with patch("agent.display._friendly_tool_labels", False):
        relay._flush()
    preview_raw = parent_cb.call_args_list[-1].kwargs.get("preview") or parent_cb.call_args_list[-1].args[2]
    assert preview_raw == "🔀 [1/4] read_file ×2, terminal"

    # SubagentPulseTicker._parse_action_from_content respects friendly_tool_labels=False
    ticker = SubagentPulseTicker(adapter=_make_adapter(), chat_id="12345")
    with patch("agent.display._friendly_tool_labels", False):
        assert ticker._parse_action_from_content("2026-11-19 tool -> terminal(pytest -q)") == "⚙️ terminal (pytest -q)"


@pytest.mark.asyncio
async def test_composite_keys_aliases_mixed_render_and_stop_guard():
    """BDD 4 & 5: composite keys, _subagent_key_aliases, mixed int/str _render_card, and stop('completed') guard."""
    adapter = _make_adapter("msg-comp")
    adapter.delete_message = AsyncMock(return_value=True)
    ticker = SubagentPulseTicker(adapter=adapter, chat_id="12345", cleanup_progress=True)
    ticker.current_message_id = "msg-comp"

    # Two separate delegate_task calls with task_index=0 and different delegation_id do not overwrite each other
    ticker.on_subagent_start(0, "First delegation", delegation_id="deleg_1", subagent_id="sub_1")
    ticker.on_subagent_start(0, "Second delegation", delegation_id="deleg_2", subagent_id="sub_2")
    assert "deleg_1:0" in ticker.tasks
    assert "deleg_2:0" in ticker.tasks
    assert len(ticker.tasks) == 2

    # Alias resolution: start with subagent_id only (no delegation_id), then progress/complete with delegation_id + subagent_id
    ticker.on_subagent_start(0, "Alias task", subagent_id="sub_alias")
    assert "subagent:sub_alias" in ticker.tasks
    ticker.on_subagent_progress(0, "Reading file", delegation_id="deleg_late", subagent_id="sub_alias")
    assert "deleg_late:0" not in ticker.tasks
    assert ticker.tasks["subagent:sub_alias"]["last_action"] == "Reading file"

    # Mixed int and str keys in _render_card do not raise TypeError and number #1..#N
    ticker.on_subagent_start(0, "Legacy int task")
    assert 0 in ticker.tasks
    rendered_running = ticker._render_card(final_status="completed")
    assert "⏳" in rendered_running
    assert "завершен" not in rendered_running.lower()
    assert "#1" in rendered_running and "#4" in rendered_running

    # stop(status="completed") while running tasks exist does NOT stop ticker or delete card
    ticker.active = True
    ticker.ticker_task = asyncio.create_task(asyncio.sleep(10))
    try:
        await ticker.stop(status="completed", summary="Premature")
        assert ticker.active is True
        assert not ticker.ticker_task.done()
        adapter.delete_message.assert_not_awaited()

        # stop(status="interrupted") stops even with running tasks and keeps card
        await ticker.stop(status="interrupted", summary="Stopped")
        assert ticker.active is False
        adapter.delete_message.assert_not_awaited()
    finally:
        if ticker.ticker_task and not ticker.ticker_task.done():
            ticker.ticker_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ticker.ticker_task

    # stop(status="completed", force=True) stops even with running tasks (without deleting card since not all completed)
    ticker2 = SubagentPulseTicker(adapter=_make_adapter("msg-force"), chat_id="12345", cleanup_progress=True)
    ticker2.current_message_id = "msg-force"
    ticker2.active = True
    ticker2.ticker_task = asyncio.create_task(asyncio.sleep(10))
    ticker2.on_subagent_start(0, "Running task", delegation_id="d_force", subagent_id="s_force")
    await ticker2.stop(status="completed", force=True)
    assert ticker2.active is False
    ticker2.adapter.delete_message.assert_not_called()


@pytest.mark.asyncio
async def test_turn_runner_forwards_delegation_and_subagent_ids():
    """BDD 5: TurnRunner._handle_subagent_pulse forwards delegation_id and subagent_id to SubagentPulseTicker."""
    adapter = _make_adapter("msg-fwd")
    runner = SimpleNamespace(_delivery_adapter_for=lambda _src: adapter, _is_draining=lambda: False)
    ctx = TurnContext(
        session_key="tg:12345",
        source=_make_telegram_source(),
        tool_progress_enabled=True,
        _status_adapter=adapter,
        _loop_for_step=asyncio.get_running_loop(),
        _run_still_current=lambda: True,
    )
    helper = _make_progress_helper(runner, ctx)
    helper.progress_callback(
        "subagent.start", None, "Wave A", None,
        task_index=0, goal="Wave A", delegation_id="dA", subagent_id="sA",
    )
    helper.progress_callback(
        "subagent.start", None, "Wave B", None,
        task_index=0, goal="Wave B", delegation_id="dB", subagent_id="sB",
    )
    await asyncio.sleep(0.03)
    assert ctx._pulse_ticker is not None
    assert "dA:0" in ctx._pulse_ticker.tasks
    assert "dB:0" in ctx._pulse_ticker.tasks
    helper.progress_callback(
        "subagent.progress", "terminal", "Step A", None,
        task_index=0, action="Step A", delegation_id="dA", subagent_id="sA",
    )
    helper.progress_callback(
        "subagent.complete", None, "Done A", None,
        task_index=0, status="completed", summary="Done A", delegation_id="dA", subagent_id="sA",
    )
    helper.progress_callback(
        "subagent.complete", None, "Done B", None,
        task_index=0, status="completed", summary="Done B", delegation_id="dB", subagent_id="sB",
    )
    await asyncio.sleep(0.03)
    assert ctx._pulse_ticker.tasks["dA:0"]["status"] == "completed"
    assert ctx._pulse_ticker.tasks["dB:0"]["status"] == "completed"
    await ctx._pulse_ticker.stop(status="completed", force=True)
