"""Isolated BDD-1..BDD-7 tests for WebUI -> Telegram SubagentPulseTicker mirroring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.platforms.api_server as api_server_mod
import gateway.session_api_turn as session_api_turn_mod
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult
from gateway.session_api_turn import (
    api_execution,
    publish_api_tool_progress,
    wire_api_observers,
)
from hermes_state import SessionDB

SANDBOX_ROOT = Path(__file__).resolve().parents[2].resolve()


class FakeTelegramAdapter:
    """Fake Telegram adapter matching telegram_status.py and SendResult contracts."""

    def __init__(self) -> None:
        self._status_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._status_message_ids: dict[tuple[str, str], str] = {}
        self._STATUS_MESSAGE_IDS_MAX = 256
        self.calls: list[dict] = []
        self._next_msg_id = 100

    async def send(
        self,
        chat_id: str,
        content: str = "",
        *,
        text: str | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> SendResult:
        self._next_msg_id += 1
        msg_id = str(self._next_msg_id)
        body = text if text is not None and not content else content
        self.calls.append(
            {
                "op": "send",
                "chat_id": str(chat_id),
                "message_id": msg_id,
                "content": body,
                "metadata": dict(metadata or {}),
            }
        )
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: dict | None = None,
        **kwargs,
    ) -> SendResult:
        self.calls.append(
            {
                "op": "edit_message",
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "content": content,
                "finalize": finalize,
                "metadata": dict(metadata or {}),
            }
        )
        return SendResult(success=True, message_id=str(message_id))


def _assert_sandbox_imports() -> None:
    assert Path(api_server_mod.__file__).resolve().is_relative_to(SANDBOX_ROOT)
    assert Path(session_api_turn_mod.__file__).resolve().is_relative_to(SANDBOX_ROOT)


def _make_bound_db(tmp_path: Path, session_id: str = "sess_pulse_1", *, delivery_enabled: bool = True) -> SessionDB:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="api_server")
    if hasattr(db, "apply_telegram_topic_migration"):
        db.apply_telegram_topic_migration()
    db.bind_telegram_topic(
        chat_id="-1001234567890",
        thread_id="777",
        user_id="12345",
        session_key=f"agent:main:telegram:dm:12345:thread:777",
        session_id=session_id,
        managed_mode="api",
        delivery_enabled=delivery_enabled,
    )
    return db


def _make_adapter(db: SessionDB, telegram: FakeTelegramAdapter | None) -> APIServerAdapter:
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = db
    adapter._telegram_runner_and_adapter = lambda: (None, telegram)
    return adapter


def _make_stream_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


def _parse_sse(raw: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        ev_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                ev_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((ev_name, json.loads("\n".join(data_lines))))
    return events


@pytest.mark.asyncio
async def test_bdd1_webui_bound_session_mirrors_pulse_and_sse_both_paths(tmp_path):
    """BDD-1: Bound WebUI session mirrors subagent pulse to Telegram on both execution paths."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_bdd1", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)

    async def fake_run_agent(**kwargs):
        cb = kwargs["tool_progress_callback"]
        delta_cb = kwargs["stream_delta_callback"]
        cb("subagent.start", None, "Research task", None, task_index=0, goal="Research task")
        await asyncio.sleep(0.02)
        cb("subagent.progress", "terminal", "Running pytest", None, task_index=0, action="Running pytest")
        await asyncio.sleep(0.02)
        cb("subagent.complete", None, "Done summary", None, task_index=0, status="completed", summary="Done summary")
        delta_cb("Hello ")
        return {"final_response": "Final WebUI answer", "session_id": "sess_bdd1"}, {"total_tokens": 10}

    adapter._run_agent = fake_run_agent
    app = _make_stream_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions/sess_bdd1/chat/stream",
            json={"message": "Run subagent", "duplicate_to_telegram": True},
        )
        assert resp.status == 200
        raw = await resp.text()

    await asyncio.sleep(0.05)
    sse_events = _parse_sse(raw)
    sse_names = [name for name, _ in sse_events]
    assert "assistant.delta" in sse_names
    assert "assistant.completed" in sse_names
    assert "run.completed" in sse_names
    assert "done" in sse_names

    pulse_calls = [c for c in telegram.calls if c["metadata"].get("origin") != "webui_mirror"]
    mirror_calls = [c for c in telegram.calls if c["metadata"].get("origin") == "webui_mirror"]
    assert len(pulse_calls) >= 1
    assert any(w in c["content"].lower() for c in pulse_calls for w in ("завершен", "completed"))
    assert len(mirror_calls) == 1
    assert mirror_calls[0]["content"] == "Final WebUI answer"
    assert mirror_calls[0]["metadata"].get("thread_id") == "777"

    # Path 2: session_api_turn.wire_api_observers + publish_api_tool_progress for API_SERVER session
    received: list[tuple[tuple, dict]] = []
    live_session = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.API_SERVER),
        event_stream=SimpleNamespace(lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None), execution={"admission_id": "adm_1"}),
    )
    authority = SimpleNamespace(
        sessions={"sess_bdd1": live_session},
        api_observers={"adm_1": [{"tool_progress_callback": lambda *a, **kw: received.append((a, kw))}]},
        check_approval_generation=lambda sid, gen: None,
    )
    agent = SimpleNamespace(
        status_callback=None,
        interim_assistant_callback=None,
        tool_progress_callback=None,
        reasoning_callback=None,
    )
    wire_api_observers(agent, (authority, "sess_bdd1", 1), want_interim=False)
    agent.tool_progress_callback("subagent.start", None, "Goal", None, task_index=0, goal="Goal")
    publish_api_tool_progress(authority, "sess_bdd1", 1, "subagent.progress", "terminal", "Step", None, task_index=0)
    assert len(received) == 2
    for _args, kw in received:
        assert kw["source_platform"] == Platform.API_SERVER.value
        assert kw["telegram_pulse_owned"] is False
    db.close()


@pytest.mark.asyncio
async def test_bdd2_no_telegram_calls_when_unbound_disabled_or_missing_adapter(tmp_path):
    """BDD-2: Unbound session, duplicate_to_telegram=False, or telegram_adapter=None emits 0 Telegram calls."""
    _assert_sandbox_imports()

    # Case A: unbound session (delivery_enabled=False)
    db_unbound = _make_bound_db(tmp_path / "a", "sess_unbound", delivery_enabled=False)
    (tmp_path / "a").mkdir(exist_ok=True)
    db_unbound.close()
    (tmp_path / "a").mkdir(exist_ok=True)
    db_a = _make_bound_db(tmp_path / "a", "sess_unbound", delivery_enabled=False)
    tg_a = FakeTelegramAdapter()
    adapter_a = _make_adapter(db_a, tg_a)

    async def run_subagent(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "G", None, task_index=0, goal="G")
        cb("subagent.complete", None, "S", None, task_index=0, status="completed", summary="S")
        return {"final_response": "Answer", "session_id": kwargs.get("session_id", "sess_unbound")}, {}

    adapter_a._run_agent = run_subagent
    async with TestClient(TestServer(_make_stream_app(adapter_a))) as cli:
        resp = await cli.post("/api/sessions/sess_unbound/chat/stream", json={"message": "hi"})
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.03)
    assert tg_a.calls == []
    db_a.close()

    # Case B: bound session, explicit duplicate_to_telegram=False
    (tmp_path / "b").mkdir(exist_ok=True)
    db_b = _make_bound_db(tmp_path / "b", "sess_optout", delivery_enabled=True)
    tg_b = FakeTelegramAdapter()
    adapter_b = _make_adapter(db_b, tg_b)
    adapter_b._run_agent = run_subagent
    async with TestClient(TestServer(_make_stream_app(adapter_b))) as cli:
        resp = await cli.post(
            "/api/sessions/sess_optout/chat/stream",
            json={"message": "hi", "duplicate_to_telegram": False},
        )
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.03)
    assert tg_b.calls == []
    db_b.close()

    # Case C: bound session, telegram_adapter=None
    (tmp_path / "c").mkdir(exist_ok=True)
    db_c = _make_bound_db(tmp_path / "c", "sess_noadapter", delivery_enabled=True)
    adapter_c = _make_adapter(db_c, None)
    adapter_c._run_agent = run_subagent
    async with TestClient(TestServer(_make_stream_app(adapter_c))) as cli:
        resp = await cli.post("/api/sessions/sess_noadapter/chat/stream", json={"message": "hi"})
        assert resp.status == 200
        raw = await resp.text()
    assert "run.completed" in raw
    db_c.close()


@pytest.mark.asyncio
async def test_bdd3_live_bot_chat_handoff_and_telegram_owned_pulse_dedup(tmp_path):
    """BDD-3: Live Bot Chat handoff skips local ticker; Platform.TELEGRAM marks telegram_pulse_owned=True."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_bdd3", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)

    # Part 1: _stream_through_live_bot_chat returns response when admitted is not None
    adapter._admit_to_live_bot_chat = AsyncMock(return_value=(tmp_path, {"session_id": "sess_bdd3", "delivery_id": "del_1"}))
    adapter._await_live_bot_chat_receipt = AsyncMock(
        return_value={"delivery_id": "del_1", "status": "settled", "reply": "Bot Chat reply"}
    )
    adapter._run_agent = AsyncMock(side_effect=AssertionError("_run_agent must not run on live bot chat handoff"))
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess_bdd3/chat/stream", json={"message": "hello"})
        assert resp.status == 200
        raw = await resp.text()
    events = _parse_sse(raw)
    assert any(n == "assistant.completed" and p["content"] == "Bot Chat reply" for n, p in events)
    assert telegram.calls == []

    # Part 2: Fallback when admitted is None + telegram_pulse_owned=True event is ignored by api_server ticker
    adapter._admit_to_live_bot_chat = AsyncMock(return_value=None)

    async def run_with_tg_owned(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb(
            "subagent.start",
            None,
            "TG owned",
            None,
            task_index=0,
            goal="TG owned",
            source_platform=Platform.TELEGRAM.value,
            telegram_pulse_owned=True,
        )
        cb(
            "subagent.complete",
            None,
            "TG done",
            None,
            task_index=0,
            status="completed",
            source_platform=Platform.TELEGRAM.value,
            telegram_pulse_owned=True,
        )
        return {"final_response": "Fallback reply", "session_id": "sess_bdd3"}, {}

    adapter._run_agent = run_with_tg_owned
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess_bdd3/chat/stream", json={"message": "hello"})
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.03)
    pulse_calls = [c for c in telegram.calls if c["metadata"].get("origin") != "webui_mirror"]
    assert pulse_calls == []

    # Part 3: session_api_turn marks Platform.TELEGRAM with telegram_pulse_owned=True
    tg_received: list[dict] = []
    tg_session = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.TELEGRAM),
        event_stream=SimpleNamespace(lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None), execution={"admission_id": "adm_tg"}),
    )
    authority = SimpleNamespace(
        sessions={"sess_tg": tg_session},
        api_observers={"adm_tg": [{"tool_progress_callback": lambda *a, **kw: tg_received.append(kw)}]},
        check_approval_generation=lambda sid, gen: None,
    )
    publish_api_tool_progress(authority, "sess_tg", 1, "subagent.start", None, "G", None, task_index=0)
    assert len(tg_received) == 1
    assert tg_received[0]["source_platform"] == Platform.TELEGRAM.value
    assert tg_received[0]["telegram_pulse_owned"] is True
    db.close()


@pytest.mark.asyncio
async def test_bdd4_error_cancellation_and_failure_statuses_no_double_stop(tmp_path):
    """BDD-4: Failed/error/timeout subagents preserve error status; CancelledError & exceptions stop ticker once."""
    _assert_sandbox_imports()
    from gateway.subagent_pulse_ticker import SubagentPulseTicker

    # Case A: subagent completes with failed/error/timeout -> status not overwritten to completed, stop called once
    for fail_status in ("failed", "error", "timeout"):
        sub_dir = tmp_path / fail_status
        sub_dir.mkdir()
        db = _make_bound_db(sub_dir, f"sess_{fail_status}", delivery_enabled=True)
        tg = FakeTelegramAdapter()
        adapter = _make_adapter(db, tg)
        stop_calls: list[tuple[str, str]] = []
        orig_stop = SubagentPulseTicker.stop

        async def tracked_stop(self, status="completed", summary=""):
            stop_calls.append((status, summary))
            return await orig_stop(self, status=status, summary=summary)

        async def run_fail(**kwargs):
            cb = kwargs["tool_progress_callback"]
            cb("subagent.start", None, "Fail task", None, task_index=0, goal="Fail task")
            await asyncio.sleep(0.02)
            cb("subagent.complete", None, "Boom", None, task_index=0, status=fail_status, summary="Boom")
            return {"final_response": "Handled failure", "session_id": f"sess_{fail_status}"}, {}

        adapter._run_agent = run_fail
        with patch.object(SubagentPulseTicker, "stop", tracked_stop):
            async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
                resp = await cli.post(f"/api/sessions/sess_{fail_status}/chat/stream", json={"message": "go"})
                assert resp.status == 200
                raw = await resp.text()
        await asyncio.sleep(0.03)
        assert stop_calls == [(fail_status, "Boom")]
        assert any(fail_status in c["content"] for c in tg.calls)
        assert "done" in [n for n, _ in _parse_sse(raw)]
        db.close()

    # Case B: exception inside _run_agent -> ticker stopped once with failed, SSE gets error + done
    exc_dir = tmp_path / "exc"
    exc_dir.mkdir()
    db_exc = _make_bound_db(exc_dir, "sess_exc", delivery_enabled=True)
    tg_exc = FakeTelegramAdapter()
    adapter_exc = _make_adapter(db_exc, tg_exc)
    stop_calls_exc: list[str] = []
    orig_stop = SubagentPulseTicker.stop

    async def tracked_stop_exc(self, status="completed", summary=""):
        stop_calls_exc.append(status)
        return await orig_stop(self, status=status, summary=summary)

    async def run_raises(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Crash task", None, task_index=0, goal="Crash task")
        await asyncio.sleep(0.02)
        raise RuntimeError("agent exploded")

    adapter_exc._run_agent = run_raises
    with patch.object(SubagentPulseTicker, "stop", tracked_stop_exc):
        async with TestClient(TestServer(_make_stream_app(adapter_exc))) as cli:
            resp = await cli.post("/api/sessions/sess_exc/chat/stream", json={"message": "go"})
            assert resp.status == 200
            raw = await resp.text()
    await asyncio.sleep(0.03)
    assert stop_calls_exc == ["failed"]
    names = [n for n, _ in _parse_sse(raw)]
    assert "error" in names and "done" in names
    db_exc.close()

    # Case C: CancelledError inside _run_agent -> ticker stopped once with cancelled, SSE gets done
    can_dir = tmp_path / "cancel"
    can_dir.mkdir()
    db_can = _make_bound_db(can_dir, "sess_can", delivery_enabled=True)
    tg_can = FakeTelegramAdapter()
    adapter_can = _make_adapter(db_can, tg_can)
    stop_calls_can: list[str] = []

    async def tracked_stop_can(self, status="completed", summary=""):
        stop_calls_can.append(status)
        return await orig_stop(self, status=status, summary=summary)

    async def run_cancelled(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Cancel task", None, task_index=0, goal="Cancel task")
        await asyncio.sleep(0.02)
        raise asyncio.CancelledError()

    adapter_can._run_agent = run_cancelled
    with patch.object(SubagentPulseTicker, "stop", tracked_stop_can):
        async with TestClient(TestServer(_make_stream_app(adapter_can))) as cli:
            resp = await cli.post("/api/sessions/sess_can/chat/stream", json={"message": "go"})
            assert resp.status == 200
            raw = await resp.text()
    await asyncio.sleep(0.03)
    assert stop_calls_can == ["cancelled"]
    assert "done" in [n for n, _ in _parse_sse(raw)]
    db_can.close()


@pytest.mark.asyncio
async def test_bdd5_final_pulse_card_precedes_final_response_in_telegram(tmp_path):
    """BDD-5: Final ticker card update strictly precedes final_response send in FakeTelegramAdapter log."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_bdd5", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)

    async def run_agent(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Order check", None, task_index=0, goal="Order check")
        await asyncio.sleep(0.02)
        cb("subagent.complete", None, "All good", None, task_index=0, status="completed", summary="All good")
        return {"final_response": "FINAL_MIRROR_TEXT", "session_id": "sess_bdd5"}, {}

    adapter._run_agent = run_agent
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess_bdd5/chat/stream", json={"message": "order"})
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.05)

    final_card_idx = next(
        i for i, c in enumerate(telegram.calls)
        if c["metadata"].get("origin") != "webui_mirror" and any(w in c["content"].lower() for w in ("завершен", "completed"))
    )
    mirror_idx = next(
        i for i, c in enumerate(telegram.calls)
        if c["metadata"].get("origin") == "webui_mirror" and c["content"] == "FINAL_MIRROR_TEXT"
    )
    assert final_card_idx < mirror_idx
    db.close()


@pytest.mark.asyncio
async def test_bdd6_second_delegate_wave_creates_fresh_ticker_and_status_key(tmp_path):
    """BDD-6: Second delegate_task wave in the same turn creates a new SubagentPulseTicker with clean tasks."""
    _assert_sandbox_imports()
    from gateway.subagent_pulse_ticker import SubagentPulseTicker

    db = _make_bound_db(tmp_path, "sess_bdd6", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)
    created_tickers: list[SubagentPulseTicker] = []
    orig_init = SubagentPulseTicker.__init__

    def spy_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        created_tickers.append(self)

    async def run_two_waves(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Wave 1", None, task_index=0, goal="Wave 1")
        await asyncio.sleep(0.02)
        cb("subagent.complete", None, "W1 done", None, task_index=0, status="completed", summary="W1 done")
        await asyncio.sleep(0.03)
        cb("subagent.start", None, "Wave 2", None, task_index=0, goal="Wave 2")
        await asyncio.sleep(0.02)
        cb("subagent.complete", None, "W2 done", None, task_index=0, status="completed", summary="W2 done")
        return {"final_response": "Two waves done", "session_id": "sess_bdd6"}, {}

    adapter._run_agent = run_two_waves
    with patch.object(SubagentPulseTicker, "__init__", spy_init):
        async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
            resp = await cli.post("/api/sessions/sess_bdd6/chat/stream", json={"message": "waves"})
            assert resp.status == 200
            await resp.text()
    await asyncio.sleep(0.05)

    assert len(created_tickers) == 2
    t1, t2 = created_tickers
    assert t1.status_key != t2.status_key
    assert t1.tasks[0]["goal"] == "Wave 1"
    assert t2.tasks[0]["goal"] == "Wave 2"
    assert len(t2.tasks) == 1
    db.close()


@pytest.mark.asyncio
async def test_bdd7_sse_disconnect_preserves_turn_and_delivers_telegram_pulse(tmp_path):
    """BDD-7: SSE client disconnect with preserve_on_disconnect=True completes turn and delivers to Telegram."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_bdd7", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)
    turn_finished = asyncio.Event()

    async def run_slow(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Long task", None, task_index=0, goal="Long task")
        await asyncio.sleep(0.12)
        cb("subagent.complete", None, "Finished after disconnect", None, task_index=0, status="completed", summary="Finished after disconnect")
        turn_finished.set()
        return {"final_response": "Delivered after disconnect", "session_id": "sess_bdd7"}, {}

    adapter._run_agent = run_slow
    orig_prepare = adapter._prepare_sse_response

    async def disconnecting_prepare(request, session_id, gateway_session_key):
        resp = await orig_prepare(request, session_id, gateway_session_key)
        orig_write = resp.write
        writes = 0

        async def flaky_write(data):
            nonlocal writes
            writes += 1
            if writes >= 2:
                raise ConnectionResetError("client disconnected")
            return await orig_write(data)

        resp.write = flaky_write
        return resp

    adapter._prepare_sse_response = disconnecting_prepare
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post(
            "/api/sessions/sess_bdd7/chat/stream",
            json={"message": "disconnect me", "preserve_on_disconnect": True},
        )
        assert resp.status == 200
        await resp.read()

    await asyncio.wait_for(turn_finished.wait(), timeout=2.0)
    await asyncio.sleep(0.08)

    assert any(
        any(w in c["content"].lower() for w in ("завершен", "completed"))
        for c in telegram.calls
        if c["metadata"].get("origin") != "webui_mirror"
    )
    assert any(
        c["metadata"].get("origin") == "webui_mirror" and c["content"] == "Delivered after disconnect"
        for c in telegram.calls
    )
    db.close()


def test_session_response_subagent_tool_status_localization(tmp_path):
    """BDD 4 & 5: _session_response() formats active subagent tool_status via get_tool_verb (ru/en/raw/unknown)."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_tool_status", delivery_enabled=False)
    adapter = _make_adapter(db, None)
    adapter._get_active_subagents_for_session = lambda _sid: [{"last_tool": "read_file", "status": "running"}]
    session_row = {"id": "sess_tool_status", "source": "api_server"}

    with (
        patch("agent.display._friendly_tool_labels", True),
        patch("agent.i18n.get_language", return_value="ru"),
    ):
        res_ru = adapter._session_response(session_row)
    assert res_ru["tool_status"] == "Подзадача: Чтение файла..."

    with (
        patch("agent.display._friendly_tool_labels", True),
        patch("agent.i18n.get_language", return_value="en"),
    ):
        res_en = adapter._session_response(session_row)
    assert res_en["tool_status"] == "Подзадача: Reading..."

    with patch("agent.display._friendly_tool_labels", False):
        res_raw = adapter._session_response(session_row)
    assert res_raw["tool_status"] == "Подзадача выполняет read_file..."

    adapter._get_active_subagents_for_session = lambda _sid: [{"last_tool": "custom_unknown_tool", "status": "running"}]
    with (
        patch("agent.display._friendly_tool_labels", True),
        patch("agent.i18n.get_language", return_value="ru"),
    ):
        res_unk = adapter._session_response(session_row)
    assert res_unk["tool_status"] == "Подзадача выполняет custom_unknown_tool..."
    db.close()
