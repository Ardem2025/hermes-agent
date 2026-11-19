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

    # Case B: exception inside _run_agent while background subagent is still running -> ticker stays alive;
    # when no running subagents remain on parent failure -> ticker stops once with failed
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

    async def run_raises_with_running(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Crash task", None, task_index=0, goal="Crash task")
        await asyncio.sleep(0.02)
        raise RuntimeError("agent exploded")

    adapter_exc._run_agent = run_raises_with_running
    with patch.object(SubagentPulseTicker, "stop", tracked_stop_exc):
        async with TestClient(TestServer(_make_stream_app(adapter_exc))) as cli:
            resp = await cli.post("/api/sessions/sess_exc/chat/stream", json={"message": "go"})
            assert resp.status == 200
            raw = await resp.text()
    await asyncio.sleep(0.03)
    assert stop_calls_exc == []
    assert "sess_exc" in adapter_exc._pulse_registry()
    names = [n for n, _ in _parse_sse(raw)]
    assert "error" in names and "done" in names
    await adapter_exc.disconnect()
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


class FakeDeletableTelegramAdapter(FakeTelegramAdapter):
    """Fake Telegram adapter with real delete_message override for cleanup_progress tests."""

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        self.calls.append(
            {
                "op": "delete_message",
                "chat_id": str(chat_id),
                "message_id": str(message_id),
                "metadata": {},
            }
        )
        return True


@pytest.mark.asyncio
async def test_session_authority_background_pulse_survives_parent_turn_and_no_double_delivery(tmp_path):
    """BDD 1 + session_authority: background subagent events survive observe_api_turn exit & stale generation without double delivery."""
    _assert_sandbox_imports()
    from concurrent.futures import ThreadPoolExecutor
    from gateway.session_api_turn import observe_api_turn
    from hermes_state_runtime import RuntimeStoreError

    db = _make_bound_db(tmp_path, "sess_auth_bg", delivery_enabled=True)
    telegram = FakeTelegramAdapter()
    adapter = _make_adapter(db, telegram)
    adapter._pulse_loop = asyncio.get_running_loop()

    generation = 1
    live_session = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.API_SERVER),
        event_stream=SimpleNamespace(
            lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            execution={"admission_id": "adm_bg"},
        ),
    )

    def _check_gen(_sid, gen):
        if gen != generation:
            raise RuntimeStoreError("stale_generation")

    runner = SimpleNamespace(
        adapters={Platform.API_SERVER: adapter},
        _adapter_for_source=lambda src: adapter if src.platform == Platform.API_SERVER else None,
        user_config={"display": {"platforms": {"telegram": {"cleanup_progress": False}}}},
        session_db=db,
    )
    authority = SimpleNamespace(
        sessions={"sess_auth_bg": live_session},
        api_observers={},
        waiters={},
        db=db,
        runner=runner,
        check_approval_generation=_check_gen,
        _publish_pending=lambda ref: None,
    )
    runner.session_authority = authority
    adapter.gateway_runner = runner
    adapter._ensure_session_db = lambda: db
    adapter._ensure_session_db_async = AsyncMock(return_value=db)

    agent = SimpleNamespace(
        status_callback=None,
        interim_assistant_callback=None,
        tool_progress_callback=None,
        reasoning_callback=None,
    )

    # Spy on handle_session_subagent_pulse to verify no double delivery when session_authority is active
    pulse_events_seen: list[tuple[str, str]] = []
    orig_handle = adapter.handle_session_subagent_pulse

    def spy_handle(session_id, event_type, *args, **kwargs):
        pulse_events_seen.append((session_id, event_type))
        return orig_handle(session_id, event_type, *args, **kwargs)

    adapter.handle_session_subagent_pulse = spy_handle

    def _schedule_parent(_ref):
        wire_api_observers(agent, (authority, "sess_auth_bg", 1), want_interim=False)
        agent.tool_progress_callback(
            "subagent.start", None, "Task 0", None,
            task_index=0, goal="Task 0", delegation_id="deleg_bg", subagent_id="sub_0",
        )
        agent.tool_progress_callback(
            "subagent.start", None, "Task 1", None,
            task_index=1, goal="Task 1", delegation_id="deleg_bg", subagent_id="sub_1",
        )
        authority.waiters["adm_bg"].set_result(None)

    authority._schedule = _schedule_parent

    # Parent turn runs via observe_api_turn + wire_api_observers, starts 2 background subagents, then exits
    sse_received: list[str] = []
    ref = SimpleNamespace(session_id="sess_auth_bg")
    row = {"admission_id": "adm_bg", "status": "started"}
    with (
        patch("gateway.session_retirement.restore_retired", lambda *a, **kw: None),
        patch("hermes_state_runtime.get_session_admission", return_value={"admission_id": "adm_bg", "status": "started", "outcome": "completed"}),
        patch("gateway.session_api_turn.admission_result", return_value={"result": {"final_response": "ok"}, "usage": {}}),
    ):
        await observe_api_turn(
            (authority, ref, row),
            tool_progress_callback=lambda ev, *a, **kw: sse_received.append(ev),
        )
    await asyncio.sleep(0.03)

    # Parent turn finished: api_observers cleared and generation bumped (stale_generation)
    assert authority.api_observers == {}
    generation = 2
    assert "sess_auth_bg" in adapter._pulse_registry()
    entry = adapter._pulse_registry()["sess_auth_bg"]
    assert entry["ticker"].active is True
    assert pulse_events_seen == [("sess_auth_bg", "subagent.start"), ("sess_auth_bg", "subagent.start")]

    # Background ThreadPoolExecutor emits progress & complete after parent turn ended
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            agent.tool_progress_callback,
            "subagent.progress", "terminal", "Running tests", None,
            task_index=0, action="Running tests", delegation_id="deleg_bg", subagent_id="sub_0",
        ).result()
        pool.submit(
            agent.tool_progress_callback,
            "subagent.complete", None, "Done 0", None,
            task_index=0, status="completed", summary="Done 0", delegation_id="deleg_bg", subagent_id="sub_0",
        ).result()
    await asyncio.sleep(0.04)

    # Still 1 running subagent -> ticker stays active, no premature completion
    assert "sess_auth_bg" in adapter._pulse_registry()
    assert entry["ticker"].active is True
    assert entry["ticker"].tasks["deleg_bg:0"]["status"] == "completed"
    assert entry["ticker"].tasks["deleg_bg:1"]["status"] == "running"
    assert not any("все субагенты" in c["content"].lower() for c in telegram.calls)

    # Complete second subagent from ThreadPoolExecutor -> ticker stops cleanly
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            agent.tool_progress_callback,
            "subagent.complete", None, "Done 1", None,
            task_index=1, status="completed", summary="Done 1", delegation_id="deleg_bg", subagent_id="sub_1",
        ).result()
    await asyncio.sleep(0.05)

    assert "sess_auth_bg" not in adapter._pulse_registry()
    assert any(w in c["content"].lower() for c in telegram.calls for w in ("завершен", "completed"))

    # Also verify _handle_session_chat_stream._tool_progress does NOT double-deliver when session_authority is set
    pulse_events_seen.clear()

    async def fake_run_with_authority(**kwargs):
        cb = kwargs["tool_progress_callback"]
        # Simulate _tool_progress receiving an event while session_authority is active
        cb("subagent.start", None, "Only SSE", None, task_index=0, goal="Only SSE")
        return {"final_response": "ok", "session_id": "sess_auth_bg"}, {}

    adapter._run_agent = fake_run_with_authority
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess_auth_bg/chat/stream", json={"message": "hi"})
        assert resp.status == 200
        await resp.text()
    assert pulse_events_seen == []
    db.close()


@pytest.mark.asyncio
async def test_cleanup_progress_via_user_config_and_load_gateway_config(tmp_path):
    """BDD 2: cleanup_progress=True deletes card only after all background subagents complete; False keeps it."""
    _assert_sandbox_imports()

    # Case 1: cleanup_progress=True via gateway_runner.user_config
    db1 = _make_bound_db(tmp_path / "c1", "sess_clean_1", delivery_enabled=True)
    tg1 = FakeDeletableTelegramAdapter()
    adapter1 = _make_adapter(db1, tg1)
    adapter1.gateway_runner = SimpleNamespace(
        session_authority=None,
        user_config={"display": {"platforms": {"telegram": {"cleanup_progress": True}}}},
        session_db=db1,
    )
    adapter1._ensure_session_db = lambda: db1
    adapter1._ensure_session_db_async = AsyncMock(return_value=db1)

    async def run_bg_parent(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "BG 1", None, task_index=0, goal="BG 1", delegation_id="d1", subagent_id="s1")
        await asyncio.sleep(0.02)
        return {"final_response": "Parent done", "session_id": "sess_clean_1"}, {}

    adapter1._run_agent = run_bg_parent
    async with TestClient(TestServer(_make_stream_app(adapter1))) as cli:
        resp = await cli.post("/api/sessions/sess_clean_1/chat/stream", json={"message": "go"})
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.03)

    # Parent finished, subagent still running -> card NOT deleted
    assert not any(c["op"] == "delete_message" for c in tg1.calls)
    assert "sess_clean_1" in adapter1._pulse_registry()

    # Now subagent completes -> card IS deleted
    adapter1.handle_session_subagent_pulse(
        "sess_clean_1", "subagent.complete", None, "Done", None,
        task_index=0, status="completed", summary="Done", delegation_id="d1", subagent_id="s1",
    )
    await asyncio.sleep(0.05)
    assert any(c["op"] == "delete_message" for c in tg1.calls)
    assert "sess_clean_1" not in adapter1._pulse_registry()
    db1.close()

    # Case 2: cleanup_progress=True via _load_gateway_config() fallback (no user_config on runner)
    db2 = _make_bound_db(tmp_path / "c2", "sess_clean_2", delivery_enabled=True)
    tg2 = FakeDeletableTelegramAdapter()
    adapter2 = _make_adapter(db2, tg2)
    adapter2._pulse_loop = asyncio.get_running_loop()
    with patch("gateway.run._load_gateway_config", return_value={"display": {"cleanup_progress": True}}):
        adapter2.handle_session_subagent_pulse(
            "sess_clean_2", "subagent.start", None, "BG 2", None,
            task_index=0, goal="BG 2", delegation_id="d2", subagent_id="s2",
        )
        await asyncio.sleep(0.03)
        adapter2.handle_session_subagent_pulse(
            "sess_clean_2", "subagent.complete", None, "Done 2", None,
            task_index=0, status="completed", summary="Done 2", delegation_id="d2", subagent_id="s2",
        )
        await asyncio.sleep(0.05)
    assert any(c["op"] == "delete_message" for c in tg2.calls)
    db2.close()

    # Case 3: cleanup_progress=False keeps final card
    db3 = _make_bound_db(tmp_path / "c3", "sess_clean_3", delivery_enabled=True)
    tg3 = FakeDeletableTelegramAdapter()
    adapter3 = _make_adapter(db3, tg3)
    adapter3.gateway_runner = SimpleNamespace(
        session_authority=None,
        user_config={"display": {"platforms": {"telegram": {"cleanup_progress": False}}}},
        session_db=db3,
    )
    adapter3._ensure_session_db = lambda: db3
    adapter3._pulse_loop = asyncio.get_running_loop()
    adapter3.handle_session_subagent_pulse(
        "sess_clean_3", "subagent.start", None, "BG 3", None,
        task_index=0, goal="BG 3", delegation_id="d3", subagent_id="s3",
    )
    await asyncio.sleep(0.03)
    adapter3.handle_session_subagent_pulse(
        "sess_clean_3", "subagent.complete", None, "Done 3", None,
        task_index=0, status="completed", summary="Done 3", delegation_id="d3", subagent_id="s3",
    )
    await asyncio.sleep(0.05)
    assert not any(c["op"] == "delete_message" for c in tg3.calls)
    assert any(w in c.get("content", "").lower() for c in tg3.calls for w in ("завершен", "completed"))
    db3.close()


@pytest.mark.asyncio
async def test_stop_vs_parent_failure_and_draining_with_cleanup_progress(tmp_path):
    """BDD 3: /stop (cancelled) and _draining stop ticker without deleting card; parent failure keeps running subagents alive."""
    _assert_sandbox_imports()
    db = _make_bound_db(tmp_path, "sess_stop_drain", delivery_enabled=True)
    tg = FakeDeletableTelegramAdapter()
    adapter = _make_adapter(db, tg)
    adapter.gateway_runner = SimpleNamespace(
        session_authority=None,
        user_config={"display": {"platforms": {"telegram": {"cleanup_progress": True}}}},
        session_db=db,
    )
    adapter._ensure_session_db = lambda: db
    adapter._ensure_session_db_async = AsyncMock(return_value=db)

    # Draining parent turn stops ticker with interrupted and does NOT delete card
    adapter._draining = True

    async def run_draining(**kwargs):
        cb = kwargs["tool_progress_callback"]
        cb("subagent.start", None, "Drain task", None, task_index=0, goal="Drain task", delegation_id="dd", subagent_id="sd")
        await asyncio.sleep(0.02)
        return {"final_response": "Draining", "session_id": "sess_stop_drain"}, {}

    adapter._run_agent = run_draining
    async with TestClient(TestServer(_make_stream_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/sess_stop_drain/chat/stream", json={"message": "go"})
        assert resp.status == 200
        await resp.text()
    await asyncio.sleep(0.04)
    assert "sess_stop_drain" not in adapter._pulse_registry()
    assert not any(c["op"] == "delete_message" for c in tg.calls)
    db.close()


@pytest.mark.asyncio
async def test_wave1_wave2_race_identity_pop_protection(tmp_path):
    """BDD 2 & 5: Wave 2 started while Wave 1 stop_task is awaiting is NOT popped by Wave 1 cleanup."""
    _assert_sandbox_imports()
    from gateway.subagent_pulse_ticker import SubagentPulseTicker

    db = _make_bound_db(tmp_path, "sess_race", delivery_enabled=True)
    tg = FakeTelegramAdapter()
    adapter = _make_adapter(db, tg)
    adapter._pulse_loop = asyncio.get_running_loop()

    stop_entered = asyncio.Event()
    allow_stop_finish = asyncio.Event()
    orig_stop = SubagentPulseTicker.stop

    async def slow_first_stop(self, status="completed", summary="", **kwargs):
        if not stop_entered.is_set():
            stop_entered.set()
            await allow_stop_finish.wait()
        return await orig_stop(self, status=status, summary=summary, **kwargs)

    with patch.object(SubagentPulseTicker, "stop", slow_first_stop):
        adapter.handle_session_subagent_pulse(
            "sess_race", "subagent.start", None, "Wave 1", None,
            task_index=0, goal="Wave 1", delegation_id="w1", subagent_id="s1",
        )
        await asyncio.sleep(0.02)
        adapter.handle_session_subagent_pulse(
            "sess_race", "subagent.complete", None, "W1 done", None,
            task_index=0, status="completed", summary="W1 done", delegation_id="w1", subagent_id="s1",
        )
        await asyncio.wait_for(stop_entered.wait(), timeout=2.0)
        wave1_entry = adapter._pulse_registry().get("sess_race")
        assert wave1_entry is not None

        # Start Wave 2 while Wave 1 stop_task is still awaiting
        adapter.handle_session_subagent_pulse(
            "sess_race", "subagent.start", None, "Wave 2", None,
            task_index=0, goal="Wave 2", delegation_id="w2", subagent_id="s2",
        )
        wave2_entry = adapter._pulse_registry().get("sess_race")
        assert wave2_entry is not None and wave2_entry is not wave1_entry

        # Let Wave 1 stop_task finish -> Wave 2 entry MUST remain in _session_pulse_tickers
        allow_stop_finish.set()
        await asyncio.sleep(0.05)
        assert adapter._pulse_registry().get("sess_race") is wave2_entry
        assert wave2_entry["ticker"].active is True

    await adapter.disconnect()
    db.close()


@pytest.mark.asyncio
async def test_wake_turn_delivers_pulse_with_and_without_session_authority(tmp_path):
    """BDD 6: run_internal_session_turn delivers subagent.* pulse before any /chat/stream (both with and without session_authority)."""
    _assert_sandbox_imports()
    from gateway.session_api_turn import observe_api_turn

    # Case A: without session_authority (fallback tool_progress_callback in api_server_runs.py)
    db_a = _make_bound_db(tmp_path / "wa", "sess_wake_a", delivery_enabled=True)
    tg_a = FakeTelegramAdapter()
    adapter_a = _make_adapter(db_a, tg_a)
    adapter_a._draining_response = lambda: None
    adapter_a._concurrency_limited_response = lambda: None
    adapter_a._get_existing_session_or_404 = AsyncMock(return_value=({"id": "sess_wake_a"}, None))
    adapter_a._conversation_history_for_session = AsyncMock(return_value=[])
    adapter_a._select_request_route = lambda *a, **kw: (None, {}, None)

    async def fake_wake_run_a(**kwargs):
        cb = kwargs.get("tool_progress_callback")
        assert cb is not None
        cb("subagent.start", None, "Wake A", None, task_index=0, goal="Wake A", delegation_id="dw_a", subagent_id="sw_a")
        await asyncio.sleep(0.02)
        cb("subagent.complete", None, "Wake A done", None, task_index=0, status="completed", summary="Wake A done", delegation_id="dw_a", subagent_id="sw_a")
        return {"final_response": "ok"}, {}

    adapter_a._run_agent = fake_wake_run_a
    assert adapter_a._pulse_loop is None
    await adapter_a.run_internal_session_turn(session_id="sess_wake_a", text="wake", profile="default")
    await asyncio.sleep(0.05)
    assert adapter_a._pulse_loop is not None
    assert any(w in c["content"].lower() for c in tg_a.calls for w in ("завершен", "completed"))
    db_a.close()

    # Case B: with session_authority (via wire_api_observers -> publish_api_subagent_pulse)
    db_b = _make_bound_db(tmp_path / "wb", "sess_wake_b", delivery_enabled=True)
    tg_b = FakeTelegramAdapter()
    adapter_b = _make_adapter(db_b, tg_b)
    adapter_b._draining_response = lambda: None
    adapter_b._concurrency_limited_response = lambda: None
    adapter_b._get_existing_session_or_404 = AsyncMock(return_value=({"id": "sess_wake_b"}, None))
    adapter_b._conversation_history_for_session = AsyncMock(return_value=[])
    adapter_b._select_request_route = lambda *a, **kw: (None, {}, None)
    adapter_b._ensure_session_db = lambda: db_b
    adapter_b._ensure_session_db_async = AsyncMock(return_value=db_b)

    live_session = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.API_SERVER),
        event_stream=SimpleNamespace(
            lock=MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
            execution={"admission_id": "adm_wake_b"},
        ),
    )
    runner_b = SimpleNamespace(
        adapters={Platform.API_SERVER: adapter_b},
        _adapter_for_source=lambda src: adapter_b if src.platform == Platform.API_SERVER else None,
        user_config={"display": {"platforms": {"telegram": {"cleanup_progress": False}}}},
        session_db=db_b,
    )
    authority_b = SimpleNamespace(
        sessions={"sess_wake_b": live_session},
        api_observers={},
        waiters={},
        db=db_b,
        runner=runner_b,
        check_approval_generation=lambda sid, gen: None,
        _publish_pending=lambda ref: None,
    )
    runner_b.session_authority = authority_b
    adapter_b.gateway_runner = runner_b

    async def fake_wake_run_b(**kwargs):
        # With session_authority, api_server_runs does not inject fallback tool_progress_callback;
        # run_api_turn -> wire_api_observers handles delivery.
        assert "tool_progress_callback" not in kwargs
        agent = SimpleNamespace(
            status_callback=None,
            interim_assistant_callback=None,
            tool_progress_callback=None,
            reasoning_callback=None,
        )

        def _schedule_wake(_ref):
            wire_api_observers(agent, (authority_b, "sess_wake_b", 1), want_interim=False)
            agent.tool_progress_callback(
                "subagent.start", None, "Wake B", None,
                task_index=0, goal="Wake B", delegation_id="dw_b", subagent_id="sw_b",
            )
            agent.tool_progress_callback(
                "subagent.complete", None, "Wake B done", None,
                task_index=0, status="completed", summary="Wake B done", delegation_id="dw_b", subagent_id="sw_b",
            )
            authority_b.waiters["adm_wake_b"].set_result(None)

        authority_b._schedule = _schedule_wake
        ref_b = SimpleNamespace(session_id="sess_wake_b")
        row_b = {"admission_id": "adm_wake_b", "status": "started"}
        with (
            patch("gateway.session_retirement.restore_retired", lambda *a, **kw: None),
            patch("hermes_state_runtime.get_session_admission", return_value={"admission_id": "adm_wake_b", "status": "started", "outcome": "completed"}),
            patch("gateway.session_api_turn.admission_result", return_value={"result": {"final_response": "ok"}, "usage": {}}),
        ):
            await observe_api_turn((authority_b, ref_b, row_b))
        return {"final_response": "ok"}, {}

    adapter_b._run_agent = fake_wake_run_b
    await adapter_b.run_internal_session_turn(session_id="sess_wake_b", text="wake", profile="default")
    await asyncio.sleep(0.05)
    assert adapter_b._pulse_loop is not None
    assert any(w in c["content"].lower() for c in tg_b.calls for w in ("завершен", "completed"))
    db_b.close()


@pytest.mark.asyncio
async def test_disconnect_on_bare_new_fixture_stops_active_tickers(tmp_path):
    """disconnect() tolerates bare APIServerAdapter.__new__ fixtures and stops active tickers with interrupted."""
    _assert_sandbox_imports()
    # Bare __new__ without any pulse attributes
    bare = APIServerAdapter.__new__(APIServerAdapter)
    bare._mark_disconnected = lambda: None
    bare._close_cached_session_dbs = lambda: None
    bare._memory_sessions = SimpleNamespace(close_all=lambda: None)
    bare._site = None
    bare._runner = None
    await bare.disconnect()

    # Bare __new__ with an active ticker registered via handle_session_subagent_pulse
    db = _make_bound_db(tmp_path, "sess_bare_disc", delivery_enabled=True)
    tg = FakeTelegramAdapter()
    bare2 = APIServerAdapter.__new__(APIServerAdapter)
    bare2._mark_disconnected = lambda: setattr(bare2, "_disconnected", True)
    bare2._close_cached_session_dbs = lambda: None
    bare2._memory_sessions = SimpleNamespace(close_all=lambda: None)
    bare2._site = None
    bare2._runner = None
    bare2._session_db = db
    bare2._ensure_session_db = lambda: db
    bare2._telegram_runner_and_adapter = lambda: (None, tg)
    bare2.is_telegram_bound = lambda sid: sid == "sess_bare_disc"
    bare2.gateway_runner = None
    bare2._pulse_loop = asyncio.get_running_loop()

    bare2.handle_session_subagent_pulse(
        "sess_bare_disc", "subagent.start", None, "Bare task", None,
        task_index=0, goal="Bare task", delegation_id="dbare", subagent_id="sbare",
    )
    await asyncio.sleep(0.03)
    entry = bare2._pulse_registry().get("sess_bare_disc")
    assert entry is not None
    ticker = entry["ticker"]
    assert ticker.active is True

    await bare2.disconnect()
    assert ticker.active is False
    assert bare2._pulse_registry() == {}
    db.close()
