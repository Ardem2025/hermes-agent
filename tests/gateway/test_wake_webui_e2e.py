"""BDD tests for WebUI / api_server coordinator wake eligibility & propagation."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.api_server import APIServerAdapter
from gateway.run_notifications import GatewayNotificationsMixin
from gateway.session_context import (
    auto_resume_on_delegation_supported,
    clear_session_vars,
    reset_session_vars,
    set_session_vars,
)
from gateway.wake import (
    MAX_CONSECUTIVE_AUTO_WAKES,
    _CONSECUTIVE_AUTO_WAKES,
    check_coordinator_wake_eligibility,
    deliver_wake,
    get_consecutive_auto_wakes,
    increment_consecutive_auto_wakes,
)
from hermes_state import SessionDB
from tools.async_delegation import (
    dispatch_async_delegation,
    recover_abandoned_delegations,
)
from tools.delegate_tool_dispatch import (
    _Batch,
    _capture_auto_resume_on_delegation,
    _dispatch_background,
)


@pytest.fixture(autouse=True)
def _clean_wake_counters():
    _CONSECUTIVE_AUTO_WAKES.clear()
    reset_session_vars()
    yield
    _CONSECUTIVE_AUTO_WAKES.clear()
    reset_session_vars()


def _make_db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _make_adapter(db):
    return SimpleNamespace(_ensure_session_db=lambda: db, _session_db=db)


def test_bdd_1_and_6_real_session_db_webui_eligibility(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db = _make_db(tmp_path)
    with db._read_ctx() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
    assert "metadata" not in cols

    db.create_session(
        session_id="webui-mc",
        source="api_server",
        model_config={"auto_resume_on_delegation": True},
    )
    adapter = _make_adapter(db)
    assert check_coordinator_wake_eligibility("webui-mc", adapter, evt={}) is True

    db.create_session(
        session_id="webui-origin",
        source="api_server",
        origin_json=json.dumps({"auto_resume_on_delegation": True}),
    )
    assert check_coordinator_wake_eligibility("webui-origin", adapter, evt={}) is True

    db.create_session(session_id="webui-cs", source="api_server")
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE sessions SET created_source = 'webui' WHERE id = ?", ("webui-cs",))
        conn.commit()
    assert check_coordinator_wake_eligibility("webui-cs", adapter, evt={}) is True

    db.create_session(session_id="webui-evt", source="api_server")
    assert check_coordinator_wake_eligibility("webui-evt", adapter, evt={"auto_resume_on_delegation": True}) is True


def test_bdd_2_isolation_and_stateless_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    tasks_dir = tmp_path / ".hermes" / "tasks" / "foreign-task"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "state.json").write_text(json.dumps({"status": "in_progress", "session_id": "other-sid"}))
    unowned_dir = tmp_path / ".hermes" / "tasks" / "unowned-task"
    unowned_dir.mkdir(parents=True)
    (unowned_dir / "state.json").write_text(json.dumps({"status": "in_progress"}))

    db = _make_db(tmp_path)
    db.create_session(session_id="stateless-sid", source="api_server")
    adapter = _make_adapter(db)
    assert check_coordinator_wake_eligibility("stateless-sid", adapter, evt={}) is False


def test_bdd_3_clarify_and_busy_suppress_wake(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db = _make_db(tmp_path)
    db.create_session(
        session_id="busy-sid",
        source="api_server",
        model_config={"auto_resume_on_delegation": True},
    )
    adapter = _make_adapter(db)
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
            ("busy-sid", "turn-1", time.time(), time.time() + 60.0),
        )
        conn.commit()
    assert check_coordinator_wake_eligibility("busy-sid", adapter, evt={"auto_resume_on_delegation": True}) is False

    with sqlite3.connect(db.db_path) as conn:
        conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = ?", ("busy-sid",))
        conn.commit()
    from tools import clarify_gateway
    monkeypatch.setattr(
        clarify_gateway,
        "get_pending_for_session",
        lambda sid, include_choice_prompts=True: {"id": "p1"} if sid == "busy-sid" else None,
    )
    assert check_coordinator_wake_eligibility("busy-sid", adapter, evt={"auto_resume_on_delegation": True}) is False


def test_bdd_4_spdd_passport_session_id_and_coordinator_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    t1 = tmp_path / ".hermes" / "tasks" / "t1"
    t1.mkdir(parents=True)
    (t1 / "state.json").write_text(json.dumps({"status": "in_progress", "session_id": "spdd-1"}))
    t2 = tmp_path / ".hermes" / "tasks" / "t2"
    t2.mkdir(parents=True)
    (t2 / "state.json").write_text(json.dumps({"status": "active", "coordinator_session_id": "spdd-2"}))

    db = _make_db(tmp_path)
    adapter = _make_adapter(db)
    assert check_coordinator_wake_eligibility("spdd-1", adapter, evt={}) is True
    assert check_coordinator_wake_eligibility("spdd-2", adapter, evt={}) is True
    assert check_coordinator_wake_eligibility("spdd-other", adapter, evt={}) is False


def test_bdd_5_consecutive_wake_ceiling_and_reset_routes(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db = _make_db(tmp_path)
    db.create_session(
        session_id="loop-sid",
        source="api_server",
        model_config={"auto_resume_on_delegation": True},
    )
    adapter = _make_adapter(db)
    for _ in range(MAX_CONSECUTIVE_AUTO_WAKES):
        increment_consecutive_auto_wakes("loop-sid")
    assert get_consecutive_auto_wakes("loop-sid") == 10
    assert check_coordinator_wake_eligibility("loop-sid", adapter, evt={"auto_resume_on_delegation": True}) is False

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    api = APIServerAdapter.__new__(APIServerAdapter)
    api._session_db = db
    api.gateway_runner = None
    api._model_name = "test-model"
    api._api_key = ""
    api._pending_agent_requests = 0
    api._gateway_is_draining = lambda: False
    api._background_tasks = set()
    api._active_run_agents = {}
    api._run_approval_sessions = {}
    api._release_run_owner_if_forgotten = lambda rid: None
    api._run_owners = {}
    api._run_statuses = {}
    api._run_admission_aliases = {}
    api._ensure_session_db = lambda: db
    api._run_idempotency_scope = lambda req: "scope"
    api._set_run_status = lambda *a, **kw: None
    api._conversation_history_for_session = AsyncMock(return_value=[])
    api._turn_transcript_messages = lambda *a, **kw: []
    api._effective_turn_runtime = lambda *a, **kw: {}
    api._register_session_stream_approval = lambda *a, **kw: None
    api._telegram_runner_and_adapter = lambda: (None, None)
    api.is_telegram_bound = lambda sid: False
    api._sanitize_runtime_metadata = lambda **kw: {}
    api._new_telegram_mirror = lambda **kw: SimpleNamespace(
        enqueue_commentary=lambda *a, **k: None,
        drain=AsyncMock(return_value=None),
        send_final=AsyncMock(return_value=None),
    )
    api._concurrency_limited_response = lambda: None
    api._session_headers = lambda sid, key: {}
    seen_run_kwargs = []

    async def fake_run_agent(**kwargs):
        seen_run_kwargs.append(kwargs)
        return {"final_response": "ok", "session_id": "loop-sid"}, {}

    api._run_agent = fake_run_agent

    async def _prepare(req):
        sid = req.match_info["session_id"]
        return {
            "session_id": sid,
            "gateway_session_key": f"api:{sid}",
            "user_message": "hi",
            "body": {},
            "runtime_request": {},
            "runtime_meta": {},
            "lock_active": False,
            "run_kwargs": {"session_id": sid, "session_history_delivery": "1"},
        }, None

    api._prepare_session_chat = _prepare

    # 5a. Bot Chat handoff must NOT reset counter
    api._answer_through_live_bot_chat = AsyncMock(return_value=web.json_response({"handed_off": True}))
    req_handoff = make_mocked_request("POST", "/api/sessions/loop-sid/chat", match_info={"session_id": "loop-sid"})
    asyncio.run(api._handle_session_chat(req_handoff))
    assert get_consecutive_auto_wakes("loop-sid") == 10

    # 5b. Normal /api/sessions/{id}/chat resets counter and sets auto_resume_on_delegation
    api._answer_through_live_bot_chat = AsyncMock(return_value=None)
    req_chat = make_mocked_request("POST", "/api/sessions/loop-sid/chat", match_info={"session_id": "loop-sid"})
    resp = asyncio.run(api._handle_session_chat(req_chat))
    assert resp.status == 200
    assert get_consecutive_auto_wakes("loop-sid") == 0
    assert seen_run_kwargs[-1]["auto_resume_on_delegation"] is True
    assert check_coordinator_wake_eligibility("loop-sid", adapter, evt={}) is True

    # 5c. /api/sessions/{id}/chat/stream also resets counter
    for _ in range(MAX_CONSECUTIVE_AUTO_WAKES):
        increment_consecutive_auto_wakes("loop-sid")
    assert get_consecutive_auto_wakes("loop-sid") == 10
    api._stream_through_live_bot_chat = AsyncMock(return_value=None)
    req_stream = make_mocked_request("POST", "/api/sessions/loop-sid/chat/stream", match_info={"session_id": "loop-sid"})
    monkeypatch.setattr(web.StreamResponse, "prepare", AsyncMock(return_value=None))
    monkeypatch.setattr(web.StreamResponse, "write", AsyncMock(return_value=None))
    monkeypatch.setattr(web.StreamResponse, "write_eof", AsyncMock(return_value=None))
    asyncio.run(api._handle_session_chat_stream(req_stream))
    assert get_consecutive_auto_wakes("loop-sid") == 0

    # 5d. /v1/chat/completions must NOT reset counter
    for _ in range(MAX_CONSECUTIVE_AUTO_WAKES):
        increment_consecutive_auto_wakes("loop-sid")
    api._api_key = "test-key"
    api._parse_session_key_header = lambda req: (None, None)
    api._ensure_session_db_async = AsyncMock(return_value=db)
    api._select_request_route = lambda *a, **kw: (None, {}, None)
    api._authority_chat_kwargs = lambda req, provided: {}

    async def _run_idempotent(req, body, fn, **kw):
        return await fn(), None

    api._run_idempotent = _run_idempotent

    async def _json_body():
        return {"model": "test-model", "messages": [{"role": "user", "content": "wake"}]}

    req_oai = make_mocked_request(
        "POST",
        "/v1/chat/completions",
        headers={"X-Hermes-Session-Id": "loop-sid", "Authorization": "Bearer test-key"},
    )
    req_oai.json = _json_body
    resp_oai = asyncio.run(api._handle_chat_completions(req_oai))
    assert resp_oai.status == 200
    assert get_consecutive_auto_wakes("loop-sid") == 10
    assert "auto_resume_on_delegation" not in seen_run_kwargs[-1]


def test_bdd_7_propagation_webui_vs_stateless(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    SessionDB(db_path=hermes_home / "state.db")

    tokens = set_session_vars(
        platform="api_server",
        session_id="webui-prop-1",
        session_history_delivery="1",
        auto_resume_on_delegation=True,
    )
    try:
        assert auto_resume_on_delegation_supported() is True
        assert _capture_auto_resume_on_delegation(None) is True
    finally:
        clear_session_vars(tokens)

    from tools.process_registry import process_registry
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    handle_a = dispatch_async_delegation(
        goal="subtask A",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="api:webui-prop-1",
        origin_session_id="webui-prop-1",
        runner=lambda: {"status": "completed", "summary": "done A"},
        auto_resume_on_delegation=True,
    )
    assert handle_a["status"] == "dispatched"
    tid_a = handle_a["delegation_id"]
    evt_a = process_registry.completion_queue.get(timeout=5.0)
    assert evt_a.get("delegation_id") == tid_a
    assert evt_a.get("auto_resume_on_delegation") is True

    with sqlite3.connect(hermes_home / "state.db") as conn:
        row_a = conn.execute(
            "SELECT task_json, event_json FROM async_delegations WHERE delegation_id=?",
            (tid_a,),
        ).fetchone()
    assert json.loads(row_a[0]).get("auto_resume_on_delegation") is True
    assert json.loads(row_a[1]).get("auto_resume_on_delegation") is True

    tokens_b = set_session_vars(
        platform="api_server",
        session_id="stateless-prop-2",
        session_history_delivery="1",
    )
    try:
        assert auto_resume_on_delegation_supported() is False
        assert _capture_auto_resume_on_delegation(None) is False
    finally:
        clear_session_vars(tokens_b)

    handle_b = dispatch_async_delegation(
        goal="subtask B",
        context=None,
        toolsets=None,
        role="leaf",
        model=None,
        session_key="api:stateless-prop-2",
        origin_session_id="stateless-prop-2",
        runner=lambda: {"status": "completed", "summary": "done B"},
        auto_resume_on_delegation=False,
    )
    assert handle_b["status"] == "dispatched"
    tid_b = handle_b["delegation_id"]
    evt_b = process_registry.completion_queue.get(timeout=5.0)
    assert evt_b.get("delegation_id") == tid_b
    assert "auto_resume_on_delegation" not in evt_b

    with sqlite3.connect(hermes_home / "state.db") as conn:
        row_b = conn.execute(
            "SELECT task_json, event_json FROM async_delegations WHERE delegation_id=?",
            (tid_b,),
        ).fetchone()
    assert "auto_resume_on_delegation" not in json.loads(row_b[0])
    assert "auto_resume_on_delegation" not in json.loads(row_b[1])

    seen_dispatch = {}
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **kw: seen_dispatch.update(kw) or {"status": "dispatched", "delegation_id": "t1"},
    )
    batch = _Batch(
        task_list=[{"goal": "g"}],
        children=[(0, {"goal": "g"}, SimpleNamespace())],
        parent_agent=SimpleNamespace(_session_db=None),
        creds={"model": None},
        context="",
        top_role="leaf",
        max_children=1,
        live_deleg_id="d1",
        live_writers=[],
        live_paths=[],
        origin_wake_sid="webui-prop-1",
        origin_ui_session_id="",
        origin_owner_transport=None,
        origin_owner_session_record=None,
        origin_session_history_delivery=True,
        overall_start=0.0,
        origin_auto_resume_on_delegation=True,
    )
    _dispatch_background(batch)
    assert seen_dispatch.get("auto_resume_on_delegation") is True

    abandoned_id = "abandoned-1"
    with sqlite3.connect(hermes_home / "state.db") as conn:
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id, parent_session_id,
                state, dispatched_at, updated_at, delivery_state, delivery_attempts,
                owner_pid, owner_started_at, task_json, origin_session_id)
               VALUES (?, ?, ?, ?, 'running', ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (
                abandoned_id,
                "api:webui-prop-1",
                "",
                None,
                time.time(),
                time.time(),
                999999,
                1.0,
                json.dumps({"goal": "lost", "auto_resume_on_delegation": True}),
                "webui-prop-1",
            ),
        )
        conn.commit()
    recovered_count = recover_abandoned_delegations()
    assert recovered_count >= 1
    with sqlite3.connect(hermes_home / "state.db") as conn:
        ev_row = conn.execute(
            "SELECT event_json FROM async_delegations WHERE delegation_id=?",
            (abandoned_id,),
        ).fetchone()
    assert json.loads(ev_row[0]).get("auto_resume_on_delegation") is True


def test_bdd_8_telegram_non_regression(monkeypatch):
    from gateway.session import SessionSource

    class _DummyRunner(GatewayNotificationsMixin):
        def __init__(self, adapters):
            self.adapters = adapters
            self.session_db = None

        def _build_process_event_source(self, evt):
            return SessionSource(platform=Platform.TELEGRAM, chat_id="12345")

        def _resolve_injection_adapter(self, platform_name, source):
            return self.adapters.get(Platform.TELEGRAM)

    tg_adapter = SimpleNamespace(supports_async_delivery=True)
    runner = _DummyRunner({Platform.TELEGRAM: tg_adapter})
    admitted = []

    async def fake_admit(adapter, synth_event):
        admitted.append((adapter, synth_event))

    monkeypatch.setattr("gateway.wake.admit_internal_event", fake_admit)
    monkeypatch.setattr("gateway.session_authorities.active_authority", lambda r: None)
    monkeypatch.setattr("gateway.session_local_recovery.restore_completion_route", lambda r, k: None)

    async def fake_follow(r, evt):
        return evt

    monkeypatch.setattr("gateway.session_detached.follow_detached_pin", fake_follow)
    monkeypatch.setattr(
        runner,
        "_self_post_api_server",
        AsyncMock(side_effect=AssertionError("must not self-post for Telegram")),
    )
    ok = asyncio.run(
        runner._inject_watch_notification(
            "done",
            {"type": "async_delegation", "task_id": "t1", "origin_session_id": "tg-sid-1"},
        )
    )
    assert ok is True
    assert len(admitted) == 1

    synced = []
    from aiohttp import web

    async def handler(request):
        return web.json_response({"choices": [{"message": {"content": "wake reply"}}]})

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        web_runner = web.AppRunner(app)
        await web_runner.setup()
        site = web.TCPSite(web_runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            api_adapter = SimpleNamespace(
                supports_async_delivery=False,
                _host="127.0.0.1",
                _port=port,
                _api_key="k",
                _model_name="m",
                _sync_telegram_history=lambda sid: synced.append(sid),
            )
            await deliver_wake(api_adapter, text="wake", session_id="tg-mirror-sid")
        finally:
            await web_runner.cleanup()

    asyncio.run(run())
    assert synced == ["tg-mirror-sid"]
