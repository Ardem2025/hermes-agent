"""BDD tests for async-wake-dedup-fix (BDD-1..BDD-5 + Spec Audit contracts)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from agent.session_persistence import (
    _DB_PERSISTED_MARKER,
    _db_flush_collect,
    adopt_delegation_delivery_turn,
)
from agent.turn_context import _stage_turn_user_message
from gateway.config import Platform
from gateway.platforms.api_server import APIServerAdapter
from gateway.run_notifications import (
    INTERNAL_NOTIFICATION_FOOTER,
    GatewayNotificationsMixin,
    _mark_internal_notification,
)
from gateway.session import SessionSource
from gateway.session_api_turn import check_api_turn, prepare_api_execution
from gateway.session_contract import SessionRef
from gateway.wake import _CONSECUTIVE_AUTO_WAKES, deliver_wake, persist_delegation_delivery
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError


@pytest.fixture(autouse=True)
def _clean_wake_state():
    _CONSECUTIVE_AUTO_WAKES.clear()
    yield
    _CONSECUTIVE_AUTO_WAKES.clear()


def _make_db(tmp_path: Path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _make_coordinator_session(db: SessionDB, session_id: str = "coord-sid-1") -> None:
    db.create_session(
        session_id=session_id,
        source="api_server",
        model_config={"auto_resume_on_delegation": True},
    )


def _make_adapter(db: SessionDB, *, api_key: str = "test-secret-key", host: str = "127.0.0.1", port: int = 0):
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter.platform = Platform.API_SERVER
    adapter._host = host
    adapter._port = port
    adapter._api_key = api_key
    adapter._model_name = "hermes-agent"
    adapter._model_routes = {}
    adapter._session_db = db
    adapter._ensure_session_db = lambda: db
    adapter._ensure_session_db_async = AsyncMock(return_value=db)
    adapter._active_run_agents = {}
    adapter._pending_agent_requests = 0
    adapter._shutdown_interruptible_agents = {}
    adapter._run_statuses = {}
    adapter._idempotency_cache = {}
    adapter._idempotency_inflight = {}
    adapter._idempotency_lock = asyncio.Lock()
    adapter._max_concurrent_runs = 0
    adapter._draining = False
    adapter._pulse_loop = None
    adapter._ensure_pulse_loop = lambda: None
    adapter.gateway_runner = None
    adapter._cors_origins = ()
    adapter._MAX_SESSION_HEADER_LEN = 256
    adapter._select_request_route = lambda body, **kw: (None, {}, None)
    adapter._parse_session_key_header = lambda req: (None, None)
    adapter._check_auth = lambda req: (
        None
        if (not api_key or req.headers.get("Authorization") == f"Bearer {api_key}")
        else web.json_response({"error": "unauthorized"}, status=401)
    )
    adapter._expected_api_key = lambda: api_key
    adapter._draining_response = lambda: None
    adapter._concurrency_limited_response = lambda: None
    adapter._get_existing_session_or_404 = AsyncMock(return_value=({"id": "coord-sid-1"}, None))
    adapter._conversation_history_for_session = AsyncMock(
        side_effect=lambda sid: db.get_messages_as_conversation(sid)
    )
    return adapter


def _simulate_agent_turn(db: SessionDB, session_id: str, user_message: str, conversation_history: list,
                         wake_delegation_id: str = "", wake_delivery_notice: str = "",
                         assistant_reply: str = "Coordinator processed batch."):
    agent = SimpleNamespace(
        session_id=session_id,
        _session_db=db,
        _pending_cli_user_message=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=session_id,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=(),
        _persist_disabled=False,
        _mute_notification_reply=False,
    )
    history = list(conversation_history)
    if wake_delegation_id:
        adopt_delegation_delivery_turn(
            history, wake_delegation_id, wake_delivery_notice, user_message, agent
        )
    user_msg, _ = _stage_turn_user_message(
        agent, user_message, None, None, None, None, None
    )
    llm_messages = [*history, user_msg]
    assistant_msg = {"role": "assistant", "content": assistant_reply}
    full_messages = [*llm_messages, assistant_msg]
    rows, msgs = _db_flush_collect(agent, full_messages, history)
    if rows:
        db.append_messages_batch(session_id, rows)
        for m in msgs:
            m[_DB_PERSISTED_MARKER] = True
    return llm_messages, rows


def test_bdd1_http_self_post_single_db_row_and_single_llm_block(tmp_path):
    """BDD-1 (HTTP): _self_post_api_server -> persist_delegation_delivery + deliver_wake ->
    /v1/chat/completions produces exactly 1 user row (display_kind='async_delegation_complete')
    in SessionDB and 1 block with INTERNAL_NOTIFICATION_FOOTER in LLM prompt."""
    db = _make_db(tmp_path)
    sid = "coord-http-1"
    _make_coordinator_session(db, sid)
    adapter = _make_adapter(db, api_key="sekrit-1")

    captured_llm_messages = []

    async def fake_run_agent(
        user_message, conversation_history, session_id=None,
        wake_delegation_id="", wake_delivery_notice="", **kwargs
    ):
        llm_msgs, _ = _simulate_agent_turn(
            db, session_id, user_message, conversation_history,
            wake_delegation_id=wake_delegation_id,
            wake_delivery_notice=wake_delivery_notice,
            assistant_reply="Done from coordinator.",
        )
        captured_llm_messages.append(llm_msgs)
        return {"final_response": "Done from coordinator.", "completed": True}, {"total_tokens": 10}

    adapter._run_agent = fake_run_agent

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        adapter._port = site._server.sockets[0].getsockname()[1]
        try:
            evt = {
                "type": "async_delegation",
                "delegation_id": "deleg_abc123",
                "results": [{"task_index": 0, "status": "completed"}],
            }
            synth_text = "[ASYNC DELEGATION BATCH COMPLETE — deleg_abc123]\nTask 0 finished."

            class DummyRunner(GatewayNotificationsMixin):
                pass

            gr = DummyRunner()
            gr._adapters = {Platform.API_SERVER: adapter}
            gr.config = SimpleNamespace(multiplex_profiles=False)
            ok = await gr._self_post_api_server(adapter, synth_text, sid, evt)
            assert ok is True
            for _ in range(50):
                if captured_llm_messages:
                    break
                await asyncio.sleep(0.02)
        finally:
            await runner.cleanup()

    asyncio.run(run())

    assert len(captured_llm_messages) == 1
    llm_msgs = captured_llm_messages[0]
    user_blocks = [m for m in llm_msgs if m.get("role") == "user" and "[ASYNC DELEGATION BATCH COMPLETE — deleg_abc123]" in str(m.get("content"))]
    assert len(user_blocks) == 1
    assert INTERNAL_NOTIFICATION_FOOTER in user_blocks[0]["content"]

    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT role, content, display_kind, display_metadata FROM messages WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    user_rows = [r for r in rows if r[0] == "user"]
    assistant_rows = [r for r in rows if r[0] == "assistant"]
    assert len(user_rows) == 1
    assert user_rows[0][2] == "async_delegation_complete"
    assert json.loads(user_rows[0][3])["delegation_id"] == "deleg_abc123"
    assert len(assistant_rows) == 1
    assert assistant_rows[0][1] == "Done from coordinator."


def test_bdd1_served_profile_in_process_single_row_and_single_llm_block(tmp_path):
    """BDD-1 (served profile): deliver_wake(..., profile='builder', delegation_id=...) ->
    run_internal_session_turn deduplicates against persisted delivery row."""
    db = _make_db(tmp_path)
    sid = "coord-served-1"
    _make_coordinator_session(db, sid)
    adapter = _make_adapter(db)
    adapter._get_existing_session_or_404 = AsyncMock(return_value=({"id": sid}, None))

    captured_llm_messages = []

    async def fake_run_agent(
        user_message, conversation_history, session_id=None,
        wake_delegation_id="", wake_delivery_notice="", **kwargs
    ):
        llm_msgs, _ = _simulate_agent_turn(
            db, session_id, user_message, conversation_history,
            wake_delegation_id=wake_delegation_id,
            wake_delivery_notice=wake_delivery_notice,
            assistant_reply="Served profile reply.",
        )
        captured_llm_messages.append(llm_msgs)
        return {"final_response": "Served profile reply.", "completed": True}, {}

    adapter._run_agent = fake_run_agent

    async def run():
        evt = {"type": "async_delegation", "delegation_id": "deleg_srv1", "results": [{"status": "completed"}]}
        synth_text = "[ASYNC DELEGATION BATCH COMPLETE — deleg_srv1]\nResult ok."
        await persist_delegation_delivery(adapter, text=synth_text, session_id=sid, evt=evt)
        await deliver_wake(
            adapter,
            text=_mark_internal_notification(synth_text),
            session_id=sid,
            profile="builder",
            delegation_id="deleg_srv1",
        )

    asyncio.run(run())

    assert len(captured_llm_messages) == 1
    user_blocks = [m for m in captured_llm_messages[0] if m.get("role") == "user" and "deleg_srv1" in str(m.get("content"))]
    assert len(user_blocks) == 1
    assert INTERNAL_NOTIFICATION_FOOTER in user_blocks[0]["content"]

    with db._read_ctx() as conn:
        user_rows = conn.execute(
            "SELECT display_kind, display_metadata FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchall()
    assert len(user_rows) == 1
    assert user_rows[0][0] == "async_delegation_complete"


def test_spec_audit_note2_non_tail_delivery_row_deduped_without_losing_intervening_turns(tmp_path):
    """Spec Audit #2: If another user+assistant turn landed between persist_delegation_delivery
    and the wake turn, deduplication still keeps 1 DB row with display_kind='async_delegation_complete',
    preserves intervening messages, and passes the block once with INTERNAL_NOTIFICATION_FOOTER."""
    db = _make_db(tmp_path)
    sid = "coord-nontail-1"
    _make_coordinator_session(db, sid)
    adapter = _make_adapter(db)

    synth_text = "[ASYNC DELEGATION BATCH COMPLETE — deleg_mid1]\nWorker output."
    evt = {"type": "async_delegation", "delegation_id": "deleg_mid1", "results": [{"status": "completed"}]}
    asyncio.run(persist_delegation_delivery(adapter, text=synth_text, session_id=sid, evt=evt))
    db.append_messages_batch(sid, [
        {"role": "user", "content": "Intervening user question"},
        {"role": "assistant", "content": "Intervening assistant answer"},
    ])

    history = db.get_messages_as_conversation(sid)
    wake_text = _mark_internal_notification(synth_text)
    llm_msgs, inserted_rows = _simulate_agent_turn(
        db, sid, wake_text, history,
        wake_delegation_id="deleg_mid1",
        assistant_reply="Wake reply after intervening turn.",
    )

    contents = [m.get("content") for m in llm_msgs]
    assert any("Intervening user question" in str(c) for c in contents)
    assert any("Intervening assistant answer" in str(c) for c in contents)
    deleg_msgs = [m for m in llm_msgs if "deleg_mid1" in str(m.get("content"))]
    assert len(deleg_msgs) == 1
    assert INTERNAL_NOTIFICATION_FOOTER in deleg_msgs[0]["content"]

    assert len(inserted_rows) == 1
    assert inserted_rows[0]["role"] == "assistant"
    with db._read_ctx() as conn:
        deleg_rows = conn.execute(
            "SELECT display_kind FROM messages WHERE session_id = ? AND role = 'user' AND content LIKE '%deleg_mid1%'",
            (sid,),
        ).fetchall()
    assert len(deleg_rows) == 1
    assert deleg_rows[0][0] == "async_delegation_complete"


def test_bdd2_retry_after_429_and_busy_skip_preserve_single_delivery(tmp_path, monkeypatch):
    """BDD-2: HTTP 429 retry succeeds with 1 DB row; busy session skips wake while keeping
    the persisted delivery row available for the next turn."""
    import gateway.wake as wake_mod
    monkeypatch.setattr(wake_mod, "_RETRY_DELAYS_SECONDS", (0.01, 0.01))

    db = _make_db(tmp_path)
    sid = "coord-retry-1"
    _make_coordinator_session(db, sid)
    adapter = _make_adapter(db, api_key="sekrit-retry")

    attempts = {"n": 0}
    captured_llm = []

    async def fake_run_agent(
        user_message, conversation_history, session_id=None,
        wake_delegation_id="", wake_delivery_notice="", **kwargs
    ):
        llm_msgs, _ = _simulate_agent_turn(
            db, session_id, user_message, conversation_history,
            wake_delegation_id=wake_delegation_id,
            wake_delivery_notice=wake_delivery_notice,
        )
        captured_llm.append(llm_msgs)
        return {"final_response": "ok", "completed": True}, {}

    adapter._run_agent = fake_run_agent

    async def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return web.json_response({"error": "rate limited"}, status=429)
        return await adapter._handle_chat_completions(request)

    async def run():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        adapter._port = site._server.sockets[0].getsockname()[1]
        try:
            synth = "[ASYNC DELEGATION BATCH COMPLETE — deleg_retry1]\nDone."
            evt = {"type": "async_delegation", "delegation_id": "deleg_retry1", "results": [{"status": "completed"}]}
            await persist_delegation_delivery(adapter, text=synth, session_id=sid, evt=evt)
            await deliver_wake(
                adapter,
                text=_mark_internal_notification(synth),
                session_id=sid,
                delegation_id="deleg_retry1",
            )
        finally:
            await runner.cleanup()

    asyncio.run(run())
    assert attempts["n"] == 2
    assert len(captured_llm) == 1
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT display_kind FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchall()
    assert [r[0] for r in rows] == ["async_delegation_complete"]

    sid_busy = "coord-busy-1"
    _make_coordinator_session(db, sid_busy)
    synth_busy = "[ASYNC DELEGATION BATCH COMPLETE — deleg_busy1]\nBusy result."
    evt_busy = {"type": "async_delegation", "delegation_id": "deleg_busy1", "results": [{"status": "completed"}]}
    asyncio.run(persist_delegation_delivery(adapter, text=synth_busy, session_id=sid_busy, evt=evt_busy))
    monkeypatch.setattr(wake_mod, "is_session_turn_busy", lambda s, a=None: True)
    asyncio.run(deliver_wake(adapter, text=_mark_internal_notification(synth_busy), session_id=sid_busy, delegation_id="deleg_busy1"))
    hist_busy = db.get_messages_as_conversation(sid_busy)
    assert len([m for m in hist_busy if m.get("role") == "user" and "deleg_busy1" in str(m.get("content"))]) == 1


def test_bdd3_non_eligible_session_persists_without_wake(tmp_path, monkeypatch):
    """BDD-3: When check_coordinator_wake_eligibility is False, persist_delegation_delivery
    writes 1 row and deliver_wake is never called."""
    db = _make_db(tmp_path)
    sid = "coord-inelig-1"
    _make_coordinator_session(db, sid)
    adapter = _make_adapter(db)
    _CONSECUTIVE_AUTO_WAKES[sid] = 10

    wake_mock = AsyncMock()
    monkeypatch.setattr("gateway.wake.deliver_wake", wake_mock)

    class DummyRunner(GatewayNotificationsMixin):
        pass

    gr = DummyRunner()
    gr._adapters = {Platform.API_SERVER: adapter}
    gr.config = SimpleNamespace(multiplex_profiles=False)
    evt = {"type": "async_delegation", "delegation_id": "deleg_cap1", "results": [{"status": "completed"}]}
    ok = asyncio.run(gr._self_post_api_server(adapter, "[ASYNC DELEGATION BATCH COMPLETE — deleg_cap1]", sid, evt))
    assert ok is True
    wake_mock.assert_not_awaited()
    with db._read_ctx() as conn:
        rows = conn.execute(
            "SELECT display_kind FROM messages WHERE session_id = ? AND role = 'user'",
            (sid,),
        ).fetchall()
    assert [r[0] for r in rows] == ["async_delegation_complete"]


def test_bdd5_security_and_spec_audit_contracts(tmp_path, monkeypatch):
    """BDD-5 + Spec Audit #1 & #3."""
    db = _make_db(tmp_path)
    sid = "coord-sec-1"
    _make_coordinator_session(db, sid)

    adapter_no_key = _make_adapter(db, api_key="")
    async def run_403():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter_no_key._handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Hermes-Session-Id": sid, "X-Hermes-Wake-Delegation-Id": "deleg_fake"},
                ) as resp:
                    return resp.status
        finally:
            await runner.cleanup()

    assert asyncio.run(run_403()) == 403

    adapter = _make_adapter(db, api_key="sekrit-sec")
    seen_kwargs = []

    async def fake_run_agent(**kwargs):
        seen_kwargs.append(kwargs)
        return {"final_response": "ok", "completed": True}, {}

    adapter._run_agent = fake_run_agent

    async def run_unknown_and_headerless():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "real user msg"}]},
                    headers={
                        "Authorization": "Bearer sekrit-sec",
                        "X-Hermes-Session-Id": sid,
                        "X-Hermes-Wake-Delegation-Id": "deleg_nonexistent",
                    },
                ) as resp:
                    assert resp.status == 200
                async with s.post(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "headerless msg"}]},
                    headers={
                        "Authorization": "Bearer sekrit-sec",
                        "X-Hermes-Wake-Delegation-Id": "deleg_nonexistent",
                    },
                ) as resp:
                    assert resp.status == 200
        finally:
            await runner.cleanup()

    asyncio.run(run_unknown_and_headerless())
    assert len(seen_kwargs) == 2
    assert "wake_delegation_id" not in seen_kwargs[0]
    assert seen_kwargs[0]["user_message"] == "real user msg"
    assert "wake_delegation_id" not in seen_kwargs[1]

    live = SimpleNamespace(source=SessionSource(platform=Platform.API_SERVER, chat_id=sid, user_id="api", chat_type="dm"))
    runner_stub = SimpleNamespace(_adapter_for_source=lambda src: adapter)
    adapter.gateway_runner = runner_stub
    authority = SimpleNamespace(db=db, runner=runner_stub, sessions={sid: live})
    monkeypatch.setattr("gateway.session_api_turn.restore_api_session", lambda auth, s: live)
    ref = SessionRef("default", sid)
    valid_payload = {
        "text": "wake text",
        "api_turn_v1": {
            "history": None,
            "settings": {},
            "wake_delegation_id": "deleg_1",
            "wake_delivery_notice": "task_failure:0",
        },
    }
    assert check_api_turn(authority, ref, valid_payload) is adapter
    prepared = prepare_api_execution(authority, ref, valid_payload)
    assert prepared["wake_delegation_id"] == "deleg_1"
    assert prepared["wake_delivery_notice"] == "task_failure:0"

    bad_payload = {
        "text": "wake text",
        "api_turn_v1": {"history": None, "settings": {}, "wake_delegation_id": 123},
    }
    with pytest.raises(RuntimeStoreError):
        check_api_turn(authority, ref, bad_payload)
