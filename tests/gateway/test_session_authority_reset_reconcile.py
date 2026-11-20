"""Tests for startup and live /reset reconciliation of orphaned native_text_v1 admissions."""
import asyncio
import logging

import pytest

from tests.waiting import async_wait_until


def _native_payload(text: str, msg_id: str = "m1", **extra_native) -> dict:
    native = {"event": {"text": text, "message_id": msg_id}}
    native.update(extra_native)
    return {"text": text, "native_text_v1": native}


def _api_payload(text: str) -> dict:
    return {"text": text, "api_turn_v1": {"history": None, "settings": {}}}


def _messaging_principal(chat_id: str = "42") -> str:
    return f'messaging:[null,"telegram","{chat_id}",null,null]'


def _make_runner(monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", get_hermes_home() / "gateway_voice_mode.json", raising=False)
    runner = GatewayRunner(GatewayConfig())
    monkeypatch.setattr(runner, "_read_user_config", lambda: {"approvals": {"destructive_slash_confirm": False}})
    return runner


def _statuses(db, session_id: str) -> dict[str, tuple[str, str | None]]:
    from hermes_state_runtime import list_session_admissions

    return {
        r["admission_id"]: (r["status"], r["outcome"])
        for r in list_session_admissions(db, session_id=session_id, pending_only=False)
    }


@pytest.mark.asyncio
async def test_bdd1_startup_reconcile_closes_reset_native_unknown_and_queued(monkeypatch, caplog):
    """BDD-1: Startup reconcile settles unknown->interrupted and queued->cancelled on session_reset."""
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from gateway.session_unattended_unknown import settle_reset_native_admissions
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    runner = _make_runner(monkeypatch)
    db = runner.session_store._db
    db.create_session("s1", source="telegram")

    epoch1 = begin_runtime_epoch(db, instance_id="owner-1")
    r_unk = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s1",
        request_id="req-unk", payload=_native_payload("lost turn", "m1"),
    )
    r_q1 = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s1",
        request_id="req-q1", payload=_native_payload("queued 1", "m2"),
    )
    r_q2 = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s1",
        request_id="req-q2", payload=_native_payload("queued 2", "m3"),
    )
    claim_session_input(db, epoch=epoch1, session_id="s1")
    db.append_message("s1", "user", "lost turn")
    db._execute_write(
        lambda c: c.execute("UPDATE sessions SET ended_at=1000.0, end_reason='session_reset' WHERE id='s1'")
    )

    with caplog.at_level(logging.INFO):
        authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner-2", db=db)
        outcome = await recover_profile_native_sessions(runner, authority)

    assert outcome == {}
    assert "s1" not in authority.sessions
    assert "paused: unknown_execution" not in caplog.text

    st = _statuses(db, "s1")
    assert st[r_unk["admission_id"]] == ("terminal", "interrupted")
    assert st[r_q1["admission_id"]] == ("terminal", "cancelled")
    assert st[r_q2["admission_id"]] == ("terminal", "cancelled")

    # Transcript boundary closed so no open user tail remains
    msgs = db.get_messages("s1")
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1].get("display_kind") == "failed_turn"

    # Idempotent on subsequent restart
    authority2 = await initialize_session_authority(runner, profile_id="default", instance_id="owner-3", db=db)
    assert settle_reset_native_admissions(db, authority2.epoch) == 0
    assert _statuses(db, "s1") == st
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_bdd1b_startup_reconcile_follows_compression_lineage_to_reset_tip(monkeypatch):
    """BDD-1b: Admissions target logical_owner (root) while session_reset is on compression tip."""
    from gateway.session_authority import initialize_session_authority
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    runner = _make_runner(monkeypatch)
    db = runner.session_store._db
    db.create_session("s_root", source="telegram")
    db.create_session("s_tip", source="telegram", parent_session_id="s_root")

    epoch1 = begin_runtime_epoch(db, instance_id="owner-1")
    r_unk = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_root",
        request_id="req-root-unk", payload=_native_payload("lost compressed", "m1"),
    )
    r_q = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_root",
        request_id="req-root-q", payload=_native_payload("queued compressed", "m2"),
    )
    claim_session_input(db, epoch=epoch1, session_id="s_root")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=900.0, end_reason='compression' WHERE id='s_root'"
    ))
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=1000.0, end_reason='session_reset' WHERE id='s_tip'"
    ))

    await initialize_session_authority(runner, profile_id="default", instance_id="owner-2", db=db)
    st = _statuses(db, "s_root")
    assert st[r_unk["admission_id"]] == ("terminal", "interrupted")
    assert st[r_q["admission_id"]] == ("terminal", "cancelled")
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_bdd2_active_sessions_and_current_epoch_untouched(monkeypatch):
    """BDD-2: Active sessions (ended_at IS NULL) and current-epoch rows are never touched."""
    from gateway.session_authority import initialize_session_authority
    from gateway.session_unattended_unknown import settle_reset_native_admissions
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input

    runner = _make_runner(monkeypatch)
    db = runner.session_store._db
    db.create_session("s_active", source="telegram")
    db.create_session("s_running", source="api")
    db.create_session("s_reset_cur", source="telegram")

    epoch1 = begin_runtime_epoch(db, instance_id="owner-1")
    a_unk = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_active",
        request_id="act-unk", payload=_native_payload("active lost", "m1"),
    )
    a_q = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_active",
        request_id="act-q", payload=_native_payload("active queued", "m2"),
    )
    claim_session_input(db, epoch=epoch1, session_id="s_active")

    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner-2", db=db)
    epoch2 = authority.epoch

    # Active session rows from prior epoch remain unknown / queued
    st_active = _statuses(db, "s_active")
    assert st_active[a_unk["admission_id"]] == ("unknown", None)
    assert st_active[a_q["admission_id"]] == ("queued", None)

    # Current-epoch started row on active session and current-epoch queued row on reset session
    r_run = admit_session_input(
        db, epoch=epoch2, principal_id="api", session_id="s_running",
        request_id="run-cur", payload=_api_payload("running now"),
    )
    claim_session_input(db, epoch=epoch2, session_id="s_running")

    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=1000.0, end_reason='session_reset' WHERE id='s_reset_cur'"
    ))
    r_cur_q = admit_session_input(
        db, epoch=epoch2, principal_id=_messaging_principal(), session_id="s_reset_cur",
        request_id="cur-q", payload=_native_payload("current epoch queued", "m3"),
    )

    assert settle_reset_native_admissions(db, epoch2) == 0
    assert _statuses(db, "s_running")[r_run["admission_id"]] == ("started", None)
    assert _statuses(db, "s_reset_cur")[r_cur_q["admission_id"]] == ("queued", None)
    runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_bdd3_detached_api_turn_and_unattended_preserved(monkeypatch):
    """BDD-3: Detached api_turn_v1 on session_reset session stays queued and runs; unattended excluded."""
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from gateway.session_unattended_unknown import is_reset_reconcilable_native
    from hermes_state_keys import CRON_PRINCIPAL_ID
    from hermes_state_runtime import admit_session_input
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    runner = _make_runner(monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="s")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority1 = await initialize_session_authority(runner, profile_id="default", instance_id="owner-1")
    runner._wire_adapter_handlers(adapter)
    ran = []

    async def agent_turn(event, source, session_entry, session_key, *args):
        ran.append((event.text, session_entry.session_id, session_key))
        reply = f"reply:{event.text}"
        execution_result.get()["result"] = {"final_response": reply, "messages": []}
        return reply, None

    monkeypatch.setattr(runner, "_hmwa_prepare_turn", agent_turn)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")

    await adapter.handle_message(MessageEvent(text="hello", source=source, message_id="m1"))
    await async_wait_until(lambda: len(sent) >= 1, what="first reply")
    lane = runner.session_store._generate_session_key(source)
    old_sid = runner.session_store.peek_session_id(lane)

    await adapter.handle_message(MessageEvent(text="/new", source=source, message_id="m2"))
    await async_wait_until(lambda: runner.session_store.peek_session_id(lane) != old_sid, what="route rotated")

    # Admit a detached app prompt (api_turn_v1) and verify unattended exclusion
    db = runner.session_store._db
    app_row = admit_session_input(
        db, epoch=authority1.epoch, principal_id="api", session_id=old_sid,
        request_id="app-detached-1", payload=_api_payload("app prompt after reset"),
    )
    webhook_row = {
        "status": "unknown",
        "principal_id": 'messaging:[null,"webhook","r",null,null]',
        "payload": _native_payload("hook", "wh1", webhook_delivery={"deliver": "log"}),
    }
    cron_row = {"status": "unknown", "principal_id": CRON_PRINCIPAL_ID, "payload": {"text": "cron"}}
    assert not is_reset_reconcilable_native(webhook_row)
    assert not is_reset_reconcilable_native(cron_row)

    # Restart: startup reconcile must NOT cancel app_row; recover_profile_native_sessions runs it detached
    runner2 = _make_runner(monkeypatch)
    adapter2 = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    monkeypatch.setattr(adapter2, "send", send)
    runner2.adapters = {Platform.DISCORD: adapter2}
    runner2._wire_adapter_handlers(adapter2)
    monkeypatch.setattr(runner2, "_hmwa_prepare_turn", agent_turn)
    authority2 = await initialize_session_authority(runner2, profile_id="default", instance_id="owner-2", db=db)
    assert _statuses(db, old_sid)[app_row["admission_id"]] == ("queued", None)

    outcome = await recover_profile_native_sessions(runner2, authority2)
    assert outcome.get(old_sid) == "detached"
    await async_wait_until(
        lambda: _statuses(db, old_sid)[app_row["admission_id"]] == ("terminal", "completed"),
        what="detached api_turn_v1 completed",
    )
    assert any(t == "app prompt after reset" and sid == old_sid for t, sid, _ in ran)
    runner.session_store.close_all_db_handles()
    runner2.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_bdd4_live_reset_settles_native_preserves_api_and_supports_multiprofile(tmp_path, monkeypatch, caplog):
    """BDD-4: Live /reset settles unknown+queued native_text_v1, resolves waiters, preserves api_turn_v1,
    and routes via active_authority in multi-profile mode."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authorities import SessionAuthorities, active_authority
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, claim_session_input
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    launch_home = get_hermes_home()
    sec_home = tmp_path / "profiles" / "secondary"
    sec_home.mkdir(parents=True, exist_ok=True)

    runner = _make_runner(monkeypatch)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id="s")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}

    async def agent_turn(event, source, session_entry, session_key, *args):
        reply = f"reply:{event.text}"
        execution_result.get()["result"] = {"final_response": reply, "messages": []}
        return reply, None

    monkeypatch.setattr(runner, "_hmwa_prepare_turn", agent_turn)

    # Build launch authority and secondary profile authority (each with its own state.db)
    launch_auth = await initialize_session_authority(runner, profile_id="default", instance_id="l-1")
    sec_db = SessionDB(db_path=sec_home / "state.db")
    sec_auth = await initialize_session_authority(
        runner, profile_id="secondary", instance_id="s-1", db=sec_db, register=False
    )
    registry = SessionAuthorities(launch_home, multiplexed=True)
    registry.add(launch_home, launch_auth, name=None)
    registry.add(sec_home, sec_auth, name="secondary")
    runner.session_authorities = registry

    # Create a paused unknown + 2 queued native_text_v1 + 1 queued api_turn_v1 on launch profile
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    runner._wire_adapter_handlers(adapter)
    await adapter.handle_message(MessageEvent(text="hi", source=source, message_id="m0"))
    await async_wait_until(lambda: len(sent) >= 1, what="initial reply")
    lane = runner.session_store._generate_session_key(source)
    old_sid = runner.session_store.peek_session_id(lane)

    db = launch_auth.db
    r_unk = admit_session_input(
        db, epoch=launch_auth.epoch, principal_id=_messaging_principal(), session_id=old_sid,
        request_id="live-unk", payload=_native_payload("lost live", "m1"),
    )
    claim_session_input(db, epoch=launch_auth.epoch, session_id=old_sid)
    db._execute_write(lambda c: c.execute(
        "UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (r_unk["admission_id"],)
    ))
    r_q1 = admit_session_input(
        db, epoch=launch_auth.epoch, principal_id=_messaging_principal(), session_id=old_sid,
        request_id="live-q1", payload=_native_payload("queued live 1", "m2"),
    )
    r_q2 = admit_session_input(
        db, epoch=launch_auth.epoch, principal_id=_messaging_principal(), session_id=old_sid,
        request_id="live-q2", payload=_native_payload("queued live 2", "m3"),
    )
    r_api = admit_session_input(
        db, epoch=launch_auth.epoch, principal_id="api", session_id=old_sid,
        request_id="live-api", payload=_api_payload("detached app prompt"),
    )
    loop = asyncio.get_running_loop()
    w_q1 = launch_auth.waiters[r_q1["admission_id"]] = loop.create_future()
    launch_auth.native_waiters.add(r_q1["admission_id"])
    launch_auth.sessions[old_sid].pause_notified = True

    monkeypatch.setattr(launch_auth, "_schedule", lambda ref: None)
    await runner._handle_reset_command(MessageEvent(text="/reset", source=source, message_id="reset-1"))

    assert db.get_session(old_sid)["end_reason"] == "session_reset"
    st = _statuses(db, old_sid)
    assert st[r_unk["admission_id"]] == ("terminal", "interrupted")
    assert st[r_q1["admission_id"]] == ("terminal", "cancelled")
    assert st[r_q2["admission_id"]] == ("terminal", "cancelled")
    assert st[r_api["admission_id"]] == ("queued", None)
    assert w_q1.done() and r_q1["admission_id"] not in launch_auth.native_waiters
    assert launch_auth.sessions[old_sid].pause_notified is False

    # Multi-profile: active_authority(runner) under secondary scope resolves to sec_auth and settles sec_db
    sec_db.create_session("sec-old", source="telegram")
    sec_q = admit_session_input(
        sec_db, epoch=sec_auth.epoch, principal_id=_messaging_principal("99"), session_id="sec-old",
        request_id="sec-q1", payload=_native_payload("sec queued", "sm1"),
    )
    sec_db.end_session("sec-old", end_reason="session_reset")
    token = set_hermes_home_override(sec_home)
    try:
        scoped_auth = active_authority(runner)
        assert scoped_auth is sec_auth
        await scoped_auth.settle_reset_native_inputs("sec-old")
    finally:
        reset_hermes_home_override(token)
    assert _statuses(sec_db, "sec-old")[sec_q["admission_id"]] == ("terminal", "cancelled")

    # Subsequent restart sees no pending native_text_v1 and logs no unknown_execution
    with caplog.at_level(logging.WARNING):
        auth_next = await initialize_session_authority(runner, profile_id="default", instance_id="l-2", db=db)
        await recover_profile_native_sessions(runner, auth_next)
    assert "paused: unknown_execution" not in caplog.text
    runner.session_store.close_all_db_handles()
    sec_db.close()


@pytest.mark.asyncio
async def test_bdd5_idempotency_and_runtime_store_error_resilience(monkeypatch):
    """BDD-5: Already-terminal rows and RuntimeStoreError races do not crash startup or /reset."""
    from gateway.session_authority import initialize_session_authority
    import hermes_state_runtime as hsr
    from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input

    runner = _make_runner(monkeypatch)
    db = runner.session_store._db
    db.create_session("s_race", source="telegram")

    epoch1 = begin_runtime_epoch(db, instance_id="owner-1")
    r_unk = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_race",
        request_id="race-unk", payload=_native_payload("race lost", "m1"),
    )
    r_q = admit_session_input(
        db, epoch=epoch1, principal_id=_messaging_principal(), session_id="s_race",
        request_id="race-q", payload=_native_payload("race queued", "m2"),
    )
    claim_session_input(db, epoch=epoch1, session_id="s_race")
    db._execute_write(lambda c: c.execute(
        "UPDATE sessions SET ended_at=1000.0, end_reason='session_reset' WHERE id='s_race'"
    ))

    orig_resolve = hsr.resolve_unknown_session_input
    orig_cancel = hsr.cancel_session_input

    def fail_resolve(*args, **kwargs):
        raise RuntimeStoreError("stale_generation")

    def fail_cancel(*args, **kwargs):
        raise RuntimeStoreError("not_found")

    monkeypatch.setattr(hsr, "resolve_unknown_session_input", fail_resolve)
    monkeypatch.setattr(hsr, "cancel_session_input", fail_cancel)
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="owner-2", db=db)

    monkeypatch.setattr(hsr, "resolve_unknown_session_input", orig_resolve)
    monkeypatch.setattr(hsr, "cancel_session_input", orig_cancel)
    assert "s_race" not in authority.sessions
    settled = await authority.settle_reset_native_inputs("s_race")
    assert settled == 2
    st = _statuses(db, "s_race")
    assert st[r_unk["admission_id"]] == ("terminal", "interrupted")
    assert st[r_q["admission_id"]] == ("terminal", "cancelled")

    assert await authority.settle_reset_native_inputs("s_race") == 0
    runner.session_store.close_all_db_handles()
