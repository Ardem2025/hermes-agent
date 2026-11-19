"""API source restoration cannot mint a parallel native conversation."""

import pytest

from gateway.config import Platform
from gateway.session_contract import SessionActor, SessionRef, Submission
from hermes_state_runtime import RuntimeStoreError


def test_private_api_binding_restores_exact_route(owner):
    from gateway.session_api import bind_api_session, restore_api_session
    db = owner.db
    db.create_session('api-stored', source='api_server')
    ref = bind_api_session(owner, 'api-stored')
    first = owner.sessions[ref.session_id]
    assert first.source.platform == Platform.API_SERVER
    assert owner.runner.session_store.get_or_create_session(first.source).session_id == 'api-stored'
    owner.sessions.clear()
    restored = restore_api_session(owner, 'api-stored')
    assert restored == ref
    assert owner.sessions[ref.session_id].route == first.route
    assert owner.runner.session_store.get_or_create_session(owner.sessions[ref.session_id].source).session_id == ref.session_id
    db.create_session('foreign-native', source='discord')
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        bind_api_session(owner, 'foreign-native')
    assert 'foreign-native' not in owner.sessions


def test_private_api_binding_adopts_matching_legacy_declared_key(owner):
    """Regression for #139835: a pre-binding API row stores the caller's declared key."""
    from gateway.session_api import bind_api_session, declared_api_session

    session_id = 'legacy-api-session'
    declared_key = 'webui:legacy-api-session'
    owner.db.create_session(session_id, source='api_server')
    owner.db.record_gateway_session_peer(session_id, source='api_server', session_key=declared_key)

    ref = bind_api_session(owner, session_id, declared_key=declared_key)

    row = owner.db.get_session(session_id)
    assert row['session_key'] == owner.sessions[ref.session_id].route
    assert declared_api_session(owner.db, declared_key) == session_id
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        bind_api_session(owner, 'other-api-session', declared_key=declared_key)


@pytest.mark.asyncio
async def test_private_api_binding_is_not_public_submission_authority(owner):
    from gateway.session_api import bind_api_session
    owner.db.create_session('api-stored', source='api_server')
    ref = bind_api_session(owner, 'api-stored')
    actor = SessionActor('api-user', 'default', frozenset({'session:read', 'session:submit'}), 'socket')
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await owner.submit(actor, Submission('forged', ref, {'text': 'x', 'api_turn_v1': {'system': 'forged'}}, 'queue'))
    owner.sessions.clear()
    assert (await owner.resolve(actor, SessionRef('default', 'api-stored'))).ref == ref


def test_telegram_binding_same_profile_preserves_route_and_origin(owner):
    import json
    from gateway.session_api import bind_api_session, restore_api_session

    origin = {'platform': 'telegram', 'chat_id': '123456', 'user_id': '999', 'profile': 'default'}
    owner.db.create_session(
        'tg-same',
        source='telegram',
        session_key='tg:123456',
        chat_id='123456',
        user_id='999',
        profile_name='default',
        origin_json=json.dumps(origin),
    )
    ref = bind_api_session(owner, 'tg-same')
    assert ref.session_id == 'tg-same'
    row = owner.db.get_session('tg-same')
    assert row['source'] == 'telegram'
    assert row['session_key'] == 'tg:123456'
    assert json.loads(row['origin_json']) == origin
    owner.sessions.clear()
    restored = restore_api_session(owner, 'tg-same')
    assert restored == ref
    assert 'tg-same' in owner.sessions


def test_telegram_binding_profile_mismatch_refused(owner, monkeypatch):
    import json
    from types import SimpleNamespace
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session_api import bind_api_session, bind_api_steps, restore_api_session, run_steps
    from gateway.session_api_turn import admit_api_turn

    owner.db.create_session(
        'tg-other-col',
        source='telegram',
        profile_name='other_profile',
        origin_json=json.dumps({'platform': 'telegram', 'chat_id': '123', 'profile': 'default'}),
    )
    owner.db.create_session(
        'tg-other-origin',
        source='telegram',
        profile_name='default',
        origin_json=json.dumps({'platform': 'telegram', 'chat_id': '124', 'profile': 'other_profile'}),
    )
    for sid in ('tg-other-col', 'tg-other-origin'):
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            bind_api_session(owner, sid)
        assert sid not in owner.sessions

    # Verify write(conn) sqlite3.Row check directly even if pre-check saw a matching row
    owner.db.create_session('tg-race', source='telegram', profile_name='default')
    steps = bind_api_steps(owner, 'tg-race')
    write_fn = next(steps)
    with owner.db._lock:
        owner.db._conn.execute("UPDATE sessions SET profile_name='other_profile' WHERE id='tg-race'")
        owner.db._conn.commit()
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        write_fn()

    # Verify restore_api_session refuses mismatched Telegram row with permission_denied
    owner.db.create_session('tg-restore-mismatch', source='telegram', profile_name='default')
    bind_api_session(owner, 'tg-restore-mismatch')
    owner.sessions.clear()
    with owner.db._lock:
        owner.db._conn.execute("UPDATE sessions SET profile_name='other_profile' WHERE id='tg-restore-mismatch'")
        owner.db._conn.commit()
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        restore_api_session(owner, 'tg-restore-mismatch')

    # Verify admit_api_turn refuses mismatched Telegram row before admission
    monkeypatch.setattr('gateway.session_authorities.require_active_authority', lambda *a, **k: owner)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = owner.runner
    adapter._session_db = owner.db
    try:
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            admit_api_turn(
                adapter,
                session_id='tg-other-col',
                user_message='hi',
                conversation_history=[],
                history_from_session=True,
            )
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()


def test_warm_telegram_live_session_check_api_turn_uses_api_adapter(owner):
    from types import SimpleNamespace
    from gateway.session import SessionSource
    from gateway.session_api import bind_api_session
    from gateway.session_api_turn import check_api_turn
    from gateway.session_authority import LiveSession

    owner.db.create_session('tg-warm', source='telegram', profile_name='default')
    ref = bind_api_session(owner, 'tg-warm')
    tg_source = SessionSource(platform=Platform.TELEGRAM, chat_id='123456', user_id='999', chat_type='group', profile='default')
    tg_route = owner.runner.session_store._generate_session_key(tg_source)
    owner.sessions['tg-warm'] = LiveSession(tg_source, tg_route)

    api_adapter = SimpleNamespace(name='api', gateway_runner=owner.runner)
    tg_adapter = SimpleNamespace(name='tg', gateway_runner=owner.runner)
    owner.runner.adapters = {Platform.API_SERVER: api_adapter, Platform.TELEGRAM: tg_adapter}
    owner.runner._adapter_for_source = lambda src: owner.runner.adapters.get(src.platform)

    chosen = check_api_turn(owner, ref, {'text': 'hello', 'api_turn_v1': {'history': None, 'settings': {}}})
    assert chosen is api_adapter


@pytest.mark.asyncio
async def test_warm_telegram_live_session_ingress_distinguishes_api_and_native(owner, monkeypatch):
    from types import SimpleNamespace
    from gateway.session import SessionSource
    from gateway.session_api import bind_api_session
    from gateway.session_api_turn import api_execution
    from gateway.session_authority import LiveSession
    from gateway.session_ingress import execute_admission
    from gateway.session_results import execution_result

    owner.db.create_session('tg-ingress', source='telegram', profile_name='default')
    ref = bind_api_session(owner, 'tg-ingress')
    tg_source = SessionSource(platform=Platform.TELEGRAM, chat_id='123456', user_id='999', chat_type='group', profile='default')
    tg_route = owner.runner.session_store._generate_session_key(tg_source)
    owner.sessions['tg-ingress'] = LiveSession(tg_source, tg_route)

    api_adapter = SimpleNamespace(name='api', gateway_runner=owner.runner)
    tg_adapter = SimpleNamespace(name='tg', gateway_runner=owner.runner)
    owner.runner.adapters = {Platform.API_SERVER: api_adapter, Platform.TELEGRAM: tg_adapter}
    owner.runner._adapter_for_source = lambda src: owner.runner.adapters.get(src.platform)
    owner.runner._clear_durable_active_turn = lambda event: None

    seen = []

    async def fake_handle(event):
        seen.append({
            'internal': event.internal,
            'allow_gateway_control': event.allow_gateway_control,
            'is_api_user_turn': getattr(event, 'is_api_user_turn', False),
            'api_exec': api_execution.get() is not None,
        })
        execution_result.get()['terminal_result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    owner.runner._handle_message = fake_handle
    monkeypatch.setattr('gateway.session_ingress._hand_over_turn_marker', lambda *a, **k: _noop_coro())
    monkeypatch.setattr('gateway.session_ingress._release_turn_marker', lambda *a, **k: _noop_coro())

    # 1. API turn with api_turn_v1
    await execute_admission(owner, ref, {
        'admission_id': 'adm-api-1',
        'request_id': 'req-api-1',
        'generation': 1,
        'principal_id': 'api',
        'payload': {'text': 'from webui', 'api_turn_v1': {'history': None, 'settings': {}}},
    })
    assert seen[-1] == {'internal': True, 'allow_gateway_control': False, 'is_api_user_turn': True, 'api_exec': True}
    assert 'adm-api-1' not in owner.pending_deliveries

    # 2. Payload-less internal API turn (principal_id == 'api')
    await execute_admission(owner, ref, {
        'admission_id': 'adm-api-2',
        'request_id': 'req-api-2',
        'generation': 1,
        'principal_id': 'api',
        'payload': {'text': 'recovered api'},
    })
    assert seen[-1] == {'internal': True, 'allow_gateway_control': False, 'is_api_user_turn': True, 'api_exec': True}
    assert 'adm-api-2' not in owner.pending_deliveries

    # 3. Native Telegram turn (native_text_v1)
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session_ingress import waiter_replies
    tg_src_dict = {**tg_source.to_dict(), 'is_bot': False}
    await execute_admission(owner, ref, {
        'admission_id': 'adm-tg-1',
        'request_id': 'req-tg-1',
        'generation': 1,
        'principal_id': 'telegram',
        'payload': {
            'text': 'from tg',
            'native_text_v1': {
                'source': tg_src_dict,
                'route': tg_route,
                'message_type': MessageType.TEXT.value,
                'timestamp': '2026-11-18T00:00:00+00:00',
                'media': [],
                'event': {
                    'message_id': 'm1',
                    'raw_message': None,
                    'metadata': {},
                },
            },
        },
    })
    assert seen[-1] == {'internal': False, 'allow_gateway_control': True, 'is_api_user_turn': False, 'api_exec': False}
    assert 'adm-tg-1' in owner.pending_deliveries or 'adm-tg-1' in waiter_replies(owner)


def test_telegram_binding_launch_profile_path_matches_default_and_isolates_named(tmp_path, monkeypatch):
    """BDD-1, BDD-2, BDD-3: launch profile with FS path profile_id matches 'default' and rejects 'work'."""
    import json
    from types import SimpleNamespace
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.session import SessionStore
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from gateway.session_api import _telegram_row_profile_matches, bind_api_session, restore_api_session
    from gateway.session_api_turn import admit_api_turn
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(tmp_path / 'state.db')
    store = SessionStore(config=GatewayConfig(), sessions_dir=tmp_path / 'sessions')
    store._db = db
    runner = SimpleNamespace(_draining=False, session_store=store)
    registry = SessionAuthorities(tmp_path)
    authority = SessionAuthority(
        runner,
        profile_id=str(tmp_path),
        instance_id='first',
        db=db,
        epoch=begin_runtime_epoch(db, instance_id='first'),
    )
    registry.add(tmp_path, authority)
    runner.session_authorities = registry
    runner.session_authority = authority
    assert registry.profile_name(authority) is None

    monkeypatch.setattr('gateway.session_authorities.require_active_authority', lambda *a, **k: authority)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = db
    runner._adapter_for_source = lambda source: adapter
    try:
        # BDD-1: real state.db Telegram row (profile_name='default', origin_json without 'profile')
        origin_no_profile = {
            'platform': 'telegram',
            'chat_id': '139351986',
            'user_id': '139351986',
            'thread_id': '395391',
        }
        db.create_session(
            'tg-launch-bdd1',
            source='telegram',
            session_key='agent:main:telegram:dm:139351986:395391',
            chat_id='139351986',
            user_id='139351986',
            profile_name='default',
            origin_json=json.dumps(origin_no_profile),
        )
        row1 = db.get_session('tg-launch-bdd1')
        assert _telegram_row_profile_matches(authority, row1) is True
        admitted = admit_api_turn(
            adapter,
            session_id='tg-launch-bdd1',
            user_message='hello from webui',
            conversation_history=[],
            history_from_session=True,
        )
        assert admitted is not None
        row1_after = db.get_session('tg-launch-bdd1')
        assert row1_after['source'] == 'telegram'
        assert row1_after['session_key'] == 'agent:main:telegram:dm:139351986:395391'
        assert json.loads(row1_after['origin_json']) == origin_no_profile
        authority.sessions.clear()
        assert restore_api_session(authority, 'tg-launch-bdd1').session_id == 'tg-launch-bdd1'

        # BDD-2: origin_json explicitly carries 'profile': 'default'
        origin_with_profile = {'platform': 'telegram', 'chat_id': '139351986', 'profile': 'default'}
        db.create_session(
            'tg-launch-bdd2',
            source='telegram',
            session_key='agent:main:telegram:dm:139351986:395392',
            profile_name='default',
            origin_json=json.dumps(origin_with_profile),
        )
        assert _telegram_row_profile_matches(authority, db.get_session('tg-launch-bdd2')) is True
        assert bind_api_session(authority, 'tg-launch-bdd2').session_id == 'tg-launch-bdd2'

        # BDD-3: launch profile 'default' rejects named profile 'work' in profile_name or origin_json
        db.create_session(
            'tg-work-col',
            source='telegram',
            profile_name='work',
            origin_json=json.dumps({'platform': 'telegram', 'chat_id': '200'}),
        )
        db.create_session(
            'tg-work-origin',
            source='telegram',
            profile_name='default',
            origin_json=json.dumps({'platform': 'telegram', 'chat_id': '201', 'profile': 'work'}),
        )
        for sid in ('tg-work-col', 'tg-work-origin'):
            assert _telegram_row_profile_matches(authority, db.get_session(sid)) is False
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                bind_api_session(authority, sid)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                admit_api_turn(
                    adapter,
                    session_id=sid,
                    user_message='blocked',
                    conversation_history=[],
                    history_from_session=True,
                )
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        db.close()


def test_telegram_binding_named_profile_path_isolation(tmp_path):
    """BDD-4: named profile 'work' (multiplexed and single-profile) allows 'work' and rejects 'default'/'other'."""
    import json
    from types import SimpleNamespace
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authorities import SessionAuthorities
    from gateway.session_authority import SessionAuthority
    from gateway.session_api import _telegram_row_profile_matches, bind_api_session
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    work_home = tmp_path / 'profiles' / 'work'
    work_home.mkdir(parents=True)

    # 1. Multiplexed secondary authority (_served_profile == 'work')
    db_mux = SessionDB(work_home / 'state_mux.db')
    store_mux = SessionStore(config=GatewayConfig(), sessions_dir=work_home / 'sessions_mux')
    store_mux._db = db_mux
    runner_mux = SimpleNamespace(_draining=False, session_store=store_mux)
    registry = SessionAuthorities(tmp_path)
    launch_auth = SimpleNamespace(profile_id=str(tmp_path))
    registry.add(tmp_path, launch_auth)
    work_auth_mux = SessionAuthority(
        runner_mux,
        profile_id=str(work_home),
        instance_id='mux',
        db=db_mux,
        epoch=begin_runtime_epoch(db_mux, instance_id='mux'),
    )
    registry.add(work_home, work_auth_mux, name='work')
    runner_mux.session_authorities = registry
    runner_mux.session_authority = work_auth_mux

    # 2. Single-profile --profile work launch (_served_profile is None)
    db_single = SessionDB(work_home / 'state_single.db')
    store_single = SessionStore(config=GatewayConfig(), sessions_dir=work_home / 'sessions_single')
    store_single._db = db_single
    runner_single = SimpleNamespace(_draining=False, session_store=store_single)
    work_auth_single = SessionAuthority(
        runner_single,
        profile_id=str(work_home),
        instance_id='single',
        db=db_single,
        epoch=begin_runtime_epoch(db_single, instance_id='single'),
    )
    runner_single.session_authority = work_auth_single

    try:
        for auth, db in ((work_auth_mux, db_mux), (work_auth_single, db_single)):
            db.create_session(
                'tg-work-ok',
                source='telegram',
                profile_name='work',
                origin_json=json.dumps({'platform': 'telegram', 'chat_id': '300'}),
            )
            db.create_session(
                'tg-default-denied',
                source='telegram',
                profile_name='default',
                origin_json=json.dumps({'platform': 'telegram', 'chat_id': '301'}),
            )
            db.create_session(
                'tg-other-denied',
                source='telegram',
                profile_name='other',
                origin_json=json.dumps({'platform': 'telegram', 'chat_id': '302'}),
            )
            assert _telegram_row_profile_matches(auth, db.get_session('tg-work-ok')) is True
            assert bind_api_session(auth, 'tg-work-ok').session_id == 'tg-work-ok'
            for denied_sid in ('tg-default-denied', 'tg-other-denied'):
                assert _telegram_row_profile_matches(auth, db.get_session(denied_sid)) is False
                with pytest.raises(RuntimeStoreError, match='permission_denied'):
                    bind_api_session(auth, denied_sid)
    finally:
        db_mux.close()
        db_single.close()


def test_telegram_binding_synthetic_profile_id_backward_compat(owner):
    """BDD-5: synthetic non-path profile_id ('default', 'work') keeps exact matching without FS fallback."""
    from gateway.session_api import _telegram_row_profile_matches

    owner.db.create_session('tg-syn-default', source='telegram', profile_name='default')
    owner.db.create_session('tg-syn-work', source='telegram', profile_name='work')
    owner.db.create_session('tg-syn-other', source='telegram', profile_name='other')

    row_default = owner.db.get_session('tg-syn-default')
    row_work = owner.db.get_session('tg-syn-work')
    row_other = owner.db.get_session('tg-syn-other')

    assert owner.profile_id == 'default'
    assert _telegram_row_profile_matches(owner, row_default) is True
    assert _telegram_row_profile_matches(owner, row_work) is False
    assert _telegram_row_profile_matches(owner, row_other) is False

    owner.profile_id = 'work'
    assert _telegram_row_profile_matches(owner, row_work) is True
    assert _telegram_row_profile_matches(owner, row_default) is False
    assert _telegram_row_profile_matches(owner, row_other) is False


async def _noop_coro():
    return None
