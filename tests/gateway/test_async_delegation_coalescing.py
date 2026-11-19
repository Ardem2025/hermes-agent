"""BDD 1 & BDD 2 tests for coordinator-polling-anti-churn."""
import asyncio
import queue
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_automation import admit_process_notice
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter
from tools.process_registry_notifications import async_delegation_batch_display_text


@pytest.mark.asyncio
async def test_bdd1_coalesces_three_async_delegations_on_telegram_style_route(tmp_path, monkeypatch):
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='bdd1')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    human = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, human, get_hermes_home()):
        await authority.admit_native(human)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    human_rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    authority.db._write_sql(
        "UPDATE session_admissions SET status='started', owner_epoch=1, generation=1 WHERE admission_id=?",
        (human_rows[0]['admission_id'],),
    )

    for did in ('deleg_A', 'deleg_B', 'deleg_C'):
        evt = {'type': 'async_delegation', 'delegation_id': did, 'session_key': key, 'parent_session_id': sid}
        msg = MessageEvent(
            text=f'Result for {did}',
            source=source,
            message_type=MessageType.TEXT,
            internal=True,
            metadata={'gateway_session_id': sid, 'gateway_session_key': key},
        )
        await admit_process_notice(runner, authority, adapter, msg, evt)

    queued = [r for r in list_session_admissions(authority.db, session_id=sid, pending_only=False) if r['status'] == 'queued']
    assert len(queued) == 1
    text = queued[0]['payload']['text']
    assert text.count('[IMPORTANT:') == 1
    assert '3 background subagent delegations completed for this session' in text
    for did in ('deleg_A', 'deleg_B', 'deleg_C'):
        assert f'Result for {did}' in text
    identities = queued[0]['payload']['native_text_v1']['automation']['identities']
    assert len(identities) == 3


@pytest.mark.asyncio
async def test_bdd1_never_folds_across_types_task_failure_or_human_message(tmp_path, monkeypatch):
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='bdd1-guards')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    human = MessageEvent(text='human', source=source, message_id='human')
    with native_callback(runner, human, get_hermes_home()):
        await authority.admit_native(human)
    key = runner.session_store._generate_session_key(source)
    sid = runner.session_store.peek_session_id(key)
    human_rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    authority.db._write_sql(
        "UPDATE session_admissions SET status='started', owner_epoch=1, generation=1 WHERE admission_id=?",
        (human_rows[0]['admission_id'],),
    )

    # 1. completion + async_delegation must not fold
    await admit_process_notice(
        runner, authority, adapter,
        MessageEvent(text='proc done', source=source, message_type=MessageType.TEXT, internal=True,
                     metadata={'gateway_session_id': sid, 'gateway_session_key': key}),
        {'type': 'completion', 'session_id': 'proc-1', 'started_at': 1.0, 'session_key': key},
    )
    await admit_process_notice(
        runner, authority, adapter,
        MessageEvent(text='deleg 1', source=source, message_type=MessageType.TEXT, internal=True,
                     metadata={'gateway_session_id': sid, 'gateway_session_key': key}),
        {'type': 'async_delegation', 'delegation_id': 'deleg_1', 'session_key': key},
    )
    # 2. task_failure_notice must not fold with async_delegation
    await admit_process_notice(
        runner, authority, adapter,
        MessageEvent(text='task failed', source=source, message_type=MessageType.TEXT, internal=True,
                     metadata={'gateway_session_id': sid, 'gateway_session_key': key}),
        {'type': 'async_delegation', 'delegation_id': 'deleg_2', 'task_failure_notice': True, 'results': [{'task_index': 0}], 'session_key': key},
    )
    await admit_process_notice(
        runner, authority, adapter,
        MessageEvent(text='deleg 2', source=source, message_type=MessageType.TEXT, internal=True,
                     metadata={'gateway_session_id': sid, 'gateway_session_key': key}),
        {'type': 'async_delegation', 'delegation_id': 'deleg_2', 'session_key': key},
    )
    # 3. human message between deleg_2 and deleg_3 prevents folding
    human2 = MessageEvent(text='human-2', source=source, message_id='human-2')
    with native_callback(runner, human2, get_hermes_home()):
        await authority.admit_native(human2)
    await admit_process_notice(
        runner, authority, adapter,
        MessageEvent(text='deleg 3', source=source, message_type=MessageType.TEXT, internal=True,
                     metadata={'gateway_session_id': sid, 'gateway_session_key': key}),
        {'type': 'async_delegation', 'delegation_id': 'deleg_3', 'session_key': key},
    )

    queued = [r for r in list_session_admissions(authority.db, session_id=sid, pending_only=False) if r['status'] == 'queued']
    assert len(queued) == 6


def test_bdd2_route_worker_merges_ready_groups_and_respects_origin_session_id(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import tools.process_registry as pr_module
    isolated = queue.Queue()
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(registry, 'completion_queue', isolated)
    monkeypatch.setattr(pr_module, 'process_registry', registry)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__('threading').Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._sweep_orphaned_completion_ledgers = lambda: None
    runner._drain_watch_notifications = AsyncMock()

    delivered_groups = []

    async def fake_deliver(group):
        delivered_groups.append([e['delegation_id'] for e in group])
        return True

    runner._deliver_async_delegation_group = fake_deliver

    # Two events with different parent_session_id -> two route_keys? No, same route_key if same fields,
    # so simulate two ticks or pre-populated route_pending via two events with same route_key arriving across ticks
    # while sem is held, OR test directly by putting two events with same route_key and checking merge.
    # Wait: in one tick, same route_key forms one group; to get >=2 groups in route_pending[route_key],
    # we can make the first delivery yield once while a second tick appends another group!
    e1 = {'type': 'async_delegation', 'delegation_id': 'd1', 'session_key': 'agent:main:telegram:dm:1'}
    e2 = {'type': 'async_delegation', 'delegation_id': 'd2', 'session_key': 'agent:main:telegram:dm:1'}
    e3 = {'type': 'async_delegation', 'delegation_id': 'd3', 'session_key': 'agent:main:telegram:dm:1'}
    api1 = {'type': 'async_delegation', 'delegation_id': 'a1', 'session_key': 'api-1', 'origin_session_id': 'api-1'}
    api2 = {'type': 'async_delegation', 'delegation_id': 'a2', 'session_key': 'api-1', 'origin_session_id': 'api-1'}

    tick = 0
    gate = asyncio.Event()

    async def slow_deliver(group):
        delivered_groups.append([e['delegation_id'] for e in group])
        if len(delivered_groups) == 1:
            await gate.wait()
        return True

    runner._deliver_async_delegation_group = slow_deliver

    real_sleep = asyncio.sleep

    async def fake_sleep(_delay):
        nonlocal tick
        tick += 1
        if tick == 2:
            isolated.put(dict(e2))
        elif tick == 3:
            isolated.put(dict(e3))
        elif tick == 4:
            gate.set()
        elif tick >= 7:
            runner._running = False
        await real_sleep(0.05)

    isolated.put(dict(e1))
    monkeypatch.setattr(asyncio, 'sleep', fake_sleep)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    # First call got [d1]; while blocked, ticks 1 & 2 queued [d2] and [d3] into route_pending,
    # which _route_worker merged into ONE call ['d2', 'd3']!
    assert delivered_groups == [['d1'], ['d2', 'd3']]

    # Now verify origin_session_id groups are NOT merged across ticks
    delivered_groups.clear()
    runner._running = True
    tick = 0
    gate = asyncio.Event()
    isolated.put(dict(api1))

    async def fake_sleep_api(_delay):
        nonlocal tick
        tick += 1
        if tick == 2:
            isolated.put(dict(api2))
        elif tick == 3:
            isolated.put(dict(api2, delegation_id='a3'))
        elif tick == 4:
            gate.set()
        elif tick >= 7:
            runner._running = False
        await real_sleep(0.05)

    monkeypatch.setattr(asyncio, 'sleep', fake_sleep_api)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    assert delivered_groups == [['a1'], ['a2'], ['a3']]


def test_bdd2_route_worker_requeues_entire_merged_group_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import tools.process_registry as pr_module
    isolated = queue.Queue()
    registry = pr_module.ProcessRegistry()
    monkeypatch.setattr(registry, 'completion_queue', isolated)
    monkeypatch.setattr(pr_module, 'process_registry', registry)

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={})
    runner._session_source_cache = {}
    runner._completion_delivery_lock = __import__('threading').Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._sweep_orphaned_completion_ledgers = lambda: None
    runner._drain_watch_notifications = AsyncMock()

    attempts = []
    gate = asyncio.Event()

    async def fail_then_succeed(group):
        attempts.append([e['delegation_id'] for e in group])
        if len(attempts) == 1:
            await gate.wait()
            return True
        if len(attempts) == 2:
            return False
        return True

    runner._deliver_async_delegation_group = fail_then_succeed
    e1 = {'type': 'async_delegation', 'delegation_id': 'r1', 'session_key': 'agent:main:telegram:dm:2'}
    e2 = {'type': 'async_delegation', 'delegation_id': 'r2', 'session_key': 'agent:main:telegram:dm:2'}
    e3 = {'type': 'async_delegation', 'delegation_id': 'r3', 'session_key': 'agent:main:telegram:dm:2'}
    isolated.put(dict(e1))

    tick = 0
    from gateway import run_notifications
    from tests.module_clock import install_module_clock
    clock = install_module_clock(monkeypatch, run_notifications)

    real_sleep = asyncio.sleep

    async def fake_sleep(_delay):
        nonlocal tick
        tick += 1
        if tick == 2:
            isolated.put(dict(e2))
        elif tick == 3:
            isolated.put(dict(e3))
        elif tick == 4:
            gate.set()
        elif tick == 5:
            clock.now += 60
        elif tick >= 8:
            runner._running = False
        await real_sleep(0.05)

    monkeypatch.setattr(asyncio, 'sleep', fake_sleep)
    asyncio.run(runner._async_delegation_watcher(interval=0))
    assert attempts == [['r1'], ['r2', 'r3'], ['r2', 'r3']]
