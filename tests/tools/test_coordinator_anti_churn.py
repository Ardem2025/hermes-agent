"""BDD 3, BDD 4, BDD 5 tests for coordinator-polling-anti-churn."""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools import async_delegation as ad
from tools.delegate_tool import DELEGATE_TASK_SCHEMA, delegate_task
from tools.delegate_tool_dispatch import _BACKGROUND_NOTES, _Batch, _dispatched_payload
from tools.delegate_tool_registry import (
    _find_active_duplicate_subagent,
    _list_payload,
    _register_subagent,
    _unregister_subagent,
)
from tools.process_registry_notifications import (
    async_delegation_batch_display_text,
    format_process_notification,
)


def test_bdd3_active_siblings_and_no_reply_in_notifications(monkeypatch):
    monkeypatch.setattr(
        ad,
        '_records',
        {
            'deleg_done': {
                'delegation_id': 'deleg_done',
                'session_key': 'sess-A',
                'parent_session_id': 'p-A',
                'status': 'completed',
                'dispatched_at': time.time() - 30,
                'goal': 'Finished task',
            },
            'deleg_running': {
                'delegation_id': 'deleg_running',
                'session_key': 'sess-A',
                'parent_session_id': 'p-A',
                'status': 'running',
                'dispatched_at': time.time() - 15,
                'goal': 'Still working on analysis ' + ('x' * 150),
                'context': 'SECRET_CONTEXT_MUST_NOT_LEAK',
            },
            'deleg_other_session': {
                'delegation_id': 'deleg_other_session',
                'session_key': 'sess-B',
                'parent_session_id': 'p-B',
                'status': 'running',
                'dispatched_at': time.time() - 10,
                'goal': 'Foreign session goal',
            },
        },
    )
    siblings = ad.list_active_delegations_for_session('sess-A', exclude_delegation_id='deleg_done')
    assert len(siblings) == 1
    assert siblings[0]['delegation_id'] == 'deleg_running'
    assert len(siblings[0]['goal']) <= 120
    assert 'context' not in siblings[0]

    single_msg = format_process_notification({
        'type': 'async_delegation',
        'delegation_id': 'deleg_done',
        'session_key': 'sess-A',
        'status': 'completed',
        'summary': 'Done',
        'goal': 'Finished task',
    })
    assert 'Still running in this session (1):' in single_msg
    assert 'deleg_running' in single_msg
    assert 'Foreign session goal' not in single_msg
    assert 'SECRET_CONTEXT_MUST_NOT_LEAK' not in single_msg
    assert 'NO_REPLY' not in single_msg
    assert 'Briefly report the current status or act on this result, then end your turn and wait for the remaining subagents to finish.' in single_msg
    assert "delegate_task(action='list')" in single_msg

    batch_msg = format_process_notification({
        'type': 'async_delegation',
        'delegation_id': 'deleg_done',
        'session_key': 'sess-A',
        'is_batch': True,
        'goals': ['Task 1'],
        'results': [{'task_index': 0, 'status': 'completed', 'summary': 'OK'}],
    })
    assert 'Still running in this session (1):' in batch_msg
    assert 'NO_REPLY' not in batch_msg
    assert 'Briefly report the current status or act on this result, then end your turn and wait for the remaining subagents to finish.' in batch_msg
    assert "delegate_task(action='list')" in batch_msg

    # When no siblings remain running
    ad._records['deleg_running']['status'] = 'completed'
    idle_msg = format_process_notification({
        'type': 'async_delegation',
        'delegation_id': 'deleg_done',
        'session_key': 'sess-A',
        'status': 'completed',
        'summary': 'Done',
        'goal': 'Finished task',
    })
    assert 'No other background subagents are currently running in this session.' in idle_msg
    assert 'NO_REPLY' not in idle_msg
    assert 'Act on this result or summarize the outcome for the user.' in idle_msg
    assert "delegate_task(action='list')" in idle_msg
    assert async_delegation_batch_display_text(3) == '3 Subagent Tasks Completed'


def test_bdd4_hides_live_transcripts_and_updates_anti_polling_contract(monkeypatch):
    class _Dummy:
        pass
    parent = _Dummy()
    parent._delegate_depth = 0
    parent.session_id = 'p-1'
    child = _Dummy()
    child._subagent_id = 'sa-1'
    child._live_transcript_path = '/tmp/live/sa-1.log'
    batch = _Batch(
        task_list=[{'goal': 'Do work'}],
        children=[(0, {'goal': 'Do work'}, child)],
        parent_agent=parent,
        creds={'model': 'm'},
        context=None,
        top_role='leaf',
        max_children=3,
        live_deleg_id='deleg-1',
        live_writers=[],
        live_paths=['/tmp/live/sa-1.log'],
        origin_wake_sid='wake-1',
        origin_ui_session_id='ui-1',
        origin_owner_transport=None,
        origin_owner_session_record=None,
        origin_session_history_delivery=False,
        overall_start=time.time(),
    )
    payload = _dispatched_payload(batch, [(batch, 'deleg-1')])
    assert 'live_transcripts' not in payload
    assert 'live_transcripts_hint' not in payload
    assert "Do NOT call action='list' to poll progress" in payload['control_hint']
    assert 'live_transcripts_hint' not in _BACKGROUND_NOTES

    # Registry dedup & list must not expose live_transcript
    import weakref
    child._delegate_parent_ref = weakref.ref(parent)
    _register_subagent({
        'subagent_id': 'sa-bdd4',
        'delegation_id': 'deleg-bdd4',
        'parent_id': None,
        'depth': 0,
        'goal': 'Unique BDD4 goal',
        'model': 'm',
        'started_at': time.time(),
        'status': 'running',
        'tool_count': 0,
        'agent': child,
    })
    try:
        dup = _find_active_duplicate_subagent(parent, 'Unique BDD4 goal')
        assert dup is not None
        assert 'live_transcript' not in dup

        listed = _list_payload(parent)
        assert listed['count'] == 1
        assert 'live_transcript' not in listed['subagents'][0]
        assert 'NO_REPLY' in listed['note']

        dedup_json = json.loads(delegate_task(goal='Unique BDD4 goal', parent_agent=parent))
        assert dedup_json['deduplicated'] is True
        assert 'live_transcripts' not in dedup_json
        assert 'live_transcripts_hint' not in dedup_json
    finally:
        _unregister_subagent('sa-bdd4')

    action_desc = DELEGATE_TASK_SCHEMA['parameters']['properties']['action']['description']
    assert '/transcripts' not in action_desc
    assert 'transcripts' not in action_desc
    assert 'do NOT use to poll progress' in action_desc


def test_bdd5_same_turn_warning_and_turn_reset(monkeypatch):
    from run_agent import AIAgent
    import tools.delegate_tool as dt

    agent = object.__new__(AIAgent)
    agent._delegate_depth = 0
    agent._turn_dispatched_delegations = []

    calls = iter([
        json.dumps({'status': 'dispatched', 'delegation_id': 'd-1'}),
        json.dumps({'status': 'dispatched', 'delegation_id': 'd-2'}),
        json.dumps({'status': 'dispatched', 'delegation_id': 'd-3', 'deduplicated': True}),
        json.dumps({'action': 'list', 'count': 1, 'subagents': []}),
    ])
    monkeypatch.setattr(dt, 'delegate_task', lambda **_kw: next(calls))

    r1 = json.loads(agent._dispatch_delegate_task({'goal': 'first'}))
    assert 'same_turn_warning' not in r1

    r2 = json.loads(agent._dispatch_delegate_task({'goal': 'second'}))
    assert 'same_turn_warning' in r2
    assert 'END YOUR TURN' in r2['same_turn_warning']

    r3 = json.loads(agent._dispatch_delegate_task({'goal': 'dup'}))
    assert 'same_turn_warning' not in r3

    r4 = json.loads(agent._dispatch_delegate_task({'action': 'list'}))
    assert 'same_turn_warning' not in r4

    # Verify run_conversation resets _turn_dispatched_delegations
    import agent.conversation_loop as cl
    import agent.turn_context as tc
    import agent.turn_finalizer as tf
    import agent.voice_turn_route as vtr
    monkeypatch.setattr(cl, '_run_conversation_turn', lambda *_a, **_kw: {'final_response': 'ok', 'messages': []})
    monkeypatch.setattr(tf, 'settle_held_final', lambda *_a, **_kw: None)
    monkeypatch.setattr(tc, 'export_current_turn_boundary', lambda _a, res, _u: res)
    monkeypatch.setattr(vtr, 'end_voice_turn_route', lambda *_a, **_kw: None)
    agent.session_id = 's-1'
    cl.run_conversation(agent, 'hello')
    assert agent._turn_dispatched_delegations == []
