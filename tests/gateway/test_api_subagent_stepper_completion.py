"""Tests for subagent terminal buffer (_recent_subagents), recent_subagents API field, and pending delivery separation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def _make_adapter(monkeypatch, tmp_path: Path) -> APIServerAdapter:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    return APIServerAdapter(PlatformConfig(enabled=True))


@pytest.fixture(autouse=True)
def _clean_subagent_registries():
    from tools import delegate_tool_registry as reg

    saved_active = dict(reg._active_subagents)
    saved_recent = dict(reg._recent_subagents)
    with reg._active_subagents_lock:
        reg._active_subagents.clear()
        reg._recent_subagents.clear()
    try:
        yield
    finally:
        with reg._active_subagents_lock:
            reg._active_subagents.clear()
            reg._active_subagents.update(saved_active)
            reg._recent_subagents.clear()
            reg._recent_subagents.update(saved_recent)


def test_emit_complete_and_finish_failed_preserve_terminal_fields_in_recent_subagents():
    from tools.delegate_tool_child_run import _ChildRun
    from tools.delegate_tool_registry import (
        _active_subagents,
        _recent_subagents,
        _register_subagent,
        _unregister_subagent,
        get_recent_subagents_for_session,
        get_subagent_attribution,
    )

    child_ok = SimpleNamespace(
        session_id="child-ok",
        _delegate_output_schema=None,
        session_prompt_tokens=10,
        session_completion_tokens=5,
        session_reasoning_tokens=0,
        session_estimated_cost_usd=0.0,
    )
    _register_subagent({
        "subagent_id": "sa-ok",
        "parent_id": None,
        "depth": 0,
        "goal": "Analyze logs",
        "delegation_id": "deleg-1",
        "model": "test-model",
        "started_at": time.time() - 12.5,
        "status": "running",
        "tool_count": 4,
        "last_tool": "read_file",
        "owner_agent_session_id": "sess-parent-1",
        "owner_session_id": "ui-1",
        "agent": child_ok,
    })
    run_ok = _ChildRun(child_ok, None, 0, "Analyze logs", "sa-ok", None)
    run_ok.emit_complete({"final_response": "Done"}, {"status": "completed", "api_calls": 2, "summary": "Done"}, 12.5)
    _unregister_subagent("sa-ok", agent=child_ok)

    assert "sa-ok" not in _active_subagents
    assert "sa-ok" in _recent_subagents
    rec_ok = _recent_subagents["sa-ok"]
    assert rec_ok["status"] == "completed"
    assert rec_ok["duration_seconds"] == 12.5
    assert rec_ok["tool_count"] == 4
    assert rec_ok["last_tool"] == "read_file"
    assert isinstance(rec_ok["ended_at"], float)
    assert rec_ok["completed_at"] == rec_ok["ended_at"]
    assert get_subagent_attribution("sa-ok") == {
        "subagent_id": "sa-ok",
        "goal": "Analyze logs",
        "delegation_id": "deleg-1",
    }

    child_fail = SimpleNamespace(session_id="child-fail", _delegate_output_schema=None)
    _register_subagent({
        "subagent_id": "sa-fail",
        "parent_id": None,
        "depth": 0,
        "goal": "Run failing check",
        "delegation_id": "deleg-2",
        "model": "test-model",
        "started_at": time.time() - 4.2,
        "status": "running",
        "tool_count": 1,
        "last_tool": "terminal",
        "owner_agent_session_id": "sess-parent-1",
        "agent": child_fail,
    })
    run_fail = _ChildRun(child_fail, None, 1, "Run failing check", "sa-fail", None)
    run_fail.finish_failed(
        {"status": "error", "duration_seconds": 4.2, "error": "boom", "failure_reason": "tool_error"},
        None,
        preview="boom",
        summary="boom",
        status="failed",
    )
    _unregister_subagent("sa-fail", agent=child_fail)

    rec_fail = _recent_subagents["sa-fail"]
    assert rec_fail["status"] == "failed"
    assert rec_fail["duration_seconds"] == 4.2
    assert rec_fail["failure_reason"] == "tool_error"
    assert rec_fail["error"] == "boom"

    recent_for_sess = get_recent_subagents_for_session("sess-parent-1")
    assert {r["subagent_id"] for r in recent_for_sess} == {"sa-ok", "sa-fail"}
    assert get_recent_subagents_for_session("sess-other") == []


def test_recent_subagents_ttl_and_cap_eviction():
    from tools.delegate_tool_registry import (
        _RECENT_SUBAGENTS_CAP,
        _RECENT_SUBAGENTS_TTL_S,
        _recent_subagents,
        _register_subagent,
        _unregister_subagent,
        get_recent_subagents_for_session,
    )

    now = time.time()
    _register_subagent({
        "subagent_id": "sa-expired",
        "goal": "Old task",
        "owner_agent_session_id": "sess-ttl",
        "started_at": now - _RECENT_SUBAGENTS_TTL_S - 20.0,
        "status": "running",
    })
    _unregister_subagent("sa-expired", status="completed", ended_at=now - _RECENT_SUBAGENTS_TTL_S - 5.0)
    assert get_recent_subagents_for_session("sess-ttl") == []
    assert "sa-expired" not in _recent_subagents

    for i in range(_RECENT_SUBAGENTS_CAP + 15):
        sid = f"sa-cap-{i}"
        _register_subagent({
            "subagent_id": sid,
            "goal": f"Task {i}",
            "owner_agent_session_id": "sess-cap",
            "started_at": now - 1.0,
            "status": "running",
        })
        _unregister_subagent(sid, status="completed", ended_at=now)

    assert len(_recent_subagents) == _RECENT_SUBAGENTS_CAP
    assert "sa-cap-0" not in _recent_subagents
    assert f"sa-cap-{_RECENT_SUBAGENTS_CAP + 14}" in _recent_subagents


@pytest.mark.asyncio
async def test_session_endpoint_returns_recent_subagents_without_breaking_active_count(tmp_path, monkeypatch):
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent

    adapter = _make_adapter(monkeypatch, tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-api-1", source="webui", model="test-model")
    adapter._session_db = db

    now = time.time()
    _register_subagent({
        "subagent_id": "sa-done-1",
        "goal": "Finished branch",
        "owner_agent_session_id": "sess-api-1",
        "started_at": now - 15.0,
        "status": "running",
        "tool_count": 3,
        "last_tool": "search_files",
    })
    _unregister_subagent("sa-done-1", status="completed", ended_at=now - 2.0, duration_seconds=13.0)

    _register_subagent({
        "subagent_id": "sa-live-1",
        "goal": "Running branch",
        "owner_agent_session_id": "sess-api-1",
        "started_at": now - 3.0,
        "status": "running",
        "tool_count": 1,
        "last_tool": "read_file",
    })

    get_handler = getattr(APIServerAdapter._handle_get_session, "__wrapped__", APIServerAdapter._handle_get_session)
    resp = await get_handler(
        adapter,
        make_mocked_request("GET", "/api/sessions/sess-api-1", match_info={"session_id": "sess-api-1"}),
    )
    assert resp.status == 200
    session_payload = json.loads(resp.text)["session"]

    assert session_payload["active_subagents_count"] == 1
    assert [s["subagent_id"] for s in session_payload["active_subagents"]] == ["sa-live-1"]
    assert [s["subagent_id"] for s in session_payload["recent_subagents"]] == ["sa-done-1"]
    assert session_payload["recent_subagents"][0]["status"] == "completed"
    assert session_payload["recent_subagents"][0]["duration_seconds"] == 13.0
    assert session_payload["recent_subagents"][0]["tool_count"] == 3

    # Unregister the last live subagent -> active_subagents_count drops to 0, both appear in recent_subagents
    _unregister_subagent("sa-live-1", status="failed", ended_at=now, duration_seconds=3.0)
    resp2 = await get_handler(
        adapter,
        make_mocked_request("GET", "/api/sessions/sess-api-1", match_info={"session_id": "sess-api-1"}),
    )
    session_payload2 = json.loads(resp2.text)["session"]
    assert session_payload2["active_subagents_count"] == 0
    assert session_payload2["has_active_subagents"] is False
    assert session_payload2["is_generating"] is False
    assert session_payload2["current_tool"] is None
    assert {s["subagent_id"] for s in session_payload2["recent_subagents"]} == {"sa-done-1", "sa-live-1"}
    db.close()


def test_pending_delivery_without_live_workers_does_not_stick_active_subagents(tmp_path, monkeypatch):
    from tools import async_delegation

    adapter = _make_adapter(monkeypatch, tmp_path)
    monkeypatch.setattr(async_delegation, "_records", {})

    now = time.time()
    with async_delegation._DB_LOCK, async_delegation._transaction() as conn:
        conn.execute(
            """
            INSERT INTO async_delegations (
                delegation_id, origin_session, origin_session_id, origin_ui_session_id,
                parent_session_id, state, delivery_state, delivery_attempts,
                dispatched_at, updated_at, task_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("deleg-pending-1", "sess-pend", "sess-pend", "sess-pend", "sess-pend", "completed", "pending", 0, now - 20, now - 5, "{}"),
        )

    assert async_delegation.has_live_for_session(
        session_key="sess-pend", origin_ui_session_id="sess-pend", parent_session_id="sess-pend"
    ) is False
    assert async_delegation.has_pending_delivery_for_session(
        session_key="sess-pend", origin_ui_session_id="sess-pend", parent_session_id="sess-pend"
    ) is True
    assert async_delegation.has_active_or_pending_for_session(
        session_key="sess-pend", origin_ui_session_id="sess-pend", parent_session_id="sess-pend"
    ) is True

    res = adapter._session_response(
        {"id": "sess-pend", "started_at": now - 30, "last_active": now - 5, "last_read_at": now - 1},
        last_msg={"role": "assistant", "content": "Done.", "has_tool_calls": False, "finish_reason": "stop"},
    )
    assert res["has_active_subagents"] is False
    assert res["has_pending_delegation_delivery"] is True
    assert res["active_subagents_count"] == 0
    assert res["is_generating"] is False
    assert res["current_tool"] is None
    assert res["status"] == "idle"
