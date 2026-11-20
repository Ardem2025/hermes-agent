"""Tests for Wave 1 Phase 2A Backend Stability (BDD-5)."""

from __future__ import annotations

import ast
import asyncio
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms import api_server_runs as api_runs_mod
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_TELEGRAM_SYNC_LOCKS,
    _IdempotencyCache,
    _transform_media_paths,
    _transform_media_paths_async,
)
from gateway.platforms.api_server_runs import MAX_RUN_STATUSES
from hermes_state import SessionDB


def _make_adapter(monkeypatch, tmp_path: Path) -> APIServerAdapter:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    return APIServerAdapter(PlatformConfig(enabled=True))


@pytest.mark.asyncio
async def test_transform_media_paths_async_uses_to_thread(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    img_dir = hermes_home / "image_cache"
    img_dir.mkdir(parents=True, exist_ok=True)
    src = img_dir / "sample.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(api_server_mod, "_get_hermes_home_path", lambda: hermes_home)
    monkeypatch.setattr(api_server_mod.time, "time", lambda: 1_700_000_000.0)

    calls = []
    orig_to_thread = asyncio.to_thread

    async def spy_to_thread(func, /, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", spy_to_thread)

    sync_out = _transform_media_paths(f"MEDIA:{src}", api_key="k1")
    async_out = await _transform_media_paths_async(f"MEDIA:{src}", api_key="k1")

    assert "/api/media/sample.png" in sync_out
    assert async_out == sync_out
    assert len(calls) == 1
    assert calls[0][0] is _transform_media_paths


@pytest.mark.asyncio
async def test_track_background_task_logs_exception_and_ignores_cancel(tmp_path, monkeypatch, caplog):
    adapter = _make_adapter(monkeypatch, tmp_path)

    async def _boom():
        raise RuntimeError("boom-bg-task")

    async def _sleep_forever():
        await asyncio.sleep(60)

    with caplog.at_level(logging.ERROR, logger="gateway.platforms.api_server"):
        failing_task = adapter._spawn_tracked_task(_boom())
        assert failing_task in adapter._background_tasks
        with pytest.raises(RuntimeError, match="boom-bg-task"):
            await failing_task
        await asyncio.sleep(0)

    assert failing_task not in adapter._background_tasks
    assert any("Background task failed: boom-bg-task" in rec.message for rec in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="gateway.platforms.api_server"):
        cancelled_task = adapter._spawn_tracked_task(_sleep_forever())
        cancelled_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_task
        await asyncio.sleep(0)

    assert cancelled_task not in adapter._background_tasks
    assert not any("Background task failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_idempotency_cache_logs_background_exception(caplog):
    cache = _IdempotencyCache()

    async def _fail():
        raise ValueError("idem-boom")

    with caplog.at_level(logging.ERROR, logger="gateway.platforms.api_server"):
        with pytest.raises(ValueError, match="idem-boom"):
            await cache.get_or_set("k1", "fp1", _fail)
        await asyncio.sleep(0)

    assert any("Idempotency cache background task failed: idem-boom" in rec.message for rec in caplog.records)


def test_no_bare_asyncio_create_task_in_api_server():
    src_path = Path(api_server_mod.__file__)
    tree = ast.parse(src_path.read_text(encoding="utf-8"))

    allowed_parents = {"_spawn_tracked_task", "get_or_set"}
    bare_calls = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_task"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "asyncio"
            ):
                enclosing = self.stack[-1] if self.stack else "<module>"
                if enclosing not in allowed_parents:
                    bare_calls.append((node.lineno, enclosing))
            self.generic_visit(node)

    _Visitor().visit(tree)
    assert bare_calls == [], f"Found bare asyncio.create_task calls: {bare_calls}"


def test_dead_immediately_forwarded_user_hashes_removed(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    assert not hasattr(adapter, "_immediately_forwarded_user_hashes")


@pytest.mark.asyncio
async def test_telegram_sync_locks_lru_cap_and_delete_cleanup(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)

    # Hold lock for s-0 so it must NOT be evicted even when oldest
    held_lock = adapter._get_telegram_sync_lock("s-0")
    await held_lock.acquire()
    try:
        # Touch s-1 so s-2 becomes older than s-1
        _ = adapter._get_telegram_sync_lock("s-1")
        _ = adapter._get_telegram_sync_lock("s-2")
        _ = adapter._get_telegram_sync_lock("s-1")  # refresh s-1

        for i in range(3, MAX_TELEGRAM_SYNC_LOCKS + 20):
            adapter._get_telegram_sync_lock(f"s-{i}")

        assert len(adapter._telegram_sync_locks) == MAX_TELEGRAM_SYNC_LOCKS
        assert "s-0" in adapter._telegram_sync_locks  # locked entry preserved
        assert "s-2" not in adapter._telegram_sync_locks  # oldest unlocked evicted
    finally:
        held_lock.release()

    # Verify DELETE /api/sessions/{session_id} removes unlocked lock
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-del-1", source="webui", model="test-model")
    adapter._session_db = db
    adapter._get_telegram_sync_lock("sess-del-1")
    assert "sess-del-1" in adapter._telegram_sync_locks

    req = make_mocked_request("DELETE", "/api/sessions/sess-del-1", match_info={"session_id": "sess-del-1"})
    with patch.object(adapter, "_check_auth", return_value=None):
        resp = await adapter._handle_delete_session(req)
    assert resp.status == 200
    assert "sess-del-1" not in adapter._telegram_sync_locks


def test_run_statuses_sweep_includes_interrupted_and_enforces_cap(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    now = time.time()
    adapter._RUN_STATUS_TTL = 60.0

    # Expired interrupted + completed should be swept; active running should stay
    adapter._run_statuses["run-interrupted-old"] = {
        "run_id": "run-interrupted-old",
        "status": "interrupted",
        "updated_at": now - 120.0,
    }
    adapter._run_statuses["run-completed-old"] = {
        "run_id": "run-completed-old",
        "status": "completed",
        "updated_at": now - 120.0,
    }
    adapter._run_statuses["run-running-old"] = {
        "run_id": "run-running-old",
        "status": "running",
        "updated_at": now - 120.0,
    }

    api_runs_mod._sweep_orphaned_runs_once(adapter, now=now)
    assert "run-interrupted-old" not in adapter._run_statuses
    assert "run-completed-old" not in adapter._run_statuses
    assert "run-running-old" in adapter._run_statuses

    # Exceed MAX_RUN_STATUSES with fresh terminal entries -> oldest terminal evicted
    tick = [now]
    monkeypatch.setattr(api_runs_mod.time, "time", lambda: tick[0])
    for i in range(MAX_RUN_STATUSES + 50):
        tick[0] = now + (i + 1) * 0.001
        adapter._set_run_status(f"term-{i}", "interrupted" if i % 2 == 0 else "completed")

    assert len(adapter._run_statuses) <= MAX_RUN_STATUSES
    assert "run-running-old" in adapter._run_statuses
    assert "term-0" not in adapter._run_statuses
    assert f"term-{MAX_RUN_STATUSES + 49}" in adapter._run_statuses


def test_unified_active_agent_registry_and_interrupt(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    agent_a = MagicMock()
    agent_b = MagicMock()

    adapter._register_active_agent(agent_a, run_id="run-1")
    adapter._register_active_agent(agent_a, run_id="run-2")
    adapter._register_active_agent(agent_b)

    assert adapter._active_agents_by_id is adapter._shutdown_interruptible_agents
    assert id(agent_a) in adapter._active_agents_by_id
    assert id(agent_b) in adapter._active_agents_by_id
    assert adapter._active_run_agents["run-1"] is agent_a
    assert adapter._active_run_agents["run-2"] is agent_a

    # Interrupts each unique agent exactly once
    assert adapter.interrupt_active_runs("gateway shutdown") == 2
    agent_a.interrupt.assert_called_once_with("gateway shutdown", tool_reason="gateway shutdown")
    agent_b.interrupt.assert_called_once_with("gateway shutdown", tool_reason="gateway shutdown")

    # Unregistering run-1 keeps agent_a in canonical registry because run-2 still holds it
    adapter._unregister_active_agent(agent_a, run_id="run-1")
    assert "run-1" not in adapter._active_run_agents
    assert id(agent_a) in adapter._active_agents_by_id

    # Unregistering run-2 removes agent_a from canonical registry
    adapter._unregister_active_agent(agent_a, run_id="run-2")
    assert "run-2" not in adapter._active_run_agents
    assert id(agent_a) not in adapter._active_agents_by_id

    adapter._unregister_active_agent(agent_b)
    assert adapter._active_agents_by_id == {}
