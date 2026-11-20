"""Tests for Wave 2 Phase 2.1 Backend Architecture, Delta-Sync, BoundedExecutor & CSRF Guard."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms import api_server_runs as api_runs_mod
from gateway.platforms.api_server import APIServerAdapter, cors_middleware
from gateway.platforms.api_server_chat import _SessionChatMixin
from gateway.platforms.api_server_sessions import _SessionsMixin
from gateway.platforms.api_server_telegram_sync import _TelegramSyncMixin
from gateway.turn_executor import BoundedThreadExecutor
from hermes_state import SessionDB


def _make_adapter(monkeypatch, tmp_path: Path, *, api_key: str = "", cors_origins=None) -> APIServerAdapter:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    if api_key:
        monkeypatch.setenv("API_SERVER_KEY", api_key)
    else:
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
    extra = {}
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _build_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware,) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


def test_api_server_decomposition_line_budget_and_mro():
    api_path = Path(api_server_mod.__file__)
    lines = api_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 3960, f"api_server.py has {len(lines)} lines (> 3960)"
    assert issubclass(APIServerAdapter, _SessionsMixin)
    assert issubclass(APIServerAdapter, _TelegramSyncMixin)
    assert issubclass(APIServerAdapter, _SessionChatMixin)


@pytest.mark.asyncio
async def test_list_sessions_etag_and_304(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-etag-1", source="webui", model="test-model")
    db.set_session_title("sess-etag-1", "First")
    adapter._session_db = db

    app = _build_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        r1 = await cli.get("/api/sessions")
        assert r1.status == 200
        etag1 = r1.headers.get("ETag")
        assert etag1
        assert r1.headers.get("Cache-Control") == "no-cache"
        body1 = await r1.json()
        assert len(body1["data"]) == 1

        r2 = await cli.get("/api/sessions", headers={"If-None-Match": etag1})
        assert r2.status == 304
        assert r2.headers.get("ETag") == etag1

        # Mutate session -> ETag must change and return 200
        db.append_message("sess-etag-1", "user", "hello")
        r3 = await cli.get("/api/sessions", headers={"If-None-Match": etag1})
        assert r3.status == 200
        etag2 = r3.headers.get("ETag")
        assert etag2 and etag2 != etag1


@pytest.mark.asyncio
async def test_session_messages_after_id_etag_and_validation(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-delta-1", source="webui", model="test-model")
    id1 = db.append_message("sess-delta-1", "user", "msg 1")
    id2 = db.append_message("sess-delta-1", "assistant", "msg 2")
    adapter._session_db = db

    app = _build_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        # Full fetch returns ETag and 304 on repeat
        r_full = await cli.get("/api/sessions/sess-delta-1/messages")
        assert r_full.status == 200
        etag_full = r_full.headers.get("ETag")
        assert etag_full
        r_full_304 = await cli.get("/api/sessions/sess-delta-1/messages", headers={"If-None-Match": etag_full})
        assert r_full_304.status == 304

        # Delta fetch after id1 returns only id2
        r_delta = await cli.get(f"/api/sessions/sess-delta-1/messages?after_id={id1}")
        assert r_delta.status == 200
        delta_body = await r_delta.json()
        assert [m["id"] for m in delta_body["data"]] == [id2]
        assert delta_body["pagination"]["last_id"] == id2
        etag_delta = r_delta.headers.get("ETag")
        assert etag_delta and etag_delta != etag_full

        # Repeat delta with If-None-Match -> 304
        r_delta_304 = await cli.get(
            f"/api/sessions/sess-delta-1/messages?after_id={id1}",
            headers={"If-None-Match": etag_delta},
        )
        assert r_delta_304.status == 304

        # New message arrives -> delta after id2 returns id3
        id3 = db.append_message("sess-delta-1", "user", "msg 3")
        r_delta2 = await cli.get(
            f"/api/sessions/sess-delta-1/messages?after_id={id2}",
            headers={"If-None-Match": etag_delta},
        )
        assert r_delta2.status == 200
        delta2_body = await r_delta2.json()
        assert [m["id"] for m in delta2_body["data"]] == [id3]
        assert delta2_body["pagination"]["last_id"] == id3

        # Invalid after_id and incompatible combinations -> 400 invalid_pagination
        for bad_qs in (
            "after_id=-1",
            "after_id=abc",
            f"after_id={id1}&offset=1",
            f"after_id={id1}&include_compacted=true",
            f"after_id={id1}&order=latest",
        ):
            r_bad = await cli.get(f"/api/sessions/sess-delta-1/messages?{bad_qs}")
            assert r_bad.status == 400, bad_qs
            err_body = await r_bad.json()
            assert err_body["error"]["code"] == "invalid_pagination"


@pytest.mark.asyncio
async def test_session_messages_after_id_compaction_reset(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="sess-parent", source="webui", model="test-model")
    id1 = db.append_message("sess-parent", "user", "old msg")
    db.end_session("sess-parent", "compression")
    db.create_session(session_id="sess-child", source="webui", model="test-model", parent_session_id="sess-parent")
    db.append_message("sess-child", "assistant", "new msg")
    adapter._session_db = db

    app = _build_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/sess-parent/messages?after_id={id1}")
        assert resp.status == 200
        assert resp.headers.get("X-Hermes-Sync-Reset") == "1"
        body = await resp.json()
        assert body["session_id"] == "sess-child"
        assert body["sync_reset"] is True
        assert body["full_reload_required"] is True
        assert body["data"] == []


def test_bounded_thread_executor_20_concurrent_calls():
    max_workers = 4
    executor = BoundedThreadExecutor(max_workers=max_workers, thread_name_prefix="test-bounded")
    lock = threading.Lock()
    active = 0
    peak = 0

    def _work(idx: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return idx * 2

    try:
        futures = [executor.submit(_work, i) for i in range(20)]
        results = [f.result(timeout=10) for f in futures]
        assert results == [i * 2 for i in range(20)]
        assert 1 <= peak <= max_workers
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_unified_session_coordinator_compatibility(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    lock = adapter._get_telegram_sync_lock("sess-coord-1")
    assert adapter._telegram_sync_locks["sess-coord-1"] is lock

    agent = MagicMock()
    adapter._register_active_agent(agent, run_id="run-coord-1", session_id="sess-coord-1")
    assert adapter._active_agents_by_id is adapter._shutdown_interruptible_agents
    assert id(agent) in adapter._active_agents_by_id
    assert adapter._active_run_agents["run-coord-1"] is agent
    assert adapter._session_coordinator.has_active_turn("sess-coord-1") is True

    adapter._unregister_active_agent(agent, run_id="run-coord-1", session_id="sess-coord-1")
    assert id(agent) not in adapter._active_agents_by_id
    assert "run-coord-1" not in adapter._active_run_agents
    assert adapter._session_coordinator.has_active_turn("sess-coord-1") is False


@pytest.mark.asyncio
async def test_csrf_origin_guard_blocks_cross_site_without_breaking_bearer_or_cli(tmp_path, monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        tmp_path,
        api_key="secret-token-1234567890",
        cors_origins=["http://localhost:3000"],
    )
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter._session_db = db
    app = _build_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        # 1. Cross-site Origin without Bearer -> 403
        r1 = await cli.post(
            "/api/sessions",
            json={"title": "evil"},
            headers={"Origin": "http://evil.example"},
        )
        assert r1.status == 403

        # 2. Cross-site Referer without Origin and without Bearer -> 403
        r2 = await cli.post(
            "/api/sessions",
            json={"title": "evil-ref"},
            headers={"Referer": "http://evil.example/attack.html"},
        )
        assert r2.status == 403

        # 3. Sec-Fetch-Site: cross-site without Bearer -> 403
        r3 = await cli.post(
            "/api/sessions",
            json={"title": "evil-sfs"},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        assert r3.status == 403

        # 4. Cookie without Origin/Referer/Bearer -> 403
        r4 = await cli.post(
            "/api/sessions",
            json={"title": "evil-cookie"},
            headers={"Cookie": "hermes_sid=abc"},
        )
        assert r4.status == 403

        # 5. Bearer client with Referer (no Origin) -> allowed (201)
        r5 = await cli.post(
            "/api/sessions",
            json={"title": "bearer-ok"},
            headers={
                "Authorization": "Bearer secret-token-1234567890",
                "Referer": "http://evil.example/tool",
            },
        )
        assert r5.status == 201

        # 6. CLI / TestClient without browser headers + Bearer -> allowed (201)
        r6 = await cli.post(
            "/api/sessions",
            json={"title": "cli-ok"},
            headers={"Authorization": "Bearer secret-token-1234567890"},
        )
        assert r6.status == 201

        # 7. Allowed browser Origin + Bearer -> allowed (201)
        r7 = await cli.post(
            "/api/sessions",
            json={"title": "origin-ok"},
            headers={
                "Authorization": "Bearer secret-token-1234567890",
                "Origin": "http://localhost:3000",
            },
        )
        assert r7.status == 201
