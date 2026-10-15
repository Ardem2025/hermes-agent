import pytest
from unittest.mock import MagicMock
from aiohttp import web
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

@pytest.mark.asyncio
async def test_telegram_binding_route_registered(tmp_path):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"api_key": "12345678901234567890"}))
    routes = adapter._http_route_table()
    paths = [(m, p) for m, p, _ in routes]
    assert ("GET", "/api/sessions/{session_id}/telegram-binding") in paths
    assert ("POST", "/api/sessions/{session_id}/telegram-binding") in paths
    assert ("DELETE", "/api/sessions/{session_id}/telegram-binding") in paths

@pytest.mark.asyncio
async def test_get_telegram_binding_not_found(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"api_key": "12345678901234567890"}))
    adapter._session_db = db

    req = MagicMock(spec=web.Request)
    req.match_info = {"session_id": "non_existent"}
    req.headers = {"Authorization": "Bearer 12345678901234567890"}

    # Should return 404 for non-existent session
    resp = await adapter._handle_get_telegram_binding(req)
    assert resp.status == 404
