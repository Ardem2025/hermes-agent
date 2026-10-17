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

@pytest.mark.asyncio
async def test_post_and_toggle_telegram_binding(tmp_path):
    import json
    from unittest.mock import AsyncMock
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_test_123", source="api")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"api_key": "12345678901234567890"}))
    adapter._session_db = db

    # 1. POST binding with delivery_enabled=True
    req = MagicMock(spec=web.Request)
    req.match_info = {"session_id": "sess_test_123"}
    req.headers = {"Authorization": "Bearer 12345678901234567890"}
    req.can_read_body = True
    payload = {"chat_id": "139351986", "thread_id": "999", "delivery_enabled": True}
    req.json = AsyncMock(return_value=payload)

    resp = await adapter._handle_post_telegram_binding(req)
    assert resp.status == 201
    body = json.loads(resp.text)
    assert body["bound"] is True
    assert body["delivery_enabled"] is True

    binding = db.get_telegram_topic_binding_by_session(session_id="sess_test_123")
    assert binding is not None
    assert binding["delivery_enabled"] == 1

    # 2. Toggle delivery_enabled=False on existing binding
    req_toggle = MagicMock(spec=web.Request)
    req_toggle.match_info = {"session_id": "sess_test_123"}
    req_toggle.headers = {"Authorization": "Bearer 12345678901234567890"}
    req_toggle.can_read_body = True
    req_toggle.json = AsyncMock(return_value={"delivery_enabled": False})

    resp_toggle = await adapter._handle_post_telegram_binding(req_toggle)
    assert resp_toggle.status == 200
    body_toggle = json.loads(resp_toggle.text)
    assert body_toggle["bound"] is True
    assert body_toggle["delivery_enabled"] is False

    binding_after = db.get_telegram_topic_binding_by_session(session_id="sess_test_123")
    assert binding_after["delivery_enabled"] == 0

    # 3. DELETE binding
    req_del = MagicMock(spec=web.Request)
    req_del.match_info = {"session_id": "sess_test_123"}
    req_del.headers = {"Authorization": "Bearer 12345678901234567890"}
    resp_del = await adapter._handle_delete_telegram_binding(req_del)
    assert resp_del.status == 200
    body_del = json.loads(resp_del.text)
    assert body_del["deleted"] is True

    binding_deleted = db.get_telegram_topic_binding_by_session(session_id="sess_test_123")
    assert binding_deleted is None

