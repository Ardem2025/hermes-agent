"""Tests for Phase 2A Backend: secure media delivery & session status/subagent badges (BDD-1..3, BDD-5..8)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_UPLOAD_BYTES,
    _assistant_expects_user_reply,
    _compute_media_ref_hmac,
    _sign_media_ref_url,
)
from hermes_state import SessionDB


def _make_adapter(monkeypatch, tmp_path: Path, *, api_key: str = "") -> APIServerAdapter:
    import gateway.platforms.base as gw_base

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(gw_base, "_HERMES_HOME", hermes_home)
    monkeypatch.setattr(gw_base, "_HERMES_ROOT", hermes_home)
    monkeypatch.setattr(
        gw_base,
        "MEDIA_DELIVERY_SAFE_ROOTS",
        (
            hermes_home / "cache" / "images",
            hermes_home / "cache" / "audio",
            hermes_home / "cache" / "videos",
            hermes_home / "cache" / "documents",
            hermes_home / "cache" / "screenshots",
        ),
    )
    if api_key:
        monkeypatch.setenv("API_SERVER_KEY", api_key)
    else:
        monkeypatch.delenv("API_SERVER_KEY", raising=False)
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key} if api_key else {}))


def _extract_query_params(url: str) -> dict[str, str]:
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    return {k: v[0] for k, v in parse_qs(parsed.query).items()}


@pytest.mark.asyncio
async def test_bdd1_media_from_tasks_dir_resolved_and_served(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path, api_key="")
    task_dir = tmp_path / ".hermes" / "tasks" / "t1"
    task_dir.mkdir(parents=True, exist_ok=True)
    img_path = task_dir / "mobile_390x844_v2.png"
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    img_path.write_bytes(png_bytes)

    msg = {
        "id": 1,
        "session_id": "s1",
        "role": "assistant",
        "content": f"Готово:\nMEDIA:{img_path}\nИ с пробелом: MEDIA: {img_path}\nИ markdown: ![shot]({img_path})",
        "timestamp": time.time(),
    }
    projected = adapter._message_response(msg)
    content = projected["content"]
    assert "data:image" not in content
    assert "![image](/api/media/mobile_390x844_v2.png?ref=" in content
    assert "![shot](/api/media/mobile_390x844_v2.png?ref=" in content

    # Extract signed URL and fetch via _handle_get_media
    import re

    m = re.search(r"\((/api/media/[^)]+)\)", content)
    assert m is not None
    signed_url = m.group(1)
    params = _extract_query_params(signed_url)
    req = make_mocked_request(
        "GET",
        signed_url,
        match_info={"filename": "mobile_390x844_v2.png"},
    )
    # make_mocked_request parses query from path
    assert req.query["ref"] == params["ref"]
    resp = await adapter._handle_get_media(req)
    assert resp.status == 200
    assert resp.content_type == "image/png"
    assert resp.body == png_bytes
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"


@pytest.mark.asyncio
async def test_bdd2_chat_and_stream_resolve_media_to_signed_urls(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path, api_key="secret-key-1234567890")
    task_dir = tmp_path / ".hermes" / "tasks" / "t2"
    task_dir.mkdir(parents=True, exist_ok=True)
    shot = task_dir / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n")

    db = SessionDB(db_path=tmp_path / "test_state.db")
    db.create_session(session_id="sess-stream-1", source="webui", model="test-model")
    adapter._session_db = db

    async def fake_run_agent(*, stream_delta_callback=None, interim_assistant_callback=None, **kwargs):
        if interim_assistant_callback:
            interim_assistant_callback(f"Промежуточный скрин: MEDIA:{shot}")
        if stream_delta_callback:
            stream_delta_callback("Готово")
        return {"final_response": f"Итог:\nMEDIA:{shot}", "session_id": "sess-stream-1"}, {"total_tokens": 10}

    adapter._run_agent = fake_run_agent
    adapter._prepare_session_chat = AsyncMock(
        return_value=(
            {
                "gateway_session_key": "gw-1",
                "session_id": "sess-stream-1",
                "body": {"message": "hi"},
                "user_message": "hi",
                "runtime_request": {"requested": {}, "route_source": "global"},
                "lock_active": False,
                "run_kwargs": {},
            },
            None,
        )
    )

    # 1. Non-streaming _handle_session_chat
    chat_handler = getattr(APIServerAdapter._handle_session_chat, "__wrapped__", APIServerAdapter._handle_session_chat)
    chat_resp = await chat_handler(adapter, make_mocked_request("POST", "/api/sessions/sess-stream-1/chat"))
    assert chat_resp.status == 200
    chat_data = json.loads(chat_resp.text)
    assert "![image](/api/media/shot.png?ref=" in chat_data["message"]["content"]
    assert "data:image" not in chat_data["message"]["content"]

    # Verify last_read_at watermark was initialized at turn start
    sess_row = db.get_session("sess-stream-1")
    assert sess_row["last_read_at"] is not None

    # 2. Streaming _handle_session_chat_stream
    written_chunks: list[bytes] = []

    class _FakeSSEResponse:
        async def write(self, data: bytes) -> None:
            written_chunks.append(data)

    adapter._prepare_sse_response = AsyncMock(return_value=_FakeSSEResponse())
    stream_handler = getattr(
        APIServerAdapter._handle_session_chat_stream,
        "__wrapped__",
        APIServerAdapter._handle_session_chat_stream,
    )
    await stream_handler(adapter, make_mocked_request("POST", "/api/sessions/sess-stream-1/chat/stream"))
    sse_text = b"".join(written_chunks).decode("utf-8")
    assert "event: assistant.commentary" in sse_text
    assert "event: assistant.completed" in sse_text
    assert "/api/media/shot.png?ref=" in sse_text
    assert "data:image" not in sse_text
    db.close()


@pytest.mark.asyncio
async def test_bdd3_non_images_svg_csp_and_missing_fallback(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path, api_key="secret-key-1234567890")
    media_dir = tmp_path / ".hermes" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    pdf_file = media_dir / "report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 test")
    svg_file = media_dir / "diagram.svg"
    svg_file.write_text("<svg><script>alert(1)</script></svg>", encoding="utf-8")

    text = f"Отчёт: MEDIA:{pdf_file}\nСхема: MEDIA:{svg_file}\nБитый: MEDIA:/tmp/nonexistent_12345.png"
    resolved = adapter._resolve_message_media(text)
    assert "[📎 report.pdf](/api/media/report.pdf?" in resolved
    assert "[📎 diagram.svg](/api/media/diagram.svg?" in resolved
    assert "![image](/api/media/diagram.svg" not in resolved
    assert "⚠️ Файл недоступен: nonexistent_12345.png" in resolved
    assert "![image](/api/media/nonexistent_12345.png" not in resolved

    # Fetch SVG via _handle_get_media with Bearer auth -> must include CSP sandbox + attachment
    req_svg = make_mocked_request(
        "GET",
        "/api/media/diagram.svg",
        headers={"Authorization": "Bearer secret-key-1234567890"},
        match_info={"filename": "diagram.svg"},
    )
    resp_svg = await adapter._handle_get_media(req_svg)
    assert resp_svg.status == 200
    assert resp_svg.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp_svg.headers.get("Content-Security-Policy") == "default-src 'none'; sandbox"
    assert resp_svg.headers.get("Content-Disposition") == 'attachment; filename="diagram.svg"'

    # Fetch PDF -> also sandboxed attachment
    req_pdf = make_mocked_request(
        "GET",
        "/api/media/report.pdf",
        headers={"Authorization": "Bearer secret-key-1234567890"},
        match_info={"filename": "report.pdf"},
    )
    resp_pdf = await adapter._handle_get_media(req_pdf)
    assert resp_pdf.status == 200
    assert resp_pdf.headers.get("Content-Security-Policy") == "default-src 'none'; sandbox"
    assert resp_pdf.headers.get("Content-Disposition") == 'attachment; filename="report.pdf"'


@pytest.mark.asyncio
async def test_bdd5_security_denylist_hmac_and_xss(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path, api_key="")
    env_file = tmp_path / ".hermes" / ".env"
    env_file.write_text("SECRET=123\n", encoding="utf-8")
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    id_rsa = ssh_dir / "id_rsa"
    id_rsa.write_text("PRIVATE KEY", encoding="utf-8")

    dangerous = (
        f"MEDIA:/etc/passwd\n"
        f"MEDIA:{env_file}\n"
        f"MEDIA:{id_rsa}\n"
        f"![x](javascript:alert(1))\n"
        f"![y](data:text/html,<script>alert(1)</script>)"
    )
    resolved = adapter._resolve_message_media(dangerous)
    assert "/api/media/" not in resolved
    assert "⚠️ Файл недоступен: passwd" in resolved
    assert "⚠️ Файл недоступен: .env" in resolved
    assert "⚠️ Файл недоступен: id_rsa" in resolved
    assert "javascript:" not in resolved
    assert "data:text/html" not in resolved

    # Forged / missing / expired ref signature must be rejected with 403 even when API_SERVER_KEY is empty
    task_dir = tmp_path / ".hermes" / "tasks" / "t5"
    task_dir.mkdir(parents=True, exist_ok=True)
    ok_img = task_dir / "ok.png"
    ok_img.write_bytes(b"\x89PNG\r\n\x1a\n")
    ref_b64 = base64.urlsafe_b64encode(str(ok_img).encode("utf-8")).decode("ascii").rstrip("=")

    # Missing sig
    req_no_sig = make_mocked_request(
        "GET",
        f"/api/media/ok.png?ref={ref_b64}&exp={int(time.time()) + 60}",
        match_info={"filename": "ok.png"},
    )
    assert (await adapter._handle_get_media(req_no_sig)).status == 403

    # Forged sig
    req_bad_sig = make_mocked_request(
        "GET",
        f"/api/media/ok.png?ref={ref_b64}&exp={int(time.time()) + 60}&sig=deadbeef",
        match_info={"filename": "ok.png"},
    )
    assert (await adapter._handle_get_media(req_bad_sig)).status == 403

    # Expired sig
    exp_past = int(time.time()) - 10
    sig_past = _compute_media_ref_hmac("ok.png", ref_b64, exp_past, adapter._media_ref_secret)
    req_exp = make_mocked_request(
        "GET",
        f"/api/media/ok.png?ref={ref_b64}&exp={exp_past}&sig={sig_past}",
        match_info={"filename": "ok.png"},
    )
    assert (await adapter._handle_get_media(req_exp)).status == 403

    # Valid HMAC for /etc/passwd must still be blocked by validate_media_delivery_path -> 403
    passwd_url = _sign_media_ref_url("/etc/passwd", secret=adapter._media_ref_secret)
    req_passwd = make_mocked_request("GET", passwd_url, match_info={"filename": "passwd"})
    assert (await adapter._handle_get_media(req_passwd)).status == 403

    # Oversized file (> 20 MB) rejected with 413
    big_file = task_dir / "huge.png"
    big_file.write_bytes(b"\x89PNG")
    monkeypatch.setattr("gateway.platforms.api_server.MAX_UPLOAD_BYTES", 2)
    big_url = _sign_media_ref_url(str(big_file), secret=adapter._media_ref_secret)
    req_big = make_mocked_request("GET", big_url, match_info={"filename": "huge.png"})
    assert (await adapter._handle_get_media(req_big)).status == 413


def test_bdd6_question_detector_and_waiting_input_states(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)

    # Question detector unit checks
    assert _assistant_expects_user_reply("Подтверждаешь деплой в прод?") is True
    assert _assistant_expects_user_reply("Выберите вариант:\n1. Вариант А\n2. Вариант Б") is True
    assert _assistant_expects_user_reply("Готово. Ссылка: https://example.com/view?session_id=123&tab=1") is False
    assert _assistant_expects_user_reply("```python\nif x == '?':\n    pass\n```\nЗадача выполнена.") is False
    assert _assistant_expects_user_reply("Смотри [скрин](/api/media/shot.png?ref=abc&exp=123&sig=xyz).") is False

    # waiting_for_approval -> waiting_input (approval), NOT running
    adapter._run_statuses["run-app-1"] = {
        "session_id": "sess-app",
        "status": "waiting_for_approval",
        "tool_status": "Ожидает подтверждения команды",
    }
    res_app = adapter._session_response({"id": "sess-app", "started_at": 100.0})
    assert res_app["status"] == "waiting_input"
    assert res_app["waiting_reason"] == "approval"
    assert res_app["is_generating"] is False

    # Assistant question -> waiting_input (question)
    res_q = adapter._session_response(
        {"id": "sess-q", "started_at": 100.0, "last_active": 200.0, "last_read_at": 150.0},
        last_msg={"role": "assistant", "content": "Какой вариант выбираем?", "has_tool_calls": False, "finish_reason": "stop"},
    )
    assert res_q["status"] == "waiting_input"
    assert res_q["waiting_reason"] == "question"
    assert res_q["last_role"] == "assistant"
    assert res_q["is_unread_completion"] is False


@pytest.mark.asyncio
async def test_bdd7_completed_unread_and_post_read_endpoint(tmp_path, monkeypatch):
    adapter = _make_adapter(monkeypatch, tmp_path)
    db = SessionDB(db_path=tmp_path / "test_state.db")
    adapter._session_db = db

    # 1. Historical session with last_read_at=NULL -> idle (not completed)
    db.create_session(session_id="sess-hist", source="webui", model="m")
    db.append_message("sess-hist", role="assistant", content="Старый отчёт.", finish_reason="stop")

    # 2. Active turn session: watermark set at turn start, then assistant finishes
    db.create_session(session_id="sess-done", source="webui", model="m")
    db.set_session_read("sess-done", read=True)
    time.sleep(0.02)
    db.append_message("sess-done", role="assistant", content="Итоговый отчёт готов.", finish_reason="stop")

    # 3. Tool-calls intermediate message -> NOT completed
    db.create_session(session_id="sess-tools", source="webui", model="m")
    db.set_session_read("sess-tools", read=True)
    time.sleep(0.02)
    db.append_message(
        "sess-tools",
        role="assistant",
        content="Вызываю инструмент",
        tool_calls=[{"id": "tc1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )

    # Batch query check
    last_msgs = db.session_last_messages(["sess-hist", "sess-done", "sess-tools"])
    assert set(last_msgs.keys()) == {"sess-hist", "sess-done", "sess-tools"}

    list_handler = getattr(APIServerAdapter._handle_list_sessions, "__wrapped__", APIServerAdapter._handle_list_sessions)
    list_resp = await list_handler(adapter, make_mocked_request("GET", "/api/sessions"))
    assert list_resp.status == 200
    by_id = {s["id"]: s for s in json.loads(list_resp.text)["data"]}

    assert by_id["sess-hist"]["status"] == "idle"
    assert by_id["sess-done"]["status"] == "completed"
    assert by_id["sess-done"]["is_unread_completion"] is True
    assert by_id["sess-tools"]["status"] == "idle"

    # POST /api/sessions/sess-done/read clears completed -> idle
    read_handler = getattr(APIServerAdapter._handle_read_session, "__wrapped__", APIServerAdapter._handle_read_session)
    read_resp = await read_handler(
        adapter,
        make_mocked_request("POST", "/api/sessions/sess-done/read", match_info={"session_id": "sess-done"}),
    )
    assert read_resp.status == 200
    read_payload = json.loads(read_resp.text)["session"]
    assert read_payload["status"] == "idle"
    assert read_payload["is_unread_completion"] is False

    # Persists on subsequent GET /api/sessions/{id}
    get_handler = getattr(APIServerAdapter._handle_get_session, "__wrapped__", APIServerAdapter._handle_get_session)
    get_resp = await get_handler(
        adapter,
        make_mocked_request("GET", "/api/sessions/sess-done", match_info={"session_id": "sess-done"}),
    )
    assert json.loads(get_resp.text)["session"]["status"] == "idle"
    db.close()


def test_bdd8_exact_subagent_matching_and_running_count(tmp_path, monkeypatch):
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    adapter = _make_adapter(monkeypatch, tmp_path)
    now = time.time()
    with _active_subagents_lock:
        _active_subagents.clear()
        # 2 running + 3 completed for sess-1
        for i in range(2):
            _active_subagents[f"sub-a-run-{i}"] = {
                "subagent_id": f"sub-a-run-{i}",
                "owner_agent_session_id": "sess-1",
                "status": "running",
                "started_at": now,
            }
        for i in range(3):
            _active_subagents[f"sub-a-done-{i}"] = {
                "subagent_id": f"sub-a-done-{i}",
                "owner_agent_session_id": "sess-1",
                "status": "completed",
                "started_at": now,
            }
        # 5 running for superstring sess-10 and substring sess
        for i in range(3):
            _active_subagents[f"sub-b-{i}"] = {
                "subagent_id": f"sub-b-{i}",
                "owner_agent_session_id": "sess-10",
                "status": "running",
                "started_at": now,
            }
        for i in range(2):
            _active_subagents[f"sub-c-{i}"] = {
                "subagent_id": f"sub-c-{i}",
                "owner_agent_session_id": "sess",
                "status": "running",
                "started_at": now,
            }

    try:
        res = adapter._session_response(
            {"id": "sess-1", "started_at": now - 10, "last_active": now, "last_read_at": now - 5},
            last_msg={"role": "assistant", "content": "Готово.", "has_tool_calls": False, "finish_reason": "stop"},
        )
        assert res["active_subagents_count"] == 2
        assert res["has_active_subagents"] is True
        assert res["status"] == "running"
        matched_ids = {s["subagent_id"] for s in res["active_subagents"]}
        assert not any(mid.startswith("sub-b-") or mid.startswith("sub-c-") for mid in matched_ids)
    finally:
        with _active_subagents_lock:
            _active_subagents.clear()
