"""Tests for S2.1: media auth/path-confinement, upload limits/sanitization, and fail-closed telegram-binding."""

import io
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms.api_server import (
    MAX_UPLOAD_BYTES,
    APIServerAdapter,
    _compute_media_hmac,
    _sign_media_url,
    _transform_media_paths,
)


def _make_adapter(api_key: str = "test-secret-key-1234567890") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[api_server_mod.body_limit_middleware])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


class TestMediaAuthAndConfinement:
    @pytest.mark.asyncio
    async def test_get_media_auth_matrix(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        media_dir = hermes_home / "media"
        media_dir.mkdir(parents=True)
        (media_dir / "pic.png").write_bytes(b"PNGDATA")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        adapter = _make_adapter("super-secret-api-key-1234")
        app = _make_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            # 1. Unauthenticated -> 401
            r = await cli.get("/api/media/pic.png")
            assert r.status == 401

            # 2. Valid Bearer -> 200
            r = await cli.get("/api/media/pic.png", headers={"Authorization": "Bearer super-secret-api-key-1234"})
            assert r.status == 200
            assert await r.read() == b"PNGDATA"

            # 3. Valid HMAC query token -> 200
            signed = _sign_media_url("pic.png", api_key="super-secret-api-key-1234")
            r = await cli.get(signed)
            assert r.status == 200
            assert await r.read() == b"PNGDATA"

            # 4. Expired HMAC -> 401
            expired_exp = int(time.time()) - 10
            expired_sig = _compute_media_hmac("pic.png", expired_exp, "super-secret-api-key-1234")
            r = await cli.get(f"/api/media/pic.png?exp={expired_exp}&sig={expired_sig}")
            assert r.status == 401

            # 5. Forged HMAC -> 401
            future_exp = int(time.time()) + 3600
            r = await cli.get(f"/api/media/pic.png?exp={future_exp}&sig={'0' * 64}")
            assert r.status == 401

            # 6. HMAC minted for another filename -> 401
            other_sig = _compute_media_hmac("other.png", future_exp, "super-secret-api-key-1234")
            r = await cli.get(f"/api/media/pic.png?exp={future_exp}&sig={other_sig}")
            assert r.status == 401

    @pytest.mark.asyncio
    async def test_get_media_allowlist_and_traversal(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        for sub in ("media", "image_cache", "cache/images", "audio_cache", "sandbox"):
            (hermes_home / sub).mkdir(parents=True)
        (hermes_home / "image_cache" / "cached.png").write_bytes(b"CACHED")
        (hermes_home / "media" / ".env").write_text("SECRET=1")
        (hermes_home / "media" / "exact.jpg").write_bytes(b"JPG")
        (hermes_home / "sandbox" / "leak.txt").write_text("SANDBOX")
        outside = tmp_path / "outside.txt"
        outside.write_text("OUTSIDE")
        (hermes_home / "media" / "escape.txt").symlink_to(outside)

        tmp_probe = Path("/tmp") / f"hermes_test_probe_{time.time_ns()}.txt"
        tmp_probe.write_text("TMP_SECRET")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        try:
            adapter = _make_adapter("super-secret-api-key-1234")
            app = _make_app(adapter)
            headers = {"Authorization": "Bearer super-secret-api-key-1234"}

            async with TestClient(TestServer(app)) as cli:
                # Allowlisted dir works
                r = await cli.get("/api/media/cached.png", headers=headers)
                assert r.status == 200
                assert await r.read() == b"CACHED"

                # Wildcard stem fallback removed (exact.png must NOT match exact.jpg)
                r = await cli.get("/api/media/exact.png", headers=headers)
                assert r.status == 404

                # Sandbox dir removed
                r = await cli.get("/api/media/leak.txt", headers=headers)
                assert r.status == 404

                # /tmp removed
                r = await cli.get(f"/api/media/{tmp_probe.name}", headers=headers)
                assert r.status == 404

                # Hidden dotfile rejected
                r = await cli.get("/api/media/.env", headers=headers)
                assert r.status in (400, 403, 404)

                # Encoded traversal rejected
                r = await cli.get("/api/media/..%2Foutside.txt", headers=headers)
                assert r.status in (400, 403, 404)

                # Symlink escaping allowlist rejected with 403
                r = await cli.get("/api/media/escape.txt", headers=headers)
                assert r.status == 403
        finally:
            tmp_probe.unlink(missing_ok=True)

    def test_transform_media_paths_allowlist_and_signing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        img_cache = hermes_home / "image_cache"
        img_cache.mkdir(parents=True)
        allowed_img = img_cache / "gen.png"
        allowed_img.write_bytes(b"IMG")

        outside_img = tmp_path / "secret.png"
        outside_img.write_bytes(b"SECRET")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        text = f"MEDIA:{allowed_img} and ![x]({outside_img})"
        out = _transform_media_paths(text, api_key="super-secret-api-key-1234")

        assert "/api/media/gen.png?exp=" in out
        assert "&sig=" in out
        assert (hermes_home / "media" / "gen.png").is_file()
        # Outside file must NOT be copied or rewritten
        assert f"![x]({outside_img})" in out
        assert not (hermes_home / "media" / "secret.png").exists()


class TestFileUploadSecurity:
    def test_max_upload_bytes_constant(self):
        assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_upload_requires_bearer_and_handles_collisions_and_sanitization(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        adapter = _make_adapter("super-secret-api-key-1234")
        app = _make_app(adapter)
        headers = {"Authorization": "Bearer super-secret-api-key-1234"}

        async with TestClient(TestServer(app)) as cli:
            # Unauthenticated -> 401
            form = aiohttp.FormData()
            form.add_field("file", io.BytesIO(b"hello"), filename="note.txt")
            r = await cli.post("/api/files/upload", data=form)
            assert r.status == 401

            # HMAC query params must NOT bypass @_require_auth on upload
            signed = _sign_media_url("note.txt", api_key="super-secret-api-key-1234")
            query = signed.split("?", 1)[1]
            form_hmac = aiohttp.FormData()
            form_hmac.add_field("file", io.BytesIO(b"hello"), filename="note.txt")
            r = await cli.post(f"/api/files/upload?{query}", data=form_hmac)
            assert r.status == 401

            # First upload -> signed URL
            form1 = aiohttp.FormData()
            form1.add_field("file", io.BytesIO(b"first"), filename="report.txt")
            r = await cli.post("/api/files/upload", data=form1, headers=headers)
            assert r.status == 200
            body1 = await r.json()
            assert body1["files"][0]["filename"] == "report.txt"
            assert body1["files"][0]["url"].startswith("/api/media/report.txt?exp=")
            assert "&sig=" in body1["files"][0]["url"]

            # Second upload with same name -> collision suffix, original preserved
            form2 = aiohttp.FormData()
            form2.add_field("file", io.BytesIO(b"second"), filename="report.txt")
            r = await cli.post("/api/files/upload", data=form2, headers=headers)
            assert r.status == 200
            body2 = await r.json()
            new_name = body2["files"][0]["filename"]
            assert new_name != "report.txt"
            assert new_name.startswith("report_") and new_name.endswith(".txt")
            assert (hermes_home / "media" / "report.txt").read_bytes() == b"first"
            assert (hermes_home / "media" / new_name).read_bytes() == b"second"

            # Hidden / traversal filename sanitized
            form3 = aiohttp.FormData()
            form3.add_field("file", io.BytesIO(b"env"), filename="../../.env")
            r = await cli.post("/api/files/upload", data=form3, headers=headers)
            assert r.status == 200
            body3 = await r.json()
            sanitized = body3["files"][0]["filename"]
            assert not sanitized.startswith(".")
            assert "/" not in sanitized and ".." not in sanitized

    @pytest.mark.asyncio
    async def test_upload_rejects_over_limit_with_413_and_no_partial_file(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes_home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr(api_server_mod, "MAX_UPLOAD_BYTES", 1024)

        adapter = _make_adapter("super-secret-api-key-1234")
        app = _make_app(adapter)
        headers = {"Authorization": "Bearer super-secret-api-key-1234"}

        async with TestClient(TestServer(app)) as cli:
            form = aiohttp.FormData()
            form.add_field("file", io.BytesIO(b"A" * 2048), filename="big.bin")
            r = await cli.post("/api/files/upload", data=form, headers=headers)
            assert r.status == 413
            media_dir = hermes_home / "media"
            assert not (media_dir / "big.bin").exists()


class TestTelegramBindingFailClosed:
    def test_no_hardcoded_chat_id_in_api_server(self):
        src = Path(api_server_mod.__file__).read_text(encoding="utf-8")
        assert "139351986" not in src

    @pytest.mark.asyncio
    async def test_get_telegram_binding_without_chat_id_returns_null_and_skips_recovery(self):
        adapter = _make_adapter("super-secret-api-key-1234")
        db = MagicMock()
        db.get_telegram_topic_binding_by_session.return_value = None
        db.get_telegram_topic_binding = MagicMock()
        db.bind_telegram_topic = MagicMock()

        session_row = {"id": "sess-1", "thread_id": "42", "user_id": "", "chat_id": ""}
        with (
            patch.object(adapter, "_get_existing_session_or_404", AsyncMock(return_value=(session_row, None))),
            patch.object(adapter, "_ensure_session_db_async", AsyncMock(return_value=db)),
        ):
            app = _make_app(adapter)
            headers = {"Authorization": "Bearer super-secret-api-key-1234"}
            async with TestClient(TestServer(app)) as cli:
                r = await cli.get("/api/sessions/sess-1/telegram-binding", headers=headers)
                assert r.status == 200
                data = await r.json()
                assert data["bound"] is False
                assert data["chat_id"] is None
                db.bind_telegram_topic.assert_not_called()
                db.get_telegram_topic_binding.assert_not_called()

                # Ended session without chat_id also skips lookup and returns chat_id: null
                session_row["ended_at"] = 1234567890
                r2 = await cli.get("/api/sessions/sess-1/telegram-binding", headers=headers)
                assert r2.status == 200
                data2 = await r2.json()
                assert data2["ended"] is True
                assert data2["chat_id"] is None
                db.get_telegram_topic_binding.assert_not_called()

    @pytest.mark.asyncio
    async def test_post_telegram_binding_without_chat_id_returns_400(self):
        adapter = _make_adapter("super-secret-api-key-1234")
        db = MagicMock()
        db.get_telegram_topic_binding_by_session.return_value = None

        session_row = {"id": "sess-2", "user_id": "", "chat_id": "", "title": "Test"}
        with (
            patch.object(adapter, "_get_existing_session_or_404", AsyncMock(return_value=(session_row, None))),
            patch.object(adapter, "_ensure_session_db_async", AsyncMock(return_value=db)),
        ):
            app = _make_app(adapter)
            headers = {"Authorization": "Bearer super-secret-api-key-1234"}
            async with TestClient(TestServer(app)) as cli:
                r = await cli.post("/api/sessions/sess-2/telegram-binding", json={}, headers=headers)
                assert r.status == 400
                data = await r.json()
                assert data["error"]["code"] == "missing_chat_id"
