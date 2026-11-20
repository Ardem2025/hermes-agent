"""Tests for Telegram standalone flood cooldown sharing & fail-closed behavior (BDD-3, BDD-4, BDD-5, BDD-8)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.delivery_ledger import flood_wait_seconds, is_flood_error
from gateway.platforms.base import SendResult
from plugins.platforms.telegram import standalone as tg_standalone
from plugins.platforms.telegram import telegram_outbound
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_flood_cooldown import (
    clear_send_flood_cooldowns,
    send_flood_cooldown_remaining,
    token_fingerprint,
)


class _FakeRetryAfter(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after
        super().__init__(f"Flood control exceeded. Retry in {retry_after} seconds")


@pytest.fixture(autouse=True)
def _isolate_cooldowns_and_home(tmp_path, monkeypatch):
    clear_send_flood_cooldowns()
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    yield
    clear_send_flood_cooldowns()


def _make_fake_bot(token: str):
    bot = MagicMock()
    bot.token = token
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=101))
    bot.send_photo = AsyncMock(return_value=SimpleNamespace(message_id=102))
    bot.send_media_group = AsyncMock(return_value=[SimpleNamespace(message_id=103)])
    bot.do_api_request = AsyncMock(return_value={"message_id": 104})
    return bot


@pytest.mark.asyncio
async def test_bdd3_active_flood_window_blocks_standalone_without_api_calls(tmp_path, monkeypatch):
    token = "123456:ABC-bdd3-token"
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    adapter._record_send_flood_cooldown("139351986", 3600)

    bot_factory_calls = []
    monkeypatch.setattr(
        tg_standalone,
        "_telegram_bot",
        lambda tok, extra=None: bot_factory_calls.append((tok, extra)) or _make_fake_bot(tok),
    )

    img1 = tmp_path / "a.jpg"
    img2 = tmp_path / "b.jpg"
    img1.write_bytes(b"fake-jpg-1")
    img2.write_bytes(b"fake-jpg-2")

    # Plain text
    res_plain = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token), "139351986", "Cron report"
    )
    # Rich messages enabled
    res_rich = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token, extra={"rich_messages": True}),
        "139351986",
        "Cron **report**",
    )
    # Media album
    res_album = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token),
        "139351986",
        "Album report",
        media_files=[(str(img1), False), (str(img2), False)],
    )

    assert bot_factory_calls == []
    for res in (res_plain, res_rich, res_album):
        err = res.get("error", "")
        assert not err.startswith("Telegram send failed:")
        assert is_flood_error(err) is True
        assert flood_wait_seconds(err) > 3500.0


@pytest.mark.asyncio
async def test_bdd4_standalone_send_message_retry_after_arms_shared_cooldown(monkeypatch):
    token = "123456:ABC-bdd4-text"
    bot = _make_fake_bot(token)
    bot.send_message = AsyncMock(side_effect=_FakeRetryAfter(120))
    monkeypatch.setattr(tg_standalone, "_telegram_bot", lambda tok, extra=None: bot)

    res1 = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token), "139351986", "Cron report"
    )
    assert is_flood_error(res1.get("error")) is True
    assert not res1["error"].startswith("Telegram send failed:")
    assert 115.0 <= flood_wait_seconds(res1["error"]) <= 120.0
    assert bot.send_message.await_count == 1

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    rem = adapter._send_flood_cooldown_remaining("139351986")
    assert rem is not None and 115.0 <= rem <= 120.0

    res2 = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token), "139351986", "Second report"
    )
    assert is_flood_error(res2.get("error")) is True
    assert bot.send_message.await_count == 1  # no second API call


@pytest.mark.asyncio
async def test_bdd4_send_rich_standalone_retry_after_arms_shared_cooldown(monkeypatch):
    token = "123456:ABC-bdd4-rich"
    bot = _make_fake_bot(token)
    bot.do_api_request = AsyncMock(side_effect=_FakeRetryAfter(120))
    monkeypatch.setattr(tg_standalone, "_telegram_bot", lambda tok, extra=None: bot)
    monkeypatch.setattr(TelegramAdapter, "_should_attempt_rich", lambda self, content, metadata=None: True)

    res1 = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token, extra={"rich_messages": True}),
        "139351986",
        "Rich cron report",
    )
    assert is_flood_error(res1.get("error")) is True
    assert not res1["error"].startswith("Telegram send failed:")
    assert 115.0 <= flood_wait_seconds(res1["error"]) <= 120.0
    assert bot.do_api_request.await_count == 1
    assert bot.send_message.await_count == 0  # no legacy fallback

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    rem = adapter._send_flood_cooldown_remaining("139351986")
    assert rem is not None and 115.0 <= rem <= 120.0

    res2 = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token, extra={"rich_messages": True}),
        "139351986",
        "Second rich report",
    )
    assert is_flood_error(res2.get("error")) is True
    assert bot.do_api_request.await_count == 1


@pytest.mark.asyncio
async def test_bdd4_send_album_standalone_retry_after_fails_closed_without_per_photo_fallback(tmp_path, monkeypatch):
    token = "123456:ABC-bdd4-album"
    bot = _make_fake_bot(token)
    bot.send_media_group = AsyncMock(side_effect=_FakeRetryAfter(120))
    monkeypatch.setattr(tg_standalone, "_telegram_bot", lambda tok, extra=None: bot)

    img1 = tmp_path / "p1.jpg"
    img2 = tmp_path / "p2.jpg"
    img1.write_bytes(b"jpg1")
    img2.write_bytes(b"jpg2")

    per_photo_calls = []
    monkeypatch.setattr(
        tg_standalone,
        "_telegram_send_one_media",
        AsyncMock(side_effect=lambda *a, **k: per_photo_calls.append(a)),
    )

    res1 = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token),
        "139351986",
        "",
        media_files=[(str(img1), False), (str(img2), False)],
    )
    assert is_flood_error(res1.get("error")) is True
    assert 115.0 <= flood_wait_seconds(res1["error"]) <= 120.0
    assert bot.send_media_group.await_count == 1
    assert per_photo_calls == []  # fail-closed: no per-photo fallback

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=token))
    rem = adapter._send_flood_cooldown_remaining("139351986")
    assert rem is not None and 115.0 <= rem <= 120.0

    # Also verify fail-closed even if send_album_standalone returns non-empty detail["refused"] alongside flood_control
    clear_send_flood_cooldowns()
    monkeypatch.setattr(
        telegram_outbound,
        "send_album_standalone",
        AsyncMock(
            return_value=SendResult(
                success=False,
                error="flood_control:120.0",
                retry_after=120.0,
                raw_response={"refused": [str(img1), str(img2)]},
            )
        ),
    )
    res_refused = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token),
        "139351986",
        "",
        media_files=[(str(img1), False), (str(img2), False)],
    )
    assert res_refused.get("error") == "flood_control:120.0"
    assert per_photo_calls == []


@pytest.mark.asyncio
async def test_bdd5_cooldown_isolated_by_token_and_chat_id(monkeypatch):
    token_a = "111111:TOKEN-A"
    token_b = "222222:TOKEN-B"
    bots = {token_a: _make_fake_bot(token_a), token_b: _make_fake_bot(token_b)}
    monkeypatch.setattr(tg_standalone, "_telegram_bot", lambda tok, extra=None: bots[tok])

    adapter_a = TelegramAdapter(PlatformConfig(enabled=True, token=token_a))
    adapter_a._record_send_flood_cooldown("139351986", 3600)

    # Same token, different chat_id -> allowed
    res_other_chat = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token_a), "999999", "Hello other chat"
    )
    assert res_other_chat.get("success") is True
    assert bots[token_a].send_message.await_count == 1

    # Different token, same chat_id -> allowed
    res_other_token = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token_b), "139351986", "Hello other bot"
    )
    assert res_other_token.get("success") is True
    assert bots[token_b].send_message.await_count == 1

    # Same token, same chat_id -> blocked
    res_blocked = await tg_standalone.standalone_send(
        PlatformConfig(enabled=True, token=token_a), "139351986", "Blocked"
    )
    assert is_flood_error(res_blocked.get("error")) is True
    assert bots[token_a].send_message.await_count == 1


@pytest.mark.asyncio
async def test_bdd8_telegram_adapter_without_token_backwards_compatible():
    adapter = TelegramAdapter(PlatformConfig(enabled=True))
    assert adapter._flood_cooldown_token() == ""
    assert token_fingerprint("") == ""
    assert token_fingerprint(None) == ""

    res = adapter._record_send_flood_cooldown("12345", 120.0)
    assert res.success is False
    assert res.error == "flood_control:120.0"
    assert "12345" in adapter._telegram_send_cooldown_until

    rem_adapter = adapter._send_flood_cooldown_remaining("12345")
    rem_shared = send_flood_cooldown_remaining("", "12345")
    assert rem_adapter is not None and 115.0 <= rem_adapter <= 120.0
    assert rem_shared is not None and 115.0 <= rem_shared <= 120.0
