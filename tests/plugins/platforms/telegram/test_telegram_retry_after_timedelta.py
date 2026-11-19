"""Tests for datetime.timedelta RetryAfter handling across Telegram plugin modules."""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram.error import RetryAfter

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram import standalone
from plugins.platforms.telegram.adapter import TelegramAdapter, _FLOOD_INLINE_WAIT_CAP_SECS
from plugins.platforms.telegram.telegram_media_flood import flood_wait_seconds


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    return adapter


def test_flood_wait_seconds_handles_timedelta_and_hms_text():
    assert flood_wait_seconds(RetryAfter(timedelta(seconds=1488))) == 1488.0
    assert flood_wait_seconds(RetryAfter(timedelta(seconds=2))) == 2.0
    assert flood_wait_seconds(RetryAfter(3)) == 3.0
    assert flood_wait_seconds(RetryAfter(0.2)) == 0.2
    assert flood_wait_seconds(Exception("Flood control exceeded. Retry in 0:24:46")) == 1486.0
    assert flood_wait_seconds(Exception("Bad Request: message not found")) is None


@pytest.mark.asyncio
async def test_send_chunk_with_retries_long_timedelta_fails_closed_without_sleep():
    adapter = _make_adapter()
    adapter._send_chunk_markdown_or_plain = AsyncMock(
        side_effect=RetryAfter(timedelta(seconds=1488))
    )
    error_types = adapter._telegram_error_types()

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter._send_chunk_with_retries(
            "4242", "hello", 0, None, None, None, False, error_types
        )

    assert isinstance(result, SendResult)
    assert result.success is False
    assert result.error == "flood_control:1488.0"
    assert result.retry_after == 1488.0
    assert isinstance(result.retry_after, float)
    mock_sleep.assert_not_awaited()
    remaining = adapter._send_flood_cooldown_remaining("4242")
    assert remaining is not None and remaining > 0


@pytest.mark.asyncio
async def test_send_chunk_with_retries_short_timedelta_sleeps_float_and_retries():
    adapter = _make_adapter()
    sent_msg = MagicMock(message_id=101)
    adapter._send_chunk_markdown_or_plain = AsyncMock(
        side_effect=[RetryAfter(timedelta(seconds=2)), sent_msg]
    )
    error_types = adapter._telegram_error_types()

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        msg, used_fallback = await adapter._send_chunk_with_retries(
            "4242", "hello", 0, None, None, None, False, error_types
        )

    assert msg is sent_msg
    assert used_fallback is False
    mock_sleep.assert_awaited_once_with(2.0)
    assert isinstance(mock_sleep.await_args.args[0], float)


@pytest.mark.asyncio
async def test_send_chunk_with_retries_backward_compat_int_and_float():
    adapter = _make_adapter()
    sent_msg = MagicMock(message_id=102)
    adapter._send_chunk_markdown_or_plain = AsyncMock(
        side_effect=[RetryAfter(3), RetryAfter(0.2), sent_msg]
    )
    error_types = adapter._telegram_error_types()

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        msg, _ = await adapter._send_chunk_with_retries(
            "4242", "hello", 0, None, None, None, False, error_types
        )

    assert msg is sent_msg
    assert [call.args[0] for call in mock_sleep.await_args_list] == [3.0, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize("finalize", [False, True])
async def test_edit_message_long_timedelta_arms_cooldown_and_fails_closed(finalize: bool):
    adapter = _make_adapter()
    exc = RetryAfter(timedelta(seconds=45))
    adapter._edit_text = AsyncMock(side_effect=exc)
    adapter._edit_markdown_or_plain = AsyncMock(side_effect=exc)

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.edit_message("4242", "55", "hello", finalize=finalize)

    assert result.success is False
    assert result.error == "flood_control:45.0"
    assert result.retry_after == 45.0
    assert isinstance(result.retry_after, float)
    mock_sleep.assert_not_awaited()
    remaining = adapter._send_flood_cooldown_remaining("4242")
    assert remaining is not None and remaining > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("finalize", [False, True])
async def test_edit_message_short_then_retry_err_timedelta_arms_cooldown(finalize: bool):
    adapter = _make_adapter()
    short_exc = RetryAfter(timedelta(seconds=2))
    long_exc = RetryAfter(timedelta(seconds=60))
    if finalize:
        adapter._edit_markdown_or_plain = AsyncMock(side_effect=short_exc)
        adapter._edit_text = AsyncMock(side_effect=long_exc)
    else:
        adapter._edit_text = AsyncMock(side_effect=[short_exc, long_exc])

    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await adapter.edit_message("4242", "55", "hello", finalize=finalize)

    mock_sleep.assert_awaited_once_with(2.0)
    assert isinstance(mock_sleep.await_args.args[0], float)
    assert result.success is False
    assert result.error == "flood_control:60.0"
    assert result.retry_after == 60.0
    assert isinstance(result.retry_after, float)
    remaining = adapter._send_flood_cooldown_remaining("4242")
    assert remaining is not None and remaining > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("finalize", [False, True])
async def test_edit_message_respects_active_flood_cooldown_without_api_call(finalize: bool):
    adapter = _make_adapter()
    adapter._record_send_flood_cooldown("4242", 45.0)
    adapter._edit_text = AsyncMock()
    adapter._edit_markdown_or_plain = AsyncMock()

    result = await adapter.edit_message("4242", "55", "hello", finalize=finalize)

    assert result.success is False
    assert result.error.startswith("flood_control:")
    assert isinstance(result.retry_after, float)
    assert result.retry_after > 0
    adapter._edit_text.assert_not_awaited()
    adapter._edit_markdown_or_plain.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_rich_message_normalizes_timedelta_and_hms_fallback():
    adapter = _make_adapter()
    adapter._bot.do_api_request = AsyncMock(
        side_effect=RetryAfter(timedelta(seconds=1488))
    )
    res1 = await adapter._try_send_rich("4242", "hello", None, None)
    assert res1 is not None
    assert res1.success is False
    assert res1.retry_after == 1488.0
    assert isinstance(res1.retry_after, float)

    adapter._bot.do_api_request = AsyncMock(
        side_effect=Exception("Flood control exceeded. Retry in 0:24:46")
    )
    res2 = await adapter._try_send_rich("4242", "hello", None, None)
    assert res2 is not None
    assert res2.success is False
    assert res2.retry_after == 1486.0
    assert isinstance(res2.retry_after, float)


@pytest.mark.asyncio
async def test_record_typing_cooldown_handles_timedelta():
    adapter = _make_adapter()
    adapter._telegram_typing_cooldown_seconds = 5.0

    adapter._record_typing_cooldown("4242", RetryAfter(timedelta(seconds=45)))
    until_45 = adapter._telegram_typing_cooldown_until["4242"]

    adapter._record_typing_cooldown("4242", RetryAfter(timedelta(seconds=1488)))
    until_1488 = adapter._telegram_typing_cooldown_until["4242"]

    # 45s must not fall back to default 5s; 1488s is clamped to 300s.
    assert until_1488 - until_45 > 200.0


def test_standalone_telegram_retry_delay_contract():
    assert standalone.MAX_RETRY_WAIT_SECS == 30.0
    assert standalone._telegram_retry_delay(RetryAfter(timedelta(seconds=2)), 0) == 2.0
    assert standalone._telegram_retry_delay(RetryAfter(timedelta(seconds=30)), 0) == 30.0
    assert standalone._telegram_retry_delay(RetryAfter(timedelta(seconds=1488)), 0) is None

    class _FloodNoNum(Exception):
        retry_after = "unknown"

    assert standalone._telegram_retry_delay(_FloodNoNum("Flood control exceeded"), 0) == 1.0


@pytest.mark.asyncio
async def test_standalone_send_message_with_retry_short_vs_long_timedelta():
    bot = MagicMock()
    sent = MagicMock(message_id=777)
    bot.send_message = AsyncMock(side_effect=[RetryAfter(timedelta(seconds=2)), sent])

    with patch("plugins.platforms.telegram.standalone.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await standalone._send_telegram_message_with_retry(bot, chat_id=4242, text="hi")

    assert res is sent
    mock_sleep.assert_awaited_once_with(2.0)
    assert bot.send_message.await_count == 2

    bot2 = MagicMock()
    bot2.send_message = AsyncMock(side_effect=RetryAfter(timedelta(seconds=1488)))
    with patch("plugins.platforms.telegram.standalone.asyncio.sleep", new_callable=AsyncMock) as mock_sleep2:
        with pytest.raises(RetryAfter):
            await standalone._send_telegram_message_with_retry(bot2, chat_id=4242, text="hi")

    mock_sleep2.assert_not_awaited()
    assert bot2.send_message.await_count == 1
