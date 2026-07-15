"""Tests for gateway /quota command."""

import asyncio
from unittest.mock import MagicMock, patch
import subprocess
import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.config import Platform


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = None
    runner.config = None
    return runner


def _make_event() -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-123",
        chat_id="chat-123",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(text="/quota", source=source)


@pytest.mark.asyncio
async def test_quota_command_success():
    runner = _make_runner()
    event = _make_event()

    mock_completed_process = MagicMock(spec=subprocess.CompletedProcess)
    mock_completed_process.stdout = "| Provider | Used | Limit |\n|---|---|---|\n| Gemini | 5 | 10 |"
    
    with patch("subprocess.run", return_value=mock_completed_process) as mock_run:
        res = await runner._handle_quota_command(event)
        
        # Verify subprocess.run was called with correct command
        mock_run.assert_called_once()
        cmd_arg = mock_run.call_args[0][0]
        assert "check_gemini_quotas.py" in cmd_arg[-2]
        assert "--always-table" in cmd_arg[-1]
        
        # Verify the returned markdown table
        assert "| Provider | Used | Limit |" in res


@pytest.mark.asyncio
async def test_quota_command_caller_failure():
    runner = _make_runner()
    event = _make_event()

    # Raise CalledProcessError
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, cmd=["python3"], stderr="Quota limit checked but script failed")):
        res = await runner._handle_quota_command(event)
        assert "❌ Failed to query quotas: Quota limit checked but script failed" in res


@pytest.mark.asyncio
async def test_quota_command_unexpected_exception():
    runner = _make_runner()
    event = _make_event()

    # Raise general Exception
    with patch("subprocess.run", side_effect=ValueError("Some random value error")):
        res = await runner._handle_quota_command(event)
        assert "❌ Unexpected error checking quotas: Some random value error" in res
