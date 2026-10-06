"""Tests for cross-channel live activity & tool progress sync between Telegram and WebUI (BDD-1..BDD-5)."""

import asyncio
from unittest.mock import MagicMock
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionSource, build_session_key
from gateway.turn_context import TurnContext
from gateway.run_turn_runner import TurnRunner


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata,
        })
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "stopped": True, "metadata": metadata})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_event(chat_id: str, thread_id: str = None, message_id: str = "1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="group" if thread_id else "direct",
            thread_id=thread_id,
        ),
        message_id=message_id,
    )


class TestTelegramWebuiLiveActivitySync:

    @pytest.mark.asyncio
    async def test_bdd1_turn_start_initializes_status(self):
        """BDD-1: Initializing status at turn start in BasePlatformAdapter."""
        adapter = DummyTelegramAdapter()
        event = _make_event("123456", "789")
        session_key = build_session_key(event.source)

        started_event = asyncio.Event()

        async def fake_handler(_event):
            started_event.set()
            # Assert inside execution before completion
            assert session_key in adapter._active_sessions
            assert adapter._last_status.get(session_key) == "Ассистент думает над задачей..."
            assert adapter._current_tool.get(session_key) is None
            return "ok"

        adapter.set_message_handler(fake_handler)
        await adapter._process_message_background(event, session_key)
        assert started_event.is_set()

    def test_bdd2_turn_runner_progress_callback_live_tool_status(self):
        """BDD-2: Live tool status translation on tool.started and tool.completed."""
        adapter = DummyTelegramAdapter()
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="123456", thread_id="789")
        session_key = "telegram:123456:789"
        adapter._last_status[session_key] = "Ассистент думает над задачей..."
        adapter._current_tool[session_key] = None

        ctx = TurnContext(
            source=source,
            session_key=session_key,
            _status_adapter=adapter,
            _run_still_current=lambda: True,
        )
        fake_runner = MagicMock()
        runner = TurnRunner(fake_runner, ctx)

        # 1. tool.started
        runner.progress_callback("tool.started", tool_name="execute_code", args={"command": "echo test"})
        assert adapter._last_status.get(session_key) == "Выполняется execute_code..."
        assert adapter._current_tool.get(session_key) == "execute_code"

        # 2. _thinking should not overwrite current tool
        runner.progress_callback("tool.started", tool_name="_thinking", args={})
        assert adapter._last_status.get(session_key) == "Выполняется execute_code..."
        assert adapter._current_tool.get(session_key) == "execute_code"

        # 3. tool.completed
        runner.progress_callback("tool.completed", tool_name="execute_code", args={})
        assert adapter._last_status.get(session_key) == "Ассистент думает над задачей..."
        assert adapter._current_tool.get(session_key) is None

    def test_bdd3_api_server_session_response_contract_telegram_topics(self):
        """BDD-3: REST API GET /api/sessions/{session_id} contract returns is_generating, tool_status, current_tool."""
        server = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8080}))
        server._run_statuses = {}

        topic_key = "agent:main:telegram:group:-100123456:998877"
        adapter = DummyTelegramAdapter()
        adapter._active_sessions = {topic_key: asyncio.Event()}
        adapter._last_status = {topic_key: "Выполняется execute_code..."}
        adapter._current_tool = {topic_key: "execute_code"}

        runner = MagicMock()
        runner.adapters = {"telegram": adapter}
        runner._profile_adapters = {}
        server.gateway_runner = runner

        session_data = {
            "id": "session_tg_topic_1",
            "chat_id": "-100123456",
            "thread_id": "998877",
            "session_key": "agent:main:telegram:group:-100123456:998877",
        }
        resp = server._session_response(session_data)
        assert resp["is_generating"] is True
        assert resp["tool_status"] == "Выполняется execute_code..."
        assert resp["current_tool"] == "execute_code"

    def test_bdd4_api_server_concurrency_safe_iteration(self):
        """BDD-4: Concurrency safe snapshot iteration in api_server _session_response."""
        server = APIServerAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 8080}))
        server._run_statuses = {}

        adapter = DummyTelegramAdapter()
        # Prepopulate with multiple sessions
        for i in range(10):
            adapter._active_sessions[f"telegram:chat_{i}"] = asyncio.Event()
            adapter._last_status[f"telegram:chat_{i}"] = f"Выполняется tool_{i}..."
            adapter._current_tool[f"telegram:chat_{i}"] = f"tool_{i}"

        runner = MagicMock()
        runner.adapters = {"telegram": adapter}
        runner._profile_adapters = {}
        server.gateway_runner = runner

        session_data = {
            "id": "sess_target",
            "chat_id": "chat_5",
            "thread_id": None,
            "session_key": "telegram:chat_5",
        }
        resp = server._session_response(session_data)
        assert resp["is_generating"] is True
        assert resp["tool_status"] == "Выполняется tool_5..."
        assert resp["current_tool"] == "tool_5"

    @pytest.mark.asyncio
    async def test_bdd5_teardown_and_cleanup(self):
        """BDD-5: Teardown & cleanup removes session_key from _last_status and _current_tool."""
        adapter = DummyTelegramAdapter()
        session_key = "telegram:direct:999"
        guard = asyncio.Event()

        adapter._active_sessions[session_key] = guard
        adapter._last_status[session_key] = "Выполняется web_search..."
        adapter._current_tool[session_key] = "web_search"

        # 1. Test _release_session_guard cleans up
        adapter._release_session_guard(session_key, guard=guard)
        assert session_key not in adapter._active_sessions
        assert session_key not in adapter._last_status
        assert session_key not in adapter._current_tool

        # 2. Test _heal_stale_session_lock cleans up
        adapter._active_sessions[session_key] = guard
        adapter._last_status[session_key] = "Выполняется web_search..."
        adapter._current_tool[session_key] = "web_search"
        adapter._session_tasks[session_key] = MagicMock(done=lambda: True)

        healed = adapter._heal_stale_session_lock(session_key)
        assert healed is True
        assert session_key not in adapter._active_sessions
        assert session_key not in adapter._last_status
        assert session_key not in adapter._current_tool

        # 3. Test background task finally block cleans up
        event = _make_event("999")
        clean_key = build_session_key(event.source)
        adapter._session_tasks[clean_key] = asyncio.current_task()

        async def dummy_handler(_event):
            assert adapter._last_status.get(clean_key) is not None
            return "done"

        adapter.set_message_handler(dummy_handler)
        await adapter._process_message_background(event, clean_key)
        assert clean_key not in adapter._active_sessions
        assert clean_key not in adapter._last_status
        assert clean_key not in adapter._current_tool
