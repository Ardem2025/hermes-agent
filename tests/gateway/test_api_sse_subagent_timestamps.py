"""Tests for WebUI SSE subagent.* forwarding and event timestamps in api_server."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms import api_server as api_server_mod
from gateway.platforms.api_server import APIServerAdapter, _SessionEventQueue


def _parse_sse_frames(chunks: list[bytes]) -> list[tuple[str, dict]]:
    raw = b"".join(chunks).decode("utf-8")
    frames: list[tuple[str, dict]] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_name and data_lines:
            frames.append((event_name, json.loads("\n".join(data_lines))))
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_to_telegram", [False, True])
async def test_session_chat_stream_enqueues_subagent_events_and_timestamps(
    tmp_path, monkeypatch, duplicate_to_telegram
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    enqueued: list[tuple[str, dict]] = []
    orig_enqueue = _SessionEventQueue.enqueue

    def spy_enqueue(self, name: str, payload: dict) -> None:
        enqueued.append((name, dict(payload)))
        return orig_enqueue(self, name, payload)

    monkeypatch.setattr(_SessionEventQueue, "enqueue", spy_enqueue)

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    written_chunks: list[bytes] = []

    class _FakeSSEResponse:
        async def write(self, data: bytes) -> None:
            written_chunks.append(data)

    async def fake_run_agent(
        *,
        conversation_history,
        stream_delta_callback,
        tool_progress_callback,
        interim_assistant_callback,
        active_run_id,
        approval_notify_callback,
        approval_session_key,
        **kwargs,
    ):
        tool_progress_callback(
            "subagent.start",
            None,
            "Analyze codebase",
            {"goal": "Analyze codebase", "subagent_id": "sub-1"},
            task_index=0,
            goal="Analyze codebase",
            subagent_id="sub-1",
        )
        tool_progress_callback(
            "subagent_progress",
            "read_file",
            "Reading api_server.py",
            {"path": Path("/tmp/example.py"), "tags": {"sse"}},
            task_index=0,
            subagent_id="sub-1",
            action="Reading api_server.py",
        )
        tool_progress_callback(
            "subagent.tool",
            "search_files",
            "Searching gateway/",
            {"pattern": "subagent"},
            task_index=0,
            subagent_id="sub-1",
            started_at=1700000000.0,
        )
        tool_progress_callback(
            "subagent.complete",
            None,
            "Finished analysis",
            None,
            task_index=0,
            subagent_id="sub-1",
            status="completed",
            summary="Finished analysis",
        )
        stream_delta_callback("Hello")
        return {"final_response": "Hello", "session_id": "sess-1"}, {"total_tokens": 10}

    try:
        adapter._concurrency_limited_response = lambda: None
        adapter._prepare_session_chat = AsyncMock(
            return_value=(
                {
                    "gateway_session_key": "api:sess-1",
                    "session_id": "sess-1",
                    "user_message": "hi",
                    "runtime_request": {},
                    "lock_active": False,
                    "body": {"duplicate_to_telegram": duplicate_to_telegram},
                    "run_kwargs": {},
                },
                None,
            )
        )
        adapter._stream_through_live_bot_chat = AsyncMock(return_value=None)
        adapter.is_telegram_bound = lambda sid: False
        adapter._new_telegram_mirror = lambda **kw: SimpleNamespace(
            enqueue_commentary=lambda *a, **k: None,
            drain=AsyncMock(),
            send_final=AsyncMock(),
        )
        adapter._telegram_runner_and_adapter = lambda: (None, None)
        adapter._run_idempotency_scope = lambda req: "scope"
        adapter._set_run_status = lambda *a, **k: None
        adapter._conversation_history_for_session = AsyncMock(return_value=[])
        adapter._run_agent = fake_run_agent
        adapter._turn_transcript_messages = lambda *a, **k: []
        adapter._effective_turn_runtime = lambda *a, **k: {}
        adapter._release_run_owner_if_forgotten = lambda *a, **k: None
        adapter._prepare_sse_response = AsyncMock(return_value=_FakeSSEResponse())

        handler = getattr(
            APIServerAdapter._handle_session_chat_stream,
            "__wrapped__",
            APIServerAdapter._handle_session_chat_stream,
        )
        fake_request = SimpleNamespace(headers={}, match_info={"session_id": "sess-1"})
        await handler(adapter, fake_request)
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()

    subagent_enqueued = [(n, p) for n, p in enqueued if n.startswith("subagent.")]
    assert [n for n, _ in subagent_enqueued] == [
        "subagent.start",
        "subagent.progress",
        "subagent.tool",
        "subagent.complete",
    ]
    for name, payload in subagent_enqueued:
        assert isinstance(payload.get("timestamp"), (int, float))
        assert payload["timestamp"] > 0
        assert payload["subagent_id"] == "sub-1"
        assert payload["event_type"] == name
        json.dumps(payload)

    assert subagent_enqueued[0][1]["goal"] == "Analyze codebase"
    assert subagent_enqueued[0][1]["status"] == "running"
    assert subagent_enqueued[1][1]["tool_name"] == "read_file"
    assert subagent_enqueued[1][1]["args"]["path"] == "/tmp/example.py"
    assert subagent_enqueued[2][1]["tool_name"] == "search_files"
    assert subagent_enqueued[2][1]["started_at"] == 1700000000.0
    assert isinstance(subagent_enqueued[2][1]["running_seconds"], float)
    assert subagent_enqueued[2][1]["running_seconds"] > 0
    assert subagent_enqueued[3][1]["status"] == "completed"

    frames = _parse_sse_frames(written_chunks)
    by_event = {name: payload for name, payload in frames}
    for required_event in (
        "run.started",
        "message.started",
        "subagent.start",
        "subagent.progress",
        "subagent.tool",
        "subagent.complete",
        "assistant.completed",
    ):
        assert required_event in by_event, f"Missing SSE event {required_event}"
        assert isinstance(by_event[required_event].get("timestamp"), (int, float))
        assert by_event[required_event]["timestamp"] > 0

    assert by_event["subagent.tool"]["tool_name"] == "search_files"
    assert by_event["subagent.tool"]["started_at"] == 1700000000.0
    assert isinstance(by_event["subagent.tool"]["running_seconds"], float)
    assert isinstance(by_event["message.started"]["message"].get("timestamp"), (int, float))
    assert by_event["message.started"]["message"]["timestamp"] > 0


@pytest.mark.asyncio
async def test_subagent_started_at_from_registry_in_sse_and_get_session(tmp_path, monkeypatch):
    import time
    from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    written_chunks: list[bytes] = []

    class _FakeSSEResponse:
        async def write(self, data: bytes) -> None:
            written_chunks.append(data)

    started_ts = time.time() - 15.0
    with _active_subagents_lock:
        _active_subagents["sub-reg-1"] = {
            "subagent_id": "sub-reg-1",
            "owner_agent_session_id": "sess-reg-1",
            "task_index": 0,
            "goal": "Inspect registry",
            "model": "test-model",
            "status": "running",
            "last_tool": "terminal",
            "tool_count": 3,
            "started_at": started_ts,
        }

    async def fake_run_agent(
        *,
        conversation_history,
        stream_delta_callback,
        tool_progress_callback,
        interim_assistant_callback,
        active_run_id,
        approval_notify_callback,
        approval_session_key,
        **kwargs,
    ):
        tool_progress_callback(
            "subagent_tool",
            "terminal",
            "Running pytest",
            {"command": "pytest"},
            task_index=0,
            subagent_id="sub-reg-1",
        )
        stream_delta_callback("Done")
        return {"final_response": "Done", "session_id": "sess-reg-1"}, {"total_tokens": 5}

    try:
        adapter._get_existing_session_or_404 = AsyncMock(
            return_value=(
                {
                    "id": "sess-reg-1",
                    "source": "api_server",
                    "user_id": "u1",
                    "model": "test-model",
                    "started_at": started_ts - 5.0,
                },
                None,
            )
        )
        get_handler = getattr(
            APIServerAdapter._handle_get_session,
            "__wrapped__",
            APIServerAdapter._handle_get_session,
        )
        get_resp = await get_handler(adapter, SimpleNamespace(headers={}, match_info={"session_id": "sess-reg-1"}))
        session_body = json.loads(get_resp.text)
        active_subs = session_body["session"]["active_subagents"]
        assert len(active_subs) == 1
        assert active_subs[0]["subagent_id"] == "sub-reg-1"
        assert active_subs[0]["started_at"] == pytest.approx(started_ts, abs=1e-6)
        assert isinstance(active_subs[0]["running_seconds"], float)
        assert active_subs[0]["running_seconds"] >= 14.5

        adapter._concurrency_limited_response = lambda: None
        adapter._prepare_session_chat = AsyncMock(
            return_value=(
                {
                    "gateway_session_key": "api:sess-reg-1",
                    "session_id": "sess-reg-1",
                    "user_message": "hi",
                    "runtime_request": {},
                    "lock_active": False,
                    "body": {"duplicate_to_telegram": False},
                    "run_kwargs": {},
                },
                None,
            )
        )
        adapter._stream_through_live_bot_chat = AsyncMock(return_value=None)
        adapter.is_telegram_bound = lambda sid: False
        adapter._new_telegram_mirror = lambda **kw: SimpleNamespace(
            enqueue_commentary=lambda *a, **k: None,
            drain=AsyncMock(),
            send_final=AsyncMock(),
        )
        adapter._telegram_runner_and_adapter = lambda: (None, None)
        adapter._run_idempotency_scope = lambda req: "scope"
        adapter._set_run_status = lambda *a, **k: None
        adapter._conversation_history_for_session = AsyncMock(return_value=[])
        adapter._run_agent = fake_run_agent
        adapter._turn_transcript_messages = lambda *a, **k: []
        adapter._effective_turn_runtime = lambda *a, **k: {}
        adapter._release_run_owner_if_forgotten = lambda *a, **k: None
        adapter._prepare_sse_response = AsyncMock(return_value=_FakeSSEResponse())

        stream_handler = getattr(
            APIServerAdapter._handle_session_chat_stream,
            "__wrapped__",
            APIServerAdapter._handle_session_chat_stream,
        )
        await stream_handler(adapter, SimpleNamespace(headers={}, match_info={"session_id": "sess-reg-1"}))
    finally:
        with _active_subagents_lock:
            _active_subagents.pop("sub-reg-1", None)
        adapter._response_store.close()
        adapter._run_idempotency_store.close()

    frames = _parse_sse_frames(written_chunks)
    by_event = {name: payload for name, payload in frames}
    assert "subagent.tool" in by_event
    tool_ev = by_event["subagent.tool"]
    assert tool_ev["subagent_id"] == "sub-reg-1"
    assert tool_ev["tool_name"] == "terminal"
    assert tool_ev["started_at"] == pytest.approx(started_ts, abs=1e-6)
    assert isinstance(tool_ev["running_seconds"], float)
    assert tool_ev["running_seconds"] >= 14.5
