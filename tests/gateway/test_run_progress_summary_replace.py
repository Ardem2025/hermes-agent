"""Tests for summary/replace progress modes and tool_progress_command independence (BDD 2.1, 2.2)."""

import queue
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin


class _FakeAdapter:
    name = "telegram"
    MAX_MESSAGE_LENGTH = 4000
    supports_code_blocks = True

    def __init__(self):
        self.sent = []
        self.edits = []
        self._seq = 0

    def format_tool_preview(self, prepared):
        return prepared.text

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self._seq += 1
        msg_id = f"msg-{self._seq}"
        self.sent.append({"chat_id": chat_id, "content": content, "message_id": msg_id})
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(self, chat_id, message_id, content, **kwargs):
        self.edits.append({"chat_id": chat_id, "message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)


class _DummyProgressRunner(GatewayTurnProgressMixin):
    def __init__(self, ctx, adapter):
        self._ctx = ctx
        self._runner = SimpleNamespace(_delivery_adapter_for=lambda _src: adapter)


def _make_ctx(*, progress_mode="summary", progress_grouping="replace", user_config=None):
    return SimpleNamespace(
        source=SimpleNamespace(platform=Platform.TELEGRAM, chat_id="12345", thread_id=None),
        user_config=user_config or {},
        progress_mode=progress_mode,
        progress_grouping=progress_grouping,
        tool_progress_enabled=progress_mode not in {"off", "log"},
        _thinking_enabled=False,
        _native_slack_task_cards=False,
        _live_status_adapter=None,
        _live_status_mode="off",
        log_queue=None,
        progress_queue=queue.Queue(),
        long_tool_hint_fired=[False],
        last_tool=[None],
        last_was_terminal_block=[False],
        last_progress_msg=[None],
        repeat_count=[0],
        agent_holder=[None],
        stream_consumer_holder=[None],
        _cleanup_progress=False,
        _cleanup_msg_ids=[],
        _progress_reply_to=None,
        _progress_metadata=None,
        _run_still_current=lambda: True,
    )


@pytest.mark.asyncio
async def test_replace_grouping_edits_single_bubble_with_latest_line_only():
    adapter = _FakeAdapter()
    ctx = _make_ctx(progress_mode="summary", progress_grouping="replace")
    runner = _DummyProgressRunner(ctx, adapter)
    st = runner._progress_edit_state(adapter)

    assert st.can_edit is True
    assert st.replace_mode is True

    msg1 = runner._progress_absorb(st, "🔍 Поиск в сети: first")
    assert st.progress_lines == ["🔍 Поиск в сети: first"]
    assert await runner._progress_send_or_edit(st, msg1) is True
    assert len(adapter.sent) == 1
    assert adapter.sent[0]["content"] == "🔍 Поиск в сети: first"

    msg2 = runner._progress_absorb(st, "📖 Чтение файла: second.py")
    assert st.progress_lines == ["📖 Чтение файла: second.py"]
    assert await runner._progress_send_or_edit(st, msg2) is True
    assert len(adapter.edits) == 1
    assert adapter.edits[0]["message_id"] == "msg-1"
    assert adapter.edits[0]["content"] == "📖 Чтение файла: second.py"


@pytest.mark.asyncio
async def test_accumulate_and_separate_groupings_preserve_existing_behavior():
    adapter = _FakeAdapter()

    ctx_acc = _make_ctx(progress_mode="all", progress_grouping="accumulate")
    runner_acc = _DummyProgressRunner(ctx_acc, adapter)
    st_acc = runner_acc._progress_edit_state(adapter)
    assert st_acc.can_edit is True
    assert st_acc.replace_mode is False
    runner_acc._progress_absorb(st_acc, "line 1")
    runner_acc._progress_absorb(st_acc, "line 2")
    assert st_acc.progress_lines == ["line 1", "line 2"]

    ctx_sep = _make_ctx(progress_mode="all", progress_grouping="separate")
    runner_sep = _DummyProgressRunner(ctx_sep, adapter)
    st_sep = runner_sep._progress_edit_state(adapter)
    assert st_sep.can_edit is False


def test_summary_mode_emits_compact_progress_line(monkeypatch):
    from agent.display import set_friendly_tool_labels

    set_friendly_tool_labels(True)
    monkeypatch.setattr("agent.display.get_language", lambda: "ru")

    adapter = _FakeAdapter()
    ctx = _make_ctx(progress_mode="summary", progress_grouping="replace")
    runner = _DummyProgressRunner(ctx, adapter)

    runner.progress_callback("tool.started", "web_search", "hermes agent", {"query": "hermes agent"})
    runner.progress_callback("tool.started", "web_search", "second query", {"query": "second query"})

    items = []
    while not ctx.progress_queue.empty():
        items.append(ctx.progress_queue.get_nowait())
    assert len(items) == 2
    assert "Поиск в сети: hermes agent" in items[0]
    assert "Поиск в сети: second query" in items[1]


def test_tool_progress_command_independent_of_friendly_tool_labels(monkeypatch):
    from agent.display import set_friendly_tool_labels

    set_friendly_tool_labels(True)
    monkeypatch.setattr("agent.display.get_language", lambda: "ru")

    adapter = _FakeAdapter()

    # tool_progress_command: true -> shows terminal command even with friendly_tool_labels=True
    ctx_on = _make_ctx(
        progress_mode="summary",
        progress_grouping="replace",
        user_config={"display": {"friendly_tool_labels": True, "tool_progress_command": True}},
    )
    runner_on = _DummyProgressRunner(ctx_on, adapter)
    msg_on = runner_on._progress_build_message("terminal", "ls -la", {"command": "ls -la"})
    assert msg_on is not None
    assert "ls -la" in msg_on
    assert "```" in msg_on

    # tool_progress_command: false -> hides command, shows only friendly RU label
    ctx_off = _make_ctx(
        progress_mode="summary",
        progress_grouping="replace",
        user_config={"display": {"friendly_tool_labels": True, "tool_progress_command": False}},
    )
    runner_off = _DummyProgressRunner(ctx_off, adapter)
    msg_off = runner_off._progress_build_message("terminal", "ls -la", {"command": "ls -la"})
    assert msg_off is not None
    assert "Выполнение команды" in msg_off
    assert "ls -la" not in msg_off
