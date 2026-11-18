"""Tests for GatewayTurnProgressMixin._progress_terminal_blocks and friendly tool labels."""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest

import agent.display as agent_display
from agent import i18n
from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


class _MarkdownAdapter:
    supports_code_blocks = True

    @staticmethod
    def format_tool_preview(prepared) -> str:
        return prepared.text


class _DummyRunner:
    def __init__(self, adapter):
        self._adapter = adapter

    def _delivery_adapter_for(self, source):
        return self._adapter


@pytest.fixture(autouse=True)
def _restore_display_and_i18n(monkeypatch):
    prev_friendly = agent_display._friendly_tool_labels
    prev_preview_len = agent_display.get_tool_preview_max_len()
    i18n.reset_language_cache()
    try:
        yield
    finally:
        agent_display.set_friendly_tool_labels(prev_friendly)
        agent_display.set_tool_preview_max_len(prev_preview_len)
        monkeypatch.undo()
        i18n.reset_language_cache()


def _make_turn_runner(
    *,
    user_config,
    progress_mode: str = "all",
    platform: Platform = Platform.TELEGRAM,
    resolver=resolve_display_setting,
    adapter=None,
) -> TurnRunner:
    if adapter is None:
        adapter = _MarkdownAdapter()
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id="12345"),
        _run_still_current=lambda: True,
        progress_mode=progress_mode,
        tool_progress_enabled=True,
        progress_queue=queue.Queue(),
        user_config=user_config,
        resolve_display_setting=resolver,
    )
    tr = TurnRunner(_DummyRunner(adapter), ctx)
    assert not hasattr(tr, "config")
    return tr


def test_scenario1_ru_friendly_labels_without_terminal_blocks(monkeypatch):
    """Scenario 1: language=ru, friendly_tool_labels=true, tool_progress_command=false."""
    monkeypatch.setenv("HERMES_LANGUAGE", "ru")
    i18n.reset_language_cache()
    agent_display.set_friendly_tool_labels(True)
    agent_display.set_tool_preview_max_len(40)

    cfg = {
        "display": {
            "language": "ru",
            "friendly_tool_labels": True,
            "tool_progress_command": False,
            "tool_progress": "all",
        }
    }
    tr = _make_turn_runner(user_config=cfg, progress_mode="all")
    adapter = _MarkdownAdapter()

    assert tr._progress_terminal_blocks(adapter, "terminal", {"command": "ls -la /tmp"}, "💻") == (None, None)

    msg_terminal = tr._progress_build_message("terminal", "ls -la /tmp", {"command": "ls -la /tmp"})
    assert msg_terminal is not None
    assert "Выполнение команды" in msg_terminal
    assert "```" not in msg_terminal
    assert "ls -la /tmp" not in msg_terminal

    msg_exec = tr._progress_build_message("execute_code", "print(1)", {"code": "print(1)"})
    assert msg_exec is not None
    assert "Выполнение скрипта" in msg_exec
    assert "print(1)" not in msg_exec

    msg_read = tr._progress_build_message("read_file", "/tmp/a.txt", {"path": "/tmp/a.txt"})
    assert msg_read is not None
    assert "Чтение файла: " in msg_read
    assert "a.txt" in msg_read

    msg_delegate = tr._progress_build_message(
        "delegate_task", "check", {"tasks": [{"goal": "check"}]}
    )
    assert msg_delegate is not None
    assert "Делегирование задачи: " in msg_delegate
    assert "check" in msg_delegate

    tr.progress_callback("tool.started", "terminal", "ls -la /tmp", {"command": "ls -la /tmp"})
    queued = tr._ctx.progress_queue.get_nowait()
    assert "Выполнение команды" in queued


def test_scenario2_terminal_command_blocks_all_and_verbose(monkeypatch):
    """Scenario 2: tool_progress_command=true, friendly_tool_labels=false in all and verbose modes."""
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    agent_display.set_friendly_tool_labels(False)
    agent_display.set_tool_preview_max_len(40)

    cfg = {
        "display": {
            "tool_progress_command": True,
            "friendly_tool_labels": False,
            "tool_progress": "all",
        }
    }
    adapter = _MarkdownAdapter()
    cmd = "echo hello\necho world"

    tr_all = _make_turn_runner(user_config=cfg, progress_mode="all", adapter=adapter)
    emoji = agent_display.get_tool_emoji("terminal", default="⚙️")
    code_full, code_short = tr_all._progress_terminal_blocks(adapter, "terminal", {"command": cmd}, emoji)
    assert code_full is not None and code_short is not None
    assert f"```\n{cmd}\n```" in code_full
    assert "```\necho hello ...\n```" in code_short

    msg_all = tr_all._progress_build_message("terminal", cmd, {"command": cmd})
    assert msg_all == code_short

    # Non-terminal tools return (None, None) and render standard preview without friendly verbs
    for tool_name, preview, args in (
        ("read_file", "/tmp/a.txt", {"path": "/tmp/a.txt"}),
        ("execute_code", "print(1)", {"code": "print(1)"}),
        ("delegate_task", "check", {"tasks": [{"goal": "check"}]}),
    ):
        assert tr_all._progress_terminal_blocks(adapter, tool_name, args, "⚙️") == (None, None)
        non_term_msg = tr_all._progress_build_message(tool_name, preview, args)
        assert non_term_msg is not None
        assert tool_name in non_term_msg
        assert "```" not in non_term_msg

    # Verbose mode queues full command block directly into progress_queue
    cfg_verbose = {
        "display": {
            "tool_progress_command": True,
            "friendly_tool_labels": False,
            "tool_progress": "verbose",
        }
    }
    tr_verbose = _make_turn_runner(user_config=cfg_verbose, progress_mode="verbose", adapter=adapter)
    ret = tr_verbose._progress_build_message("terminal", cmd, {"command": cmd})
    assert ret is None
    queued_verbose = tr_verbose._ctx.progress_queue.get_nowait()
    assert f"```\n{cmd}\n```" in queued_verbose


def test_scenario3_platform_override_string_yaml_and_none_context():
    """Scenario 3: per-platform string 'false' override and safe fallback when ctx/user_config is None."""
    adapter = _MarkdownAdapter()

    # Variant A: display.platforms.telegram.tool_progress_command="false" overrides global True
    cfg_override = {
        "display": {
            "tool_progress_command": True,
            "friendly_tool_labels": False,
            "platforms": {
                "telegram": {
                    "tool_progress_command": "false",
                }
            },
        }
    }
    tr_override = _make_turn_runner(
        user_config=cfg_override,
        platform=Platform.TELEGRAM,
        adapter=adapter,
    )
    assert tr_override._progress_terminal_blocks(adapter, "terminal", {"command": "ls"}, "💻") == (None, None)

    # Variant B: ctx.user_config=None and ctx.resolve_display_setting=None (and even _ctx=None)
    tr_none = _make_turn_runner(
        user_config=None,
        resolver=None,
        adapter=adapter,
    )
    assert tr_none._progress_terminal_blocks(adapter, "terminal", {"command": "ls"}, "💻") == (None, None)

    tr_bare = TurnRunner(_DummyRunner(adapter), SimpleNamespace(user_config=None, resolve_display_setting=None))
    assert tr_bare._progress_terminal_blocks(adapter, "terminal", {"command": "ls"}, "💻") == (None, None)
