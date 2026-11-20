"""Tests for unified display settings and tool_progress_details across main agent and subagents."""

from __future__ import annotations

import asyncio
import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.display_config import OVERRIDEABLE_KEYS, resolve_display_setting
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.stream_events import ToolCallChunk
from gateway.subagent_pulse_ticker import SubagentPulseTicker
from gateway.turn_context import TurnContext
from hermes_cli.config_defaults import DEFAULT_CONFIG


class _FakeProgressAdapter(BasePlatformAdapter):
    name = "telegram"
    MAX_MESSAGE_LENGTH = 4000
    supports_code_blocks = True

    def __init__(self) -> None:
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent: list[dict] = []
        self.edits: list[dict] = []
        self.deleted: list[tuple[str, str]] = []
        self._status_locks = {}
        self._status_message_ids = {}
        self._STATUS_MESSAGE_IDS_MAX = 256
        self._seq = 0

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None, **kwargs) -> SendResult:
        self._seq += 1
        msg_id = f"msg-{self._seq}"
        self.sent.append({"chat_id": str(chat_id), "content": content, "message_id": msg_id})
        return SendResult(success=True, message_id=msg_id)

    async def edit_message(self, chat_id: str, message_id: str, content: str, **kwargs) -> SendResult:
        self.edits.append({"chat_id": str(chat_id), "message_id": str(message_id), "content": content})
        return SendResult(success=True, message_id=str(message_id))

    async def delete_message(self, chat_id: str, message_id: str, **kwargs) -> bool:
        self.deleted.append((str(chat_id), str(message_id)))
        return True

    async def get_chat_info(self, chat_id: str) -> dict:
        return {"name": str(chat_id), "type": "dm"}


class _DummyProgressRunner(GatewayTurnProgressMixin):
    def __init__(self, ctx: TurnContext, adapter: _FakeProgressAdapter) -> None:
        self._ctx = ctx
        self._runner = SimpleNamespace(
            _delivery_adapter_for=lambda _src: adapter,
            _is_draining=lambda: False,
        )
        self.scheduled: list[tuple[object, str]] = []

    def _schedule(self, coro, label: str = "") -> None:
        self.scheduled.append((coro, label))
        if asyncio.iscoroutine(coro):
            coro.close()


def _make_turn_ctx(
    *,
    platform: Platform = Platform.TELEGRAM,
    progress_mode: str = "all",
    tool_progress_enabled: bool = True,
    log_mode_enabled: bool = False,
    tool_progress_details: bool = True,
    friendly_tool_labels: bool = True,
    tool_preview_length: int = 0,
    progress_grouping: str = "accumulate",
    user_config: dict | None = None,
) -> TurnContext:
    return TurnContext(
        session_key="tg:12345",
        source=SimpleNamespace(platform=platform, chat_id="12345", thread_id=None),
        user_config=user_config or {},
        progress_mode=progress_mode,
        progress_grouping=progress_grouping,
        tool_progress_enabled=tool_progress_enabled,
        log_mode_enabled=log_mode_enabled,
        tool_progress_details=tool_progress_details,
        friendly_tool_labels=friendly_tool_labels,
        tool_preview_length=tool_preview_length,
        progress_queue=queue.Queue(),
        _run_still_current=lambda: True,
    )


# ---------------------------------------------------------------------------
# 1. display_config.py & config_defaults.py
# ---------------------------------------------------------------------------


class TestDisplayConfigUnifiedKeys:
    def test_keys_in_overrideable_and_defaults_true(self):
        assert "tool_progress_details" in OVERRIDEABLE_KEYS
        assert "friendly_tool_labels" in OVERRIDEABLE_KEYS
        assert resolve_display_setting({}, "telegram", "tool_progress_details") is True
        assert resolve_display_setting({}, "telegram", "friendly_tool_labels") is True
        assert DEFAULT_CONFIG["display"]["tool_progress_details"] is True
        assert DEFAULT_CONFIG["display"]["friendly_tool_labels"] is True

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("false", False),
            ("no", False),
            (False, False),
            ("true", True),
            ("yes", True),
            (True, True),
        ],
    )
    def test_norm_bool_values(self, raw, expected):
        cfg = {"display": {"tool_progress_details": raw, "friendly_tool_labels": raw}}
        assert resolve_display_setting(cfg, "telegram", "tool_progress_details") is expected
        assert resolve_display_setting(cfg, "telegram", "friendly_tool_labels") is expected

    def test_platform_override_beats_global(self):
        cfg = {
            "display": {
                "tool_progress_details": True,
                "friendly_tool_labels": True,
                "platforms": {
                    "telegram": {
                        "tool_progress_details": "false",
                        "friendly_tool_labels": "no",
                    }
                },
            }
        }
        assert resolve_display_setting(cfg, "telegram", "tool_progress_details") is False
        assert resolve_display_setting(cfg, "telegram", "friendly_tool_labels") is False
        assert resolve_display_setting(cfg, "discord", "tool_progress_details") is True
        assert resolve_display_setting(cfg, "discord", "friendly_tool_labels") is True


# ---------------------------------------------------------------------------
# 2. TurnRunner._progress_build_message & _progress_terminal_blocks (BDD 1)
# ---------------------------------------------------------------------------


class TestTurnRunnerToolProgressDetails:
    def test_details_false_friendly_true_hides_paths_args_and_terminal_blocks(self, monkeypatch):
        monkeypatch.setattr("agent.display.get_language", lambda: "ru")
        adapter = _FakeProgressAdapter()
        ctx = _make_turn_ctx(
            progress_mode="all",
            tool_progress_details=False,
            friendly_tool_labels=True,
            user_config={"display": {"tool_progress_command": True}},
        )
        runner = _DummyProgressRunner(ctx, adapter)

        # Built-in tool with verb -> only "{emoji} {verb}", no path
        msg_read = runner._progress_build_message("read_file", "/etc/passwd", {"path": "/etc/passwd"})
        assert msg_read is not None
        assert "/etc/passwd" not in msg_read
        assert "Чтение файла" in msg_read

        # Terminal with tool_progress_command=True -> no fenced block, no command text
        full_blk, short_blk = runner._progress_terminal_blocks(
            adapter, "terminal", {"command": "cat /secret/token"}, "💻"
        )
        assert (full_blk, short_blk) == (None, None)
        msg_term = runner._progress_build_message("terminal", "cat /secret/token", {"command": "cat /secret/token"})
        assert msg_term is not None
        assert "```" not in msg_term
        assert "/secret/token" not in msg_term
        assert "Выполнение команды" in msg_term

        # Custom/MCP tool without verb -> "{emoji} {tool_name}", no args/preview
        msg_mcp = runner._progress_build_message("mcp_custom_lookup", "secret query", {"query": "secret query"})
        assert msg_mcp == "⚙️ mcp_custom_lookup"

    def test_details_false_friendly_false_uses_raw_tool_name(self, monkeypatch):
        monkeypatch.setattr("agent.display.get_language", lambda: "ru")
        adapter = _FakeProgressAdapter()
        ctx = _make_turn_ctx(
            progress_mode="all",
            tool_progress_details=False,
            friendly_tool_labels=False,
        )
        runner = _DummyProgressRunner(ctx, adapter)

        msg = runner._progress_build_message("read_file", "/secret/file.txt", {"path": "/secret/file.txt"})
        assert msg == "⚙️ read_file"
        assert "/secret/file.txt" not in msg

    def test_details_false_verbose_queues_clean_line_and_returns_none(self, monkeypatch):
        monkeypatch.setattr("agent.display.get_language", lambda: "ru")
        adapter = _FakeProgressAdapter()
        ctx = _make_turn_ctx(
            progress_mode="verbose",
            tool_progress_details=False,
            friendly_tool_labels=True,
            user_config={"display": {"tool_progress_command": True}},
        )
        runner = _DummyProgressRunner(ctx, adapter)

        ret = runner._progress_build_message("terminal", "echo secret", {"command": "echo secret"})
        assert ret is None
        queued = ctx.progress_queue.get_nowait()
        assert "echo secret" not in queued
        assert "```" not in queued
        assert "Выполнение команды" in queued

    def test_details_true_respects_tool_preview_length_and_friendly_flag(self, monkeypatch):
        monkeypatch.setattr("agent.display.get_language", lambda: "ru")
        adapter = _FakeProgressAdapter()
        long_query = "a" * 60

        ctx_friendly = _make_turn_ctx(
            progress_mode="all",
            tool_progress_details=True,
            friendly_tool_labels=True,
            tool_preview_length=20,
        )
        runner_friendly = _DummyProgressRunner(ctx_friendly, adapter)
        msg_friendly = runner_friendly._progress_build_message("web_search", long_query, {"query": long_query})
        assert msg_friendly is not None
        assert "Поиск в сети" in msg_friendly
        assert long_query not in msg_friendly

        ctx_raw = _make_turn_ctx(
            progress_mode="all",
            tool_progress_details=True,
            friendly_tool_labels=False,
            tool_preview_length=20,
        )
        runner_raw = _DummyProgressRunner(ctx_raw, adapter)
        msg_raw = runner_raw._progress_build_message("web_search", long_query, {"query": long_query})
        assert msg_raw is not None
        assert "web_search" in msg_raw
        assert "Поиск в сети" not in msg_raw
        assert long_query not in msg_raw


# ---------------------------------------------------------------------------
# 3. TurnRunner._handle_subagent_pulse & _uses_telegram_pulse_ticker (BDD 3)
# ---------------------------------------------------------------------------


class TestTurnRunnerSubagentPulseGating:
    @pytest.mark.parametrize(
        ("tool_progress_enabled", "log_mode_enabled"),
        [
            (False, False),  # tool_progress: off
            (True, True),    # tool_progress: log
        ],
    )
    def test_off_or_log_disables_pulse_ticker(self, tool_progress_enabled, log_mode_enabled):
        adapter = _FakeProgressAdapter()
        ctx = _make_turn_ctx(
            platform=Platform.TELEGRAM,
            progress_mode="off" if not tool_progress_enabled else "log",
            tool_progress_enabled=tool_progress_enabled,
            log_mode_enabled=log_mode_enabled,
        )
        ctx._status_adapter = adapter
        runner = _DummyProgressRunner(ctx, adapter)

        assert runner._uses_telegram_pulse_ticker() is False
        runner._handle_subagent_pulse("subagent.start", "delegate_task", "secret goal", None, {"task_index": 0})
        assert getattr(ctx, "_pulse_ticker", None) is None
        assert runner.scheduled == []

    def test_enabled_forwards_unified_display_settings_to_ticker(self):
        adapter = _FakeProgressAdapter()
        ctx = _make_turn_ctx(
            platform=Platform.TELEGRAM,
            progress_mode="summary",
            tool_progress_enabled=True,
            log_mode_enabled=False,
            tool_progress_details=False,
            friendly_tool_labels=False,
            tool_preview_length=25,
            progress_grouping="replace",
        )
        ctx._status_adapter = adapter
        runner = _DummyProgressRunner(ctx, adapter)

        assert runner._uses_telegram_pulse_ticker() is True
        runner._handle_subagent_pulse(
            "subagent.start",
            "delegate_task",
            "Audit secret repo",
            None,
            {"task_index": 0, "goal": "Audit secret repo"},
        )
        ticker = getattr(ctx, "_pulse_ticker", None)
        assert isinstance(ticker, SubagentPulseTicker)
        assert ticker.tool_progress_details is False
        assert ticker.friendly_tool_labels is False
        assert ticker.tool_preview_length == 25
        assert ticker._preview_cap == 25
        assert ticker.tool_progress_grouping == "replace"


# ---------------------------------------------------------------------------
# 4. SubagentPulseTicker (BDD 2, 3, 4)
# ---------------------------------------------------------------------------


@pytest.fixture
def _ru_lang(monkeypatch):
    monkeypatch.setattr("agent.display.get_language", lambda: "ru")
    monkeypatch.setattr("agent.i18n.get_language", lambda: "ru")


class TestSubagentPulseTickerUnifiedDisplay:
    def test_parse_action_from_content_details_false_and_friendly_toggle(self, _ru_lang):

        # tool -> name(args) with details=False, friendly=True
        line_call = "2026-01-01 tool -> read_file(/home/srvadm/.hermes/config.yaml)"
        parsed_friendly = SubagentPulseTicker._parse_action_from_content(
            line_call, tool_progress_details=False, friendly_tool_labels=True
        )
        assert parsed_friendly == "⚙️ Чтение файла"
        assert "config.yaml" not in parsed_friendly

        # tool -> name(args) with details=False, friendly=False
        parsed_raw = SubagentPulseTicker._parse_action_from_content(
            line_call, tool_progress_details=False, friendly_tool_labels=False
        )
        assert parsed_raw == "⚙️ read_file"
        assert "config.yaml" not in parsed_raw

        # tool -> action without parens with details=False
        line_noparen = "2026-01-01 tool -> web_search query=secret"
        assert (
            SubagentPulseTicker._parse_action_from_content(
                line_noparen, tool_progress_details=False, friendly_tool_labels=True
            )
            == "⚙️ Поиск в сети"
        )

        # result ... with details=False
        line_res = "2026-01-01 result secret_token_value_123"
        parsed_res = SubagentPulseTicker._parse_action_from_content(
            line_res, tool_progress_details=False, friendly_tool_labels=True
        )
        assert parsed_res == "⚙️ Обработка"
        assert "secret_token_value_123" not in parsed_res

    def test_parse_action_from_content_preview_length_cap(self, _ru_lang):
        long_arg = "x" * 60
        line_call = f"2026-01-01 tool -> read_file({long_arg})"

        # preview_length=0 -> fallback cap 40
        parsed_default = SubagentPulseTicker._parse_action_from_content(
            line_call, tool_progress_details=True, friendly_tool_labels=True, preview_length=0
        )
        assert f"({'x' * 40})" in parsed_default

        # custom preview_length=20 -> cap 20
        parsed_20 = SubagentPulseTicker._parse_action_from_content(
            line_call, tool_progress_details=True, friendly_tool_labels=False, preview_length=20
        )
        assert parsed_20 == f"⚙️ read_file ({'x' * 20})"

    def test_render_card_details_false_single_and_batch(self, _ru_lang, tmp_path: Path):
        transcript = tmp_path / "subagent.log"
        transcript.write_text("2026-01-01 tool -> read_file(/secret/path.txt)\n", encoding="utf-8")

        adapter = _FakeProgressAdapter()
        ticker = SubagentPulseTicker(
            adapter=adapter,
            chat_id="12345",
            tool_progress_details=False,
            friendly_tool_labels=True,
        )
        ticker.on_subagent_start(0, "Secret goal text", transcript_path=str(transcript))
        card_running = ticker._render_card()
        assert card_running == "⏳ 🔀 Субагент 1/1\n↳ ⚙️ Чтение файла"
        assert "Secret goal text" not in card_running
        assert "/secret/path.txt" not in card_running

        # Completed hides summary
        ticker.on_subagent_complete(0, "completed", "Sensitive summary output")
        card_done = ticker._render_card(final_status="completed")
        assert card_done == "✅ 🔀 Субагент 1/1 завершен"
        assert "Sensitive summary output" not in card_done

        # Error hides summary
        ticker_err = SubagentPulseTicker(adapter=adapter, chat_id="12345", tool_progress_details=False)
        ticker_err.on_subagent_start(0, "Secret goal")
        ticker_err.on_subagent_complete(0, "failed", "Traceback secret")
        card_err = ticker_err._render_card(final_status="failed")
        assert card_err == "❌ 🔀 Субагент 1/1 ошибка (failed)"
        assert "Traceback secret" not in card_err

        # Batch mode hides args/paths in both transcript and raw progress
        ticker_batch = SubagentPulseTicker(
            adapter=adapter,
            chat_id="12345",
            tool_progress_details=False,
            friendly_tool_labels=True,
        )
        ticker_batch.on_subagent_start(0, "Goal A", transcript_path=str(transcript))
        ticker_batch.on_subagent_start(1, "Goal B")
        ticker_batch.on_subagent_progress(1, "terminal(cat /etc/shadow)")
        card_batch = ticker_batch._render_card()
        assert "⏳ 🔀 Субагенты (2 активны):" in card_batch
        assert " • #1 ⚙️ Чтение файла" in card_batch
        assert " • #2 ⚙️ Выполнение команды" in card_batch
        assert "/secret/path.txt" not in card_batch
        assert "/etc/shadow" not in card_batch

    def test_render_card_details_true_uses_preview_cap(self, _ru_lang):
        adapter = _FakeProgressAdapter()
        ticker = SubagentPulseTicker(
            adapter=adapter,
            chat_id="12345",
            tool_progress_details=True,
            tool_preview_length=20,
        )
        assert ticker._preview_cap == 20
        ticker.on_subagent_start(0, "G" * 50)
        card_running = ticker._render_card()
        assert f"Субагент 1/1: {'G' * 20}\n" in card_running
        assert "G" * 21 not in card_running

        ticker.on_subagent_complete(0, "completed", "S" * 50)
        card_done = ticker._render_card(final_status="completed")
        assert card_done == f"✅ 🔀 Субагент 1/1 завершен: {'S' * 20}"

        ticker_default = SubagentPulseTicker(adapter=adapter, chat_id="12345", tool_preview_length=0)
        assert ticker_default._preview_cap == 40

    @pytest.mark.asyncio
    async def test_cleanup_progress_deletes_card_on_completed(self, _ru_lang):
        adapter = _FakeProgressAdapter()
        ticker = SubagentPulseTicker(
            adapter=adapter,
            chat_id="12345",
            cleanup_progress=True,
            tool_progress_details=False,
        )
        ticker.on_subagent_start(0, "Goal")
        await ticker._send_or_edit(ticker._render_card())
        assert ticker.current_message_id == "msg-1"

        ticker.on_subagent_complete(0, "completed", "Done")
        await ticker.stop(status="completed", summary="Done")
        assert adapter.deleted == [("12345", "msg-1")]


# ---------------------------------------------------------------------------
# 5. BasePlatformAdapter.format_tool_event (BDD 1, 4)
# ---------------------------------------------------------------------------


class TestBasePlatformAdapterFormatToolEvent:
    def test_format_tool_event_details_false_and_friendly_toggle(self, monkeypatch):
        monkeypatch.setattr("agent.display.get_language", lambda: "ru")
        adapter = _FakeProgressAdapter()

        ev_builtin = ToolCallChunk(tool_name="read_file", preview="/etc/passwd", args={"path": "/etc/passwd"})
        assert (
            adapter.format_tool_event(ev_builtin, tool_progress_details=False, friendly_tool_labels=True)
            == "⚙️ Чтение файла"
        )
        assert (
            adapter.format_tool_event(ev_builtin, tool_progress_details=False, friendly_tool_labels=False)
            == "⚙️ read_file..."
        )

        ev_custom = ToolCallChunk(tool_name="mcp_custom", preview="secret", args={"q": "secret"})
        assert (
            adapter.format_tool_event(ev_custom, tool_progress_details=False, friendly_tool_labels=True)
            == "⚙️ mcp_custom..."
        )

        # Default tool_progress_details=True preserves preview
        with_details = adapter.format_tool_event(ev_builtin, mode="all", preview_max_len=40)
        assert with_details is not None
        assert "read_file" in with_details
        assert "passwd" in with_details


# ---------------------------------------------------------------------------
# 6. api_server_telegram_sync.py (BDD 3)
# ---------------------------------------------------------------------------


class TestApiServerTelegramSyncDisplaySettings:
    def _make_api_adapter(self, tmp_path: Path, user_config: dict) -> tuple[APIServerAdapter, _FakeProgressAdapter]:
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_sync_1", source="api_server")
        if hasattr(db, "apply_telegram_topic_migration"):
            db.apply_telegram_topic_migration()
        db.bind_telegram_topic(
            chat_id="-1001234567890",
            thread_id="777",
            user_id="12345",
            session_key="agent:main:telegram:dm:12345:thread:777",
            session_id="sess_sync_1",
            managed_mode="api",
            delivery_enabled=True,
        )
        telegram = _FakeProgressAdapter()
        api = APIServerAdapter(PlatformConfig(enabled=True))
        api._session_db = db
        api._ensure_session_db = lambda: db
        api.gateway_runner = SimpleNamespace(user_config=user_config)
        api._telegram_runner_and_adapter = lambda: (api.gateway_runner, telegram)
        api._pulse_prefs()["sess_sync_1"] = True
        return api, telegram

    def test_resolve_telegram_display_settings(self, tmp_path: Path):
        cfg = {
            "display": {
                "tool_progress": "all",
                "tool_progress_details": True,
                "friendly_tool_labels": True,
                "tool_preview_length": 40,
                "platforms": {
                    "telegram": {
                        "tool_progress": "summary",
                        "tool_progress_details": False,
                        "friendly_tool_labels": False,
                        "tool_preview_length": 25,
                        "tool_progress_grouping": "replace",
                        "cleanup_progress": True,
                    }
                },
            }
        }
        api, telegram = self._make_api_adapter(tmp_path, cfg)
        resolved = api._resolve_telegram_display_settings(telegram)
        assert resolved == {
            "tool_progress": "summary",
            "tool_progress_details": False,
            "friendly_tool_labels": False,
            "tool_preview_length": 25,
            "tool_progress_grouping": "replace",
            "cleanup_progress": True,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["off", "log"])
    async def test_apply_pulse_event_skips_ticker_when_tool_progress_off_or_log(self, tmp_path: Path, mode: str, monkeypatch):
        cfg = {"display": {"platforms": {"telegram": {"tool_progress": mode}}}}
        api, _telegram = self._make_api_adapter(tmp_path, cfg)
        called = []
        orig_resolve = api._resolve_telegram_display_settings
        def _spy(tg):
            res = orig_resolve(tg)
            called.append(res)
            return res
        monkeypatch.setattr(api, "_resolve_telegram_display_settings", _spy)
        api._apply_pulse_event_on_loop(
            "sess_sync_1",
            "subagent.start",
            None,
            "Secret goal",
            None,
            {"task_index": 0, "goal": "Secret goal"},
        )
        assert len(called) == 1
        assert called[0]["tool_progress"] == mode
        assert api._pulse_registry() == {}

    @pytest.mark.asyncio
    async def test_apply_pulse_event_forwards_display_settings_to_ticker(self, tmp_path: Path):
        cfg = {
            "display": {
                "platforms": {
                    "telegram": {
                        "tool_progress": "summary",
                        "tool_progress_details": False,
                        "friendly_tool_labels": False,
                        "tool_preview_length": 25,
                        "tool_progress_grouping": "replace",
                        "cleanup_progress": True,
                    }
                }
            }
        }
        api, _telegram = self._make_api_adapter(tmp_path, cfg)
        api._apply_pulse_event_on_loop(
            "sess_sync_1",
            "subagent.start",
            None,
            "Secret goal",
            None,
            {"task_index": 0, "goal": "Secret goal"},
        )
        entry = api._pulse_registry().get("sess_sync_1")
        assert entry is not None
        ticker = entry["ticker"]
        assert ticker.tool_progress_details is False
        assert ticker.friendly_tool_labels is False
        assert ticker.tool_preview_length == 25
        assert ticker._preview_cap == 25
        assert ticker.tool_progress_grouping == "replace"
        assert ticker.cleanup_progress is True

        # Clean up background start task cleanly
        start_task = entry.get("start_task")
        if start_task is not None:
            start_task.cancel()
            try:
                await start_task
            except BaseException:
                pass
        if ticker.ticker_task is not None:
            ticker.ticker_task.cancel()
            try:
                await ticker.ticker_task
            except BaseException:
                pass
