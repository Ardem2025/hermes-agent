"""BDD tests for MarkdownV2-aware chunking (truncate_markdown_v2) and adapter/standalone integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import utf16_len
from plugins.platforms.telegram import adapter as tg_adapter_mod
from plugins.platforms.telegram import standalone as tg_standalone
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram.telegram_markdown import to_markdown_v2, truncate_markdown_v2
from tests.fakes.platforms.telegram_standin import mdv2_error


def _assert_valid_chunks(chunks: list[str], max_length: int) -> None:
    assert chunks, "Expected at least one chunk"
    for idx, chunk in enumerate(chunks):
        assert utf16_len(chunk) <= max_length, f"Chunk {idx} exceeded {max_length}: {utf16_len(chunk)}"
        err = mdv2_error(chunk)
        assert err is None, f"Chunk {idx} invalid MarkdownV2 ({err}): {chunk!r}"


def test_scenario_1_spoiler_split():
    raw = "||" + ("секретный текст " * 300).strip() + "||"
    formatted = to_markdown_v2(raw)
    assert utf16_len(formatted) > 4096

    chunks = truncate_markdown_v2(formatted, 4096, len_fn=utf16_len)
    assert len(chunks) >= 2
    _assert_valid_chunks(chunks, 4096)
    assert chunks[0].endswith("|| \\(1/" + str(len(chunks)) + "\\)")
    assert chunks[1].startswith("||")


def test_scenario_2_inline_entities_cyrillic():
    for open_m, close_m in [("_", "_"), ("*", "*"), ("~", "~"), ("__", "__")]:
        if open_m == "__":
            # Hand-crafted MarkdownV2 underline (to_markdown_v2 does not emit __)
            formatted = "__" + ("подчёркнутый кириллический текст " * 180).strip() + "__"
        else:
            raw = open_m + ("длинный кириллический отчёт " * 200).strip() + close_m
            formatted = to_markdown_v2(raw)
        assert utf16_len(formatted) > 4096
        chunks = truncate_markdown_v2(formatted, 4096, len_fn=utf16_len)
        assert len(chunks) >= 2
        _assert_valid_chunks(chunks, 4096)


def test_scenario_3_nested_lifo_and_dynamic_budget():
    # Nested *bold _italic ||spoiler||_*
    formatted = "*жирный _курсив ||" + ("секретное слово " * 20) + "||_*"
    chunks = truncate_markdown_v2(formatted, max_length=120, len_fn=utf16_len)
    assert len(chunks) >= 2
    _assert_valid_chunks(chunks, 120)
    assert "||_*" in chunks[0]
    assert chunks[1].startswith("*_||")

    # Repeated marker when top of stack differs (open_marks = ["*", "_"], then "*" pushes again)
    repeated = "*outer _inner *nested " + ("word " * 30) + "* inner_ outer*"
    chunks_rep = truncate_markdown_v2(repeated, max_length=120, len_fn=utf16_len)
    assert len(chunks_rep) >= 2
    _assert_valid_chunks(chunks_rep, 120)

    # ["__", "_"] stack adjacency must not emit "___"
    adj = "__underline _italic " + ("word " * 25) + "_ end__"
    chunks_adj = truncate_markdown_v2(adj, max_length=120, len_fn=utf16_len)
    assert len(chunks_adj) >= 2
    _assert_valid_chunks(chunks_adj, 120)


def test_scenario_4_escape_pairs_boundary():
    raw = " ".join(
        f"snake_case_{i} a*b_{i} x|y_{i} t~u_{i} c`d_{i} \\path\\{i}"
        for i in range(40)
    )
    formatted = to_markdown_v2(raw)
    chunks = truncate_markdown_v2(formatted, max_length=120, len_fn=utf16_len)
    assert len(chunks) >= 3
    _assert_valid_chunks(chunks, 120)
    for chunk in chunks:
        body = chunk.rsplit(" \\(", 1)[0]
        # Count trailing backslashes: must be even (never a dangling single backslash)
        trailing_bs = len(body) - len(body.rstrip("\\"))
        assert trailing_bs % 2 == 0, f"Dangling backslash in {chunk!r}"


def test_scenario_5_fenced_and_inline_code():
    # Fenced code block with \\, \`, \``` (not closing), and \\``` (closing)
    code_lines = [
        "```python",
        "def process_items(items):",
        "    path = 'C:\\\\dir\\\\file'",
        "    tick = '\\`inline\\`'",
        "    fake_close = '\\```not_close'",
    ] + [f"    val_{i} = {i} * 2  # indented line" for i in range(25)] + [
        "    real_end = '\\\\'```",
    ]
    # Build valid MarkdownV2 fenced block via to_markdown_v2 + explicit \\``` test
    raw_fence = "```python\n" + "\n".join(
        [f"    val_{i} = {i} * 2  # indented line with `tick` and \\slash" for i in range(20)]
    ) + "\n```"
    formatted_fence = to_markdown_v2(raw_fence)
    chunks = truncate_markdown_v2(formatted_fence, max_length=250, len_fn=utf16_len)
    assert len(chunks) >= 2
    _assert_valid_chunks(chunks, 250)
    assert chunks[0].endswith("\n```\n \\(1/" + str(len(chunks)) + "\\)")
    assert chunks[1].startswith("```python\n    val_")

    # Explicit even vs odd backslash before ``` inside fence
    manual_fence = (
        "```python\n"
        + r"    a = \`\`\`" + "\n"  # odd backslash before ``` -> does NOT close
        + ("    x = 1234567890\n" * 15)
        + "    b = \\\\```\n"  # even backslash (\\) before ``` -> closes fence
        + "after fence"
    )
    chunks_manual = truncate_markdown_v2(manual_fence, max_length=250, len_fn=utf16_len)
    assert len(chunks_manual) >= 2
    _assert_valid_chunks(chunks_manual, 250)

    # Short inline code stays intact; super-long inline code splits cleanly
    short_and_long = (
        "prefix `cmd --flag --opt` middle "
        + "`" + ("long_token_" * 35) + "`"
        + " suffix"
    )
    chunks_ic = truncate_markdown_v2(short_and_long, max_length=250, len_fn=utf16_len)
    assert len(chunks_ic) >= 2
    _assert_valid_chunks(chunks_ic, 250)
    assert any(r"`cmd \-\-flag \-\-opt`" in c or "`cmd --flag --opt`" in c for c in chunks_ic)


def test_scenario_6_links_and_blockquotes():
    # Links with escaped parens + ordinary quote
    raw_links = (
        "> " + " ".join(f"[описание (v{i})](https://example.com/a)b/{i})" for i in range(12))
        + "\n> продолжение цитаты " + ("слово " * 20)
        + "\nобычный текст после цитаты"
    )
    formatted_links = to_markdown_v2(raw_links)
    chunks_links = truncate_markdown_v2(formatted_links, max_length=120, len_fn=utf16_len)
    assert len(chunks_links) >= 2
    _assert_valid_chunks(chunks_links, 120)

    # Multi-line expandable blockquote split across lines and mid-line
    raw_eq = (
        "**> строка 1 " + ("слово " * 15)
        + "\n> строка 2 " + ("текст " * 15)
        + "\n> строка 3||"
    )
    formatted_eq = to_markdown_v2(raw_eq)
    chunks_eq = truncate_markdown_v2(formatted_eq, max_length=120, len_fn=utf16_len)
    assert len(chunks_eq) >= 3
    _assert_valid_chunks(chunks_eq, 120)
    assert chunks_eq[0].endswith("||\n \\(1/" + str(len(chunks_eq)) + "\\)")
    assert chunks_eq[1].startswith("**>")

    # Single-line long expandable blockquote
    raw_eq_single = "**> " + ("длинный текст цитаты " * 25) + "||"
    formatted_eq_single = to_markdown_v2(raw_eq_single)
    chunks_eq_single = truncate_markdown_v2(formatted_eq_single, max_length=120, len_fn=utf16_len)
    assert len(chunks_eq_single) >= 2
    _assert_valid_chunks(chunks_eq_single, 120)
    assert chunks_eq_single[0].endswith("||\n \\(1/" + str(len(chunks_eq_single)) + "\\)")
    assert chunks_eq_single[1].startswith("**> ")


def test_scenario_7_utf16_emoji_and_short_message():
    short = to_markdown_v2("Короткое *сообщение* с ||спойлером||")
    assert truncate_markdown_v2(short, 4096, len_fn=utf16_len) == [short]

    # 2100 emoji = 2100 codepoints (< 4096), 4200 UTF-16 units (> 4096)
    raw_emoji = "||" + ("😀" * 2100) + "||"
    formatted_emoji = to_markdown_v2(raw_emoji)
    assert len(formatted_emoji) < 4096
    assert utf16_len(formatted_emoji) > 4096

    chunks = truncate_markdown_v2(formatted_emoji, 4096, len_fn=utf16_len)
    assert len(chunks) == 2
    _assert_valid_chunks(chunks, 4096)


@pytest.mark.asyncio
async def test_scenario_8_adapter_and_standalone_integration(monkeypatch):
    raw = "||" + ("секретный текст " * 280) + "|| и _" + ("курсивный текст " * 120) + "_"
    sent_adapter: list[tuple[str, str]] = []

    async def fake_adapter_send(text: str, parse_mode=None, **_kw):
        err = mdv2_error(text) if parse_mode == "MarkdownV2" else None
        if err is not None:
            raise AssertionError(f"Adapter sent invalid MarkdownV2: {err}")
        sent_adapter.append((parse_mode, text))
        return MagicMock(message_id=len(sent_adapter))

    if tg_adapter_mod.ParseMode is None:
        monkeypatch.setattr(tg_adapter_mod, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML"))
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))
    adapter._rich_send_disabled = True
    adapter._bot = MagicMock()
    adapter._bot.send_message = AsyncMock(side_effect=fake_adapter_send)

    res = await adapter.send("4242", raw)
    assert res.success is True
    assert len(sent_adapter) >= 2
    assert all(pm == "MarkdownV2" for pm, _ in sent_adapter)

    # Standalone _send_telegram integration
    sent_standalone: list[tuple[str, str]] = []

    async def fake_standalone_send(chat_id, text, parse_mode=None, **_kw):
        err = mdv2_error(text) if parse_mode == "MarkdownV2" else None
        if err is not None:
            raise AssertionError(f"Standalone sent invalid MarkdownV2: {err}")
        sent_standalone.append((parse_mode, text))
        return SimpleNamespace(message_id=len(sent_standalone))

    import sys
    import types
    from plugins.platforms.telegram import telegram_outbound as tg_outbound

    if "telegram.constants" not in sys.modules:
        tg_mod = sys.modules.get("telegram") or types.ModuleType("telegram")
        tg_const = types.ModuleType("telegram.constants")
        setattr(tg_const, "ParseMode", SimpleNamespace(MARKDOWN_V2="MarkdownV2", HTML="HTML"))
        setattr(tg_mod, "constants", tg_const)
        monkeypatch.setitem(sys.modules, "telegram", tg_mod)
        monkeypatch.setitem(sys.modules, "telegram.constants", tg_const)

    fake_bot = SimpleNamespace(send_message=AsyncMock(side_effect=fake_standalone_send))
    monkeypatch.setattr(tg_standalone, "_telegram_bot", lambda *_a, **_kw: fake_bot)
    monkeypatch.setattr(tg_outbound, "send_rich_standalone", AsyncMock(return_value=None))

    st_res = await tg_standalone._send_telegram(
        token="123:abc",
        chat_id="4242",
        message=raw,
        extra={"rich_messages": False},
    )
    assert st_res["success"] is True
    assert len(sent_standalone) >= 2
    assert all(pm == "MarkdownV2" for pm, _ in sent_standalone)


def test_scenario_9_collision_and_empty_entity_prevention():
    # (a) Split right before closing _ or right after opening _
    for max_len in range(64, 110):
        s1 = "_слово1 слово2 слово3 слово4 слово5_"
        c1 = truncate_markdown_v2(s1, max_length=max_len, len_fn=utf16_len)
        _assert_valid_chunks(c1, max_len)
        for ch in c1:
            assert "__" not in ch, f"Unexpected __ collision in {ch!r}"

        # (b) Split near || boundaries
        s2 = "||секрет1 секрет2 секрет3 секрет4 секрет5||"
        c2 = truncate_markdown_v2(s2, max_length=max_len, len_fn=utf16_len)
        _assert_valid_chunks(c2, max_len)
        for ch in c2:
            assert "||||" not in ch, f"Empty spoiler in {ch!r}"

        # (c) _italic __underline___ collision
        s3 = "_курсив слово1 слово2 __подчёркнутый текст__ конец_"
        c3 = truncate_markdown_v2(s3, max_length=max_len, len_fn=utf16_len)
        _assert_valid_chunks(c3, max_len)
        for ch in c3:
            assert "___" not in ch, f"Unexpected ___ collision in {ch!r}"
