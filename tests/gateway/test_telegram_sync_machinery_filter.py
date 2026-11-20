"""Tests for filtering machinery/tool/system/silence messages in _sync_telegram_history."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.response_filters import MACHINERY_DISPLAY_KINDS
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from hermes_state import SessionDB


def _make_adapter(tmp_path, session_id: str = "sess_sync_filter"):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="api")
    db.bind_telegram_topic(
        chat_id="139351986",
        thread_id="999",
        user_id="139351986",
        session_key=f"telegram:dm:139351986:topic:999",
        session_id=session_id,
        delivery_enabled=True,
    )

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"api_key": "123456...7890"}))
    adapter._session_db = db

    telegram_adapter = MagicMock()
    telegram_adapter.send = AsyncMock(return_value=SimpleNamespace(success=True))
    adapter._telegram_runner_and_adapter = MagicMock(return_value=(None, telegram_adapter))
    return adapter, db, telegram_adapter


@pytest.mark.asyncio
async def test_sync_skips_tool_and_system_roles_and_advances_checkpoint(tmp_path):
    session_id = "sess_roles"
    adapter, db, telegram_adapter = _make_adapter(tmp_path, session_id)

    anchor_id = db.append_message(session_id, role="user", content="Anchor", platform_message_id="tg_1")
    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=anchor_id)

    db.append_message(session_id, role="system", content="System prompt")
    db.append_message(session_id, role="tool", content='{"ok": true}', tool_name="terminal", tool_call_id="call_1")
    last_id = db.append_message(session_id, role="assistant", content="Готово")

    delivery = await adapter._sync_telegram_history(session_id)

    assert delivery == {
        "mode": "incremental",
        "status": "completed",
        "sent": 1,
        "failed": 0,
        "skipped": 2,
    }
    assert telegram_adapter.send.await_count == 1
    assert telegram_adapter.send.await_args.kwargs["content"] == "Готово"
    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["last_synced_message_id"] == last_id


@pytest.mark.asyncio
async def test_sync_skips_machinery_display_kinds_including_metadata(tmp_path):
    session_id = "sess_display_kinds"
    adapter, db, telegram_adapter = _make_adapter(tmp_path, session_id)

    anchor_id = db.append_message(session_id, role="user", content="Anchor", platform_message_id="tg_1")
    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=anchor_id)

    for kind in sorted(MACHINERY_DISPLAY_KINDS):
        db.append_message(session_id, role="user", content=f"Machinery {kind}", display_kind=kind)

    # Also verify metadata.display_kind fallback when top-level display_kind is absent
    db.get_messages = MagicMock(
        return_value=[
            {"id": anchor_id, "role": "user", "content": "Anchor", "platform_message_id": "tg_1"},
            {"id": anchor_id + 1, "role": "user", "content": "m1", "display_kind": "internal_notification"},
            {"id": anchor_id + 2, "role": "user", "content": "m2", "display_kind": "process_complete"},
            {"id": anchor_id + 3, "role": "user", "content": "m3", "display_kind": "async_delegation_complete"},
            {
                "id": anchor_id + 4,
                "role": "user",
                "content": "m4",
                "metadata": {"display_kind": "internal_notification"},
            },
            {"id": anchor_id + 5, "role": "user", "content": "Привет"},
        ]
    )

    delivery = await adapter._sync_telegram_history(session_id)

    assert delivery == {
        "mode": "incremental",
        "status": "completed",
        "sent": 1,
        "failed": 0,
        "skipped": 4,
    }
    assert telegram_adapter.send.await_count == 1
    assert telegram_adapter.send.await_args.kwargs["content"] == "👤 **User:**\nПривет"
    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["last_synced_message_id"] == anchor_id + 5


@pytest.mark.asyncio
async def test_sync_skips_internal_wake_compaction_and_footer_markers(tmp_path):
    session_id = "sess_markers"
    adapter, db, telegram_adapter = _make_adapter(tmp_path, session_id)

    anchor_id = db.append_message(session_id, role="user", content="Anchor", platform_message_id="tg_1")
    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=anchor_id)

    db.append_message(session_id, role="user", content="[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\nDone")
    db.append_message(session_id, role="user", content="[ASYNC DELEGATION COMPLETE — task_1]\nDone")
    db.append_message(session_id, role="user", content="[INTERNAL NOTIFICATION] Process exited")
    db.append_message(session_id, role="user", content="[CONTEXT COMPACTION — REFERENCE ONLY]\nSummary")
    last_id = db.append_message(
        session_id,
        role="user",
        content=f"Background task finished\n\n{INTERNAL_NOTIFICATION_FOOTER}",
    )

    delivery = await adapter._sync_telegram_history(session_id)

    assert delivery == {
        "mode": "incremental",
        "status": "completed",
        "sent": 0,
        "failed": 0,
        "skipped": 5,
    }
    assert telegram_adapter.send.await_count == 0
    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["last_synced_message_id"] == last_id


@pytest.mark.asyncio
async def test_sync_skips_silence_and_blank_but_keeps_failed_turn_and_strips_trailing_marker(tmp_path):
    session_id = "sess_silence_and_failed"
    adapter, db, telegram_adapter = _make_adapter(tmp_path, session_id)

    anchor_id = db.append_message(session_id, role="user", content="Anchor", platform_message_id="tg_1")
    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=anchor_id)

    db.append_message(session_id, role="assistant", content="   ")
    db.append_message(session_id, role="assistant", content="NO_REPLY")
    db.append_message(session_id, role="assistant", content="[SILENT]")
    db.append_message(session_id, role="assistant", content="Полезный ответ\n\nNO_REPLY")
    db.append_message(session_id, role="assistant", content="Ошибка модели", display_kind="failed_turn")
    last_id = db.append_message(session_id, role="user", content="Следующий вопрос из WebUI")

    delivery = await adapter._sync_telegram_history(session_id)

    assert delivery == {
        "mode": "incremental",
        "status": "completed",
        "sent": 3,
        "failed": 0,
        "skipped": 3,
    }
    sent_contents = [call.kwargs["content"] for call in telegram_adapter.send.await_args_list]
    assert sent_contents == [
        "Полезный ответ",
        "⚠️ Ошибка модели",
        "👤 **User:**\nСледующий вопрос из WebUI",
    ]
    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["last_synced_message_id"] == last_id


@pytest.mark.asyncio
async def test_sync_delivers_normal_webui_user_and_assistant_messages_in_order(tmp_path):
    session_id = "sess_normal_webui"
    adapter, db, telegram_adapter = _make_adapter(tmp_path, session_id)

    anchor_id = db.append_message(session_id, role="user", content="Anchor", platform_message_id="tg_1")
    db.advance_telegram_topic_sync_checkpoint(session_id=session_id, message_id=anchor_id)

    db.append_message(session_id, role="user", content="Как дела?")
    last_id = db.append_message(session_id, role="assistant", content="Всё отлично!")

    delivery = await adapter._sync_telegram_history(session_id)

    assert delivery == {
        "mode": "incremental",
        "status": "completed",
        "sent": 2,
        "failed": 0,
        "skipped": 0,
    }
    sent_contents = [call.kwargs["content"] for call in telegram_adapter.send.await_args_list]
    assert sent_contents == [
        "👤 **User:**\nКак дела?",
        "Всё отлично!",
    ]
    binding = db.get_telegram_topic_binding_by_session(session_id=session_id)
    assert binding["last_synced_message_id"] == last_id
