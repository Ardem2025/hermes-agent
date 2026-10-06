"""Tests for Smart-Pause Session Hygiene Gate, Continuity Packaging, and Rotation."""

import time
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.continuity_packager import ContinuityCard, ContinuityPackager, scrub_secrets
from gateway.session_hygiene import RehydrationInjector, SeamlessSessionRotator, SessionHygieneGate


def test_safe_pause_trigger_and_gates():
    gate = SessionHygieneGate(
        idle_seconds_threshold=600.0,
        min_messages=30,
        cooldown_seconds=7200.0,
        cooldown_messages=40,
    )

    entry = SimpleNamespace(message_count=10, metadata={})
    can_rotate, reason = gate.evaluate(entry, idle_seconds=700.0, is_generating=False, active_subagents_count=0)
    assert not can_rotate
    assert "Triggers not met" in reason

    entry.metadata = {"compacted": True}
    can_rotate, reason = gate.evaluate(entry, idle_seconds=700.0, is_generating=True, active_subagents_count=0)
    assert not can_rotate
    assert "currently generating" in reason

    can_rotate, reason = gate.evaluate(entry, idle_seconds=700.0, is_generating=False, active_subagents_count=2)
    assert not can_rotate
    assert "running subagents" in reason

    can_rotate, reason = gate.evaluate(entry, idle_seconds=300.0, is_generating=False, active_subagents_count=0)
    assert not can_rotate
    assert "idle time" in reason

    can_rotate, reason = gate.evaluate(entry, idle_seconds=650.0, is_generating=False, active_subagents_count=0)
    assert can_rotate
    assert "Eligible for rotation" in reason

    entry.metadata["last_rotated_at"] = time.time()
    can_rotate, reason = gate.evaluate(entry, idle_seconds=700.0, is_generating=False, active_subagents_count=0)
    assert not can_rotate
    assert "cooldown active" in reason


def test_continuity_card_serialization_and_scrubbing():
    packager = ContinuityPackager()
    card = packager.package(
        parent_session_id="sess_old_123",
        topic_title="Deploy Task with sk-1234567890abcdefghijklmn",
        summary="Bearer secret_token_value_abc1234567890def",
        artifacts=["/app/config.yaml", "token: 123456789:ABCDefGhIjKlMnOpQrStUvWxYz123456789"],
        decisions=["Use PostgreSQL"],
        pending=["Run migrations"],
    )

    assert "sk-" not in card.topic_title
    assert "[REDACTED_SECRET]" in card.topic_title
    assert "Bearer secret" not in card.summary
    assert "[REDACTED_SECRET]" in card.summary
    assert "[REDACTED_SECRET]" in card.recent_artifacts[1]

    d = card.to_dict()
    restored = ContinuityCard.from_dict(d)
    assert restored.parent_session_id == "sess_old_123"
    assert len(restored.pending_actions) == 1

    md = restored.to_markdown()
    assert "[SESSION REHYDRATION CONTEXT]" in md
    assert "Run migrations" in md
    assert "[/SESSION REHYDRATION CONTEXT]" in md


@pytest.mark.asyncio
async def test_seamless_session_rotator():
    rotator = SeamlessSessionRotator()

    old_entry = SimpleNamespace(
        session_id="sid_old_999",
        message_count=45,
        metadata={"chat_id": 12345, "thread_id": 55, "last_synced_message_id": 140},
    )

    new_entry = SimpleNamespace(
        session_id="sid_new_100",
        message_count=0,
        parent_session_id="sid_old_999",
        metadata={},
    )

    mock_store = MagicMock()
    mock_store.rotate_session_with_continuity = AsyncMock(return_value=new_entry)

    mock_telegram_sync = AsyncMock()

    card = ContinuityCard(parent_session_id="sid_old_999", topic_title="Refactoring")

    result = await rotator.rotate(
        session_store=mock_store,
        session_key="telegram:dm:12345:55",
        current_entry=old_entry,
        card=card,
        telegram_sync_fn=mock_telegram_sync,
        source="telegram",
    )

    assert result.session_id == "sid_new_100"
    mock_store.rotate_session_with_continuity.assert_awaited_once()
    mock_telegram_sync.assert_awaited_once_with("telegram", new_entry, reason="smart_pause_hygiene")


def test_rehydration_injector_and_user_notice():
    injector = RehydrationInjector()
    card = ContinuityCard(
        parent_session_id="sid_abc",
        topic_title="Feature X",
        pending_actions=["Тестирование API"],
    )

    preamble = injector.format_system_preamble(card)
    assert "[SESSION REHYDRATION CONTEXT]" in preamble
    assert "Feature X" in preamble

    notice = injector.format_user_notice(card)
    assert "Диалог аккуратно обновлён" in notice
    assert "Тестирование API" in notice
