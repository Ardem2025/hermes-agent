"""Session Hygiene and Smart-Pause Auto-Rotation Gatekeeper.

Manages autonomous background session rotation during user idle windows
to prevent LLM context degradation without disrupting user conversation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

from gateway.continuity_packager import ContinuityCard, ContinuityPackager, scrub_secrets

logger = logging.getLogger(__name__)


class SessionHygieneGate:
    """Evaluates whether an active session qualifies for Smart-Pause rotation."""

    def __init__(
        self,
        idle_seconds_threshold: float = 600.0,
        min_messages: int = 30,
        min_tokens_pct: float = 0.70,
        cooldown_seconds: float = 7200.0,
        cooldown_messages: int = 40,
    ) -> None:
        self.idle_seconds_threshold = idle_seconds_threshold
        self.min_messages = min_messages
        self.min_tokens_pct = min_tokens_pct
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_messages = cooldown_messages

    def evaluate(
        self,
        session_entry: Any,
        idle_seconds: float,
        is_generating: bool = False,
        active_subagents_count: int = 0,
        current_tokens: int = 0,
        context_length: int = 512000,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        now = now or time.time()
        metadata = getattr(session_entry, "metadata", {}) or {}
        msg_count = getattr(session_entry, "message_count", 0)

        has_compact_flag = bool(metadata.get("needs_compact") or metadata.get("compacted"))
        has_msg_threshold = msg_count >= self.min_messages
        has_token_threshold = (
            current_tokens > 0 and current_tokens >= int(self.min_tokens_pct * context_length)
        )

        triggered = has_compact_flag or has_msg_threshold or has_token_threshold
        if not triggered:
            return False, "Triggers not met (neither compaction, msg count, nor token threshold reached)"

        if is_generating:
            return False, "Session is currently generating response (turn active)"

        if active_subagents_count > 0:
            return False, f"Session has {active_subagents_count} running subagents"

        if idle_seconds < self.idle_seconds_threshold:
            return False, f"User idle time ({idle_seconds:.0f}s) < threshold ({self.idle_seconds_threshold:.0f}s)"

        last_rotated_at = float(metadata.get("last_rotated_at", 0.0))
        if last_rotated_at > 0.0:
            if (now - last_rotated_at) < self.cooldown_seconds:
                remaining = int(self.cooldown_seconds - (now - last_rotated_at))
                return False, f"Rotation cooldown active ({remaining}s remaining)"

            last_rotated_msg_count = int(metadata.get("last_rotated_msg_count", 0))
            if (msg_count - last_rotated_msg_count) < self.cooldown_messages:
                return False, f"Messages since last rotation < cooldown ({self.cooldown_messages})"

        trigger_reason = (
            "compaction" if has_compact_flag else ("msg_count" if has_msg_threshold else "token_threshold")
        )
        return True, f"Eligible for rotation: trigger={trigger_reason}, idle={idle_seconds:.0f}s"


class SeamlessSessionRotator:
    """Executes atomic session rotation and state handoff."""

    def __init__(self, packager: Optional[ContinuityPackager] = None) -> None:
        self.packager = packager or ContinuityPackager()

    async def rotate(
        self,
        session_store: Any,
        session_key: str,
        current_entry: Any,
        card: ContinuityCard,
        telegram_sync_fn: Optional[Callable[..., Any]] = None,
        source: str = "telegram",
    ) -> Any:
        parent_id = getattr(current_entry, "session_id", "")
        old_metadata = getattr(current_entry, "metadata", {}) or {}

        last_synced_id = old_metadata.get("last_synced_message_id", 0)

        new_metadata = {
            **old_metadata,
            "parent_session_id": parent_id,
            "last_rotated_at": time.time(),
            "last_rotated_msg_count": getattr(current_entry, "message_count", 0),
            "continuity_card": card.to_dict(),
            "last_synced_message_id": last_synced_id,
            "rehydration_pending": True,
        }

        new_entry = await session_store.rotate_session_with_continuity(
            session_key=session_key,
            parent_session_id=parent_id,
            metadata=new_metadata,
        )

        if telegram_sync_fn and source == "telegram":
            try:
                await telegram_sync_fn(source, new_entry, reason="smart_pause_hygiene")
            except Exception as exc:
                logger.warning("Telegram topic sync hook returned error on rotation: %s", exc)

        return new_entry


class RehydrationInjector:
    """Injects rehydration context into the new session turn."""

    def format_system_preamble(self, card: ContinuityCard) -> str:
        return card.to_markdown()

    def format_user_notice(self, card: ContinuityCard) -> str:
        action_note = f" Продолжаем с шага: {card.pending_actions[0]}." if card.pending_actions else ""
        return (
            "✨ Диалог аккуратно обновлён и очищен от технических логов: "
            f"сессия обновлена для максимальной скорости и точности ответов.{action_note} Все ваши цели и файлы на месте!"
        )
