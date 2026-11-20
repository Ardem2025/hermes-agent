"""Process-local thread-safe monotonic flood-control cooldown registry for Telegram.

Shared between the live gateway ``TelegramAdapter`` and the out-of-process ``standalone``
sender (which runs inside ``asyncio.run()`` / worker threads with a separate event loop,
so ``loop.time()`` cannot be shared across lanes).
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any, Optional

from plugins.platforms.telegram.telegram_ids import normalize_telegram_chat_id

_MIN_FLOOD_COOLDOWN_SECS = 1.0
_MAX_FLOOD_COOLDOWN_SECS = 86400.0

_LOCK = threading.Lock()
_COOLDOWNS: dict[tuple[str, str], float] = {}


def token_fingerprint(token: str | None) -> str:
    """Short SHA-256 fingerprint of a bot token, or ``""`` when absent."""
    raw = (token or "").strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cooldown_key(token: str | None, chat_id: Any) -> tuple[str, str]:
    return (token_fingerprint(token), str(normalize_telegram_chat_id(chat_id)))


def _clamp_wait(wait_seconds: float | int) -> float:
    try:
        wait = float(wait_seconds)
    except (TypeError, ValueError):
        wait = _MIN_FLOOD_COOLDOWN_SECS
    if not math.isfinite(wait):
        wait = _MIN_FLOOD_COOLDOWN_SECS
    return max(_MIN_FLOOD_COOLDOWN_SECS, min(wait, _MAX_FLOOD_COOLDOWN_SECS))


def record_send_flood_cooldown(token: str | None, chat_id: Any, wait_seconds: float | int) -> float:
    """Arm or extend (monotonically) the flood cooldown for ``(token, chat_id)``."""
    clamped = _clamp_wait(wait_seconds)
    key = _cooldown_key(token, chat_id)
    with _LOCK:
        now = time.monotonic()
        for stale in [k for k, deadline in _COOLDOWNS.items() if deadline <= now]:
            del _COOLDOWNS[stale]
        _COOLDOWNS[key] = max(_COOLDOWNS.get(key, 0.0), now + clamped)
    return clamped


def send_flood_cooldown_remaining(token: str | None, chat_id: Any) -> Optional[float]:
    """Seconds remaining in the active flood cooldown for ``(token, chat_id)``, or ``None``."""
    key = _cooldown_key(token, chat_id)
    with _LOCK:
        deadline = _COOLDOWNS.get(key)
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining > 0:
            return remaining
        _COOLDOWNS.pop(key, None)
        return None


def clear_send_flood_cooldowns() -> None:
    """Clear all recorded flood cooldowns (for test isolation)."""
    with _LOCK:
        _COOLDOWNS.clear()
