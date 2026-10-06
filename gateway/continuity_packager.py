"""Continuity Packager for Smart-Pause Auto-Hygiene.

Serializes and restores task continuity across seamless session rotations.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
    re.compile(r"\b\d{9,10}:[a-zA-Z0-9_\-]{35}\b"),
]


def scrub_secrets(text: str) -> str:
    """Mask credentials and tokens in text."""
    if not text:
        return ""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED_SECRET]", result)
    return result


@dataclass
class ContinuityCard:
    """Structured snapshot of task context preserved across session rotations."""

    parent_session_id: str = ""
    topic_title: str = ""
    summary: str = ""
    recent_artifacts: List[str] = field(default_factory=list)
    key_decisions: List[str] = field(default_factory=list)
    pending_actions: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContinuityCard:
        return cls(
            parent_session_id=data.get("parent_session_id", ""),
            topic_title=data.get("topic_title", ""),
            summary=data.get("summary", ""),
            recent_artifacts=list(data.get("recent_artifacts", [])),
            key_decisions=list(data.get("key_decisions", [])),
            pending_actions=list(data.get("pending_actions", [])),
            created_at=data.get("created_at", time.time()),
        )

    def to_markdown(self) -> str:
        """Render clean, high-signal context block for LLM rehydration."""
        lines = [
            "[SESSION REHYDRATION CONTEXT]",
            f"Previous Session: {self.parent_session_id}",
        ]
        if self.topic_title:
            lines.append(f"Task / Topic: {self.topic_title}")
        if self.summary:
            lines.append(f"\nSummary:\n{scrub_secrets(self.summary)}")
        if self.recent_artifacts:
            lines.append("\nKey Artifacts & Files:")
            for art in self.recent_artifacts:
                lines.append(f"- {scrub_secrets(art)}")
        if self.key_decisions:
            lines.append("\nKey Decisions & Preferences:")
            for dec in self.key_decisions:
                lines.append(f"- {scrub_secrets(dec)}")
        if self.pending_actions:
            lines.append("\nPending / Next Steps:")
            for act in self.pending_actions:
                lines.append(f"- {scrub_secrets(act)}")
        lines.append("[/SESSION REHYDRATION CONTEXT]")
        return "\n".join(lines)


class ContinuityPackager:
    """Extracts, scrubs, and packages session history into ContinuityCard."""

    def package(
        self,
        parent_session_id: str,
        topic_title: str = "",
        summary: str = "",
        artifacts: Optional[List[str]] = None,
        decisions: Optional[List[str]] = None,
        pending: Optional[List[str]] = None,
    ) -> ContinuityCard:
        card = ContinuityCard(
            parent_session_id=parent_session_id,
            topic_title=scrub_secrets(topic_title),
            summary=scrub_secrets(summary),
            recent_artifacts=[scrub_secrets(a) for a in (artifacts or [])],
            key_decisions=[scrub_secrets(d) for d in (decisions or [])],
            pending_actions=[scrub_secrets(p) for p in (pending or [])],
            created_at=time.time(),
        )
        return card
