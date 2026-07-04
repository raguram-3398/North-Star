"""Patch-note confidence branching and delivery ordering.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure) — a threshold lookup on an already-computed
confidence value, per Architecture_North_Star.md §2. Per PRD §7.9: high
confidence is prioritized into near-term delivery; low/uncertain
confidence asks the user to learn now or defer. Ordering among multiple
pending items on the same day is governed by hierarchy/dependency order,
not detection or arrival order.
"""

from typing import Any


def branch_by_confidence(confidence: str) -> str:
    """Return "prioritize" for high-confidence patch-notes or "ask_user"
    for low/uncertain confidence, per PRD §7.9.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def order_pending_patches(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order multiple pending patch-notes for same-day delivery by
    hierarchy/dependency position, per PRD §7.9.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
