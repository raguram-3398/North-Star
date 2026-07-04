"""Patch-note confidence branching and delivery ordering.

Pure, deterministic, no LLM calls, no DB reads/writes (CLAUDE.md: pure
functions stay pure) — a threshold lookup on an already-computed
confidence value, per Architecture_North_Star.md §2. Per PRD §7.9: high
confidence is prioritized into near-term delivery; low/uncertain
confidence asks the user to learn now or defer. Ordering among multiple
pending items on the same day is governed by hierarchy/dependency
position, not detection or arrival order.

This module never reads or writes an origin topic's completion/
verification status — a patch-note is always a separate,
independently-tracked unit (PRD §7.9's absolute rule; CLAUDE.md guardrail
#5). It also does not decide *what* content goes into a patch (Agent 1's
reasoning job) and does not detect significant events
(outline/significant_event.py) — it only decides how an already-generated
patch candidate is branched, tracked, and ordered for delivery.
"""

from enum import StrEnum
from typing import Any, Literal

from security.output_guard import ConfidenceTier

# Judgment call, flagged for review: PRD §7.9 names only "high confidence"
# (-> prioritize) and "low/uncertain confidence" (-> ask the user) as the
# two named branches. It does not pin down where MEDIUM falls on
# Architecture §8's six-tier ladder. Chosen here: only HIGH auto-
# prioritizes; MEDIUM, LOW, CACHED_LOW, and GENERAL_KNOWLEDGE_ONLY all
# route to needs_user_decision. Reasoning: PRD §7.3 defines HIGH as both
# sources agreeing, and MEDIUM as a single source or minor disagreement —
# MEDIUM is therefore not fully cross-validated and reads as "uncertain"
# in the PRD's own terms, consistent with the system's "honest about its
# own confidence" value proposition (PRD §3). This is an easy, low-risk
# constant to change if MEDIUM was intended to auto-prioritize too.
_AUTO_PRIORITIZED_TIERS = frozenset({ConfidenceTier.HIGH})


class PatchStatus(StrEnum):
    """Patch-note lifecycle status (Architecture_North_Star.md §5's
    `patch_notes.status` column: pending / delivered / deferred).
    """

    PENDING = "pending"
    DELIVERED = "delivered"
    DEFERRED = "deferred"


def branch_by_confidence(
    confidence: ConfidenceTier,
) -> Literal["prioritize", "needs_user_decision"]:
    """Decide whether a patch candidate's confidence tier auto-prioritizes
    it for near-term delivery, or requires a user decision (learn now vs.
    defer), per PRD §7.9.

    This function only returns the branch decision — it does not ask the
    user, and does not change the patch's status itself; the caller
    (Agent 2 / the UI layer) acts on the result.

    Raises ValueError if `confidence` is ConfidenceTier.REJECT — a patch
    candidate should never carry zero-signal confidence
    (security/output_guard.py rejects that tier before anything is ever
    persisted).
    """
    if confidence is ConfidenceTier.REJECT:
        raise ValueError("a patch candidate must never carry REJECT confidence")
    if confidence in _AUTO_PRIORITIZED_TIERS:
        return "prioritize"
    return "needs_user_decision"


def mark_patch_delivered(patch: dict[str, Any]) -> dict[str, Any]:
    """Mark a patch-note as delivered.

    Only this patch's own `status` field changes — every other field
    (including anything related to its origin topic) passes through
    untouched.
    """
    return {**patch, "status": PatchStatus.DELIVERED}


def mark_patch_deferred(patch: dict[str, Any]) -> dict[str, Any]:
    """Mark a patch-note as deferred: parked permanently, no expiry, per
    PRD §7.9. Deferred patches resurface at the goal-completion closing
    note, or on-demand if the user explicitly asks — this function only
    sets the status; resurfacing itself is the caller's job.

    Only this patch's own `status` field changes — every other field
    passes through untouched.
    """
    return {**patch, "status": PatchStatus.DEFERRED}


def order_pending_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order multiple pending items — patch-notes and/or regular topics —
    competing for the same day's delivery by `hierarchy_position`, never
    by detection time or arrival order (PRD §7.9's explicit rule).

    Each item is expected to carry a "hierarchy_position" key. For a
    patch-note, this is its origin topic's hierarchy_position, copied in
    by the caller — this module does no DB reads/joins itself, so it
    cannot look that up on its own. Reuses the exact same ordering
    concept as outline/hierarchy.py rather than inventing a separate one.
    """
    return sorted(items, key=lambda item: item["hierarchy_position"])
