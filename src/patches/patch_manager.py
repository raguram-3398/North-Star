"""Patch-note confidence branching, delivery ordering, and the learn-now-or-defer decision state machine."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from security.output_guard import ConfidenceTier

_AUTO_PRIORITIZED_TIERS = frozenset({ConfidenceTier.HIGH})


class PatchStatus(StrEnum):
    """Patch-note lifecycle status: pending, delivered, or deferred."""

    PENDING = "pending"
    DELIVERED = "delivered"
    DEFERRED = "deferred"


def branch_by_confidence(
    confidence: ConfidenceTier,
) -> Literal["prioritize", "needs_user_decision"]:
    """Decide whether confidence auto-prioritizes a patch for delivery or requires a user decision."""
    if confidence is ConfidenceTier.REJECT:
        raise ValueError("a patch candidate must never carry REJECT confidence")
    if confidence in _AUTO_PRIORITIZED_TIERS:
        return "prioritize"
    return "needs_user_decision"


def order_pending_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order pending items competing for the same day's delivery by hierarchy_position."""
    return sorted(items, key=lambda item: item["hierarchy_position"])


@dataclass(frozen=True)
class PatchDeliveryDecision:
    """The single action (insert_now, ask_user, or none) to take today for one user's pending patch-notes."""

    action: Literal["insert_now", "ask_user", "none"]
    patch_note_id: str | None
    insert_at_position: int | None


def decide_patch_delivery(
    pending_patches: list[dict[str, Any]],
    current_hierarchy_position: int,
) -> PatchDeliveryDecision:
    """Pick the single pending patch-note (if any) to surface today, prioritized then ordered by hierarchy."""
    prioritized = [
        patch
        for patch in pending_patches
        if branch_by_confidence(patch["confidence"]) == "prioritize"
    ]
    if prioritized:
        chosen = order_pending_items(prioritized)[0]
        return PatchDeliveryDecision(
            action="insert_now",
            patch_note_id=chosen["id"],
            insert_at_position=current_hierarchy_position,
        )

    needs_decision = [
        patch
        for patch in pending_patches
        if branch_by_confidence(patch["confidence"]) == "needs_user_decision"
    ]
    if needs_decision:
        chosen = order_pending_items(needs_decision)[0]
        return PatchDeliveryDecision(
            action="ask_user", patch_note_id=chosen["id"], insert_at_position=None
        )

    return PatchDeliveryDecision(
        action="none", patch_note_id=None, insert_at_position=None
    )


@dataclass(frozen=True)
class PatchDecisionState:
    """Immutable snapshot of one low/uncertain-confidence patch-note's learn-now-or-defer decision."""

    patch_note_id: str
    resolved: bool = False
    status: PatchStatus | None = None
    resolved_at: datetime | None = None


def resolve_patch_decision(
    state: PatchDecisionState,
    user_choice: Literal["learn_now", "defer"],
    resolved_at: datetime,
) -> PatchDecisionState:
    """Resolve a pending PatchDecisionState to DELIVERED (learn now) or DEFERRED, given the user's choice."""
    if state.resolved:
        raise ValueError(
            "resolve_patch_decision called on an already-resolved state "
            f"(patch_note_id={state.patch_note_id!r})"
        )
    new_status = (
        PatchStatus.DELIVERED if user_choice == "learn_now" else PatchStatus.DEFERRED
    )
    return PatchDecisionState(
        patch_note_id=state.patch_note_id,
        resolved=True,
        status=new_status,
        resolved_at=resolved_at,
    )
