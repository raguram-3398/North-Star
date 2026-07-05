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

from dataclasses import dataclass
from datetime import datetime
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


@dataclass(frozen=True)
class PatchDeliveryDecision:
    """The single action to take today for one user's pending patch-notes
    (`decide_patch_delivery`'s return value) — never more than one action
    per call, per PRD §7.9's "which single patch-note (if any) should be
    surfaced today" framing.

    `action`:
    - `"insert_now"`: a high-confidence patch was auto-prioritized (no
      user turn required). `patch_note_id` names it;
      `insert_at_position` echoes back the caller's own
      `current_hierarchy_position` — so this object alone tells a caller
      "insert this patch-note's content at/near the user's current
      position now" without needing to re-derive the position.
    - `"ask_user"`: no pending patch prioritized, but at least one needs a
      user decision — `patch_note_id` names the one to ask about first
      (same hierarchy ordering as the prioritized branch). The caller is
      expected to start a `PatchDecisionState` for it.
      `insert_at_position` is always `None` on this branch (nothing is
      inserted until the user actually chooses "learn now").
    - `"none"`: no pending patches at all. Both id fields are `None`.
    """

    action: Literal["insert_now", "ask_user", "none"]
    patch_note_id: str | None
    insert_at_position: int | None


def decide_patch_delivery(
    pending_patches: list[dict[str, Any]],
    current_hierarchy_position: int,
) -> PatchDeliveryDecision:
    """Decide which single pending patch-note (if any) should be surfaced
    today, and what happens next (PRD §7.9).

    `pending_patches`: each dict must carry at least `"id"`,
    `"confidence"` (a `ConfidenceTier`), and `"hierarchy_position"` (its
    origin topic's position, an int). This module does no DB reads/joins
    of its own (see `order_pending_items`'s docstring above) — a real
    caller assembles this shape from `data/patch_notes.py`'s
    `get_pending_patch_notes`, joined with each row's origin topic's
    `hierarchy_position` via `data/outline_topics.py`'s `get_topic`.

    Every patch is branched via the existing `branch_by_confidence` (not
    reimplemented here). If one or more patches prioritize, the single one
    surfaced is the earliest in hierarchy/dependency order among the
    prioritized set — `order_pending_items`, the same ordering concept
    `outline/hierarchy.py`'s `insert_new_topic` renumbers by, reused
    directly rather than building a second ordering scheme. If none
    prioritize but at least one needs a user decision, the earliest (same
    ordering) of those is surfaced for an "ask" instead — never both
    branches in the same call. Any pending patch not chosen this call
    stays `PENDING`, reconsidered whenever the caller calls this again
    (typically once the day's chosen item is resolved).

    Does not write to the DB and does not change any patch's `status` —
    a pure decision only, mirroring `branch_by_confidence`'s own contract
    of "returns the branch, does not act on it."
    """
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
    """Immutable snapshot of one low/uncertain-confidence patch-note's
    learn-now-or-defer decision.

    Mirrors `security/input_gate.py`'s `OutlineConfirmationState` pattern
    — a plain, frozen dataclass advanced by a pure transition function —
    scaled down to a single yes/no ask with no bounded-round counter:
    unlike the clarify gate or outline confirmation, PRD §7.9 describes
    one decision per patch-note, not a multi-round loop. There is
    deliberately no separate `begin_*` constructor: `PatchDecisionState(
    patch_note_id=...)` already *is* the initial, unresolved state, using
    the dataclass's own field defaults — adding a one-line wrapper
    function around a plain constructor call would be pure indirection.
    """

    patch_note_id: str
    resolved: bool = False
    status: PatchStatus | None = None
    resolved_at: datetime | None = None


def resolve_patch_decision(
    state: PatchDecisionState,
    user_choice: Literal["learn_now", "defer"],
    resolved_at: datetime,
) -> PatchDecisionState:
    """Advance a `PatchDecisionState` once the user has chosen to learn
    the patch-note's content now, or defer it (PRD §7.9).

    `"learn_now"` -> `PatchStatus.DELIVERED`; `"defer"` ->
    `PatchStatus.DEFERRED` (parked permanently, no expiry — see
    `data/patch_notes.py`'s `get_deferred_patch_notes`, the query function
    a future surfacing feature would use to find these again).

    Pure — does not write to the DB, like every other function in this
    module; the caller applies the returned state's `status`/
    `resolved_at` via `data/patch_notes.py`'s `update_patch_note_status`.
    Generates no conversational prompt text and makes no LLM call for the
    "learn now or defer" ask itself — that is UI-layer work with no home
    yet, explicitly out of scope for this task.

    `resolved_at` is an explicit parameter, not a call to the wall clock
    internally, keeping this function deterministic and testable like
    every other pure module in this codebase (e.g.
    `data/roles_cache.py`'s `is_stale`).

    Raises `ValueError` if `state` is already resolved.
    """
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
