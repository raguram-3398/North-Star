"""Tests for patches/patch_manager.py — confidence branching and delivery
ordering.
"""

from datetime import datetime

import pytest

from patches.patch_manager import (
    PatchDecisionState,
    PatchStatus,
    branch_by_confidence,
    decide_patch_delivery,
    order_pending_items,
    resolve_patch_decision,
)
from security.output_guard import ConfidenceTier


def test_high_confidence_branches_to_prioritize() -> None:
    assert branch_by_confidence(ConfidenceTier.HIGH) == "prioritize"


@pytest.mark.parametrize(
    "confidence",
    [
        ConfidenceTier.MEDIUM,
        ConfidenceTier.LOW,
        ConfidenceTier.CACHED_LOW,
        ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
    ],
)
def test_non_high_confidence_branches_to_needs_user_decision(
    confidence: ConfidenceTier,
) -> None:
    assert branch_by_confidence(confidence) == "needs_user_decision"


def test_reject_confidence_is_rejected_outright() -> None:
    with pytest.raises(ValueError):
        branch_by_confidence(ConfidenceTier.REJECT)


def test_order_pending_items_orders_multiple_patches_by_hierarchy_position() -> None:
    """Input order (simulating arbitrary detection/arrival order) must
    not affect the result — only hierarchy_position does (PRD §7.9).
    """
    patch_c = {"id": "patch-c", "hierarchy_position": 7}
    patch_a = {"id": "patch-a", "hierarchy_position": 2}
    patch_b = {"id": "patch-b", "hierarchy_position": 4}

    result = order_pending_items([patch_c, patch_a, patch_b])

    assert [item["id"] for item in result] == ["patch-a", "patch-b", "patch-c"]


def test_order_pending_items_mixes_a_patch_and_a_regular_topic() -> None:
    """A patch-note and a regular topic competing for the same day are
    ordered together purely by hierarchy_position, regardless of which
    kind of item each one is (PRD §7.9/§7.6).
    """
    regular_topic = {"id": "topic-5", "kind": "topic", "hierarchy_position": 5}
    patch = {"id": "patch-x", "kind": "patch", "hierarchy_position": 2}

    result = order_pending_items([regular_topic, patch])

    assert [item["id"] for item in result] == ["patch-x", "topic-5"]


def test_order_pending_items_interleaves_multiple_patches_and_topics() -> None:
    """Two patches and two regular topics whose correct combined order
    requires alternating between kinds, not grouping same-kind items
    together first. This is the case that would catch a buggy
    implementation that sorts patches and topics separately and then
    concatenates by kind — such a bug would still pass a test with only
    one patch and one topic, since a single pair can't distinguish
    "interleaved by position" from "grouped by kind, kind order happens
    to match."
    """
    topic_a = {"id": "topic-a", "kind": "topic", "hierarchy_position": 1}
    patch_b = {"id": "patch-b", "kind": "patch", "hierarchy_position": 2}
    topic_c = {"id": "topic-c", "kind": "topic", "hierarchy_position": 3}
    patch_d = {"id": "patch-d", "kind": "patch", "hierarchy_position": 4}

    # Shuffle the input order so it isn't accidentally already sorted or
    # already grouped by kind.
    result = order_pending_items([patch_d, topic_a, patch_b, topic_c])

    assert [item["id"] for item in result] == [
        "topic-a",
        "patch-b",
        "topic-c",
        "patch-d",
    ]


# --- decide_patch_delivery -------------------------------------------------


def test_decide_patch_delivery_orders_high_confidence_by_hierarchy() -> None:
    """Two prioritized (high-confidence) patches, listed in an order that
    disagrees with their origin topics' hierarchy_position — the chosen
    one must be the earliest by hierarchy_position, not the one listed
    first, so this actually exercises order_pending_items rather than
    coincidentally passing because the two orders already agree.
    """
    patch_listed_first = {
        "id": "patch-listed-first",
        "confidence": ConfidenceTier.HIGH,
        "hierarchy_position": 9,
    }
    patch_listed_second = {
        "id": "patch-listed-second",
        "confidence": ConfidenceTier.HIGH,
        "hierarchy_position": 2,
    }

    decision = decide_patch_delivery(
        [patch_listed_first, patch_listed_second], current_hierarchy_position=15
    )

    assert decision.action == "insert_now"
    assert decision.patch_note_id == "patch-listed-second"
    assert decision.insert_at_position == 15


def test_decide_patch_delivery_asks_about_earliest_needs_user_decision_patch() -> None:
    patch_a = {
        "id": "patch-a",
        "confidence": ConfidenceTier.LOW,
        "hierarchy_position": 5,
    }
    patch_b = {
        "id": "patch-b",
        "confidence": ConfidenceTier.MEDIUM,
        "hierarchy_position": 1,
    }

    decision = decide_patch_delivery([patch_a, patch_b], current_hierarchy_position=10)

    assert decision.action == "ask_user"
    assert decision.patch_note_id == "patch-b"
    assert decision.insert_at_position is None


def test_decide_patch_delivery_returns_none_action_when_no_pending_patches() -> None:
    decision = decide_patch_delivery([], current_hierarchy_position=1)

    assert decision.action == "none"
    assert decision.patch_note_id is None
    assert decision.insert_at_position is None


def test_decide_patch_delivery_prioritizes_high_confidence_over_low() -> None:
    """A high-confidence patch is chosen even when a low-confidence patch
    has an earlier hierarchy_position — the two branches never compete on
    ordering with each other; if anything prioritizes, prioritize always
    wins, regardless of the other branch's positions.
    """
    high_confidence_patch = {
        "id": "patch-high",
        "confidence": ConfidenceTier.HIGH,
        "hierarchy_position": 8,
    }
    low_confidence_patch = {
        "id": "patch-low",
        "confidence": ConfidenceTier.LOW,
        "hierarchy_position": 1,
    }

    decision = decide_patch_delivery(
        [low_confidence_patch, high_confidence_patch], current_hierarchy_position=20
    )

    assert decision.action == "insert_now"
    assert decision.patch_note_id == "patch-high"


# --- PatchDecisionState / resolve_patch_decision ---------------------------


def test_resolve_patch_decision_learn_now_marks_delivered() -> None:
    state = PatchDecisionState(patch_note_id="patch-1")
    resolved_at = datetime(2026, 7, 5, 9, 0, 0)

    result = resolve_patch_decision(state, "learn_now", resolved_at)

    assert result.resolved is True
    assert result.status == PatchStatus.DELIVERED
    assert result.resolved_at == resolved_at
    assert result.patch_note_id == "patch-1"


def test_resolve_patch_decision_defer_marks_deferred() -> None:
    state = PatchDecisionState(patch_note_id="patch-2")
    resolved_at = datetime(2026, 7, 5, 9, 0, 0)

    result = resolve_patch_decision(state, "defer", resolved_at)

    assert result.resolved is True
    assert result.status == PatchStatus.DEFERRED
    assert result.resolved_at == resolved_at


def test_resolve_patch_decision_rejects_an_already_resolved_state() -> None:
    already_resolved = PatchDecisionState(
        patch_note_id="patch-3",
        resolved=True,
        status=PatchStatus.DELIVERED,
        resolved_at=datetime(2026, 7, 1),
    )

    with pytest.raises(ValueError):
        resolve_patch_decision(already_resolved, "defer", datetime(2026, 7, 5))
