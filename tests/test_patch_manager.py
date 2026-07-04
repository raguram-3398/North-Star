"""Tests for patches/patch_manager.py — confidence branching and delivery
ordering.
"""

import pytest

from patches.patch_manager import (
    PatchStatus,
    branch_by_confidence,
    mark_patch_deferred,
    mark_patch_delivered,
    order_pending_items,
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


def test_mark_patch_delivered_sets_status_and_preserves_other_fields() -> None:
    patch = {
        "id": "patch-1",
        "origin_topic_id": "topic-1",
        "confidence": ConfidenceTier.HIGH,
        "status": PatchStatus.PENDING,
        "hierarchy_position": 3,
    }

    result = mark_patch_delivered(patch)

    assert result["status"] == PatchStatus.DELIVERED
    assert result["id"] == "patch-1"
    assert result["origin_topic_id"] == "topic-1"
    assert result["confidence"] == ConfidenceTier.HIGH
    assert result["hierarchy_position"] == 3


def test_mark_patch_deferred_sets_status_and_preserves_other_fields() -> None:
    patch = {
        "id": "patch-2",
        "origin_topic_id": "topic-2",
        "confidence": ConfidenceTier.LOW,
        "status": PatchStatus.PENDING,
        "hierarchy_position": 5,
    }

    result = mark_patch_deferred(patch)

    assert result["status"] == PatchStatus.DEFERRED
    assert result["id"] == "patch-2"
    assert result["origin_topic_id"] == "topic-2"


def test_mark_patch_never_touches_origin_topic_status_field() -> None:
    """Even if a patch dict happens to carry a field describing its
    origin topic's status, marking the patch delivered/deferred must
    never alter it — a patch-note is always independently tracked
    (PRD §7.9, CLAUDE.md guardrail #5).
    """
    patch = {
        "id": "patch-3",
        "origin_topic_id": "topic-3",
        "origin_topic_status": "completed",
        "status": PatchStatus.PENDING,
        "hierarchy_position": 1,
    }

    delivered = mark_patch_delivered(patch)
    deferred = mark_patch_deferred(patch)

    assert delivered["origin_topic_status"] == "completed"
    assert deferred["origin_topic_status"] == "completed"


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
