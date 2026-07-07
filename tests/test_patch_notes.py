"""Tests for data/patch_notes.py: patch_notes I/O, using a mocked SQLAlchemy Session."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from data.patch_notes import (
    create_patch_note,
    get_deferred_patch_notes,
    get_pending_patch_notes,
    update_patch_note_status,
)
from patches.patch_manager import PatchStatus
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from utils.exceptions import ConfidenceValidationError


def test_create_patch_note_writes_a_pending_row_and_commits() -> None:
    session = MagicMock()
    grounded_content = ValidatedGroundedContent(
        source_url="https://example.com/listing",
        source_type="job_listing",
        confidence=ConfidenceTier.HIGH,
        extra={"skill": "SQL"},
    )
    created_at = datetime(2026, 7, 5, 9, 0, 0)

    result = create_patch_note(
        session,
        user_id="user-1",
        origin_topic_id="topic-1",
        new_content="SQL is now more in-demand.",
        grounded_content=grounded_content,
        created_at=created_at,
    )

    assert result["user_id"] == "user-1"
    assert result["origin_topic_id"] == "topic-1"
    assert result["new_content"] == "SQL is now more in-demand."
    assert result["source_url"] == "https://example.com/listing"
    assert result["confidence"] == "high"
    assert result["status"] == PatchStatus.PENDING.value
    assert result["created_at"] == created_at
    assert result["resolved_at"] is None
    session.add.assert_called_once()
    session.commit.assert_called_once()


def test_create_patch_note_rejects_a_raw_dict_in_place_of_grounded_content() -> None:
    """A raw dict must not be silently accepted as a patch-note's grounding source."""
    session = MagicMock()
    unvalidated = {"source_url": "https://x", "confidence": "high", "skill": "SQL"}

    with pytest.raises(ConfidenceValidationError):
        create_patch_note(
            session,
            user_id="user-1",
            origin_topic_id="topic-1",
            new_content="anything",
            grounded_content=unvalidated,  # type: ignore[arg-type]
            created_at=datetime(2026, 7, 5),
        )

    session.add.assert_not_called()
    session.commit.assert_not_called()


def test_get_pending_patch_notes_filters_by_user_and_pending_status() -> None:
    session = MagicMock()
    row = MagicMock(
        id="patch-1",
        user_id="user-1",
        origin_topic_id="topic-1",
        new_content="content",
        source_url="https://x",
        confidence="high",
        status=PatchStatus.PENDING.value,
        created_at=datetime(2026, 7, 5),
        resolved_at=None,
    )
    session.query.return_value.filter.return_value.all.return_value = [row]

    result = get_pending_patch_notes(session, "user-1")

    assert len(result) == 1
    assert result[0]["id"] == "patch-1"
    assert result[0]["status"] == PatchStatus.PENDING.value


def test_get_deferred_patch_notes_is_queryable_after_a_note_is_deferred() -> None:
    """Deferred patch-notes are parked permanently and must remain queryable by a future caller."""
    session = MagicMock()
    deferred_row = MagicMock(
        id="patch-2",
        user_id="user-1",
        origin_topic_id="topic-2",
        new_content="content",
        source_url="https://y",
        confidence="low",
        status=PatchStatus.DEFERRED.value,
        created_at=datetime(2026, 7, 1),
        resolved_at=datetime(2026, 7, 5),
    )
    session.query.return_value.filter.return_value.all.return_value = [deferred_row]

    result = get_deferred_patch_notes(session, "user-1")

    assert len(result) == 1
    assert result[0]["id"] == "patch-2"
    assert result[0]["status"] == PatchStatus.DEFERRED.value
    assert result[0]["resolved_at"] == datetime(2026, 7, 5)


def test_update_patch_note_status_writes_status_and_resolved_at() -> None:
    session = MagicMock()
    row = MagicMock(status=PatchStatus.PENDING.value, resolved_at=None)
    session.get.return_value = row
    resolved_at = datetime(2026, 7, 5, 10, 0, 0)

    update_patch_note_status(session, "patch-1", PatchStatus.DELIVERED, resolved_at)

    assert row.status == PatchStatus.DELIVERED.value
    assert row.resolved_at == resolved_at
    session.commit.assert_called_once()


def test_update_patch_note_status_raises_if_patch_note_not_found() -> None:
    session = MagicMock()
    session.get.return_value = None

    with pytest.raises(ValueError):
        update_patch_note_status(
            session, "missing-id", PatchStatus.DEFERRED, datetime(2026, 7, 5)
        )

    session.commit.assert_not_called()
