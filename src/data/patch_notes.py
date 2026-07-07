"""Read and write patch_notes rows, including the creation path significant-event detection needs."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from models.schemas import PatchNote
from patches.patch_manager import PatchStatus
from security.output_guard import ValidatedGroundedContent
from utils.exceptions import ConfidenceValidationError


def _to_dict(row: PatchNote) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "origin_topic_id": row.origin_topic_id,
        "new_content": row.new_content,
        "source_url": row.source_url,
        "confidence": row.confidence,
        "status": row.status,
        "created_at": row.created_at,
        "resolved_at": row.resolved_at,
    }


def create_patch_note(
    session: Session,
    user_id: str,
    origin_topic_id: str,
    new_content: str,
    grounded_content: ValidatedGroundedContent,
    created_at: datetime,
) -> dict[str, Any]:
    """Insert a new pending patch-note row from an already-validated grounded content object and return it."""
    if not isinstance(grounded_content, ValidatedGroundedContent):
        raise ConfidenceValidationError(
            "patch_notes source content must be a ValidatedGroundedContent, "
            f"got {type(grounded_content).__name__!r}"
        )
    row = PatchNote(
        id=uuid.uuid4(),
        user_id=user_id,
        origin_topic_id=origin_topic_id,
        new_content=new_content,
        source_url=grounded_content.source_url,
        confidence=grounded_content.confidence.value,
        status=PatchStatus.PENDING.value,
        created_at=created_at,
        resolved_at=None,
    )
    session.add(row)
    result = _to_dict(row)
    session.commit()
    return result


def get_pending_patch_notes(session: Session, user_id: str) -> list[dict[str, Any]]:
    """Read every pending patch-note for a user."""
    rows = (
        session.query(PatchNote)
        .filter(
            PatchNote.user_id == user_id,
            PatchNote.status == PatchStatus.PENDING.value,
        )
        .all()
    )
    return [_to_dict(row) for row in rows]


def get_deferred_patch_notes(session: Session, user_id: str) -> list[dict[str, Any]]:
    """Read every permanently deferred patch-note for a user."""
    rows = (
        session.query(PatchNote)
        .filter(
            PatchNote.user_id == user_id,
            PatchNote.status == PatchStatus.DEFERRED.value,
        )
        .all()
    )
    return [_to_dict(row) for row in rows]


def update_patch_note_status(
    session: Session,
    patch_note_id: str,
    status: PatchStatus,
    resolved_at: datetime | None,
) -> None:
    """Apply a status change and resolution timestamp to an existing patch-note row."""
    row = session.get(PatchNote, patch_note_id)
    if row is None:
        raise ValueError(f"patch_note {patch_note_id!r} not found")
    row.status = status.value
    row.resolved_at = resolved_at
    session.commit()
