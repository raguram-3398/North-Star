"""patch_notes I/O.

Per Architecture_North_Star.md §5/§9. `create_patch_note` is the write
path significant-event detection needs (`src/cron/refresh_roles.py`);
`get_pending_patch_notes`/`get_deferred_patch_notes`/
`update_patch_note_status` exist so a future caller (the not-yet-built
delivery/decision UI layer) has real query/write functions to act on
`patches/patch_manager.py`'s `decide_patch_delivery`/
`resolve_patch_decision` decisions — neither of those pure functions does
any DB I/O itself (CLAUDE.md: pure functions stay pure), so applying their
output to a real row is always this module's job.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern.
"""

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
    """Insert a new patch-note row at `PENDING` status and return it.

    `grounded_content` must already be a `ValidatedGroundedContent`
    (CLAUDE.md guardrail #12) — its `source_url`/`confidence` are what get
    written to the row. A raw dict is structurally rejected via an
    explicit `isinstance` check, mirroring `data/roles_cache.py`'s
    `_to_skill_entry` — the same structural gate applied at this write
    boundary, not just outline items and roles_cache.

    Raises `ConfidenceValidationError` if `grounded_content` is not a
    `ValidatedGroundedContent` instance. Commits the transaction.
    """
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
    """Read every `PENDING` patch-note for `user_id` — the input a future
    caller assembles (joining in each row's origin topic's
    `hierarchy_position` via `data/outline_topics.py`'s `get_topic`) before
    calling `patches/patch_manager.py`'s `decide_patch_delivery`, which
    does no DB reads of its own.
    """
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
    """Read every `DEFERRED` patch-note for `user_id`. Deferred patch-notes
    are parked permanently, no expiry (PRD §7.9) — this is the query
    function a future goal-completion closing note or on-demand surfacing
    feature (both out of scope for this task) would call to find them.
    """
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
    """Apply a patch-note status change — the DB-write counterpart to
    `patches/patch_manager.py`'s pure `resolve_patch_decision` (which only
    computes the new status/`resolved_at`, never writes to the DB itself).

    Raises `ValueError` if `patch_note_id` does not exist. Commits the
    transaction.
    """
    row = session.get(PatchNote, patch_note_id)
    if row is None:
        raise ValueError(f"patch_note {patch_note_id!r} not found")
    row.status = status.value
    row.resolved_at = resolved_at
    session.commit()
