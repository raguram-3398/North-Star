"""Read, status-update, and insert outline_topics rows, including persisting a freshly generated or regenerated outline."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import func
from sqlalchemy.orm import Session

from models.schemas import OutlineTopic
from outline.hierarchy import augment_existing_topic, insert_new_topic
from security.output_guard import ConfidenceTier

NOT_STARTED_STATUS = "not_started"
COMPLETED_STATUS = "completed"
COMPLETED_TEST_OUT_STATUS = "completed_test_out"
_VALID_COMPLETION_STATUSES = frozenset({COMPLETED_STATUS, COMPLETED_TEST_OUT_STATUS})


@runtime_checkable
class SequencedOutlineTopic(Protocol):
    """The structural shape a sequenced outline topic must have to be persisted, checked at runtime instead of via an imported dataclass."""

    @property
    def topic_name(self) -> str: ...
    @property
    def hierarchy_position(self) -> int: ...
    @property
    def topic_group(self) -> str: ...
    @property
    def position_in_group(self) -> int: ...
    @property
    def source_url(self) -> str: ...
    @property
    def source_type(self) -> str: ...
    @property
    def confidence(self) -> ConfidenceTier: ...
    @property
    def is_enrichment(self) -> bool: ...
    @property
    def status(self) -> str: ...


def _to_dict(row: OutlineTopic) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "topic_name": row.topic_name,
        "hierarchy_position": row.hierarchy_position,
        "topic_group": row.topic_group,
        "position_in_group": row.position_in_group,
        "source_url": row.source_url,
        "source_type": row.source_type,
        "confidence": row.confidence,
        "is_enrichment": row.is_enrichment,
        "status": row.status,
        "completed_at": row.completed_at,
    }


def get_topic(session: Session, topic_id: str) -> dict[str, Any] | None:
    """Read a single outline topic, or None if no entry exists."""
    row = session.get(OutlineTopic, topic_id)
    if row is None:
        return None
    return _to_dict(row)


def get_topics_in_group(
    session: Session, user_id: str, topic_group: str
) -> list[dict[str, Any]]:
    """Read every topic in a topic group for a user, ordered by position within the group."""
    rows = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.user_id == user_id, OutlineTopic.topic_group == topic_group
        )
        .order_by(OutlineTopic.position_in_group)
        .all()
    )
    return [_to_dict(row) for row in rows]


def mark_topic_completed(
    session: Session, topic_id: str, status: str = COMPLETED_STATUS
) -> None:
    """Mark a topic completed (via regular coaching or test-out) and stamp its completion time."""
    if status not in _VALID_COMPLETION_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(_VALID_COMPLETION_STATUSES)}, "
            f"got {status!r}"
        )
    row = session.get(OutlineTopic, topic_id)
    if row is None:
        raise ValueError(f"outline topic {topic_id!r} not found")
    row.status = status
    row.completed_at = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    session.commit()


def augment_outline_topic(
    session: Session,
    topic_id: str,
    source_url: str,
    source_type: str,
    confidence: ConfidenceTier,
) -> dict[str, Any]:
    """Refresh an existing topic's source URL, source type, and confidence in place without touching its status or position."""
    row = session.get(OutlineTopic, topic_id)
    if row is None:
        raise ValueError(f"outline topic {topic_id!r} not found")

    existing_topics = [_to_dict(row)]
    refreshed_content = {
        "source_url": source_url,
        "source_type": source_type,
        "confidence": confidence.value,
    }
    augmented = augment_existing_topic(existing_topics, topic_id, refreshed_content)[0]

    row.source_url = source_url
    row.source_type = source_type
    row.confidence = confidence.value
    session.commit()
    return augmented


def insert_outline_topics(
    session: Session,
    user_id: str,
    topics: Sequence[SequencedOutlineTopic],
) -> list[dict[str, Any]]:
    """Replace a user's entire outline_topics rows with a freshly generated or regenerated set, refusing to overwrite progressed rows."""
    if not topics:
        raise ValueError("insert_outline_topics requires at least one topic")
    for topic in topics:
        if not isinstance(topic, SequencedOutlineTopic):
            raise TypeError(
                "insert_outline_topics requires already-sequenced topic "
                f"objects (e.g. InitialOutlineTopic), got {type(topic).__name__!r}"
            )

    existing_rows = (
        session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    )
    already_progressed = [
        row for row in existing_rows if row.status != NOT_STARTED_STATUS
    ]
    if already_progressed:
        raise ValueError(
            f"cannot persist a regenerated outline for user {user_id!r}: "
            f"{len(already_progressed)} existing row(s) have already "
            "progressed past 'not_started' — outline content is never "
            "overwritten once started or completed"
        )
    for row in existing_rows:
        session.delete(row)

    new_rows = [
        OutlineTopic(
            id=uuid.uuid4(),
            user_id=user_id,
            topic_name=topic.topic_name,
            hierarchy_position=topic.hierarchy_position,
            topic_group=topic.topic_group,
            position_in_group=topic.position_in_group,
            source_url=topic.source_url,
            source_type=topic.source_type,
            confidence=topic.confidence.value,
            is_enrichment=topic.is_enrichment,
            status=topic.status,
        )
        for topic in topics
    ]
    session.add_all(new_rows)
    persisted = [_to_dict(row) for row in new_rows]
    session.commit()
    return persisted


def get_completed_topics_matching_skill(
    session: Session, skill_name: str
) -> list[dict[str, Any]]:
    """Find every completed outline_topics row, across all users, whose topic name case-insensitively matches a given skill name."""
    rows = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.status.in_(_VALID_COMPLETION_STATUSES),
            func.lower(OutlineTopic.topic_name) == skill_name.lower(),
        )
        .all()
    )
    return [_to_dict(row) for row in rows]


def get_all_topics_for_user(session: Session, user_id: str) -> list[dict[str, Any]]:
    """Read every outline_topics row for a user, regardless of status or topic group."""
    rows = session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    return [_to_dict(row) for row in rows]


def has_pending_enrichment_topic(session: Session, user_id: str) -> bool:
    """True if the user already has an unresolved enrichment topic pending."""
    row = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.user_id == user_id,
            OutlineTopic.is_enrichment.is_(True),
            OutlineTopic.status.notin_(_VALID_COMPLETION_STATUSES),
        )
        .first()
    )
    return row is not None


def insert_new_outline_topic(
    session: Session,
    user_id: str,
    topic_name: str,
    topic_group: str,
    position_in_group: int,
    source_url: str,
    source_type: str,
    confidence: ConfidenceTier,
    is_enrichment: bool,
    prerequisite_topic_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Insert one new topic into a user's existing hierarchy, renumbering affected positions and persisting the new row."""
    existing_rows = (
        session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    )
    existing_topics = [_to_dict(row) for row in existing_rows]

    new_topic_id = uuid.uuid4()
    new_topic_candidate = {
        "id": new_topic_id,
        "topic_name": topic_name,
        "topic_group": topic_group,
        "position_in_group": position_in_group,
        "source_url": source_url,
        "source_type": source_type,
        "confidence": confidence.value,
        "is_enrichment": is_enrichment,
        "status": NOT_STARTED_STATUS,
    }
    renumbered = insert_new_topic(
        existing_topics, new_topic_candidate, prerequisite_topic_ids
    )
    renumbered_by_id = {topic["id"]: topic for topic in renumbered}

    for existing_row in existing_rows:
        existing_row.hierarchy_position = renumbered_by_id[existing_row.id][
            "hierarchy_position"
        ]

    new_row = OutlineTopic(
        id=new_topic_id,
        user_id=user_id,
        topic_name=topic_name,
        hierarchy_position=renumbered_by_id[new_topic_id]["hierarchy_position"],
        topic_group=topic_group,
        position_in_group=position_in_group,
        source_url=source_url,
        source_type=source_type,
        confidence=confidence.value,
        is_enrichment=is_enrichment,
        status=NOT_STARTED_STATUS,
    )
    session.add(new_row)
    session.commit()
    return _to_dict(new_row)
