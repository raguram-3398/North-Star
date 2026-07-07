"""Read and write pace_snapshots rows, the rolling-window input used to detect sustained pace drift."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from models.schemas import PaceSnapshot


def write_pace_snapshot(
    session: Session,
    user_id: str,
    topic_id: str,
    topic_score: float,
    timing_ratio: float,
    days_taken: int,
    days_expected: int,
) -> None:
    """Append one pace-snapshot row for a just-completed topic."""
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    session.add(
        PaceSnapshot(
            user_id=user_id,
            topic_id=topic_id,
            topic_score=topic_score,
            timing_ratio=timing_ratio,
            days_taken=days_taken,
            days_expected=days_expected,
            computed_at=now,
        )
    )
    session.commit()


def get_pace_snapshot_history(session: Session, user_id: str) -> list[dict[str, Any]]:
    """Read a user's full pace-snapshot history in chronological order, leaving windowing and signal computation to the caller."""
    rows = (
        session.query(PaceSnapshot)
        .filter(PaceSnapshot.user_id == user_id)
        .order_by(PaceSnapshot.computed_at)
        .all()
    )
    return [
        {
            "topic_score": row.topic_score,
            "timing_ratio": row.timing_ratio,
            "days_taken": row.days_taken,
            "days_expected": row.days_expected,
            "computed_at": row.computed_at,
        }
        for row in rows
    ]
