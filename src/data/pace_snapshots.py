"""pace_snapshots I/O.

Per Architecture_North_Star.md §5: one row per completed topic —
topic_score, timing_ratio, days_taken, days_expected — the rolling-window
input `pace/calculator.py`'s `detect_sustained_drift` reads across
topics. This module only reads and writes rows; computing topic_score/
timing_ratio/the combined signal is `pace/calculator.py`'s job, and
deciding when a topic's verification is complete enough to snapshot is
`agents/coaching_pace_agent.py`'s — never reimplemented here.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern.
"""

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
    """Append one pace-snapshot row for a just-completed topic. Always
    an append (never an upsert) — each topic completion is its own
    rolling-window data point, per PRD §7.8.

    `computed_at` is stamped here as naive UTC, matching
    `data/roles_cache.py`'s established timestamp convention. Commits the
    transaction.
    """
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
    """Read every `pace_snapshots` row for `user_id`, oldest first — the
    rolling-window input `pace/calculator.py`'s `detect_sustained_drift`
    needs.

    Returns bare `topic_score`/`timing_ratio` per row (plus
    `days_taken`/`days_expected`/`computed_at`), not a combined pace
    signal — computing that from the two raw values is
    `pace/calculator.py`'s `calculate_combined_pace_signal`'s job, done by
    the caller (`agents/coaching_pace_agent.py`), never reimplemented or
    pre-computed here. No row limit is applied: `detect_sustained_drift`
    already slices its own trailing window
    (`pace_signals[-DRIFT_WINDOW_SIZE:]`), so this function does not
    guess at that window size — it returns the full history in
    chronological order and lets the caller's downstream call handle
    windowing.
    """
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
