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


# Deliberately no read function here yet: reconstructing the rolling
# window `pace/calculator.py`'s `detect_sustained_drift` needs (the
# combined pace signal per snapshot, not bare topic_score/timing_ratio)
# is part of *acting* on drift — explicitly out of scope for this task
# (see agents/coaching_pace_agent.py's module docstring). Adding a read
# function now, before its actual consumer is built, risks guessing at a
# shape the next task hasn't decided yet.
