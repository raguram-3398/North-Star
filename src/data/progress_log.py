"""progress_log I/O.

Per Architecture_North_Star.md §5: the canonical store for verification
results, hands-on/review outcomes, reflection entries, and timing — feeds
pace calculation. This module only reads and writes rows; deciding what
content each step contains is `agents/coaching_pace_agent.py`'s job.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern — the original stub's
`get_progress_for_topic(topic_id)` had no `session` parameter at all,
which couldn't have done real I/O; corrected here to match every other
I/O module's convention.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from models.schemas import ProgressLog

# The 7 step names PRD §7.6's hands-on-eligible day structure names,
# matching Architecture §5's `step TEXT` column comment exactly.
VALID_STEPS = frozenset(
    {
        "summary",
        "theory",
        "hands_on",
        "review",
        "reflection",
        "verification",
        "preview",
    }
)


def log_progress_step(
    session: Session,
    user_id: str,
    topic_id: str,
    day_number: int,
    step: str,
    reflection_text: str | None = None,
) -> None:
    """Record a single day/step entry (summary/theory/hands_on/review/
    reflection/verification/preview) to the progress log.

    Raises `ValueError` if `step` is not one of `VALID_STEPS` — never
    silently accepts an unrecognized step name. `created_at` is stamped
    here as naive UTC, matching `data/roles_cache.py`'s established
    timestamp convention. Commits the transaction.
    """
    if step not in VALID_STEPS:
        raise ValueError(f"step must be one of {sorted(VALID_STEPS)}, got {step!r}")
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    session.add(
        ProgressLog(
            user_id=user_id,
            topic_id=topic_id,
            day_number=day_number,
            step=step,
            reflection_text=reflection_text,
            created_at=now,
        )
    )
    session.commit()


def get_progress_for_topic(session: Session, topic_id: str) -> list[dict[str, Any]]:
    """Read all recorded progress-log entries for a given topic, ordered
    by day_number then created_at.
    """
    rows = (
        session.query(ProgressLog)
        .filter(ProgressLog.topic_id == topic_id)
        .order_by(ProgressLog.day_number, ProgressLog.created_at)
        .all()
    )
    return [
        {
            "user_id": row.user_id,
            "topic_id": row.topic_id,
            "day_number": row.day_number,
            "step": row.step,
            "reflection_text": row.reflection_text,
            "created_at": row.created_at,
        }
        for row in rows
    ]
