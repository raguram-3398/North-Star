"""progress_log I/O.

Per Architecture_North_Star.md §5: the canonical store for verification
results, hands-on/review outcomes, reflection entries, and timing — feeds
pace calculation. This module only reads and writes rows; deciding what
content each step contains is `agents/coaching_pace_agent.py`'s job.

Write-only: no code path anywhere reads this log back (pace calculation
is fed by `data/pace_snapshots.py`, not by re-querying these rows), so
this module exposes only `log_progress_step`. Sessions are passed in by
the caller (dependency injection), matching `data/roles_cache.py`'s
established pattern.
"""

from datetime import UTC, datetime

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
