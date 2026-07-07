"""Write-only store for verification results, hands-on/review outcomes, reflections, and timing that feeds pace calculation."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from models.schemas import ProgressLog

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
    """Record a single day/step entry to the progress log."""
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
