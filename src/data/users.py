"""users I/O.

Per Architecture_North_Star.md §5. `get_user` is a plain read (consumed by
`agents/coaching_pace_agent.py` to look up `resolved_role` for enrichment
selection); `extend_pacing` is the write path for the sustained-behind
pacing-extension mechanism (§7.8) — see `models/schemas.py`'s `User` for
the new `pace_extension_days` column this task added.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern.
"""

from typing import Any

from sqlalchemy.orm import Session

from models.schemas import User


def _to_dict(row: User) -> dict[str, Any]:
    return {
        "id": row.id,
        "background": row.background,
        "current_job": row.current_job,
        "years_experience": row.years_experience,
        "prior_self_study": row.prior_self_study,
        "available_time_per_week": row.available_time_per_week,
        "resolved_role": row.resolved_role,
        "role_confidence": row.role_confidence,
        "pacing_profile": row.pacing_profile,
        "pace_extension_days": row.pace_extension_days,
        "created_at": row.created_at,
    }


def get_user(session: Session, user_id: str) -> dict[str, Any] | None:
    """Read a single user's profile, or None if no entry exists."""
    row = session.get(User, user_id)
    if row is None:
        return None
    return _to_dict(row)


def extend_pacing(session: Session, user_id: str, extension_days: int) -> int:
    """Add `extension_days` to `user_id`'s accumulated
    `pace_extension_days` (PRD §7.8's sustained-behind branch: "pacing
    extends only — outline content is never reduced"). Returns the new
    total. Commits the transaction.

    Raises `ValueError` if `user_id` does not exist, or if
    `extension_days` is not positive — this function only ever extends,
    matching "content is never reduced" applied to the pacing budget
    itself, not just outline rows.
    """
    if extension_days <= 0:
        raise ValueError(f"extension_days must be positive, got {extension_days}")
    row = session.get(User, user_id)
    if row is None:
        raise ValueError(f"user {user_id!r} not found")
    row.pace_extension_days = row.pace_extension_days + extension_days
    session.commit()
    return row.pace_extension_days
