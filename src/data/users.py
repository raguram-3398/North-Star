"""CRUD I/O for the users table: profile creation, resolved-role writes, and pacing-extension tracking."""

import uuid
from datetime import UTC, datetime
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


def create_user(
    session: Session,
    background: str | None,
    current_job: str | None,
    years_experience: int | None,
    prior_self_study: str | None,
    available_time_per_week: int | None,
) -> dict[str, Any]:
    """Insert a new users row from Intake's collected profile fields and return it."""
    row = User(
        id=uuid.uuid4(),
        background=background,
        current_job=current_job,
        years_experience=years_experience,
        prior_self_study=prior_self_study,
        available_time_per_week=available_time_per_week,
        created_at=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
    )
    session.add(row)
    session.commit()
    return _to_dict(row)


def set_resolved_role(
    session: Session,
    user_id: str,
    resolved_role: str,
    role_confidence: str,
) -> None:
    """Persist the resolved role and its confidence tier onto user_id's profile row."""
    row = session.get(User, user_id)
    if row is None:
        raise ValueError(f"user {user_id!r} not found")
    row.resolved_role = resolved_role
    row.role_confidence = role_confidence
    session.commit()


def get_user(session: Session, user_id: str) -> dict[str, Any] | None:
    """Read a single user's profile, or None if no entry exists."""
    row = session.get(User, user_id)
    if row is None:
        return None
    return _to_dict(row)


def extend_pacing(session: Session, user_id: str, extension_days: int) -> int:
    """Add extension_days to user_id's accumulated pace_extension_days and return the new total."""
    if extension_days <= 0:
        raise ValueError(f"extension_days must be positive, got {extension_days}")
    row = session.get(User, user_id)
    if row is None:
        raise ValueError(f"user {user_id!r} not found")
    row.pace_extension_days = row.pace_extension_days + extension_days
    session.commit()
    return row.pace_extension_days
