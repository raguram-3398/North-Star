"""users I/O.

Per Architecture_North_Star.md §5. `get_user` is a plain read (consumed by
`agents/coaching_pace_agent.py` to look up `resolved_role` for enrichment
selection); `extend_pacing` is the write path for the sustained-behind
pacing-extension mechanism (§7.8) — see `models/schemas.py`'s `User` for
the `pace_extension_days` column.

`create_user`/`set_resolved_role` were added by `src/main.py`'s
orchestration-wiring task — a discovered gap, not a pre-existing stub:
no function anywhere in this codebase created a `users` row at all (every
other function here assumed one already existed), and nothing wrote
`resolved_role`/`role_confidence` back after the Clarify Gate/Research
stage resolves them, despite `agents/coaching_pace_agent.py`'s
`maybe_trigger_enrichment` and `generate_closing_note` both reading
`user["resolved_role"]` as an existing precondition. Without
`set_resolved_role` being called somewhere, `resolved_role` would stay
`None` forever and both of those already-built functions would silently
never fire / always raise. Plain CRUD, not a decision point — no
confidence-ladder or source-validation gate applies to a user's own
profile fields (unlike outline items/patch-notes/grounding results),
matching CLAUDE.md guardrail #12's scope.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern.
"""

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
    """Insert a new `users` row from Intake's collected inputs (PRD §7.2:
    background, current job, years of experience, prior self-study,
    available time) and return it.

    `resolved_role`/`role_confidence`/`pacing_profile` are deliberately
    left unset (`None`) here — resolving a role is the Clarify Gate/
    Research stage's job, not Intake's (`set_resolved_role` below is the
    write path for the first two once they're known). No function
    anywhere in this codebase yet computes PRD §7.2's background-derived
    `pacing_profile` ("Output: a pacing profile... and a resolved role");
    flagged as a discovered spec gap, not solved here.
    `pace_extension_days` is left to the column's own mapped `default=0`.

    Commits the transaction.
    """
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
    """Persist the role the Clarify Gate/Research stage resolved, plus its
    confidence tier, onto `user_id`'s own profile row.

    A discovered wiring gap, not a pre-existing stub: without this being
    called, `agents/coaching_pace_agent.py`'s `maybe_trigger_enrichment`
    and `generate_closing_note` — both of which read `user["resolved_role"]`
    as an already-satisfied precondition — would never see it populated.

    Raises `ValueError` if `user_id` does not exist. Commits the
    transaction.
    """
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
