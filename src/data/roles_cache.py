"""Read and write structured, cron-refreshed market data per role, used as fallback data and a normalization anchor."""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models.schemas import RolesCache
from security.output_guard import ValidatedGroundedContent
from utils.exceptions import ConfidenceValidationError

STALENESS_FLOOR_DAYS = 30


def get_role(session: Session, role_name: str) -> dict[str, Any] | None:
    """Read a single role's cached market data, or None if no entry exists."""
    row = session.get(RolesCache, role_name)
    if row is None:
        return None
    return {
        "role_name": row.role_name,
        "core_skills": row.core_skills,
        "emerging_skills": row.emerging_skills,
        "last_updated": row.last_updated,
    }


def upsert_role(
    session: Session,
    role_name: str,
    core_skills: list[ValidatedGroundedContent],
    emerging_skills: list[ValidatedGroundedContent],
) -> None:
    """Write or refresh a role's cached market data from validated skill entries, stamping the current refresh time."""
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    serialized_core_skills = [_to_skill_entry(item) for item in core_skills]
    serialized_emerging_skills = [_to_skill_entry(item) for item in emerging_skills]
    statement = (
        pg_insert(RolesCache)
        .values(
            role_name=role_name,
            core_skills=serialized_core_skills,
            emerging_skills=serialized_emerging_skills,
            last_updated=now,
        )
        .on_conflict_do_update(
            index_elements=[RolesCache.role_name],
            set_={
                "core_skills": serialized_core_skills,
                "emerging_skills": serialized_emerging_skills,
                "last_updated": now,
            },
        )
    )
    session.execute(statement)
    session.commit()


def _to_skill_entry(item: ValidatedGroundedContent) -> dict[str, Any]:
    """Serialize one validated grounding result into the roles_cache skill-entry JSON shape."""
    if not isinstance(item, ValidatedGroundedContent):
        raise ConfidenceValidationError(
            f"roles_cache skill entry must be a ValidatedGroundedContent, "
            f"got {type(item).__name__!r}"
        )
    if "skill" not in item.extra:
        raise ConfidenceValidationError(
            "ValidatedGroundedContent is missing a 'skill' name in its "
            "extra field — cannot serialize to a roles_cache skill entry"
        )
    return {
        "skill": item.extra["skill"],
        "source_url": item.source_url,
        "confidence": item.confidence.value,
    }


def is_stale(
    last_updated: datetime,
    reference_time: datetime,
    max_age_days: int = STALENESS_FLOOR_DAYS,
) -> bool:
    """Determine whether a cached role's last_updated is past the staleness floor."""
    return reference_time - last_updated > timedelta(days=max_age_days)
