"""Cached-fallback and general-knowledge-only floor paths of the confidence ladder, operating only on already-cached roles data."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from data.roles_cache import get_role, is_stale
from security.output_guard import (
    ConfidenceTier,
    ValidatedGroundedContent,
    validate_output_object,
)

CACHED_SOURCE_TYPE = "roles_cache-cached"


@dataclass(frozen=True)
class CachedFallbackResult:
    """The cached-fallback rung, returned whenever the roles cache has any entry at all for the role, stale or not."""

    role_name: str
    core_skills: list[ValidatedGroundedContent]
    emerging_skills: list[ValidatedGroundedContent]
    last_updated: datetime
    is_stale: bool


@dataclass(frozen=True)
class GeneralKnowledgeFloorResult:
    """The honest, unsourced floor of the confidence ladder, used when the roles cache has no entry at all for the role."""

    role_name: str
    confidence: ConfidenceTier
    label: str


def get_cached_fallback(
    session: Session,
    role_name: str,
    reference_time: datetime,
) -> CachedFallbackResult | None:
    """Read a role's roles-cache entry and re-validate its contents at cached-low confidence, or return None if no entry exists."""
    row = get_role(session, role_name)
    if row is None:
        return None

    stale = is_stale(row["last_updated"], reference_time)
    core_skills = [
        _rehydrate_skill_entry(entry, row["last_updated"])
        for entry in row["core_skills"]
    ]
    emerging_skills = [
        _rehydrate_skill_entry(entry, row["last_updated"])
        for entry in row["emerging_skills"]
    ]
    return CachedFallbackResult(
        role_name=role_name,
        core_skills=core_skills,
        emerging_skills=emerging_skills,
        last_updated=row["last_updated"],
        is_stale=stale,
    )


def _rehydrate_skill_entry(
    entry: dict[str, Any], last_updated: datetime
) -> ValidatedGroundedContent:
    """Re-validate one persisted roles-cache skill entry at cached-low confidence and return it as a guard-validated object."""
    candidate = {
        "source_url": entry["source_url"],
        "source_type": CACHED_SOURCE_TYPE,
        "confidence": ConfidenceTier.CACHED_LOW.value,
        "skill": entry.get("skill"),
        "last_updated": last_updated,
    }
    return validate_output_object(candidate)


def get_general_knowledge_floor(role_name: str) -> GeneralKnowledgeFloorResult:
    """Return the final labeled, honestly unsourced floor result for a role with no cached data at all."""
    return GeneralKnowledgeFloorResult(
        role_name=role_name,
        confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
        label=(
            f"No cached or live market data is available for {role_name!r}. "
            "This content reflects general knowledge only and is not "
            "grounded in any source."
        ),
    )
