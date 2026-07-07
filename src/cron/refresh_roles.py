"""Shared roles-cache refresh routine used identically by the scheduled cron job and the startup staleness check."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy.orm import Session

from agents.research_outline_agent import LiveGroundingResult, ground_role
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from data.outline_topics import get_completed_topics_matching_skill
from data.patch_notes import create_patch_note
from data.roles_cache import get_role, is_stale, upsert_role
from outline.significant_event import SkillBucket, SkillSnapshot, is_significant_event
from security.output_guard import ConfidenceTier, ValidatedGroundedContent

logger = logging.getLogger(__name__)

SEED_ROLES: list[str] = [
    "Backend Engineer",
    "Frontend Engineer",
    "Data Analyst",
    "DevOps Engineer",
]

ROLE_REFRESH_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RoleRefreshResult:
    """The outcome of attempting to refresh one role within a refresh_roles_cache batch."""

    role_name: str
    status: Literal["upserted", "no_live_signal", "error"]
    detail: str | None = None
    patch_notes_created: int = 0


@dataclass(frozen=True)
class RefreshSummary:
    """The batch outcome of one refresh run, one result per role attempted in the order supplied."""

    results: list[RoleRefreshResult]

    @property
    def had_errors(self) -> bool:
        """True if any role in this batch ended in an error."""
        return any(result.status == "error" for result in self.results)


async def refresh_roles_cache(
    session: Session, role_names: list[str], reference_time: datetime
) -> RefreshSummary:
    """Re-run grounding for each given role, write any live grounding result into the roles cache, and record any resulting patch-notes."""
    results: list[RoleRefreshResult] = []
    for role_name in role_names:
        try:
            grounding_result = await asyncio.wait_for(
                ground_role(role_name, session, reference_time),
                timeout=ROLE_REFRESH_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("roles_cache refresh failed for role=%r: %s", role_name, exc)
            results.append(RoleRefreshResult(role_name, "error", str(exc)))
            continue

        if isinstance(grounding_result, LiveGroundingResult):
            previous_row = get_role(session, role_name)
            upsert_role(
                session,
                role_name,
                core_skills=grounding_result.skills,
                emerging_skills=[],
            )
            patch_notes_created = 0
            if previous_row is not None:
                patch_notes_created = create_patch_notes_for_significant_events(
                    session, role_name, previous_row, grounding_result, reference_time
                )
            results.append(
                RoleRefreshResult(
                    role_name, "upserted", patch_notes_created=patch_notes_created
                )
            )
        elif isinstance(
            grounding_result, (CachedFallbackResult, GeneralKnowledgeFloorResult)
        ):
            logger.info(
                "roles_cache refresh got no live signal for role=%r "
                "(%s) — leaving existing cache entry untouched",
                role_name,
                type(grounding_result).__name__,
            )
            results.append(RoleRefreshResult(role_name, "no_live_signal"))

    return RefreshSummary(results=results)


def _build_skill_snapshot_map(
    core_skills: list[dict[str, Any]], emerging_skills: list[dict[str, Any]]
) -> dict[str, SkillSnapshot]:
    """Build a casefolded skill-name to snapshot map from a roles-cache row's core and emerging skill lists."""
    snapshot_map: dict[str, SkillSnapshot] = {}
    for entry in core_skills:
        snapshot_map[str(entry["skill"]).casefold()] = SkillSnapshot(
            bucket=SkillBucket.CORE_SKILLS,
            confidence=ConfidenceTier(entry["confidence"]),
        )
    for entry in emerging_skills:
        snapshot_map[str(entry["skill"]).casefold()] = SkillSnapshot(
            bucket=SkillBucket.EMERGING_SKILLS,
            confidence=ConfidenceTier(entry["confidence"]),
        )
    return snapshot_map


def _build_patch_note_content(role_name: str, skill: ValidatedGroundedContent) -> str:
    skill_name = skill.extra["skill"]
    return (
        f"Market data for {role_name!r} now shows {skill_name!r} at "
        f"{skill.confidence.value!r} confidence — this topic's market "
        "relevance has increased since you completed it."
    )


def create_patch_notes_for_significant_events(
    session: Session,
    role_name: str,
    previous_row: dict[str, Any],
    grounding_result: LiveGroundingResult,
    created_at: datetime,
) -> int:
    """Diff the pre-refresh roles-cache snapshot against the freshly grounded result and create a pending patch-note for every user with a completed topic matching a skill that crossed upward in significance, returning the count created."""
    old_snapshots = _build_skill_snapshot_map(
        previous_row["core_skills"], previous_row["emerging_skills"]
    )
    new_snapshots: dict[str, SkillSnapshot] = {}
    new_skill_content: dict[str, ValidatedGroundedContent] = {}
    for skill in grounding_result.skills:
        key = str(skill.extra["skill"]).casefold()
        new_snapshots[key] = SkillSnapshot(
            bucket=SkillBucket.CORE_SKILLS, confidence=skill.confidence
        )
        new_skill_content[key] = skill

    absent = SkillSnapshot(bucket=SkillBucket.ABSENT, confidence=None)
    patch_notes_created = 0
    for key in set(old_snapshots) | set(new_snapshots):
        old_snapshot = old_snapshots.get(key, absent)
        new_snapshot = new_snapshots.get(key, absent)
        if not is_significant_event(old_snapshot, new_snapshot):
            continue

        skill_content = new_skill_content[key]
        skill_name = str(skill_content.extra["skill"])
        matching_topics = get_completed_topics_matching_skill(session, skill_name)
        for topic in matching_topics:
            create_patch_note(
                session,
                user_id=topic["user_id"],
                origin_topic_id=topic["id"],
                new_content=_build_patch_note_content(role_name, skill_content),
                grounded_content=skill_content,
                created_at=created_at,
            )
            patch_notes_created += 1

    return patch_notes_created


def get_stale_or_missing_roles(
    session: Session, role_names: list[str], reference_time: datetime
) -> list[str]:
    """Return the subset of given roles whose roles-cache entry is missing entirely or past the staleness floor."""
    stale_or_missing: list[str] = []
    for role_name in role_names:
        cached = get_role(session, role_name)
        if cached is None or is_stale(cached["last_updated"], reference_time):
            stale_or_missing.append(role_name)
    return stale_or_missing


async def check_and_refresh_stale_roles(
    session: Session, role_names: list[str], reference_time: datetime
) -> RefreshSummary:
    """Refresh only the roles among the given list whose cache entry is stale or missing, leaving already-fresh roles untouched."""
    stale_or_missing = get_stale_or_missing_roles(session, role_names, reference_time)
    if not stale_or_missing:
        return RefreshSummary(results=[])
    return await refresh_roles_cache(session, stale_or_missing, reference_time)


if __name__ == "__main__":
    import sys
    from datetime import UTC

    from db.connection import get_session

    logging.basicConfig(level=logging.INFO)

    _session = get_session()
    _reference_time = datetime.now(UTC).replace(tzinfo=None)
    _summary = asyncio.run(refresh_roles_cache(_session, SEED_ROLES, _reference_time))

    for _result in _summary.results:
        logger.info(
            "role=%r status=%r detail=%r",
            _result.role_name,
            _result.status,
            _result.detail,
        )

    if _summary.had_errors:
        sys.exit(1)
