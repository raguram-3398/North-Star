"""Shared roles_cache refresh function.

Called identically by both the GitHub Actions scheduled workflow
(`.github/workflows/refresh_roles.yml`, via this module's `__main__` block)
and the Streamlit startup staleness check (Architecture_North_Star.md §3,
`check_and_refresh_stale_roles` below) — the seed run, the recurring cron
job, and the startup resilience layer are all the same code path. CLAUDE.md
guardrail #9: never touch this function's core logic without checking all
call sites still work.

Deliberately not agentic: the trigger is wall-clock time (or startup
staleness), not judgment. This module owns no grounding or cross-validation
logic of its own — it orchestrates already-tested functions:
`agents/research_outline_agent.py`'s `ground_role` (the real grounding
pipeline), `data/roles_cache.py`'s `upsert_role` (the write path), and, per
Architecture §9, `outline/significant_event.py`'s upward-crossing diff plus
`data/outline_topics.py`'s `get_completed_topics_matching_skill` and
`data/patch_notes.py`'s `create_patch_note` — closing the previously-open
"nothing calls significant_event.py" gap. No LLM/Himalayas/Tavily call is
made directly by this module; every such call happens inside `ground_role`,
which already owns its own per-source timeout/error handling.
"""

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

# Per PRD §7.3's seed-role bootstrap list — matches the 4 roles this
# codebase already has real Himalayas/Tavily fixtures for
# (tests/fixtures/himalayas_search_jobs_*.txt, tests/fixtures/tavily_search_*.json).
# Do not add or invent additional roles here without first gathering real
# fixtures the way those 4 were gathered (see data/himalayas_parser.py /
# data/tavily_parser.py's module docstrings).
SEED_ROLES: list[str] = [
    "Backend Engineer",
    "Frontend Engineer",
    "Data Analyst",
    "DevOps Engineer",
]

# Judgment call, flagged for tuning: `ground_role` already wraps its own
# Himalayas/Tavily calls in EXTERNAL_CALL_TIMEOUT_SECONDS (10s each, run in
# parallel via asyncio.gather inside ground_role), so the realistic worst
# case for one role is ~10s plus in-process compute. This outer timeout is
# a second, independent guard at this orchestrator's own boundary (CLAUDE.md
# guardrail #14 applies here too, not just inside ground_role) — generous
# headroom in case a future change to ground_role's internals introduces a
# code path not covered by a per-source timeout. Unvalidated against real
# latency; tune once real refresh runs are measured (ship-day README item).
ROLE_REFRESH_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class RoleRefreshResult:
    """Outcome of attempting to refresh one role in a `refresh_roles_cache`
    batch.

    `"upserted"` — `ground_role` produced a live grounding result (high/
    medium/low confidence) and it was written via `upsert_role`.
    `"no_live_signal"` — `ground_role` fell through to a fallback rung
    (cached-fallback or the general-knowledge-only floor); nothing was
    written this cycle (see `refresh_roles_cache`'s docstring for why).
    `"error"` — `ground_role` itself raised (a data-integrity error, or the
    outer `ROLE_REFRESH_TIMEOUT_SECONDS` timeout elapsed); `detail` carries
    `str(exception)`.

    `patch_notes_created` is always 0 except on `"upserted"` — significant-
    event diffing (see `refresh_roles_cache`'s docstring) only ever runs
    once a fresh `LiveGroundingResult` has actually been written.
    """

    role_name: str
    status: Literal["upserted", "no_live_signal", "error"]
    detail: str | None = None
    patch_notes_created: int = 0


@dataclass(frozen=True)
class RefreshSummary:
    """Batch outcome of one `refresh_roles_cache` (or
    `check_and_refresh_stale_roles`) call — one `RoleRefreshResult` per role
    attempted, in the order supplied, regardless of outcome.
    """

    results: list[RoleRefreshResult]

    @property
    def had_errors(self) -> bool:
        """True if any role in this batch ended in `"error"` — used by the
        GitHub Actions script entry point to set a non-zero process exit
        code without the caller re-deriving this from `results` itself.
        """
        return any(result.status == "error" for result in self.results)


async def refresh_roles_cache(
    session: Session, role_names: list[str], reference_time: datetime
) -> RefreshSummary:
    """Re-run `ground_role` for each role in `role_names` and write any
    live grounding result into `roles_cache` via `upsert_role`.

    Roles are processed sequentially (not `asyncio.gather`-parallel):
    judgment call — this batch runs at most once a month (cron) or once at
    startup against a handful of stale roles, so the marginal latency of
    sequential calls is immaterial next to the simplicity of a
    straightforward per-role try/except loop and deterministic test
    ordering.

    One role's failure never aborts the batch: `ground_role` is awaited
    inside `asyncio.wait_for` (this function's own explicit timeout, see
    `ROLE_REFRESH_TIMEOUT_SECONDS`) inside a `try`/`except Exception` — a
    deliberately broad catch, flagged as a departure from this codebase's
    usual specific-exception-tuple convention: this function's entire
    purpose is "don't let one role's grounding bug or transient failure
    block every other role," so the catch boundary must be as wide as
    `ground_role`'s own possible failure surface (it can raise
    data-integrity errors like `ConfidenceValidationError`, not just a
    fixed, enumerable set). Every other module in this codebase catches
    specific exceptions because failure there is enumerable and expected;
    here it explicitly is not.

    Judgment call, flagged: only a `LiveGroundingResult` (high/medium/low
    confidence) is written to `roles_cache`. A `CachedFallbackResult` or
    `GeneralKnowledgeFloorResult` (ground_role's fallback rungs) means live
    grounding produced no usable signal this cycle — writing either back
    through `upsert_role` would re-stamp `last_updated` as "just refreshed"
    for data that was not, in fact, freshly re-verified (a
    `CachedFallbackResult` is only a re-serving of the *existing* cache
    entry; a `GeneralKnowledgeFloorResult` has no source at all, and
    Architecture §8 already states it must never be written to
    `roles_cache`). Recorded as `"no_live_signal"` instead: the existing
    cache entry (if any) is left exactly as it was, staleness-checkable on
    its own honest timestamp.

    Significant-event detection (Architecture §9), on the `"upserted"`
    path only: the *pre-refresh* `roles_cache` row is fetched via
    `get_role` before `upsert_role` overwrites it, then diffed against the
    fresh `LiveGroundingResult` via `create_patch_notes_for_significant_events`
    below. A role with no pre-existing row (first-ever refresh) has
    nothing to diff against — diffing is skipped for that role, not
    treated as an error.
    """
    results: list[RoleRefreshResult] = []
    for role_name in role_names:
        try:
            grounding_result = await asyncio.wait_for(
                ground_role(role_name, session, reference_time),
                timeout=ROLE_REFRESH_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 — see docstring: deliberately broad
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
    """Build a casefolded-skill-name -> `SkillSnapshot` map from a
    `roles_cache` row's `core_skills`/`emerging_skills` JSONB lists (each
    entry: `{"skill":..., "source_url":..., "confidence":...}` — see
    `data/roles_cache.py`'s `_to_skill_entry`).
    """
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


# Judgment call, flagged prominently: this cron job is not an agent
# (Architecture §3's "Cron job (not an agent)" section) and this task's
# scope fence explicitly forbids any LLM call here — but Architecture §9
# still requires *a* patch_notes row to exist the moment a significant
# event is detected, and CLAUDE.md guardrail #1 requires every patch-note
# to carry a real source_url/confidence (satisfied via `grounded_content`
# in create_patch_notes_for_significant_events below). This deterministic,
# mechanically-assembled sentence is a structurally-valid placeholder, not
# real Agent-1-authored narrative content: Architecture §3 assigns
# *content* authorship to Agent 1's reasoning, which this cron module
# cannot invoke without violating the scope fence. Replacing this with an
# actual Agent-1-generated explanation is real, named future work, not
# solved here.
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
    """Diff `previous_row` (the pre-refresh `roles_cache` snapshot, from
    `get_role`) against `grounding_result` (the just-grounded snapshot
    `refresh_roles_cache` just wrote) via `outline/significant_event.py`'s
    `is_significant_event` — not reimplemented here — and create a
    `PENDING` patch-note (`data/patch_notes.py`'s `create_patch_note`) for
    every user with a completed topic matching a skill that crossed
    upward (Architecture §9). Returns the number of patch-notes created.

    The "new" snapshot treats every skill in `grounding_result.skills` as
    `SkillBucket.CORE_SKILLS`: this deliberately mirrors exactly what
    `refresh_roles_cache` just wrote via `upsert_role`
    (`core_skills=grounding_result.skills, emerging_skills=[]` — the same
    degenerate-split workaround flagged in PRD Future Improvements #5,
    since `LiveGroundingResult` has no real core/emerging split). Diffing
    against anything other than what was actually persisted would make
    this function's notion of "significant" disagree with the data
    actually sitting in `roles_cache`.

    A skill matches a completed topic by exact, case-insensitive name via
    `data/outline_topics.py`'s `get_completed_topics_matching_skill` — not
    reimplemented here either.
    """
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

        # An upward crossing always means new_snapshot is not ABSENT, so
        # `key` is always present in new_skill_content here.
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
    """Return the subset of `role_names` whose `roles_cache` entry is
    either missing entirely or past `data/roles_cache.py`'s `is_stale`
    30-day floor, per Architecture §3's startup staleness check.

    Read-only — does not call `ground_role` or `upsert_role`. Staleness
    logic itself is not reimplemented here; this only calls `is_stale`.
    """
    stale_or_missing: list[str] = []
    for role_name in role_names:
        cached = get_role(session, role_name)
        if cached is None or is_stale(cached["last_updated"], reference_time):
            stale_or_missing.append(role_name)
    return stale_or_missing


async def check_and_refresh_stale_roles(
    session: Session, role_names: list[str], reference_time: datetime
) -> RefreshSummary:
    """Startup/session staleness check (Architecture §3's resilience layer,
    alongside the GitHub Actions primary trigger): refresh only the roles
    in `role_names` whose `roles_cache` entry is stale or missing, leaving
    already-fresh roles untouched.

    Judgment call, flagged: refreshes only the stale/missing subset (via
    `refresh_roles_cache`), not the full `role_names` batch — this path
    runs inline on a real user-facing startup/session, so re-grounding
    roles that are already fresh would mean unnecessary live Himalayas/
    Tavily/Gemini calls (cost and latency) with no benefit; the shared
    `refresh_roles_cache` function already accepts an arbitrary
    `role_names` list, so no new mechanism is needed to express "just
    these roles."

    Not wired into any Streamlit app or `main.py` — this is a plain,
    callable function only, per this task's explicit scope fence. It is
    deliberately independent of `SEED_ROLES`: callers decide which roles
    to check.
    """
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
