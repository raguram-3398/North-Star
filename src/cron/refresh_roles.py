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
logic of its own — it orchestrates two already-tested functions:
`agents/research_outline_agent.py`'s `ground_role` (the real grounding
pipeline) and `data/roles_cache.py`'s `upsert_role` (the write path).
No LLM/Himalayas/Tavily call is made directly by this module; every such
call happens inside `ground_role`, which already owns its own per-source
timeout/error handling.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session

from agents.research_outline_agent import LiveGroundingResult, ground_role
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from data.roles_cache import get_role, is_stale, upsert_role

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
    """

    role_name: str
    status: Literal["upserted", "no_live_signal", "error"]
    detail: str | None = None


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
            upsert_role(
                session,
                role_name,
                core_skills=grounding_result.skills,
                emerging_skills=[],
            )
            results.append(RoleRefreshResult(role_name, "upserted"))
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
