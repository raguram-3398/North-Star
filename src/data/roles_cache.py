"""roles_cache I/O.

Per Architecture_North_Star.md §5/§7.3: structured, cron-refreshed market
data (core_skills / emerging_skills per role, each with a confidence
tier), used as fallback data and a normalization anchor — never a shortcut
that skips live research for a new user.

This module is allowed real DB I/O, unlike the pure modules built so far
(CLAUDE.md's "pure functions stay pure" rule names pace/calculator.py,
outline/significant_event.py, security/output_guard.py, and
security/input_gate.py specifically — not this one). Query logic here is
intentionally minimal: this module does not decide *when* to refresh
(src/cron/refresh_roles.py), does not detect significant events
(outline/significant_event.py), and does not do cross-validation judgment
(Agent 1's reasoning). It only reads and writes rows, plus one small,
genuinely pure staleness calculation.

Sessions are passed in by the caller (dependency injection) rather than
obtained internally via db/connection.py. This keeps every function here
testable with a mocked Session and no real database — see
tests/test_roles_cache.py.

Per CLAUDE.md guardrail #12, `roles_cache` is one of the enforced
structural-gate boundaries (alongside outline items and patch-notes):
`upsert_role` requires each `core_skills`/`emerging_skills` entry to
already be a `ValidatedGroundedContent` (security/output_guard.py), not a
raw dict — the write path itself cannot accept an unvalidated skill
entry, regardless of what the caller intended.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models.schemas import RolesCache
from security.output_guard import ValidatedGroundedContent
from utils.exceptions import ConfidenceValidationError

# Per Architecture_North_Star.md §3: roles_cache is refreshed "minimum
# every 30 days, immediately on a significant event."
STALENESS_FLOOR_DAYS = 30


def get_role(session: Session, role_name: str) -> dict[str, Any] | None:
    """Read a single role's cached market data, or None if no entry
    exists.
    """
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
    """Write or refresh a role's cached market data, stamping
    `last_updated` with the current time. Commits the transaction.

    `core_skills` and `emerging_skills` must already be
    `ValidatedGroundedContent` instances (security/output_guard.py) — per
    CLAUDE.md guardrail #12, this write path structurally requires a
    post-output_guard object for each skill entry, not a raw dict. Each
    entry is serialized here to the `{skill, source_url, confidence}`
    JSONB shape documented in Architecture_North_Star.md §5.

    `last_updated` is stored as a naive UTC datetime, matching the
    `TIMESTAMP` (not `TIMESTAMPTZ`) column type in
    Architecture_North_Star.md §5 — all timestamps in this table are
    naive UTC by convention.

    Raises ConfidenceValidationError if a skill entry is not a
    ValidatedGroundedContent instance, or is missing a "skill" name in
    its `extra` field (see `_to_skill_entry`).
    """
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
    """Serialize one validated grounding result into the roles_cache
    skill-entry JSONB shape: `{skill, source_url, confidence}`
    (Architecture_North_Star.md §5).

    Raises ConfidenceValidationError if `item` is not a
    ValidatedGroundedContent instance — this is checked explicitly rather
    than left to an incidental AttributeError from a raw dict lacking
    `.extra`, so the structural gate at this write boundary is
    intentional and self-documenting, consistent with every other guard
    in this codebase (CLAUDE.md guardrail #12). Also raises
    ConfidenceValidationError if `item.extra` has no "skill" key — the
    skill name has no dedicated field on `ValidatedGroundedContent`
    itself, so it must have been included in the original candidate dict
    passed to `security.output_guard.validate_output_object`.
    """
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


# Judgment call, flagged for review: this staleness check could arguably
# live in its own pure module instead of alongside this module's I/O
# functions. Kept here because it operates on a value only this module
# reads (roles_cache.last_updated), has no other caller, and a separate
# module would just be indirection with no reuse benefit. It remains a
# genuinely pure, side-effect-free function — no DB access inside it, and
# (like pace/calculator.py) no hidden wall-clock call either: the caller
# supplies `reference_time` explicitly rather than this function calling
# datetime.now() internally, keeping it fully deterministic and testable.
def is_stale(
    last_updated: datetime,
    reference_time: datetime,
    max_age_days: int = STALENESS_FLOOR_DAYS,
) -> bool:
    """Determine whether a cached role's last_updated is past the 30-day
    floor, per Architecture_North_Star.md §3's startup staleness check.

    Both `last_updated` and `reference_time` are expected as naive UTC
    datetimes, matching the convention `upsert_role` writes under.
    """
    return reference_time - last_updated > timedelta(days=max_age_days)
