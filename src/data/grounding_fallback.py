"""Cached-fallback and general-knowledge-only floor paths of the
confidence ladder (Architecture_North_Star.md §8, PRD §7.3) — the two
lowest labeled rungs above `reject`:

    high -> medium -> low -> cached-low -> general-knowledge-only -> reject

Both paths here operate only on already-cached data (`roles_cache` via
`data/roles_cache.py`) and `security/output_guard.py`'s existing
validation gate. No live Himalayas/Tavily calls happen in this module —
those, and the decision to fall back at all, belong to Agent 1's
Research/Grounding reasoning (Architecture_North_Star.md §3). This module
answers only "what does the cache say" and "what does the honest floor
look like" once a caller has already decided live grounding failed.

DB reads are allowed (via an injected `Session`, matching
`data/roles_cache.py`'s dependency-injection pattern) but nothing here
calls an LLM and nothing here decides *when* to invoke it — kept as
close to pure as a module doing real I/O can be.
"""

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

# Judgment call, flagged for review: roles_cache persists each skill entry
# as {skill, source_url, confidence} only (Architecture_North_Star.md §5)
# — no per-skill source_type is ever written back, so one cannot be read
# honestly off the row itself (see context-transfer.md's "Asymmetry not
# yet resolved" note from the roles_cache session). Rather than fabricate
# an external-looking source_type, every cached-fallback result is stamped
# with this constant, which honestly names *this* provenance layer (a
# previously-validated grounding result now being re-served from the
# cache) instead of guessing at the original external source's type.
CACHED_SOURCE_TYPE = "roles_cache-cached"


@dataclass(frozen=True)
class CachedFallbackResult:
    """The cached-fallback rung (PRD §7.3: "cached-fallback (low, labeled
    with last_updated)"). Returned whenever `roles_cache` has *any* entry
    for the role — see `get_cached_fallback`'s docstring for the flagged
    judgment call on whether a stale entry should still qualify.
    """

    role_name: str
    core_skills: list[ValidatedGroundedContent]
    emerging_skills: list[ValidatedGroundedContent]
    last_updated: datetime
    is_stale: bool


@dataclass(frozen=True)
class GeneralKnowledgeFloorResult:
    """The honest floor of the confidence ladder (PRD §7.3):
    `roles_cache` has no entry at all for the role — not merely stale,
    genuinely absent. There is no real source of any kind to label.

    Deliberately **not** a `ValidatedGroundedContent`. That type requires
    a non-empty, structurally valid `source_url`
    (`security/output_guard.py`); fabricating one here to force this
    result through `validate_output_object` would violate CLAUDE.md
    guardrail #1 and PRD §7.3's "never silently fabricate a source" rule.
    This is a structurally distinct, lower-trust shape by design — it has
    no `source_url` field to omit or fake, not a candidate dict that
    happens to fail validation.

    Per `specs/scenarios/high_risk_flows.feature`'s "No source returns
    usable data" scenario, a result of this type must never be written
    into `outline_topics`, `patch_notes`, or `roles_cache` — it exists
    solely so a caller can report the floor explicitly and honestly to
    the user. There is deliberately no function in this module that turns
    a `GeneralKnowledgeFloorResult` into a `ValidatedGroundedContent`.
    """

    role_name: str
    confidence: ConfidenceTier
    label: str


def get_cached_fallback(
    session: Session,
    role_name: str,
    reference_time: datetime,
) -> CachedFallbackResult | None:
    """Attempt the cached-fallback rung: read `role_name` from
    `roles_cache` and re-validate its contents at confidence
    `"cached-low"`.

    Returns `None` if `roles_cache` has no entry at all for `role_name` —
    the caller should then fall through to `get_general_knowledge_floor`,
    the next and final labeled rung before `reject`.

    Judgment call, flagged for review: a *stale* cached entry (past the
    30-day floor per `data/roles_cache.py`'s `is_stale`) is still returned
    here, not treated as equivalent to "no entry." Staleness triggers a
    refresh on `roles_cache`'s own cron/startup-check schedule
    (Architecture_North_Star.md §3) — it is not, in this reading, a reason
    to discard already-grounded data when live sources have already
    failed *today*. The ladder's "labeled with last_updated" phrasing
    reads as the mechanism for surfacing a stale hit honestly (e.g. "based
    on data from 47 days ago"), rather than a signal to escalate straight
    to the weaker, sourceless general-knowledge-only floor when a real,
    labeled source is sitting right there. `is_stale` is carried on the
    result purely as informational metadata for that label.

    Alternative reading, not chosen here: a stale entry should NOT count
    as usable fallback and should escalate to the floor instead, on the
    theory that "low confidence" shouldn't be allowed to silently mean
    "arbitrarily old." Flagged explicitly rather than decided
    unilaterally — swap this function's early-return condition if that
    reading is preferred.
    """
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
    """Re-validate one persisted `roles_cache` skill entry
    (`{skill, source_url, confidence}`) at the cached-low rung, passing it
    back through `security/output_guard.py`'s `validate_output_object` so
    a caller downstream still only ever receives a post-guard object
    (CLAUDE.md guardrail #12), never a raw cache dict.

    The persisted `confidence` is overridden with `"cached-low"` per PRD
    §7.3 — the ladder rung reflects that this is being served as a
    fallback *today*, not whatever confidence it was originally graded at
    when first written to `roles_cache`. `source_type` is stamped with
    `CACHED_SOURCE_TYPE` (see module docstring) since `roles_cache`'s
    JSONB shape never persists a per-skill `source_type` to read back.
    """
    candidate = {
        "source_url": entry["source_url"],
        "source_type": CACHED_SOURCE_TYPE,
        "confidence": ConfidenceTier.CACHED_LOW.value,
        "skill": entry.get("skill"),
        "last_updated": last_updated,
    }
    return validate_output_object(candidate)


def get_general_knowledge_floor(role_name: str) -> GeneralKnowledgeFloorResult:
    """The final labeled rung before `reject` (PRD §7.3,
    Architecture_North_Star.md §8): `roles_cache` has no entry at all for
    `role_name`. Returns an explicitly labeled, honestly unsourced
    result — never a `ValidatedGroundedContent` (see
    `GeneralKnowledgeFloorResult`'s docstring for why).

    Per `specs/scenarios/high_risk_flows.feature`, a caller must never
    write this result into `outline_topics`, `patch_notes`, or
    `roles_cache` — it exists only so the system can report the floor
    explicitly to the user (PRD §7.3, CLAUDE.md guardrail #6).
    """
    return GeneralKnowledgeFloorResult(
        role_name=role_name,
        confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
        label=(
            f"No cached or live market data is available for {role_name!r}. "
            "This content reflects general knowledge only and is not "
            "grounded in any source."
        ),
    )
