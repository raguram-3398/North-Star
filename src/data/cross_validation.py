"""Deterministic cross-validation rules (PRD §7.3; Architecture
_North_Star.md §8/§9).

PRD §7.3 states cross-validation as rule application, not open-ended
LLM judgment: "Agreement/normalization judged against roles.json as a
grounding anchor (not open-ended LLM judgment)." Architecture §2's
orchestration principle is that Agent 1's "reasoning" here is exactly
this kind of deterministic rule — so, consistent with
`outline/significant_event.py` and `patches/patch_manager.py`'s
confidence-branching, the actual tier decision lives in its own plain,
testable module rather than inline in `agents/research_outline_agent.py`,
which calls this module the same way it calls
`security/output_guard.py`.

**Tavily-only signal (resolved — previously a hard scope limit):** a
Tavily-only batch can now reach `medium` confidence if it clears
`TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD`, using `data/tavily_parser.py`'s
skill extraction. Two rules from `data/tavily_parser.py`'s own
real-data finding are non-negotiable here (do not weaken them):
Tavily's `score` field is used *only* to choose which already-skill-
bearing result becomes the citation `source_url` — never to decide
whether to trust the batch at all, since `score` was shown not to
predict extractability (the highest-scored result across 4 real
fixtures had zero extractable skills). Trust is decided purely by how
much was actually extracted (`_select_tavily_citation`'s distinct-skill
count), independent of `score`.

**Known limitation, carried forward from `data/tavily_parser.py` (not
resolved here):** `TECH_SKILL_VOCABULARY` was derived from skills
`data/himalayas_parser.py` already extracted for the same 4 seed roles
(Backend Engineer, Frontend Engineer, Data Analyst, DevOps Engineer),
plus 2 manually-added terms — it does not independently discover skills
Himalayas hasn't already surfaced for a role. For any role outside
those 4 (including the rest of PRD §7.3's seed list, e.g. AI/ML
Engineer), Tavily's trust check has no special vocabulary coverage and
will likely under-count real signal until the vocabulary is expanded or
replaced with a more independent extraction method — a named future
improvement, not attempted here.
"""

from dataclasses import dataclass

from data.tavily_parser import ParsedSearchResult
from security.output_guard import ConfidenceTier

# Judgment call, flagged for review: the minimum number of skills shared
# between Himalayas's extracted skill set and roles_cache's anchor skill
# set required to call the two "in agreement" rather than a genuine
# conflict. 1 (any overlap at all) was chosen as the simplest possible
# rule that still distinguishes "these are recognizably the same role"
# from "these share nothing in common" — no real-world data exists yet to
# validate this is the right bar; tune if cross-validation runs in
# practice show it's too lenient or too strict.
ANCHOR_OVERLAP_MINIMUM = 1

# Judgment call, flagged for review — unvalidated, same status as
# ANCHOR_OVERLAP_MINIMUM and himalayas_relevance.py's constants: the
# number of *distinct* skills (not total mentions, not count of
# results-with->=1-skill) that must be found across an entire Tavily
# batch before it counts as usable signal on its own.
#
# Distinct-skill count was chosen over the alternatives because:
# - total mentions would let one generic, repeated buzzword (e.g. many
#   articles all just saying "Python") inflate trust without breadth;
# - count of skill-bearing results ignores how much each result actually
#   contributes (a result naming 5 skills and a result naming 1 would
#   count the same).
# Real data (tests/fixtures/tavily_search_*.json, the same 4 seed roles
# as data/tavily_parser.py's fixtures) found 4-11 distinct skills per
# genuine-role batch of ~9-10 results; 3 was chosen as comfortably below
# that real range while still requiring more than one or two incidental
# vocabulary hits (e.g. a stray "R" or "Mode" match) to pass.
TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD = 3


@dataclass(frozen=True)
class TavilyCitation:
    """The result of `_select_tavily_citation`: the single `source_url`
    to attribute Tavily-derived skills to, plus the full set of distinct
    skills found across the whole trust-qualifying batch.

    Judgment call, flagged for review: all skills found anywhere in the
    batch are attributed to one shared citation URL (the highest-scoring
    skill-bearing result), rather than each skill citing its own
    originating result. Given Tavily's coarse, vocabulary-based
    extraction (`data/tavily_parser.py`'s own docstring: "not real
    extraction... treat its output as directional signal"), per-result
    attribution would imply a precision this mechanism doesn't actually
    have; one representative citation for the batch is the more honest
    framing, and keeps every skill the batch found usable rather than
    discarding all but the citation result's own list.
    """

    source_url: str
    skills: frozenset[str]


@dataclass(frozen=True)
class CrossValidationDecision:
    """The outcome of applying PRD §7.3's cross-validation rules to one
    role's live grounding attempt. `confidence` uses `ConfidenceTier.REJECT`
    to mean "live grounding produced no usable signal this round" — the
    caller (`agents/research_outline_agent.py`) must treat that as a
    trigger to fall through to `data/grounding_fallback.py`, never write
    it anywhere; `security/output_guard.py`'s `validate_output_object`
    already refuses `reject` unconditionally, so this can't leak into a
    persisted record even by accident.

    `tavily_citation` is populated only on the "Tavily-only, Himalayas
    has no signal" medium-confidence branch — the one case where this
    module hands back raw ingredients (a source_url and a skill set)
    for the caller to build a `ValidatedGroundedContent` from, rather
    than Himalayas-sourced skills the caller already has on hand.
    """

    confidence: ConfidenceTier
    has_conflict: bool
    reason: str
    tavily_citation: TavilyCitation | None = None


def _select_tavily_citation(
    tavily_results: list[ParsedSearchResult],
) -> TavilyCitation | None:
    """Among Tavily results with at least one extracted skill, select the
    highest-`score` one as the citation `source_url` — never a
    skill-less result, however high its score (see module docstring and
    `data/tavily_parser.py`'s real-data finding: a high-scoring
    skill-less result must never win by default).

    Returns `None` if no result has any extracted skill, or if the
    distinct-skill count across all skill-bearing results doesn't meet
    `TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD` — both mean "no usable
    Tavily signal," the same outcome for this function's callers.
    """
    skill_bearing = [result for result in tavily_results if result.skills]
    if not skill_bearing:
        return None

    all_skills = frozenset(skill for result in skill_bearing for skill in result.skills)
    if len(all_skills) < TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD:
        return None

    citation_result = max(skill_bearing, key=lambda result: result.score)
    return TavilyCitation(source_url=citation_result.source_url, skills=all_skills)


def tavily_has_usable_signal(tavily_results: list[ParsedSearchResult]) -> bool:
    """Whether `tavily_results` clears `TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD`
    — usable independently of `decide_confidence_tier` (e.g. by a caller
    that wants to record per-source status) without duplicating the
    citation-selection logic.
    """
    return _select_tavily_citation(tavily_results) is not None


def decide_confidence_tier(
    *,
    himalayas_has_signal: bool,
    tavily_results: list[ParsedSearchResult],
    anchor_skills: frozenset[str],
    himalayas_skills: frozenset[str],
) -> CrossValidationDecision:
    """Apply PRD §7.3's cross-validation rules given Himalayas's
    already-computed usable-signal status, Tavily's parsed results
    (`data/tavily_parser.py`), Himalayas's extracted skill set, and
    roles_cache's anchor skill set (empty if no anchor entry exists, or
    it has no skills recorded).

    Rule order:
    1. No usable Himalayas signal:
       - Tavily also has no usable signal (per
         `TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD`) -> `reject`.
       - Tavily clears the trust threshold on its own -> `medium`,
         PRD's "single source only" rule, now reachable via Tavily —
         the citation is carried on the decision so the caller can
         build a writable object from it.
    2. No anchor skills at all (Himalayas has signal) -> `low` (PRD
       §7.3's niche/no-anchor rule; the LLM sanity-check pass that rule
       also describes is explicitly deferred, not attempted here).
    3. Only Himalayas has signal (Tavily doesn't clear its threshold)
       -> `medium`, PRD's "single source only" rule (Himalayas side).
    4. Both sources have signal: if Himalayas's skills overlap the
       anchor by at least `ANCHOR_OVERLAP_MINIMUM`, `high`; otherwise a
       genuine conflict — `medium` with `has_conflict=True`.

    Branches 2-4 are unchanged from before Tavily-only signal was
    possible — only branch 1's Tavily sub-case is new.
    """
    tavily_citation = _select_tavily_citation(tavily_results)

    if not himalayas_has_signal:
        if tavily_citation is None:
            return CrossValidationDecision(
                confidence=ConfidenceTier.REJECT,
                has_conflict=False,
                reason="no usable live signal from either source",
            )
        return CrossValidationDecision(
            confidence=ConfidenceTier.MEDIUM,
            has_conflict=False,
            reason=(
                f"single source only (Tavily; {len(tavily_citation.skills)} "
                "distinct skills found, Himalayas had no usable signal)"
            ),
            tavily_citation=tavily_citation,
        )

    if not anchor_skills:
        return CrossValidationDecision(
            confidence=ConfidenceTier.LOW,
            has_conflict=False,
            reason=(
                "no roles_cache anchor to validate agreement against "
                "(niche/no-anchor role)"
            ),
        )

    if tavily_citation is None:
        return CrossValidationDecision(
            confidence=ConfidenceTier.MEDIUM,
            has_conflict=False,
            reason="single source only (Himalayas; Tavily had no usable signal)",
        )

    overlap = anchor_skills & himalayas_skills
    if len(overlap) >= ANCHOR_OVERLAP_MINIMUM:
        return CrossValidationDecision(
            confidence=ConfidenceTier.HIGH,
            has_conflict=False,
            reason=(
                "both sources have live signal and agree with the "
                f"roles_cache anchor on {sorted(overlap)}"
            ),
        )

    return CrossValidationDecision(
        confidence=ConfidenceTier.MEDIUM,
        has_conflict=True,
        reason=(
            "genuine conflict: both sources have live signal but Himalayas's "
            "extracted skills do not overlap the roles_cache anchor"
        ),
    )
