"""Deterministic rules that decide a confidence tier by comparing Himalayas and Tavily signal against the roles-cache anchor."""

from dataclasses import dataclass

from data.tavily_parser import ParsedSearchResult
from security.output_guard import ConfidenceTier

ANCHOR_OVERLAP_MINIMUM = 1

TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD = 3


@dataclass(frozen=True)
class TavilyCitation:
    """The single source URL a Tavily-only trust-qualifying batch is attributed to, plus the full set of distinct skills it found."""

    source_url: str
    skills: frozenset[str]


@dataclass(frozen=True)
class CrossValidationDecision:
    """The confidence tier, conflict flag, and reasoning produced by cross-validating one role's live grounding attempt."""

    confidence: ConfidenceTier
    has_conflict: bool
    reason: str
    tavily_citation: TavilyCitation | None = None


def _select_tavily_citation(
    tavily_results: list[ParsedSearchResult],
) -> TavilyCitation | None:
    """Pick the highest-scoring skill-bearing Tavily result as the citation source, or None if no batch of results clears the trust threshold."""
    skill_bearing = [result for result in tavily_results if result.skills]
    if not skill_bearing:
        return None

    all_skills = frozenset(skill for result in skill_bearing for skill in result.skills)
    if len(all_skills) < TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD:
        return None

    citation_result = max(skill_bearing, key=lambda result: result.score)
    return TavilyCitation(source_url=citation_result.source_url, skills=all_skills)


def tavily_has_usable_signal(tavily_results: list[ParsedSearchResult]) -> bool:
    """Whether the given Tavily results contain enough distinct skills to count as usable signal on their own."""
    return _select_tavily_citation(tavily_results) is not None


def decide_confidence_tier(
    *,
    himalayas_has_signal: bool,
    tavily_results: list[ParsedSearchResult],
    anchor_skills: frozenset[str],
    himalayas_skills: frozenset[str],
) -> CrossValidationDecision:
    """Decide a confidence tier by combining Himalayas's and Tavily's usable-signal status against the roles-cache anchor skill set."""
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
