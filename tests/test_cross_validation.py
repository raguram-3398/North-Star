"""Tests for data/cross_validation.py: the pure decision function that determines confidence tier from Himalayas and Tavily signals."""

from data.cross_validation import (
    TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD,
    decide_confidence_tier,
    tavily_has_usable_signal,
)
from data.tavily_parser import ParsedSearchResult
from security.output_guard import ConfidenceTier


def _result(
    skills: list[str], score: float = 0.5, url: str = "https://x.test"
) -> ParsedSearchResult:
    return ParsedSearchResult(title="x", skills=skills, source_url=url, score=score)


def _passing_tavily_batch(url: str = "https://good.test") -> list[ParsedSearchResult]:
    """Build a Tavily batch that clears the distinct-skills trust threshold."""
    return [_result(["sql", "python", "excel"], score=0.5, url=url)]


def _failing_tavily_batch() -> list[ParsedSearchResult]:
    """Build a Tavily batch that does not clear the distinct-skills trust threshold."""
    return [_result(["sql"], score=0.9)]


def test_no_signal_from_either_source_is_reject() -> None:
    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=[],
        anchor_skills=frozenset({"sql", "python"}),
        himalayas_skills=frozenset(),
    )
    assert decision.confidence == ConfidenceTier.REJECT
    assert decision.has_conflict is False
    assert decision.tavily_citation is None


def test_himalayas_no_signal_and_tavily_below_threshold_is_reject() -> None:
    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=_failing_tavily_batch(),
        anchor_skills=frozenset({"sql", "python"}),
        himalayas_skills=frozenset(),
    )
    assert decision.confidence == ConfidenceTier.REJECT
    assert decision.tavily_citation is None


def test_himalayas_no_signal_tavily_passes_threshold_is_medium_with_citation() -> None:
    """Tavily alone, with enough distinct skills found, reaches medium confidence instead of falling through."""
    tavily_results = _passing_tavily_batch(url="https://good.test")

    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=tavily_results,
        anchor_skills=frozenset(),
        himalayas_skills=frozenset(),
    )

    assert decision.confidence == ConfidenceTier.MEDIUM
    assert decision.has_conflict is False
    assert decision.tavily_citation is not None
    assert decision.tavily_citation.source_url == "https://good.test"
    assert decision.tavily_citation.skills == frozenset({"sql", "python", "excel"})


def test_citation_never_picked_from_a_high_score_skill_less_result() -> None:
    """A skill-less result must never win citation just because it has the highest score."""
    high_score_no_skills = _result([], score=0.95, url="https://junk.test")
    lower_score_with_skills = _result(
        ["sql", "python", "excel"], score=0.3, url="https://good.test"
    )

    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=[high_score_no_skills, lower_score_with_skills],
        anchor_skills=frozenset(),
        himalayas_skills=frozenset(),
    )

    assert decision.confidence == ConfidenceTier.MEDIUM
    assert decision.tavily_citation is not None
    assert decision.tavily_citation.source_url == "https://good.test"


def test_tavily_trust_threshold_just_below_fails() -> None:
    skills = [f"skill{i}" for i in range(TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD - 1)]
    results = [_result(skills)]
    assert tavily_has_usable_signal(results) is False

    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=results,
        anchor_skills=frozenset(),
        himalayas_skills=frozenset(),
    )
    assert decision.confidence == ConfidenceTier.REJECT


def test_tavily_trust_threshold_exactly_at_passes() -> None:
    skills = [f"skill{i}" for i in range(TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD)]
    results = [_result(skills)]
    assert tavily_has_usable_signal(results) is True

    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=results,
        anchor_skills=frozenset(),
        himalayas_skills=frozenset(),
    )
    assert decision.confidence == ConfidenceTier.MEDIUM


def test_tavily_trust_threshold_just_above_passes() -> None:
    skills = [f"skill{i}" for i in range(TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD + 1)]
    results = [_result(skills)]
    assert tavily_has_usable_signal(results) is True

    decision = decide_confidence_tier(
        himalayas_has_signal=False,
        tavily_results=results,
        anchor_skills=frozenset(),
        himalayas_skills=frozenset(),
    )
    assert decision.confidence == ConfidenceTier.MEDIUM


def test_distinct_skill_count_not_total_mentions() -> None:
    """The same skill repeated across many results must not inflate the distinct-skill count."""
    results = [_result(["python"]) for _ in range(10)]
    assert tavily_has_usable_signal(results) is False


def test_no_anchor_skills_is_low_niche_role() -> None:
    """When Himalayas has signal but there is no anchor, Tavily's state must not affect the outcome."""
    for tavily_results in (_passing_tavily_batch(), _failing_tavily_batch(), []):
        decision = decide_confidence_tier(
            himalayas_has_signal=True,
            tavily_results=tavily_results,
            anchor_skills=frozenset(),
            himalayas_skills=frozenset({"rust", "webassembly"}),
        )
        assert decision.confidence == ConfidenceTier.LOW
        assert decision.has_conflict is False


def test_single_source_himalayas_only_is_medium() -> None:
    decision = decide_confidence_tier(
        himalayas_has_signal=True,
        tavily_results=_failing_tavily_batch(),
        anchor_skills=frozenset({"sql", "python"}),
        himalayas_skills=frozenset({"sql"}),
    )
    assert decision.confidence == ConfidenceTier.MEDIUM
    assert decision.has_conflict is False
    assert decision.tavily_citation is None


def test_both_sources_agree_with_anchor_is_high() -> None:
    decision = decide_confidence_tier(
        himalayas_has_signal=True,
        tavily_results=_passing_tavily_batch(),
        anchor_skills=frozenset({"sql", "python", "excel"}),
        himalayas_skills=frozenset({"sql", "tableau"}),
    )
    assert decision.confidence == ConfidenceTier.HIGH
    assert decision.has_conflict is False


def test_both_sources_but_no_anchor_overlap_is_genuine_conflict() -> None:
    decision = decide_confidence_tier(
        himalayas_has_signal=True,
        tavily_results=_passing_tavily_batch(),
        anchor_skills=frozenset({"sql", "python", "excel"}),
        himalayas_skills=frozenset({"rust", "webassembly"}),
    )
    assert decision.confidence == ConfidenceTier.MEDIUM
    assert decision.has_conflict is True


def test_high_vs_conflict_diverge_only_on_anchor_overlap() -> None:
    """The high-versus-conflict outcome must be determined only by whether anchor_skills and himalayas_skills overlap."""
    anchor = frozenset({"sql", "python"})
    agreeing = decide_confidence_tier(
        himalayas_has_signal=True,
        tavily_results=_passing_tavily_batch(),
        anchor_skills=anchor,
        himalayas_skills=frozenset({"sql"}),
    )
    conflicting = decide_confidence_tier(
        himalayas_has_signal=True,
        tavily_results=_passing_tavily_batch(),
        anchor_skills=anchor,
        himalayas_skills=frozenset({"go"}),
    )
    assert agreeing.confidence == ConfidenceTier.HIGH
    assert conflicting.confidence == ConfidenceTier.MEDIUM
    assert conflicting.has_conflict is True
    assert agreeing.has_conflict is False
