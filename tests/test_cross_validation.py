"""Tests for data/cross_validation.py — PRD §7.3's cross-validation
rules, expressed as a pure decision function.

Tavily inputs are `ParsedSearchResult`s (data/tavily_parser.py) rather
than a plain boolean, so a Tavily-only batch can be trusted on its own
merit (distinct-skill count) — see `decide_confidence_tier`'s docstring
and the module docstring's real-data finding about `score` never being
the trust signal.
"""

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
    """A Tavily batch that clears TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD
    (currently 3): one result naming exactly that many distinct skills.
    """
    return [_result(["sql", "python", "excel"], score=0.5, url=url)]


def _failing_tavily_batch() -> list[ParsedSearchResult]:
    """A Tavily batch that does not clear the trust threshold — one
    result naming fewer distinct skills than required.
    """
    return [_result(["sql"], score=0.9)]


# --- reject: neither source has usable signal -----------------------------


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


# --- the new behavior: Tavily-only signal reaching medium -----------------


def test_himalayas_no_signal_tavily_passes_threshold_is_medium_with_citation() -> None:
    """The scenario this task exists for: Tavily alone, with enough
    distinct skills found, now reaches medium confidence instead of
    always falling through to the fallback chain.
    """
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
    """The exact failure mode Task 3a's finding warns against: a
    skill-less result must never win citation just because it has the
    highest score. Batch includes a high-score result with zero skills
    and a lower-score result that actually names enough distinct skills
    to pass the trust threshold on its own.
    """
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


# --- boundary: exactly at, just below, just above the trust threshold ----


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
    """The same skill repeated across many results must not inflate the
    count — distinct skills only, per the module's documented judgment
    call. Ten results all naming the same single skill must still fail
    the threshold (currently 3).
    """
    results = [_result(["python"]) for _ in range(10)]
    assert tavily_has_usable_signal(results) is False


# --- unchanged branches: regression checks --------------------------------


def test_no_anchor_skills_is_low_niche_role() -> None:
    """Himalayas has signal, no anchor — Tavily's state must not matter
    for this branch (checked with both a passing and failing batch).
    """
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
    """Both the "high" and "conflict" branches require
    himalayas_has_signal=True, a Tavily batch that passes the trust
    threshold, and a non-empty anchor — the only variable that must
    actually determine the outcome is the overlap between anchor_skills
    and himalayas_skills. Checked explicitly with matching preconditions
    and only the skills varied, so a bug that ignores the overlap check
    entirely (as an earlier draft of this module did) cannot pass both
    assertions.
    """
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
