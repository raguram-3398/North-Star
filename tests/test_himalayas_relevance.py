"""Tests for data/himalayas_relevance.py — the relevance heuristic that
infers "no signal from Himalayas" despite a non-empty search_jobs
response (PRD §7.2/§7.3's known limitation).
"""

from pathlib import Path

from data.himalayas_parser import ParsedJobListing, parse_search_jobs_response
from data.himalayas_relevance import (
    MAX_COUNT_RELEVANCE_FRACTION,
    MAX_COUNT_THRESHOLD,
    MIN_COUNT_RELEVANCE_FRACTION,
    MIN_COUNT_THRESHOLD,
    compute_title_relevance,
    has_usable_himalayas_signal,
    is_listing_relevant,
    required_relevance_fraction,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _listing(title: str) -> ParsedJobListing:
    return ParsedJobListing(
        title=title, company="Acme", skills=["X"], source_url="https://x.test"
    )


def test_compute_title_relevance_full_and_partial_overlap() -> None:
    assert compute_title_relevance("Backend Engineer", "Senior Backend Engineer") == 1.0
    assert compute_title_relevance("Backend Engineer", "Senior Engineer") == 0.5
    assert compute_title_relevance("Backend Engineer", "Community Moderator") == 0.0


def test_compute_title_relevance_empty_role_is_zero() -> None:
    assert compute_title_relevance("", "Backend Engineer") == 0.0


def test_is_listing_relevant_uses_the_overlap_threshold() -> None:
    assert is_listing_relevant("Backend Engineer", "Senior Backend Engineer") is True
    assert (
        is_listing_relevant("Backend Engineer", "Senior Engineer") is True
    )  # 0.5 == threshold
    assert is_listing_relevant("Backend Engineer", "Community Moderator") is False


def test_required_relevance_fraction_flat_below_min_count() -> None:
    """Below (and at) MIN_COUNT_THRESHOLD, the required fraction is flat
    at MIN_COUNT_RELEVANCE_FRACTION regardless of exact count — tested at
    two different counts, not just one, so a wrong implementation that
    happens to work at a single count wouldn't pass unnoticed.
    """
    assert required_relevance_fraction(1) == MIN_COUNT_RELEVANCE_FRACTION
    assert (
        required_relevance_fraction(MIN_COUNT_THRESHOLD) == MIN_COUNT_RELEVANCE_FRACTION
    )


def test_required_relevance_fraction_flat_above_max_count() -> None:
    assert (
        required_relevance_fraction(MAX_COUNT_THRESHOLD) == MAX_COUNT_RELEVANCE_FRACTION
    )
    assert required_relevance_fraction(1000) == MAX_COUNT_RELEVANCE_FRACTION


def test_required_relevance_fraction_scales_linearly_between_bounds() -> None:
    """Two points strictly inside the band, not just the midpoint — a
    linear-interpolation bug (e.g. reversed direction, wrong slope) could
    still pass a single interior-point test by coincidence.
    """
    span = MAX_COUNT_THRESHOLD - MIN_COUNT_THRESHOLD
    quarter_point = MIN_COUNT_THRESHOLD + span // 4
    three_quarter_point = MIN_COUNT_THRESHOLD + (3 * span) // 4

    fraction_at_quarter = required_relevance_fraction(quarter_point)
    fraction_at_three_quarter = required_relevance_fraction(three_quarter_point)

    # Required fraction decreases monotonically as count grows.
    assert (
        MIN_COUNT_RELEVANCE_FRACTION
        > fraction_at_quarter
        > fraction_at_three_quarter
        > MAX_COUNT_RELEVANCE_FRACTION
    )


def test_has_usable_himalayas_signal_false_for_empty_listings() -> None:
    assert has_usable_himalayas_signal("Backend Engineer", []) is False


def test_has_usable_himalayas_signal_true_for_small_highly_relevant_batch() -> None:
    """Small batch (at MIN_COUNT_THRESHOLD), high relevant fraction —
    exercises the "below minimum count, require high fraction" end of the
    band, and the batch passes it.
    """
    listings = [_listing("Backend Engineer") for _ in range(4)] + [
        _listing("Community Moderator")
    ]
    assert len(listings) == MIN_COUNT_THRESHOLD
    assert has_usable_himalayas_signal("Backend Engineer", listings) is True


def test_has_usable_himalayas_signal_false_for_small_mostly_irrelevant_batch() -> None:
    """Same small-batch end of the band, but the relevant fraction is too
    low to clear MIN_COUNT_RELEVANCE_FRACTION — must return False, not
    just "true because non-empty."
    """
    listings = [_listing("Community Moderator") for _ in range(4)] + [
        _listing("Backend Engineer")
    ]
    assert len(listings) == MIN_COUNT_THRESHOLD
    assert has_usable_himalayas_signal("Backend Engineer", listings) is False


def test_has_usable_himalayas_signal_true_for_large_low_fraction_batch() -> None:
    """Large batch (at MAX_COUNT_THRESHOLD), low but sufficient relevant
    fraction — exercises the "above maximum count, low fraction is
    enough" end of the band.
    """
    relevant_needed = int(MAX_COUNT_THRESHOLD * MAX_COUNT_RELEVANCE_FRACTION) + 1
    listings = [_listing("Backend Engineer") for _ in range(relevant_needed)] + [
        _listing("Community Moderator")
        for _ in range(MAX_COUNT_THRESHOLD - relevant_needed)
    ]
    assert len(listings) == MAX_COUNT_THRESHOLD
    assert has_usable_himalayas_signal("Backend Engineer", listings) is True


def test_has_usable_himalayas_signal_false_for_large_batch_below_low_fraction() -> None:
    listings = [_listing("Backend Engineer")] + [
        _listing("Community Moderator") for _ in range(MAX_COUNT_THRESHOLD - 1)
    ]
    assert len(listings) == MAX_COUNT_THRESHOLD
    assert has_usable_himalayas_signal("Backend Engineer", listings) is False


def test_real_fixtures_have_usable_signal_for_their_own_role() -> None:
    """Cross-checked against three different real, live-captured roles —
    not just one — per this project's multi-item testing discipline.
    """
    cases = [
        ("Data Analyst", "himalayas_search_jobs_data_analyst.txt"),
        ("Frontend Engineer", "himalayas_search_jobs_frontend_engineer.txt"),
        ("DevOps Engineer", "himalayas_search_jobs_devops_engineer.txt"),
    ]
    for role, filename in cases:
        listings = parse_search_jobs_response((FIXTURES_DIR / filename).read_text())
        assert has_usable_himalayas_signal(role, listings) is True, filename


def test_real_fixture_mismatched_role_has_no_signal() -> None:
    """The real Data Analyst fixture's listings do not represent "Backend
    Engineer" signal — cross-role check, not just role-vs-itself.
    """
    listings = parse_search_jobs_response(
        (FIXTURES_DIR / "himalayas_search_jobs_data_analyst.txt").read_text()
    )
    assert has_usable_himalayas_signal("Backend Engineer", listings) is False


def test_real_nonsense_keyword_fixture_has_no_signal() -> None:
    """The real captured fallback response for a deliberately nonsensical
    keyword (PRD §7.2/§7.3's known limitation) — Himalayas returned 15
    unrelated listings (Moderator, etc.) rather than an empty result; the
    heuristic must still recognize this as no real signal.
    """
    listings = parse_search_jobs_response(
        (
            FIXTURES_DIR / "himalayas_search_jobs_nonsense_keyword_fallback.txt"
        ).read_text()
    )
    assert (
        len(listings) > 0
    )  # confirms this is exercising the heuristic, not the empty-list case
    assert has_usable_himalayas_signal("zzznonexistentrolezzz123456", listings) is False
