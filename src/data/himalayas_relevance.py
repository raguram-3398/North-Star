"""Relevance heuristic for Himalayas MCP `search_jobs` results (PRD
§7.2/§7.3's known limitation; Architecture_North_Star.md §8).

Himalayas's `search_jobs` never returns a genuine empty result under live
testing (documented in PRD §7.2/§7.3 and Architecture §8) — even a
deliberately nonsensical keyword returns a full page of unrelated
listings (`tests/fixtures/himalayas_search_jobs_nonsense_keyword_fallback.txt`
is a real captured example: 15 "Moderator"/"Community Manager"-type
listings for the keyword `"zzznonexistentrolezzz123456"`). So "no signal
from Himalayas" cannot be read off an empty response; it must be inferred
from whether the returned listings are actually on-topic for the searched
role. This module is that inference.

Per PRD §7.0 ("Shift Intelligence Left"): judging word overlap between a
known query string and a set of returned titles is pattern matching on
known inputs, not semantic judgment — so this lives here as a plain,
deterministic module (no I/O, no LLM calls), the same way
`data/himalayas_parser.py` does for the parsing step upstream of this one.
"""

import re

from data.himalayas_parser import ParsedJobListing

# --- Judgment-call constants ---------------------------------------------
# No real-world data exists yet to calibrate these — they are a reasonable,
# testable starting mechanism (mirrors pace/calculator.py's
# TIMING_OUTLIER_THRESHOLD/TIMING_SATURATION_DEVIATION banding pattern),
# flagged for review/tuning once real cross-validation runs are observed,
# not presented as settled.

# A listing is "relevant" if at least this fraction of the searched role's
# tokens appear in the listing's title. 0.5 means, for a two-word role
# like "Backend Engineer", a title containing just one of the two words
# (e.g. "Senior Engineer") is borderline-relevant; a title containing both
# clears it comfortably.
PER_LISTING_TOKEN_OVERLAP_THRESHOLD = 0.5

# At or below this many returned listings, the batch is small enough that
# a high fraction of them must be individually relevant before the batch
# as a whole is trusted as real signal.
MIN_COUNT_THRESHOLD = 5

# At or above this many returned listings, a much lower relevant fraction
# is sufficient — large batches are usually Himalayas's broad-match
# fallback (see module docstring), which dilutes the relevant fraction
# just by being large, so demanding the same high fraction at scale would
# make even genuine searches with many results fail.
MAX_COUNT_THRESHOLD = 25

# Required relevant-listing fraction at or below MIN_COUNT_THRESHOLD.
MIN_COUNT_RELEVANCE_FRACTION = 0.6

# Required relevant-listing fraction at or above MAX_COUNT_THRESHOLD.
MAX_COUNT_RELEVANCE_FRACTION = 0.2

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase, alphanumeric-word tokenization for a simple overlap
    check. Judgment call: no stemming or synonym handling (e.g.
    "Engineer" and "Engineering" are different tokens) — deliberately
    simple, since this is a coarse relevance filter, not a semantic
    search.
    """
    return set(_WORD_PATTERN.findall(text.lower()))


def compute_title_relevance(searched_role: str, listing_title: str) -> float:
    """Fraction of `searched_role`'s tokens that also appear in
    `listing_title` — e.g. `searched_role="Backend Engineer"` against a
    title containing both "backend" and "engineer" scores 1.0; a title
    containing only one of the two scores 0.5.

    Returns 0.0 if `searched_role` tokenizes to nothing (e.g. an empty
    string) — there is nothing to overlap with, so nothing can be
    relevant to it.
    """
    role_tokens = _tokenize(searched_role)
    if not role_tokens:
        return 0.0
    title_tokens = _tokenize(listing_title)
    return len(role_tokens & title_tokens) / len(role_tokens)


def is_listing_relevant(searched_role: str, listing_title: str) -> bool:
    """Whether a single listing's title clears `compute_title_relevance`'s
    `PER_LISTING_TOKEN_OVERLAP_THRESHOLD`.
    """
    return (
        compute_title_relevance(searched_role, listing_title)
        >= PER_LISTING_TOKEN_OVERLAP_THRESHOLD
    )


def required_relevance_fraction(listing_count: int) -> float:
    """The fraction of listings that must be individually relevant for a
    batch of `listing_count` results to count as real signal, linearly
    scaled between `MIN_COUNT_RELEVANCE_FRACTION` (at or below
    `MIN_COUNT_THRESHOLD` listings) and `MAX_COUNT_RELEVANCE_FRACTION` (at
    or above `MAX_COUNT_THRESHOLD` listings) — mirrors the linear banding
    `pace/calculator.py`'s `calculate_combined_pace_signal` uses between
    `TIMING_OUTLIER_THRESHOLD` and `TIMING_SATURATION_DEVIATION`.
    """
    if listing_count <= MIN_COUNT_THRESHOLD:
        return MIN_COUNT_RELEVANCE_FRACTION
    if listing_count >= MAX_COUNT_THRESHOLD:
        return MAX_COUNT_RELEVANCE_FRACTION

    span = MAX_COUNT_THRESHOLD - MIN_COUNT_THRESHOLD
    progress = (listing_count - MIN_COUNT_THRESHOLD) / span
    fraction_range = MIN_COUNT_RELEVANCE_FRACTION - MAX_COUNT_RELEVANCE_FRACTION
    return MIN_COUNT_RELEVANCE_FRACTION - progress * fraction_range


def has_usable_himalayas_signal(
    searched_role: str, listings: list[ParsedJobListing]
) -> bool:
    """Determine whether Himalayas's `search_jobs` results represent real,
    on-topic signal for `searched_role`, or should be treated as no
    signal at all despite being a non-empty response (PRD §7.2/§7.3's
    known limitation).

    An empty `listings` list is always "no signal" (this is the one case
    that genuinely doesn't need the heuristic — see
    `data/himalayas_parser.py`'s handling of a real "Found 0 jobs
    matching" header). Otherwise, the fraction of individually relevant
    listings (`is_listing_relevant`) must meet or exceed
    `required_relevance_fraction` for this batch's size.
    """
    if not listings:
        return False

    relevant_count = sum(
        1 for listing in listings if is_listing_relevant(searched_role, listing.title)
    )
    relevant_fraction = relevant_count / len(listings)
    return relevant_fraction >= required_relevance_fraction(len(listings))
