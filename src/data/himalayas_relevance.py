"""Relevance heuristic that infers whether Himalayas search_jobs results are actually on-topic, since a genuine empty result never occurs."""

import re

from data.himalayas_parser import ParsedJobListing

PER_LISTING_TOKEN_OVERLAP_THRESHOLD = 0.5

MIN_COUNT_THRESHOLD = 5

MAX_COUNT_THRESHOLD = 25

MIN_COUNT_RELEVANCE_FRACTION = 0.6

MAX_COUNT_RELEVANCE_FRACTION = 0.2

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Lowercase and tokenize text into alphanumeric words for a simple overlap check, with no stemming or synonym handling."""
    return set(_WORD_PATTERN.findall(text.lower()))


def compute_title_relevance(searched_role: str, listing_title: str) -> float:
    """Compute the fraction of the searched role's tokens that also appear in a listing's title, or 0.0 if the role tokenizes to nothing."""
    role_tokens = _tokenize(searched_role)
    if not role_tokens:
        return 0.0
    title_tokens = _tokenize(listing_title)
    return len(role_tokens & title_tokens) / len(role_tokens)


def is_listing_relevant(searched_role: str, listing_title: str) -> bool:
    """Whether a single listing's title clears the per-listing token overlap threshold."""
    return (
        compute_title_relevance(searched_role, listing_title)
        >= PER_LISTING_TOKEN_OVERLAP_THRESHOLD
    )


def required_relevance_fraction(listing_count: int) -> float:
    """Compute the fraction of listings that must be individually relevant for a batch of this size to count as real signal."""
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
    """Determine whether a set of Himalayas listings represents real, on-topic signal for the searched role."""
    if not listings:
        return False

    relevant_count = sum(
        1 for listing in listings if is_listing_relevant(searched_role, listing.title)
    )
    relevant_fraction = relevant_count / len(listings)
    return relevant_fraction >= required_relevance_fraction(len(listings))
