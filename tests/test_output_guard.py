"""Tests for security/output_guard.py — confidence-ladder enforcement, the
structural gate before any DB write.
"""

from dataclasses import FrozenInstanceError

import pytest

from security.output_guard import (
    ConfidenceTier,
    ValidatedGroundedContent,
    validate_output_object,
)
from utils.exceptions import ConfidenceValidationError

VALID_CANDIDATE = {
    "source_url": "https://example.com/job-postings/backend-engineer",
    "source_type": "job_listing",
    "topic_name": "SQL Joins",
}


def test_confidence_tier_values_match_canonical_ladder() -> None:
    """The enum's string values must match Architecture_North_Star.md §8's
    canonical ladder exactly, since they are compared/persisted as raw
    strings elsewhere in the system.
    """
    assert ConfidenceTier.HIGH.value == "high"
    assert ConfidenceTier.MEDIUM.value == "medium"
    assert ConfidenceTier.LOW.value == "low"
    assert ConfidenceTier.CACHED_LOW.value == "cached-low"
    assert ConfidenceTier.GENERAL_KNOWLEDGE_ONLY.value == "general-knowledge-only"
    assert ConfidenceTier.REJECT.value == "reject"


@pytest.mark.parametrize(
    "tier", ["high", "medium", "low", "cached-low", "general-knowledge-only"]
)
def test_validate_output_object_accepts_every_persistable_tier(tier: str) -> None:
    """Every tier except `reject` must produce a validated object."""
    candidate = {**VALID_CANDIDATE, "confidence": tier}

    result = validate_output_object(candidate)

    assert isinstance(result, ValidatedGroundedContent)
    assert result.source_url == candidate["source_url"]
    assert result.source_type == candidate["source_type"]
    assert result.confidence == ConfidenceTier(tier)
    assert result.extra == {"topic_name": "SQL Joins"}


def test_validate_output_object_rejects_reject_tier() -> None:
    """A candidate resolved to `reject` (zero market signal) must never
    become a writable object — PRD §7.3: no record is created.
    """
    candidate = {**VALID_CANDIDATE, "confidence": "reject"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_missing_source_url() -> None:
    candidate = {"source_type": "job_listing", "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_empty_source_url() -> None:
    candidate = {**VALID_CANDIDATE, "source_url": "   ", "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_malformed_source_url() -> None:
    candidate = {**VALID_CANDIDATE, "source_url": "not-a-url", "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_scheme_only_source_url() -> None:
    """A URL that is structurally scheme-like but has no host or path
    (e.g. "https://") must be rejected — it parses without error but
    carries no real source, and is not just fully-empty or missing.
    """
    candidate = {**VALID_CANDIDATE, "source_url": "https://", "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_missing_source_type() -> None:
    candidate = {"source_url": VALID_CANDIDATE["source_url"], "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_empty_source_type() -> None:
    candidate = {**VALID_CANDIDATE, "source_type": "", "confidence": "high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_missing_confidence() -> None:
    candidate = {
        "source_url": VALID_CANDIDATE["source_url"],
        "source_type": VALID_CANDIDATE["source_type"],
    }

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validate_output_object_rejects_unknown_confidence_string() -> None:
    candidate = {**VALID_CANDIDATE, "confidence": "super-duper-high"}

    with pytest.raises(ConfidenceValidationError):
        validate_output_object(candidate)


def test_validated_grounded_content_is_immutable() -> None:
    """The validated object type must be immutable so nothing between the
    gate and the DB write can silently alter a validated field.
    """
    candidate = {**VALID_CANDIDATE, "confidence": "high"}
    result = validate_output_object(candidate)

    with pytest.raises(FrozenInstanceError):
        result.source_url = "https://example.com/tampered"  # type: ignore[misc]
