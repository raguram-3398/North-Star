"""Confidence-ladder validation gate that every outline item, patch-note, or grounding result must pass before a DB write."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from utils.exceptions import ConfidenceValidationError


class ConfidenceTier(StrEnum):
    """The confidence ladder, in descending order of strength; REJECT means zero market signal was found."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CACHED_LOW = "cached-low"
    GENERAL_KNOWLEDGE_ONLY = "general-knowledge-only"
    REJECT = "reject"


@dataclass(frozen=True)
class ValidatedGroundedContent:
    """An outline item, patch-note, or grounding result that has passed the confidence-ladder validation gate."""

    source_url: str
    source_type: str
    confidence: ConfidenceTier
    extra: dict[str, Any] = field(default_factory=dict)


def validate_output_object(candidate: dict[str, Any]) -> ValidatedGroundedContent:
    """Validate that a candidate has a well-formed source_url, source_type, and non-reject confidence tier."""
    source_url = candidate.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise ConfidenceValidationError("candidate is missing a non-empty 'source_url'")
    parsed_url = urlparse(source_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ConfidenceValidationError(
            f"'source_url' is not a valid absolute URL: {source_url!r}"
        )

    source_type = candidate.get("source_type")
    if not isinstance(source_type, str) or not source_type.strip():
        raise ConfidenceValidationError(
            "candidate is missing a non-empty 'source_type'"
        )

    confidence_raw = candidate.get("confidence")
    if not isinstance(confidence_raw, str) or not confidence_raw.strip():
        raise ConfidenceValidationError("candidate is missing a non-empty 'confidence'")
    try:
        confidence = ConfidenceTier(confidence_raw)
    except ValueError as exc:
        raise ConfidenceValidationError(
            f"'confidence' is not a valid ladder tier: {confidence_raw!r}"
        ) from exc

    if confidence is ConfidenceTier.REJECT:
        raise ConfidenceValidationError(
            "candidate resolved to the 'reject' tier — zero market signal "
            "found; no record may be created (PRD §7.3)"
        )

    extra = {
        key: value
        for key, value in candidate.items()
        if key not in {"source_url", "source_type", "confidence"}
    }
    return ValidatedGroundedContent(
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
        extra=extra,
    )
