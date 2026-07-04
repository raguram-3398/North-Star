"""Confidence-ladder enforcement — the structural gate before any DB write.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per CLAUDE.md guardrail #1 and #12: no outline item,
patch-note, or grounding result may be written without a populated
source_url and confidence, and no DB write function may accept an
unvalidated object — only a post-output_guard object.

Canonical ladder (Architecture_North_Star.md §8):
high -> medium -> low -> cached-low -> general-knowledge-only -> reject
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from utils.exceptions import ConfidenceValidationError


class ConfidenceTier(StrEnum):
    """The canonical confidence ladder, in descending order of strength
    (Architecture_North_Star.md §8). `REJECT` means zero market signal was
    found — per PRD §7.3, no record is ever created at this tier."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CACHED_LOW = "cached-low"
    GENERAL_KNOWLEDGE_ONLY = "general-knowledge-only"
    REJECT = "reject"


@dataclass(frozen=True)
class ValidatedGroundedContent:
    """An outline item, patch-note, or grounding result that has passed
    the confidence-ladder and source-validation gate. Only an object of
    this type may reach a DB write function (CLAUDE.md guardrail #12) —
    never a raw dict.
    """

    source_url: str
    source_type: str
    confidence: ConfidenceTier
    extra: dict[str, Any] = field(default_factory=dict)


def assign_confidence_tier(sources_agree: bool, source_count: int) -> ConfidenceTier:
    """Deterministically map a cross-validation outcome to a confidence
    tier on the canonical ladder (PRD §7.3).

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def validate_output_object(candidate: dict[str, Any]) -> ValidatedGroundedContent:
    """Validate that a candidate outline item, patch-note, or grounding
    result carries a populated, well-formed source_url, source_type, and
    confidence before it may be written to Postgres.

    Raises ConfidenceValidationError if any required field is missing,
    empty, malformed, or resolves to the `reject` tier (PRD §7.3: zero
    signal means no record is ever created) — never silently passes
    ungrounded content through (CLAUDE.md guardrail #1).
    """
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
