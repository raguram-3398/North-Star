"""Confidence-ladder enforcement — the structural gate before any DB write.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per CLAUDE.md guardrail #1 and #12: no outline item,
patch-note, or grounding result may be written without a populated
source_url and confidence, and no DB write function may accept an
unvalidated object — only a post-output_guard object.

Canonical ladder (Architecture_North_Star.md §8):
high -> medium -> low -> cached-low -> general-knowledge-only -> reject
"""

from typing import Any


def assign_confidence_tier(sources_agree: bool, source_count: int) -> str:
    """Deterministically map a cross-validation outcome to a confidence
    tier on the canonical ladder (PRD §7.3).

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def validate_output_object(candidate: dict[str, Any]) -> dict[str, Any]:
    """Validate that a candidate outline item, patch-note, or grounding
    result carries a populated source_url and confidence before it may be
    written to Postgres; raise ConfidenceValidationError otherwise.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
