"""Outline insertion/positioning into an already-known dependency
hierarchy.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Distinct from initial full-hierarchy *creation*,
which requires reasoning and stays in Agent 1 (Architecture_North_Star.md
§2). Update policy is additive/refreshing only — content is never removed
(PRD §7.4, CLAUDE.md guardrail #2).
"""

from typing import Any


def insert_new_topic(
    existing_topics: list[dict[str, Any]], new_topic: dict[str, Any]
) -> list[dict[str, Any]]:
    """Insert a genuinely new topic at its correct hierarchical position
    within an existing outline, per PRD §7.4.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def augment_existing_topic(
    existing_topics: list[dict[str, Any]],
    topic_id: str,
    refreshed_content: dict[str, Any],
) -> list[dict[str, Any]]:
    """Refresh an existing topic's content in place without changing its
    hierarchical position or completion status, per PRD §7.4.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
