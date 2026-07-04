"""Outline insertion/positioning into an already-known dependency
hierarchy.

Pure, deterministic, no LLM calls, no DB reads/writes (CLAUDE.md: pure
functions stay pure). Distinct from initial full-hierarchy *creation*,
which requires reasoning and stays in Agent 1 (Architecture_North_Star.md
§2/§9). Update policy is additive/refreshing only — content is never
removed (PRD §7.4, CLAUDE.md guardrail #2).

Design note — judgment call, flagged for review: Architecture_North_Star.md
§5's outline_topics schema represents the hierarchy as a flat, contiguous
`hierarchy_position` integer; there is no prerequisite-edge/graph column.
This module therefore accepts `prerequisite_topic_ids` as a plain
parameter (a set of existing topic IDs the new topic must come after),
not a persisted field — Agent 1 is expected to have already resolved the
new topic's dependencies down to this simple set during its reasoning
step, before calling insert_new_topic. Only *prerequisites* (topics that
must precede the new one) are modeled here, not *dependents* (existing
topics that must follow it): a brand-new topic cannot already have
existing topics depending on it, since those existing topics were
positioned before this topic existed. If a future requirement needs a new
topic inserted strictly before some existing topic regardless of
prerequisites, that would need a second parameter and is out of scope
here.
"""

from typing import Any


# Known limitation: this function only models "must-follow" constraints
# (prerequisite_topic_ids). It does not support a "must-precede" constraint
# (inserting a new/enrichment topic before some existing topic). Agent 1's
# reasoning is currently responsible for never producing a prerequisite set
# that would require must-precede to be correct. If enrichment positioning
# (PRD §7.9) is found to need this, this function's signature will need a
# second parameter and conflict-resolution logic.
def insert_new_topic(
    existing_topics: list[dict[str, Any]],
    new_topic: dict[str, Any],
    prerequisite_topic_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Insert a genuinely new topic into an existing, already-ordered
    hierarchy immediately after its latest prerequisite (or at the very
    start if it has none), renumbering `hierarchy_position` for every
    topic from that point on. Existing topics are never removed, and
    never reordered relative to each other — only shifted to make room
    (PRD §7.4).

    `existing_topics` and `new_topic` are plain dicts, each expected to
    carry at least "id" and "hierarchy_position" keys. The returned
    list's `hierarchy_position` values are renumbered 1..N contiguously,
    regardless of the numbering scheme on the way in.

    Raises ValueError if `new_topic`'s id already exists in
    `existing_topics`, or if any id in `prerequisite_topic_ids` is not
    found in `existing_topics`.
    """
    new_topic_id = new_topic["id"]
    if any(topic["id"] == new_topic_id for topic in existing_topics):
        raise ValueError(f"topic id {new_topic_id!r} already exists in the hierarchy")

    ordered_topics = sorted(
        existing_topics, key=lambda topic: topic["hierarchy_position"]
    )

    if prerequisite_topic_ids:
        prerequisite_indices = [
            index
            for index, topic in enumerate(ordered_topics)
            if topic["id"] in prerequisite_topic_ids
        ]
        found_ids = {ordered_topics[index]["id"] for index in prerequisite_indices}
        missing_ids = prerequisite_topic_ids - found_ids
        if missing_ids:
            raise ValueError(
                "prerequisite topic id(s) not found in existing_topics: "
                f"{sorted(missing_ids)}"
            )
        insert_index = max(prerequisite_indices) + 1
    else:
        insert_index = 0

    spliced = (
        ordered_topics[:insert_index] + [new_topic] + ordered_topics[insert_index:]
    )

    return [
        {**topic, "hierarchy_position": position}
        for position, topic in enumerate(spliced, start=1)
    ]


def augment_existing_topic(
    existing_topics: list[dict[str, Any]],
    topic_id: str,
    refreshed_content: dict[str, Any],
) -> list[dict[str, Any]]:
    """Refresh an existing topic's content in place, without changing its
    hierarchical position or that of any other topic (PRD §7.4's
    "augmentation" update type — nearly a no-op for this module beyond
    the content merge itself).

    `id` and `hierarchy_position` are always preserved from the original
    topic regardless of what `refreshed_content` contains, so an
    augmentation can never reposition anything even if the caller
    accidentally includes those keys. The list's order is left completely
    untouched — topics are not re-sorted by `hierarchy_position`.

    Raises ValueError if `topic_id` is not found in `existing_topics`.
    """
    if not any(topic["id"] == topic_id for topic in existing_topics):
        raise ValueError(f"topic id {topic_id!r} not found in existing_topics")

    return [
        (
            {
                **topic,
                **refreshed_content,
                "id": topic["id"],
                "hierarchy_position": topic["hierarchy_position"],
            }
            if topic["id"] == topic_id
            else topic
        )
        for topic in existing_topics
    ]
