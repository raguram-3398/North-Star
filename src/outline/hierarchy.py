"""Insertion and positioning of outline topics into an existing dependency hierarchy."""

from typing import Any


def insert_new_topic(
    existing_topics: list[dict[str, Any]],
    new_topic: dict[str, Any],
    prerequisite_topic_ids: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Insert a new topic after its latest prerequisite and renumber hierarchy_position from that point on."""
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
    """Merge refreshed content into an existing topic in place, preserving its id and hierarchy_position."""
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
