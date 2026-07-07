"""Tests for outline/hierarchy.py: insertion and positioning into an existing dependency hierarchy."""

import pytest

from outline.hierarchy import augment_existing_topic, insert_new_topic


def _topics(*names: str) -> list[dict[str, str | int]]:
    """Build a simple 1-indexed ordered topic list for test fixtures."""
    return [
        {"id": f"t{i}", "hierarchy_position": i, "topic_name": name}
        for i, name in enumerate(names, start=1)
    ]


def test_insert_new_topic_in_the_middle_renumbers_everything_after() -> None:
    existing = _topics("Python Basics", "Control Flow", "Functions", "OOP")
    new_topic = {
        "id": "new",
        "hierarchy_position": -1,
        "topic_name": "List Comprehensions",
    }

    result = insert_new_topic(
        existing, new_topic, prerequisite_topic_ids=frozenset({"t2"})
    )

    assert [t["id"] for t in result] == ["t1", "t2", "new", "t3", "t4"]
    assert [t["hierarchy_position"] for t in result] == [1, 2, 3, 4, 5]
    assert [t["topic_name"] for t in result] == [
        "Python Basics",
        "Control Flow",
        "List Comprehensions",
        "Functions",
        "OOP",
    ]


def test_insert_new_topic_at_the_very_start_with_no_prerequisites() -> None:
    existing = _topics("Control Flow", "Functions")
    new_topic = {"id": "new", "hierarchy_position": -1, "topic_name": "Python Basics"}

    result = insert_new_topic(existing, new_topic)

    assert [t["id"] for t in result] == ["new", "t1", "t2"]
    assert [t["hierarchy_position"] for t in result] == [1, 2, 3]


def test_insert_new_topic_at_the_very_end() -> None:
    existing = _topics("Python Basics", "Control Flow", "Functions")
    new_topic = {"id": "new", "hierarchy_position": -1, "topic_name": "Decorators"}

    result = insert_new_topic(
        existing, new_topic, prerequisite_topic_ids=frozenset({"t3"})
    )

    assert [t["id"] for t in result] == ["t1", "t2", "t3", "new"]
    assert [t["hierarchy_position"] for t in result] == [1, 2, 3, 4]


def test_insert_new_topic_rejects_duplicate_id() -> None:
    existing = _topics("Python Basics")
    new_topic = {"id": "t1", "hierarchy_position": -1, "topic_name": "Duplicate"}

    with pytest.raises(ValueError):
        insert_new_topic(existing, new_topic)


def test_insert_new_topic_rejects_unknown_prerequisite_id() -> None:
    existing = _topics("Python Basics")
    new_topic = {"id": "new", "hierarchy_position": -1, "topic_name": "X"}

    with pytest.raises(ValueError):
        insert_new_topic(
            existing, new_topic, prerequisite_topic_ids=frozenset({"nope"})
        )


def test_augment_existing_topic_leaves_ordering_completely_untouched() -> None:
    existing = _topics("Python Basics", "Control Flow", "Functions")
    original_positions = [t["hierarchy_position"] for t in existing]
    original_order = [t["id"] for t in existing]

    result = augment_existing_topic(
        existing, "t2", {"source_url": "https://example.com/refreshed"}
    )

    assert [t["id"] for t in result] == original_order
    assert [t["hierarchy_position"] for t in result] == original_positions
    augmented = next(t for t in result if t["id"] == "t2")
    assert augmented["source_url"] == "https://example.com/refreshed"
    assert augmented["topic_name"] == "Control Flow"


def test_augment_existing_topic_ignores_attempted_id_and_position_override() -> None:
    """Augmentation must never reposition anything, even if refreshed_content tries to smuggle in a new id or position."""
    existing = _topics("Python Basics", "Control Flow")

    result = augment_existing_topic(
        existing,
        "t1",
        {"id": "sneaky", "hierarchy_position": 99, "topic_name": "New Name"},
    )

    augmented = result[0]
    assert augmented["id"] == "t1"
    assert augmented["hierarchy_position"] == 1
    assert augmented["topic_name"] == "New Name"


def test_augment_existing_topic_rejects_unknown_topic_id() -> None:
    existing = _topics("Python Basics")

    with pytest.raises(ValueError):
        augment_existing_topic(existing, "nope", {"topic_name": "X"})
