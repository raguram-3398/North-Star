"""Tests for data/outline_topics.py's `insert_outline_topics` — the
persistence-layer gap closed by this task (PRD §11 item 8 / Architecture
§10, now resolved).

Uses a mocked SQLAlchemy Session rather than a real database, matching
`data/roles_cache.py`'s established convention (see test_roles_cache.py's
module docstring): CLAUDE.md's Stack section forbids a SQLite substitute
for Neon Postgres, and `OutlineTopic.id` uses a Postgres-specific
`sqlalchemy.dialects.postgresql.UUID` column type a SQLite engine
couldn't run anyway.

`get_topic`/`get_topics_in_group` are pre-existing and out of scope here.
`mark_topic_completed` gained an optional `status` parameter as part of
the test-out (verification-first) task — see the dedicated tests below —
but its pre-existing default-`"completed"` behavior is otherwise
unchanged.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.research_outline_agent import InitialOutlineTopic
from data.outline_topics import (
    COMPLETED_STATUS,
    COMPLETED_TEST_OUT_STATUS,
    NOT_STARTED_STATUS,
    augment_outline_topic,
    get_all_topics_for_user,
    get_completed_topics_matching_skill,
    has_pending_enrichment_topic,
    insert_new_outline_topic,
    insert_outline_topics,
    mark_topic_completed,
)
from security.output_guard import ConfidenceTier


def _topic(
    topic_name: str = "Git basics",
    hierarchy_position: int = 1,
    topic_group: str = "Git",
    position_in_group: int = 1,
    source_url: str = "https://example.com/git",
    source_type: str = "job_listing",
    confidence: ConfidenceTier = ConfidenceTier.HIGH,
    is_enrichment: bool = False,
    status: str = NOT_STARTED_STATUS,
) -> InitialOutlineTopic:
    """Build a real `InitialOutlineTopic` — the exact type
    `create_initial_outline`/`regenerate_outline_with_addition` produce —
    rather than a hand-rolled stand-in, so these tests exercise the real
    output shape.
    """
    return InitialOutlineTopic(
        topic_name=topic_name,
        hierarchy_position=hierarchy_position,
        topic_group=topic_group,
        position_in_group=position_in_group,
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
        is_enrichment=is_enrichment,
        status=status,
    )


def _session_with_existing_rows(rows: list[SimpleNamespace]) -> MagicMock:
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = rows
    return session


def test_insert_outline_topics_persists_a_fresh_outline() -> None:
    """The normal path: `create_initial_outline`'s shape, no prior rows
    for this user (a plain insert, not a replacement)."""
    session = _session_with_existing_rows([])
    topics = [
        _topic("Git basics", 1, "Git", 1),
        _topic("Git branching", 2, "Git", 2),
        _topic("Python syntax", 3, "Python", 1, source_url="https://example.com/py"),
    ]

    result = insert_outline_topics(session, "user-1", topics)

    assert len(result) == 3
    assert [row["topic_name"] for row in result] == [
        "Git basics",
        "Git branching",
        "Python syntax",
    ]
    assert all(row["id"] is not None for row in result)
    assert all(row["user_id"] == "user-1" for row in result)
    assert all(row["status"] == NOT_STARTED_STATUS for row in result)
    assert result[2]["confidence"] == "high"  # stored as the .value, not the enum

    session.delete.assert_not_called()
    session.add_all.assert_called_once()
    session.commit.assert_called_once()


def test_insert_outline_topics_regeneration_replaces_prior_unstarted_rows() -> None:
    """The regeneration path: `regenerate_outline_with_addition` produces
    a brand-new full topic set (PRD §7.5's "regenerates the full outline
    from scratch") that must replace, not append to, whatever this user
    already had — every prior row is still `not_started` since
    regeneration only ever happens pre-Day-1.
    """
    prior_row_1 = SimpleNamespace(id="old-1", status=NOT_STARTED_STATUS)
    prior_row_2 = SimpleNamespace(id="old-2", status=NOT_STARTED_STATUS)
    session = _session_with_existing_rows([prior_row_1, prior_row_2])

    new_topics = [
        _topic("Git basics", 1, "Git", 1),
        _topic("Git branching", 2, "Git", 2),
        _topic("GraphQL basics", 3, "GraphQL", 1, source_url="https://example.com/gql"),
    ]

    result = insert_outline_topics(session, "user-1", new_topics)

    assert session.delete.call_count == 2
    session.delete.assert_any_call(prior_row_1)
    session.delete.assert_any_call(prior_row_2)
    assert len(result) == 3
    assert {row["topic_name"] for row in result} == {
        "Git basics",
        "Git branching",
        "GraphQL basics",
    }
    session.commit.assert_called_once()


def test_insert_outline_topics_raises_if_an_existing_row_has_progressed() -> None:
    """Guardrail #2 ('never delete or reduce outline content') must block
    regeneration from silently discarding a row the user has already
    started or completed — this should never happen given Outline
    Confirmation's pre-Day-1 scope, but the function does not trust that
    invariant blindly.
    """
    in_progress_row = SimpleNamespace(id="old-1", status="in_progress")
    session = _session_with_existing_rows([in_progress_row])

    with pytest.raises(ValueError):
        insert_outline_topics(session, "user-1", [_topic()])

    session.delete.assert_not_called()
    session.add_all.assert_not_called()
    session.commit.assert_not_called()


def test_insert_outline_topics_rejects_empty_topic_list() -> None:
    session = _session_with_existing_rows([])

    with pytest.raises(ValueError):
        insert_outline_topics(session, "user-1", [])


def test_insert_outline_topics_rejects_a_raw_dict_in_place_of_a_topic_object() -> None:
    """CLAUDE.md guardrail #12: a raw dict must not be silently accepted
    in place of an already-sequenced topic object — this is a
    `TypeError`, not a `ValueError`, since it's a caller/type contract
    violation, not a data-integrity problem with otherwise-valid input.
    """
    session = _session_with_existing_rows([])
    raw_dict_topic = {
        "topic_name": "Git basics",
        "hierarchy_position": 1,
        "topic_group": "Git",
        "position_in_group": 1,
        "source_url": "https://example.com/git",
        "source_type": "job_listing",
        "confidence": "high",
        "is_enrichment": False,
        "status": NOT_STARTED_STATUS,
    }

    with pytest.raises(TypeError):
        insert_outline_topics(session, "user-1", [raw_dict_topic])  # type: ignore[list-item]

    session.add_all.assert_not_called()
    session.commit.assert_not_called()


def test_mark_topic_completed_defaults_to_completed_status() -> None:
    session = MagicMock()
    row = SimpleNamespace(status=NOT_STARTED_STATUS, completed_at=None)
    session.get.return_value = row

    mark_topic_completed(session, "t1")

    assert row.status == COMPLETED_STATUS
    assert row.completed_at is not None
    session.commit.assert_called_once()


def test_mark_topic_completed_accepts_completed_test_out_status() -> None:
    """The test-out task's reason for adding this parameter: Architecture
    §5 lists `completed_test_out` as a schema value distinct from
    `completed`, not a synonym — a test-out full/partial pass must write
    that specific value, not the regular one."""
    session = MagicMock()
    row = SimpleNamespace(status=NOT_STARTED_STATUS, completed_at=None)
    session.get.return_value = row

    mark_topic_completed(session, "t1", status=COMPLETED_TEST_OUT_STATUS)

    assert row.status == COMPLETED_TEST_OUT_STATUS
    session.commit.assert_called_once()


def test_mark_topic_completed_rejects_an_unrecognized_status() -> None:
    session = MagicMock()
    session.get.return_value = SimpleNamespace(status=NOT_STARTED_STATUS)

    with pytest.raises(ValueError):
        mark_topic_completed(session, "t1", status="something_else")

    session.commit.assert_not_called()


def _completed_row(**overrides: object) -> SimpleNamespace:
    fields: dict[str, object] = {
        "id": "t1",
        "user_id": "u1",
        "topic_name": "SQL",
        "hierarchy_position": 3,
        "topic_group": "SQL",
        "position_in_group": 1,
        "source_url": "https://example.com/sql-old",
        "source_type": "job_listing",
        "confidence": "medium",
        "is_enrichment": False,
        "status": COMPLETED_STATUS,
        "completed_at": "2026-01-01T00:00:00",
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_augment_outline_topic_refreshes_provenance_fields_only() -> None:
    """PRD §7.4's "Augmentation" update type + CLAUDE.md guardrail #5:
    only source_url/source_type/confidence change — status, completed_at,
    hierarchy_position, and topic_name (an already-completed topic) must
    never be touched by an augmentation.
    """
    session = MagicMock()
    row = _completed_row()
    session.get.return_value = row

    result = augment_outline_topic(
        session,
        "t1",
        source_url="https://example.com/sql-new",
        source_type="patch-note",
        confidence=ConfidenceTier.HIGH,
    )

    assert row.source_url == "https://example.com/sql-new"
    assert row.source_type == "patch-note"
    assert row.confidence == ConfidenceTier.HIGH.value
    assert row.status == COMPLETED_STATUS
    assert row.completed_at == "2026-01-01T00:00:00"
    assert row.hierarchy_position == 3
    assert row.topic_name == "SQL"
    session.commit.assert_called_once()

    assert result["source_url"] == "https://example.com/sql-new"
    assert result["source_type"] == "patch-note"
    assert result["confidence"] == ConfidenceTier.HIGH.value
    assert result["id"] == "t1"
    assert result["hierarchy_position"] == 3


def test_augment_outline_topic_raises_if_topic_not_found() -> None:
    session = MagicMock()
    session.get.return_value = None

    with pytest.raises(ValueError):
        augment_outline_topic(
            session,
            "missing",
            source_url="https://example.com/x",
            source_type="patch-note",
            confidence=ConfidenceTier.HIGH,
        )

    session.commit.assert_not_called()


def test_get_completed_topics_matching_skill_matches_case_insensitively() -> None:
    """Used by src/cron/refresh_roles.py's significant-event wiring —
    outline topic names and roles_cache skill names come from two
    different pipelines with no guaranteed identical casing.
    """
    row = SimpleNamespace(
        id="topic-1",
        user_id="user-1",
        topic_name="sql",
        hierarchy_position=3,
        topic_group="Databases",
        position_in_group=1,
        source_url="https://example.com/sql",
        source_type="job_listing",
        confidence="high",
        is_enrichment=False,
        status=COMPLETED_STATUS,
        completed_at=None,
    )
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [row]

    result = get_completed_topics_matching_skill(session, "SQL")

    assert len(result) == 1
    assert result[0]["id"] == "topic-1"
    assert result[0]["user_id"] == "user-1"


def test_get_completed_topics_matching_skill_returns_empty_when_no_match() -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = []

    result = get_completed_topics_matching_skill(session, "Kafka")

    assert result == []


def _topic_row(
    topic_id: str,
    hierarchy_position: int,
    topic_name: str = "X",
    topic_group: str = "G",
    position_in_group: int = 1,
    status: str = NOT_STARTED_STATUS,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=topic_id,
        user_id="user-1",
        topic_name=topic_name,
        hierarchy_position=hierarchy_position,
        topic_group=topic_group,
        position_in_group=position_in_group,
        source_url="https://example.com/x",
        source_type="job_listing",
        confidence="high",
        is_enrichment=False,
        status=status,
        completed_at=None,
    )


def test_get_all_topics_for_user_returns_every_row() -> None:
    row = _topic_row("t1", 1, topic_name="SQL")
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [row]

    result = get_all_topics_for_user(session, "user-1")

    assert len(result) == 1
    assert result[0]["topic_name"] == "SQL"


def test_has_pending_enrichment_topic_true_when_an_unresolved_one_exists() -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = _topic_row(
        "t1", 1
    )

    assert has_pending_enrichment_topic(session, "user-1") is True


def test_has_pending_enrichment_topic_false_when_none_exists() -> None:
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None

    assert has_pending_enrichment_topic(session, "user-1") is False


def test_insert_new_outline_topic_inserts_after_prerequisite_and_renumbers() -> None:
    """old-1 (position 1) is the sole prerequisite, old-2 (position 2) is
    not — the new topic must land immediately after old-1 (position 2),
    pushing old-2 to position 3. old-1 itself is untouched.
    """
    old1 = _topic_row("old-1", 1)
    old2 = _topic_row("old-2", 2)
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [old1, old2]

    result = insert_new_outline_topic(
        session,
        user_id="user-1",
        topic_name="GraphQL",
        topic_group="GraphQL (Enrichment)",
        position_in_group=1,
        source_url="https://example.com/graphql",
        source_type="roles_cache-cached",
        confidence=ConfidenceTier.MEDIUM,
        is_enrichment=True,
        prerequisite_topic_ids=frozenset({"old-1"}),
    )

    assert old1.hierarchy_position == 1
    assert old2.hierarchy_position == 3
    assert result["hierarchy_position"] == 2
    assert result["topic_name"] == "GraphQL"
    assert result["topic_group"] == "GraphQL (Enrichment)"
    assert result["is_enrichment"] is True
    assert result["confidence"] == "medium"
    assert result["status"] == NOT_STARTED_STATUS
    session.add.assert_called_once()
    session.commit.assert_called_once()


def test_insert_new_outline_topic_with_no_prerequisites_inserts_at_the_start() -> None:
    old1 = _topic_row("old-1", 1)
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [old1]

    result = insert_new_outline_topic(
        session,
        user_id="user-1",
        topic_name="gRPC",
        topic_group="gRPC (Enrichment)",
        position_in_group=1,
        source_url="https://example.com/grpc",
        source_type="roles_cache-cached",
        confidence=ConfidenceTier.LOW,
        is_enrichment=True,
    )

    assert result["hierarchy_position"] == 1
    assert old1.hierarchy_position == 2


def test_insert_new_outline_topic_rejects_unknown_prerequisite_id() -> None:
    old1 = _topic_row("old-1", 1)
    session = MagicMock()
    session.query.return_value.filter.return_value.all.return_value = [old1]

    with pytest.raises(ValueError):
        insert_new_outline_topic(
            session,
            user_id="user-1",
            topic_name="gRPC",
            topic_group="gRPC (Enrichment)",
            position_in_group=1,
            source_url="https://example.com/grpc",
            source_type="roles_cache-cached",
            confidence=ConfidenceTier.LOW,
            is_enrichment=True,
            prerequisite_topic_ids=frozenset({"nonexistent"}),
        )

    session.add.assert_not_called()
    session.commit.assert_not_called()
