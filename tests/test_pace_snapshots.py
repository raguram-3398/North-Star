"""Tests for data/pace_snapshots.py — pace_snapshots I/O.

Uses a mocked SQLAlchemy Session, matching data/roles_cache.py's
established convention (see test_roles_cache.py's module docstring).
"""

from datetime import datetime
from unittest.mock import MagicMock

from data.pace_snapshots import get_pace_snapshot_history, write_pace_snapshot


def test_write_pace_snapshot_appends_a_row_and_commits() -> None:
    session = MagicMock()

    write_pace_snapshot(
        session,
        user_id="user-1",
        topic_id="topic-1",
        topic_score=0.9,
        timing_ratio=1.1,
        days_taken=3,
        days_expected=3,
    )

    session.add.assert_called_once()
    added_row = session.add.call_args.args[0]
    assert added_row.user_id == "user-1"
    assert added_row.topic_id == "topic-1"
    assert added_row.topic_score == 0.9
    assert added_row.timing_ratio == 1.1
    assert added_row.days_taken == 3
    assert added_row.days_expected == 3
    assert isinstance(added_row.computed_at, datetime)
    session.commit.assert_called_once()


def test_get_pace_snapshot_history_returns_rows_ordered_oldest_first() -> None:
    row1 = MagicMock(
        topic_score=0.9,
        timing_ratio=1.0,
        days_taken=3,
        days_expected=3,
        computed_at=datetime(2026, 7, 1),
    )
    row2 = MagicMock(
        topic_score=0.8,
        timing_ratio=1.2,
        days_taken=4,
        days_expected=3,
        computed_at=datetime(2026, 7, 3),
    )
    session = MagicMock()
    query_chain = session.query.return_value.filter.return_value.order_by.return_value
    query_chain.all.return_value = [row1, row2]

    result = get_pace_snapshot_history(session, "user-1")

    assert result == [
        {
            "topic_score": 0.9,
            "timing_ratio": 1.0,
            "days_taken": 3,
            "days_expected": 3,
            "computed_at": datetime(2026, 7, 1),
        },
        {
            "topic_score": 0.8,
            "timing_ratio": 1.2,
            "days_taken": 4,
            "days_expected": 3,
            "computed_at": datetime(2026, 7, 3),
        },
    ]


def test_get_pace_snapshot_history_returns_empty_list_for_no_snapshots() -> None:
    session = MagicMock()
    query_chain = session.query.return_value.filter.return_value.order_by.return_value
    query_chain.all.return_value = []

    result = get_pace_snapshot_history(session, "user-1")

    assert result == []
