"""Tests for data/users.py — users I/O.

Uses a mocked SQLAlchemy Session, matching data/roles_cache.py's
established convention (see test_roles_cache.py's module docstring).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data.users import extend_pacing, get_user


def test_get_user_returns_none_when_no_entry_exists() -> None:
    session = MagicMock()
    session.get.return_value = None

    result = get_user(session, "user-1")

    assert result is None


def test_get_user_returns_a_dict_when_an_entry_exists() -> None:
    session = MagicMock()
    session.get.return_value = SimpleNamespace(
        id="user-1",
        background="backend developer",
        current_job="SWE",
        years_experience=3,
        prior_self_study="some Udemy courses",
        available_time_per_week=10,
        resolved_role="Backend Engineer",
        role_confidence="high",
        pacing_profile="medium",
        pace_extension_days=0,
        created_at=None,
    )

    result = get_user(session, "user-1")

    assert result is not None
    assert result["resolved_role"] == "Backend Engineer"
    assert result["pace_extension_days"] == 0


def test_extend_pacing_increments_existing_total_and_commits() -> None:
    session = MagicMock()
    row = SimpleNamespace(pace_extension_days=2)
    session.get.return_value = row

    new_total = extend_pacing(session, "user-1", 3)

    assert new_total == 5
    assert row.pace_extension_days == 5
    session.commit.assert_called_once()


def test_extend_pacing_raises_if_user_not_found() -> None:
    session = MagicMock()
    session.get.return_value = None

    with pytest.raises(ValueError):
        extend_pacing(session, "missing-user", 2)

    session.commit.assert_not_called()


def test_extend_pacing_rejects_non_positive_extension() -> None:
    session = MagicMock()

    with pytest.raises(ValueError):
        extend_pacing(session, "user-1", 0)

    session.get.assert_not_called()
