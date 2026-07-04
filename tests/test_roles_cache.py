"""Tests for data/roles_cache.py — roles_cache I/O.

Uses a mocked SQLAlchemy Session rather than a real database, per the
task's explicit choice between a test database and a mocked session.
Rationale (flagged for review): CLAUDE.md's Stack section explicitly
forbids substituting SQLite for Neon Postgres ("not local SQLite"), and
`upsert_role` uses a Postgres-specific `INSERT ... ON CONFLICT` — a
SQLite-backed test database couldn't even execute that statement, let
alone verify it faithfully. A mocked Session, injected as a plain
argument (see roles_cache.py's dependency-injection design), avoids both
problems and keeps these tests fast and hitting no network at all.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data.roles_cache import STALENESS_FLOOR_DAYS, get_role, is_stale, upsert_role
from models.schemas import RolesCache
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from utils.exceptions import ConfidenceValidationError


def test_get_role_returns_none_when_no_entry_exists() -> None:
    session = MagicMock()
    session.get.return_value = None

    result = get_role(session, "backend_engineer")

    assert result is None
    session.get.assert_called_once_with(RolesCache, "backend_engineer")


def test_get_role_returns_a_dict_when_an_entry_exists() -> None:
    session = MagicMock()
    last_updated = datetime(2026, 6, 1, 12, 0, 0)
    session.get.return_value = SimpleNamespace(
        role_name="backend_engineer",
        core_skills=[{"skill": "SQL", "source_url": "https://x", "confidence": "high"}],
        emerging_skills=[],
        last_updated=last_updated,
    )

    result = get_role(session, "backend_engineer")

    assert result == {
        "role_name": "backend_engineer",
        "core_skills": [
            {"skill": "SQL", "source_url": "https://x", "confidence": "high"}
        ],
        "emerging_skills": [],
        "last_updated": last_updated,
    }


def test_upsert_role_executes_an_upsert_and_commits() -> None:
    session = MagicMock()
    core_skills = [
        ValidatedGroundedContent(
            source_url="https://x",
            source_type="job_listing",
            confidence=ConfidenceTier.HIGH,
            extra={"skill": "SQL"},
        )
    ]
    emerging_skills = [
        ValidatedGroundedContent(
            source_url="https://y",
            source_type="job_listing",
            confidence=ConfidenceTier.LOW,
            extra={"skill": "dbt"},
        )
    ]

    upsert_role(session, "backend_engineer", core_skills, emerging_skills)

    session.execute.assert_called_once()
    statement = session.execute.call_args.args[0]
    assert statement.table.name == "roles_cache"

    compiled = statement.compile()
    assert "ON CONFLICT" in str(compiled)
    assert compiled.params["role_name"] == "backend_engineer"
    assert compiled.params["core_skills"] == [
        {"skill": "SQL", "source_url": "https://x", "confidence": "high"}
    ]
    assert compiled.params["emerging_skills"] == [
        {"skill": "dbt", "source_url": "https://y", "confidence": "low"}
    ]
    assert isinstance(compiled.params["last_updated"], datetime)

    session.commit.assert_called_once()


def test_upsert_role_rejects_raw_dict_in_place_of_validated_grounded_content() -> None:
    """A raw, unvalidated dict must not be silently accepted in place of
    a ValidatedGroundedContent instance — CLAUDE.md guardrail #12
    requires a post-output_guard object at this write boundary. This is
    an explicit isinstance check raising ConfidenceValidationError, not
    an incidental AttributeError from a dict lacking `.extra` — the
    enforcement is intentional and self-documenting, consistent with
    every other guard in this codebase.
    """
    session = MagicMock()
    unvalidated_core_skills = [
        {"skill": "SQL", "source_url": "https://x", "confidence": "high"}
    ]

    with pytest.raises(ConfidenceValidationError):
        upsert_role(
            session,
            "backend_engineer",
            unvalidated_core_skills,  # type: ignore[arg-type]
            [],
        )

    session.execute.assert_not_called()
    session.commit.assert_not_called()


def test_upsert_role_rejects_skill_entry_missing_skill_name() -> None:
    session = MagicMock()
    missing_skill_name = [
        ValidatedGroundedContent(
            source_url="https://x",
            source_type="job_listing",
            confidence=ConfidenceTier.HIGH,
            extra={},
        )
    ]

    with pytest.raises(ConfidenceValidationError):
        upsert_role(session, "backend_engineer", missing_skill_name, [])


def test_is_stale_false_within_the_thirty_day_floor() -> None:
    last_updated = datetime(2026, 6, 1)
    reference_time = last_updated + timedelta(days=STALENESS_FLOOR_DAYS - 1)

    assert is_stale(last_updated, reference_time) is False


def test_is_stale_false_exactly_at_the_thirty_day_boundary() -> None:
    """Exactly 30 days old is still within the floor, not past it —
    `is_stale` uses a strict `>` comparison.
    """
    last_updated = datetime(2026, 6, 1)
    reference_time = last_updated + timedelta(days=STALENESS_FLOOR_DAYS)

    assert is_stale(last_updated, reference_time) is False


def test_is_stale_true_once_past_the_thirty_day_floor() -> None:
    last_updated = datetime(2026, 6, 1)
    reference_time = last_updated + timedelta(days=STALENESS_FLOOR_DAYS + 1)

    assert is_stale(last_updated, reference_time) is True


def test_is_stale_respects_a_custom_max_age() -> None:
    last_updated = datetime(2026, 6, 1)
    reference_time = last_updated + timedelta(days=5)

    assert is_stale(last_updated, reference_time, max_age_days=3) is True
    assert is_stale(last_updated, reference_time, max_age_days=10) is False
