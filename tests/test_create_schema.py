"""Tests for db/create_schema.py — one-time schema creation script.

This hits a real database in production use, so per CLAUDE.md's testing
expectations and this codebase's established mocked-Session/mocked-engine
convention (see test_roles_cache.py, test_connection.py), these tests never
touch a real Neon instance. Two things are verified instead: (1) a
structural check that all 7 expected tables are actually registered on
`Base.metadata` by the time `create_schema` is imported, and (2) that
`create_all_tables()` calls `Base.metadata.create_all()` with the engine
returned by `db.connection.get_engine()` — the engine itself is mocked, so
no network call is made.
"""

from unittest.mock import MagicMock

import pytest

from db.create_schema import REGISTERED_MODELS, create_all_tables
from models.schemas import Base

EXPECTED_TABLE_NAMES = {
    "users",
    "roles_cache",
    "outline_topics",
    "progress_log",
    "verification_attempts",
    "patch_notes",
    "pace_snapshots",
}


def test_all_7_expected_tables_are_registered_on_base_metadata() -> None:
    assert set(Base.metadata.tables.keys()) == EXPECTED_TABLE_NAMES


def test_registered_models_tuple_covers_all_7_tables() -> None:
    registered_table_names = {model.__tablename__ for model in REGISTERED_MODELS}
    assert registered_table_names == EXPECTED_TABLE_NAMES


def test_create_all_tables_calls_create_all_on_the_engine_from_get_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_engine = MagicMock()
    monkeypatch.setattr("db.create_schema.get_engine", lambda: fake_engine)
    monkeypatch.setattr(Base.metadata, "create_all", MagicMock())

    create_all_tables()

    Base.metadata.create_all.assert_called_once_with(fake_engine)
