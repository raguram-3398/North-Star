"""Tests for db/create_schema.py: verifies all expected tables are registered and create_all_tables calls create_all with a mocked engine."""

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
