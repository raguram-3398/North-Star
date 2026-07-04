"""Tests for db/connection.py — Neon Postgres engine/session wiring.

Uses `monkeypatch` to set a fake connection string rather than relying on
the real `.env` file, so these tests need no real secrets and make no
network call — `create_engine()` is lazy about actually connecting;
these tests only verify the wiring (dialect normalization, singleton
memoization, the missing-env-var error), never a real Neon round-trip.
"""

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

import db.connection as connection


@pytest.fixture(autouse=True)
def _reset_module_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets a clean slate: the module-level engine/session
    factory are memoized globals, so a prior test's Engine would
    otherwise leak into the next one.
    """
    monkeypatch.setattr(connection, "_engine", None)
    monkeypatch.setattr(connection, "_session_factory", None)


def test_get_engine_raises_clearly_when_env_var_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NEON_CONNECTION_STRING", raising=False)

    with pytest.raises(RuntimeError):
        connection.get_engine()


def test_get_engine_normalizes_bare_postgresql_scheme_to_psycopg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEON_CONNECTION_STRING", "postgresql://user:pw@example.com/db")

    engine = connection.get_engine()

    assert isinstance(engine, Engine)
    assert str(engine.url).startswith("postgresql+psycopg://")


def test_get_engine_is_memoized_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEON_CONNECTION_STRING", "postgresql://user:pw@example.com/db")

    first = connection.get_engine()
    second = connection.get_engine()

    assert first is second


def test_get_session_returns_a_session_bound_to_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEON_CONNECTION_STRING", "postgresql://user:pw@example.com/db")

    session = connection.get_session()

    assert isinstance(session, Session)
    assert session.get_bind() is connection.get_engine()
    session.close()
