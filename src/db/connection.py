"""Lazily-memoized Neon Postgres Engine/Session management with bounded connect and statement timeouts."""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_SECONDS = 10

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _normalized_connection_string() -> str:
    """Read NEON_CONNECTION_STRING from the environment and rewrite it to the psycopg v3 dialect."""
    connection_string = os.environ.get("NEON_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("NEON_CONNECTION_STRING environment variable is not set")
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    return connection_string


def get_engine() -> Engine:
    """Return the module-level SQLAlchemy Engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            _normalized_connection_string(),
            connect_args={
                "connect_timeout": CONNECT_TIMEOUT_SECONDS,
                "options": f"-c statement_timeout={STATEMENT_TIMEOUT_SECONDS * 1000}",
            },
        )
    return _engine


def get_session() -> Session:
    """Return a new SQLAlchemy Session bound to the module-level engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine())
    return _session_factory()
