"""Neon Postgres connection/session management.

Per CLAUDE.md's coding conventions: one client (Engine/connection pool)
per module. A fresh `Session` is still created per call — that is the
normal SQLAlchemy unit-of-work pattern, not a violation of "one client
per module": the `Engine` is the client being guarded against
re-instantiation, not the lightweight `Session` object.

Judgment call, flagged for review: rather than instantiating the engine
as the very first thing executed at import time (which would make
`NEON_CONNECTION_STRING` a hard import-time requirement for every module
that transitively imports this one, including in CI/test contexts with no
DB configured), engine creation is lazy-but-memoized — created on first
call to get_engine()/get_session(), cached in a module-level variable,
and never recreated afterward. This still satisfies the intent of "one
client per module" (exactly one Engine instance ever exists per process);
it avoids only the literal instant of creation, not the singleton
guarantee, and sidesteps the anti-pattern CLAUDE.md actually names
(instantiating a new client inside a function on every call/request).

Per CLAUDE.md guardrail #14, every external call gets an explicit
timeout: `connect_timeout` bounds the TCP handshake, and a Postgres
`statement_timeout` is set on the connection so a hung query cannot block
indefinitely either. Neither duration is specified anywhere in
PRD/Architecture — 10 seconds was chosen for both as a generous-but-bounded
default for a free-tier Neon instance; tune once real latency is measured
(CLAUDE.md's ship-day README requirement).
"""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

CONNECT_TIMEOUT_SECONDS = 10
STATEMENT_TIMEOUT_SECONDS = 10

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _normalized_connection_string() -> str:
    """Read `NEON_CONNECTION_STRING` and ensure it uses the `psycopg`
    (v3) SQLAlchemy dialect. This project installs `psycopg[binary]`
    (v3) — not the legacy `psycopg2` that a bare `postgresql://` URL
    resolves to by default — so a plain `postgresql://` prefix is
    rewritten to `postgresql+psycopg://`.

    Raises RuntimeError if the environment variable is not set.
    """
    connection_string = os.environ.get("NEON_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("NEON_CONNECTION_STRING environment variable is not set")
    if connection_string.startswith("postgresql://"):
        connection_string = connection_string.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    return connection_string


def get_engine() -> Engine:
    """Return the module-level SQLAlchemy Engine for the Neon Postgres
    connection, creating it on first use.
    """
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
