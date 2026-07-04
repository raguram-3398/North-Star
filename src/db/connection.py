"""Neon Postgres connection/session management.

Per CLAUDE.md's coding conventions: one client per module, instantiated at
module level — never inside a function or per-request. The connection
string comes from an environment variable / secret, never hardcoded.
"""

from sqlalchemy import Engine
from sqlalchemy.orm import Session


def get_engine() -> Engine:
    """Return the module-level SQLAlchemy Engine for the Neon Postgres
    connection.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def get_session() -> Session:
    """Return a new SQLAlchemy Session bound to the module-level engine.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
