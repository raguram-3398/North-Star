"""One-time database schema creation for a live Neon Postgres instance.

Hackathon scope: no Alembic, no migration tool — this is a single
`Base.metadata.create_all()` script, not a migration system (CLAUDE.md's
Stack section; Architecture_North_Star.md §5's "Known limitation" that no
schema-push mechanism exists yet). Invoked as `python -m src.db.create_schema`,
matching the existing `cron/refresh_roles.py` pattern (a `__main__` block,
not a `scripts/` file).

`Base.metadata.create_all()` is idempotent: SQLAlchemy checks each table's
existence in the connected database before issuing its `CREATE TABLE`
statement, so re-running this script against an already-provisioned
database is safe — it will not error or duplicate any existing table. This
matters because this script is expected to run more than once over the
course of the build (e.g. once to provision, again after a schema field is
added to `models/schemas.py`).

Every model class is imported explicitly (rather than relying on the
`import models.schemas` side effect alone) so a caller/test can assert each
of the 7 expected tables is actually registered on `Base.metadata` before
`create_all()` runs.
"""

import logging

from db.connection import get_engine
from models.schemas import (
    Base,
    OutlineTopic,
    PaceSnapshot,
    PatchNote,
    ProgressLog,
    RolesCache,
    User,
    VerificationAttempt,
)

logger = logging.getLogger(__name__)

# The 7 tables Architecture_North_Star.md §5 specifies. Referenced here only
# to guarantee each class has been imported (and therefore registered on
# Base.metadata) — not otherwise used.
REGISTERED_MODELS: tuple[type, ...] = (
    User,
    RolesCache,
    OutlineTopic,
    ProgressLog,
    VerificationAttempt,
    PatchNote,
    PaceSnapshot,
)


def create_all_tables() -> None:
    """Create every table registered on `Base.metadata` that does not
    already exist in the connected Neon database. Safe to call more than
    once — see this module's docstring on `create_all()`'s idempotency.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info(
        "create_all() finished; tables registered on Base.metadata: %s",
        sorted(Base.metadata.tables.keys()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_all_tables()
