"""One-time, idempotent database schema creation for a live Neon Postgres instance."""

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
    """Create every table registered on Base.metadata that doesn't already exist in the connected database."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info(
        "create_all() finished; tables registered on Base.metadata: %s",
        sorted(Base.metadata.tables.keys()),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_all_tables()
