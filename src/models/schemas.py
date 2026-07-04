"""SQLAlchemy models mirroring Architecture_North_Star.md §5's data model
exactly. If a field needs to change, the architecture doc is updated in the
same commit (CLAUDE.md coding conventions).

Field-level definitions are fleshed out incrementally, as each table's I/O
module is implemented — a class with no columns below is still a
docstring-only placeholder awaiting its own I/O module; `RolesCache` is
the first to be fully mapped, alongside `src/data/roles_cache.py`.
"""

from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models in this project."""


class User:
    """`users` — user profile: background, current job, years of
    experience, prior self-study, resolved role, role confidence, pacing
    profile.
    """


class RolesCache(Base):
    """`roles_cache` — cron-refreshed market data cache: core_skills /
    emerging_skills per role, each carrying a confidence tier, plus
    last_updated.
    """

    __tablename__ = "roles_cache"

    role_name: Mapped[str] = mapped_column(primary_key=True)
    core_skills: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    emerging_skills: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    last_updated: Mapped[datetime] = mapped_column()


class OutlineTopic:
    """`outline_topics` — dependency hierarchy per user: topic, hierarchy
    position, topic group / position-in-group (hands-on ramping), source
    metadata, enrichment flag, status.
    """


class ProgressLog:
    """`progress_log` — canonical record of everything that feeds pace:
    per-day, per-step entries (summary/theory/hands_on/review/reflection/
    verification/preview).
    """


class VerificationAttempt:
    """`verification_attempts` — per-question, per-attempt verification
    records that feed topic_score: question/grading criteria, answer,
    passed, credit (1.0 full / 0.5 half), test-out flag.
    """


class PatchNote:
    """`patch_notes` — market-driven updates to already-completed topics:
    origin topic, new content, source/confidence, status (pending /
    delivered / deferred).
    """


class PaceSnapshot:
    """`pace_snapshots` — rolling-window inputs to pace tracking:
    topic_score, timing_ratio, days_taken, days_expected per topic.
    """
