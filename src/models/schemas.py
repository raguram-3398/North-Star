"""SQLAlchemy models mirroring Architecture_North_Star.md §5's data model
exactly. If a field needs to change, the architecture doc is updated in the
same commit (CLAUDE.md coding conventions).

Field-level definitions are fleshed out incrementally, as each table's I/O
module is implemented — a class with no columns below is still a
docstring-only placeholder awaiting its own I/O module; `RolesCache` was
the first to be fully mapped, alongside `src/data/roles_cache.py`.
`OutlineTopic`/`ProgressLog`/`VerificationAttempt`/`PaceSnapshot`/`PatchNote`
are fleshed out here alongside `src/data/outline_topics.py`,
`src/data/progress_log.py`, `src/data/verification_log.py`,
`src/data/pace_snapshots.py`, and `src/data/patch_notes.py`. `user_id`/
`topic_id`/`origin_topic_id` columns below are plain UUID columns, not
`ForeignKey`-constrained — `User` remains an unmapped placeholder, and a
FK to an unmapped table isn't possible; `PatchNote.user_id` stays
unconstrained for that reason even though `PatchNote` itself is now
mapped. Add the constraints once `User` is mapped too.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB, UUID
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


class OutlineTopic(Base):
    """`outline_topics` — dependency hierarchy per user: topic, hierarchy
    position, topic group / position-in-group (hands-on ramping), source
    metadata, enrichment flag, status.
    """

    __tablename__ = "outline_topics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    topic_name: Mapped[str] = mapped_column()
    hierarchy_position: Mapped[int] = mapped_column()
    topic_group: Mapped[str] = mapped_column()
    position_in_group: Mapped[int] = mapped_column()
    source_url: Mapped[str] = mapped_column()
    source_type: Mapped[str] = mapped_column()
    confidence: Mapped[str] = mapped_column()
    is_enrichment: Mapped[bool] = mapped_column(default=False)
    status: Mapped[str] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column(default=None)


class ProgressLog(Base):
    """`progress_log` — canonical record of everything that feeds pace:
    per-day, per-step entries (summary/theory/hands_on/review/reflection/
    verification/preview).
    """

    __tablename__ = "progress_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    day_number: Mapped[int] = mapped_column()
    step: Mapped[str] = mapped_column()
    reflection_text: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column()


class VerificationAttempt(Base):
    """`verification_attempts` — per-question, per-attempt verification
    records that feed topic_score: question/grading criteria, answer,
    passed, credit (1.0 full / 0.5 half), test-out flag.
    """

    __tablename__ = "verification_attempts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    question_number: Mapped[int] = mapped_column()
    attempt_number: Mapped[int] = mapped_column()
    question_text: Mapped[str] = mapped_column()
    grading_criteria: Mapped[str] = mapped_column()
    user_answer: Mapped[str] = mapped_column()
    passed: Mapped[bool] = mapped_column()
    credit: Mapped[float] = mapped_column()
    is_test_out: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column()


class PatchNote(Base):
    """`patch_notes` — market-driven updates to already-completed topics:
    origin topic, new content, source/confidence, status (pending /
    delivered / deferred).
    """

    __tablename__ = "patch_notes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    origin_topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    new_content: Mapped[str] = mapped_column()
    source_url: Mapped[str] = mapped_column()
    confidence: Mapped[str] = mapped_column()
    status: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column()
    resolved_at: Mapped[datetime | None] = mapped_column(default=None)


class PaceSnapshot(Base):
    """`pace_snapshots` — rolling-window inputs to pace tracking:
    topic_score, timing_ratio, days_taken, days_expected per topic.
    """

    __tablename__ = "pace_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    topic_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True))
    topic_score: Mapped[float] = mapped_column()
    timing_ratio: Mapped[float] = mapped_column()
    days_taken: Mapped[int] = mapped_column()
    days_expected: Mapped[int] = mapped_column()
    computed_at: Mapped[datetime] = mapped_column()
