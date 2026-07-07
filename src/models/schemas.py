"""SQLAlchemy ORM models for the application's Postgres schema."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models in this project."""


class User(Base):
    """User profile, background, resolved role, and pacing state."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    background: Mapped[str | None] = mapped_column(default=None)
    current_job: Mapped[str | None] = mapped_column(default=None)
    years_experience: Mapped[int | None] = mapped_column(default=None)
    prior_self_study: Mapped[str | None] = mapped_column(default=None)
    available_time_per_week: Mapped[int | None] = mapped_column(default=None)
    resolved_role: Mapped[str | None] = mapped_column(default=None)
    role_confidence: Mapped[str | None] = mapped_column(default=None)
    pacing_profile: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=None)
    pace_extension_days: Mapped[int] = mapped_column(default=0)


class RolesCache(Base):
    """Cron-refreshed market data cache of core/emerging skills per role."""

    __tablename__ = "roles_cache"

    role_name: Mapped[str] = mapped_column(primary_key=True)
    core_skills: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    emerging_skills: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    last_updated: Mapped[datetime] = mapped_column()


class OutlineTopic(Base):
    """A single topic in a user's dependency hierarchy, with source metadata and status."""

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
    """A per-day, per-step progress entry that feeds pace tracking."""

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
    """A single verification question attempt, with grading result and credit, feeding topic_score."""

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
    """A market-driven content update to an already-completed topic."""

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
    """A single topic's pace inputs: topic_score, timing_ratio, days_taken, days_expected."""

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
