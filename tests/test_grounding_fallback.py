"""Tests for data/grounding_fallback.py — the cached-fallback and
general-knowledge-only floor rungs of the confidence ladder
(Architecture_North_Star.md §8, PRD §7.3).

Uses a mocked SQLAlchemy Session, same rationale as
tests/test_roles_cache.py: no SQLite substitution for Neon, and this
module's DB access is entirely delegated to data/roles_cache.py's
get_role, which is itself already covered against a mocked Session.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from data.grounding_fallback import (
    CACHED_SOURCE_TYPE,
    CachedFallbackResult,
    GeneralKnowledgeFloorResult,
    get_cached_fallback,
    get_general_knowledge_floor,
)
from data.roles_cache import STALENESS_FLOOR_DAYS
from models.schemas import RolesCache
from security.output_guard import ConfidenceTier, ValidatedGroundedContent


def _mock_session_with_role(
    role_name: str,
    core_skills: list[dict[str, str]],
    emerging_skills: list[dict[str, str]],
    last_updated: datetime,
) -> MagicMock:
    session = MagicMock()
    session.get.return_value = SimpleNamespace(
        role_name=role_name,
        core_skills=core_skills,
        emerging_skills=emerging_skills,
        last_updated=last_updated,
    )
    return session


def test_get_cached_fallback_returns_none_when_roles_cache_has_no_entry() -> None:
    """The genuinely-absent case: get_role returns None, so there is
    nothing to serve as cached-fallback. The caller falls through to
    get_general_knowledge_floor instead (the floor case, tested below).
    """
    session = MagicMock()
    session.get.return_value = None

    result = get_cached_fallback(session, "backend_engineer", datetime(2026, 7, 4))

    assert result is None
    session.get.assert_called_once_with(RolesCache, "backend_engineer")


def test_get_cached_fallback_found_and_fresh() -> None:
    """A fresh cache hit (well within the 30-day floor) still goes
    through this fallback module and is still labeled "cached-low" —
    this module is only ever invoked after live grounding has already
    failed for today's request, so the cache's own freshness (an
    orthogonal cron-refresh concept) doesn't change which rung the
    *fallback* result reads as. is_stale is surfaced as metadata (False
    here), not as a gate on whether data is returned.
    """
    last_updated = datetime(2026, 7, 1)
    reference_time = last_updated + timedelta(days=3)
    session = _mock_session_with_role(
        "backend_engineer",
        core_skills=[
            {
                "skill": "SQL",
                "source_url": "https://jobs.example.com/1",
                "confidence": "high",
            }
        ],
        emerging_skills=[
            {
                "skill": "dbt",
                "source_url": "https://jobs.example.com/2",
                "confidence": "low",
            }
        ],
        last_updated=last_updated,
    )

    result = get_cached_fallback(session, "backend_engineer", reference_time)

    assert isinstance(result, CachedFallbackResult)
    assert result.role_name == "backend_engineer"
    assert result.last_updated == last_updated
    assert result.is_stale is False

    assert len(result.core_skills) == 1
    core_entry = result.core_skills[0]
    assert isinstance(core_entry, ValidatedGroundedContent)
    assert core_entry.source_url == "https://jobs.example.com/1"
    assert core_entry.source_type == CACHED_SOURCE_TYPE
    assert core_entry.confidence == ConfidenceTier.CACHED_LOW
    assert core_entry.extra["skill"] == "SQL"
    assert core_entry.extra["last_updated"] == last_updated

    assert len(result.emerging_skills) == 1
    assert result.emerging_skills[0].confidence == ConfidenceTier.CACHED_LOW


def test_get_cached_fallback_found_and_stale() -> None:
    """A stale cache hit (past the 30-day floor) is still returned as
    cached-low fallback data, with is_stale=True surfaced so a caller can
    label the result honestly (e.g. "based on data from 47 days ago") —
    see get_cached_fallback's docstring for the flagged judgment call on
    this choice versus escalating straight to the floor.
    """
    last_updated = datetime(2026, 5, 1)
    reference_time = last_updated + timedelta(days=STALENESS_FLOOR_DAYS + 17)
    session = _mock_session_with_role(
        "backend_engineer",
        core_skills=[
            {
                "skill": "SQL",
                "source_url": "https://jobs.example.com/1",
                "confidence": "medium",
            }
        ],
        emerging_skills=[],
        last_updated=last_updated,
    )

    result = get_cached_fallback(session, "backend_engineer", reference_time)

    assert isinstance(result, CachedFallbackResult)
    assert result.is_stale is True
    # Confidence is still downgraded to cached-low regardless of the
    # originally-stored tier ("medium" here) — the cache-fallback rung
    # reflects today's re-serving, not the original grading.
    assert result.core_skills[0].confidence == ConfidenceTier.CACHED_LOW


def test_get_general_knowledge_floor_returns_labeled_unsourced_result() -> None:
    """The floor case: roles_cache has no entry at all. Per
    specs/scenarios/high_risk_flows.feature's "No source returns usable
    data" scenario, this must be an honestly labeled, structurally
    distinct result — never a ValidatedGroundedContent (no source_url to
    even omit or fake).
    """
    result = get_general_knowledge_floor("underwater_basket_weaver")

    assert isinstance(result, GeneralKnowledgeFloorResult)
    assert not isinstance(result, ValidatedGroundedContent)
    assert result.role_name == "underwater_basket_weaver"
    assert result.confidence == ConfidenceTier.GENERAL_KNOWLEDGE_ONLY
    assert "underwater_basket_weaver" in result.label
    assert not hasattr(result, "source_url")
