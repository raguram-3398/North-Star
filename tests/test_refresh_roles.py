"""Tests for src/cron/refresh_roles.py: the shared roles_cache refresh function and the startup staleness check."""

from datetime import datetime
from typing import cast
from unittest.mock import MagicMock

import pytest

import cron.refresh_roles as rr
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from security.output_guard import ConfidenceTier, ValidatedGroundedContent

REFERENCE_TIME = datetime(2026, 7, 5, 12, 0, 0)


def _live_result(role_name: str) -> rr.LiveGroundingResult:
    return rr.LiveGroundingResult(
        role_name=role_name,
        skills=[
            ValidatedGroundedContent(
                source_url="https://example.com/listing",
                source_type="job_listing",
                confidence=ConfidenceTier.HIGH,
                extra={"skill": "SQL"},
            )
        ],
        confidence=ConfidenceTier.HIGH,
        has_conflict=False,
        himalayas_status="signal",
        tavily_status="signal",
    )


def _fallback_result(role_name: str) -> CachedFallbackResult:
    return CachedFallbackResult(
        role_name=role_name,
        core_skills=[],
        emerging_skills=[],
        last_updated=datetime(2026, 5, 1),
        is_stale=True,
    )


def _floor_result(role_name: str) -> GeneralKnowledgeFloorResult:
    return GeneralKnowledgeFloorResult(
        role_name=role_name,
        confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
        label="general knowledge only",
    )


@pytest.mark.asyncio
async def test_refresh_roles_cache_upserts_each_successfully_grounded_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ground_role_mock = MagicMock(side_effect=lambda role_name: _live_result(role_name))

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return cast(rr.LiveGroundingResult, ground_role_mock(role_name))

    upsert_role_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", upsert_role_mock)
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: None)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(session, rr.SEED_ROLES, REFERENCE_TIME)

    assert [result.status for result in summary.results] == ["upserted"] * 4
    assert upsert_role_mock.call_count == 4
    assert ground_role_mock.call_count == 4
    assert not summary.had_errors
    assert all(result.patch_notes_created == 0 for result in summary.results)


@pytest.mark.asyncio
async def test_refresh_roles_cache_uses_the_exact_four_role_seed_list() -> None:
    assert rr.SEED_ROLES == [
        "Backend Engineer",
        "Frontend Engineer",
        "Data Analyst",
        "DevOps Engineer",
    ]


@pytest.mark.asyncio
async def test_refresh_roles_cache_continues_past_a_single_role_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One role's `ground_role` call raising must not abort the batch for the other roles."""

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        if role_name == "Frontend Engineer":
            raise ValueError("simulated grounding failure")
        return _live_result(role_name)

    upsert_role_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", upsert_role_mock)
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: None)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(session, rr.SEED_ROLES, REFERENCE_TIME)

    statuses = {result.role_name: result.status for result in summary.results}
    assert statuses == {
        "Backend Engineer": "upserted",
        "Frontend Engineer": "error",
        "Data Analyst": "upserted",
        "DevOps Engineer": "upserted",
    }
    assert upsert_role_mock.call_count == 3
    assert summary.had_errors
    failed = next(r for r in summary.results if r.role_name == "Frontend Engineer")
    assert failed.detail == "simulated grounding failure"


@pytest.mark.asyncio
async def test_refresh_roles_cache_does_not_upsert_a_cached_fallback_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CachedFallbackResult (ground_role fell through to the cache) must not be written back via upsert_role."""

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> CachedFallbackResult:
        return _fallback_result(role_name)

    upsert_role_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", upsert_role_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results == [
        rr.RoleRefreshResult("Backend Engineer", "no_live_signal")
    ]
    upsert_role_mock.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_roles_cache_does_not_upsert_a_general_knowledge_floor_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> GeneralKnowledgeFloorResult:
        return _floor_result(role_name)

    upsert_role_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", upsert_role_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(session, ["Data Analyst"], REFERENCE_TIME)

    assert summary.results == [rr.RoleRefreshResult("Data Analyst", "no_live_signal")]
    upsert_role_mock.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_roles_cache_times_out_a_hanging_ground_role_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def _hanging_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        await asyncio.sleep(999)
        raise AssertionError("should have been cancelled by the outer timeout")

    monkeypatch.setattr(rr, "ground_role", _hanging_ground_role)
    monkeypatch.setattr(rr, "ROLE_REFRESH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: None)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results[0].status == "error"
    assert summary.had_errors


def _previous_row(
    core_skills: list[dict[str, str]], emerging_skills: list[dict[str, str]]
) -> dict[str, object]:
    return {
        "role_name": "Backend Engineer",
        "core_skills": core_skills,
        "emerging_skills": emerging_skills,
        "last_updated": datetime(2026, 6, 1),
    }


@pytest.mark.asyncio
async def test_refresh_roles_cache_creates_patch_notes_for_an_upward_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQL strengthening medium to high within the same bucket is a significant event, so completed-topic users get a patch-note."""
    previous_row = _previous_row(
        core_skills=[
            {"skill": "SQL", "source_url": "https://old", "confidence": "medium"}
        ],
        emerging_skills=[],
    )

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return _live_result(role_name)

    matching_topics = [
        {"id": "topic-1", "user_id": "user-1"},
        {"id": "topic-2", "user_id": "user-2"},
    ]
    get_completed_topics_mock = MagicMock(return_value=matching_topics)
    create_patch_note_mock = MagicMock(return_value={})

    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: previous_row)
    monkeypatch.setattr(
        rr, "get_completed_topics_matching_skill", get_completed_topics_mock
    )
    monkeypatch.setattr(rr, "create_patch_note", create_patch_note_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results[0].patch_notes_created == 2
    get_completed_topics_mock.assert_called_once_with(session, "SQL")
    assert create_patch_note_mock.call_count == 2
    called_user_ids = {
        call.kwargs["user_id"] for call in create_patch_note_mock.call_args_list
    }
    assert called_user_ids == {"user-1", "user-2"}
    for call in create_patch_note_mock.call_args_list:
        assert call.kwargs["created_at"] == REFERENCE_TIME
        assert call.kwargs["grounded_content"].extra["skill"] == "SQL"
        assert call.kwargs["grounded_content"].confidence == ConfidenceTier.HIGH
        assert (
            isinstance(call.kwargs["new_content"], str) and call.kwargs["new_content"]
        )


@pytest.mark.asyncio
async def test_refresh_roles_cache_generates_nothing_for_a_downward_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQL weakening high to medium within the same bucket is a downward crossing and generates no patch-notes."""
    previous_row = _previous_row(
        core_skills=[
            {"skill": "SQL", "source_url": "https://old", "confidence": "high"}
        ],
        emerging_skills=[],
    )
    weakened_result = rr.LiveGroundingResult(
        role_name="Backend Engineer",
        skills=[
            ValidatedGroundedContent(
                source_url="https://new",
                source_type="job_listing",
                confidence=ConfidenceTier.MEDIUM,
                extra={"skill": "SQL"},
            )
        ],
        confidence=ConfidenceTier.MEDIUM,
        has_conflict=False,
        himalayas_status="signal",
        tavily_status="signal",
    )

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return weakened_result

    get_completed_topics_mock = MagicMock()
    create_patch_note_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: previous_row)
    monkeypatch.setattr(
        rr, "get_completed_topics_matching_skill", get_completed_topics_mock
    )
    monkeypatch.setattr(rr, "create_patch_note", create_patch_note_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results[0].patch_notes_created == 0
    get_completed_topics_mock.assert_not_called()
    create_patch_note_mock.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_roles_cache_first_ever_refresh_generates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No pre-existing roles_cache row for this role means diffing is skipped entirely, not treated as an error."""

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return _live_result(role_name)

    create_patch_note_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: None)
    monkeypatch.setattr(rr, "create_patch_note", create_patch_note_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results[0].status == "upserted"
    assert summary.results[0].patch_notes_created == 0
    create_patch_note_mock.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_roles_cache_generates_nothing_with_no_matching_completed_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQL is brand-new and significant, but nobody has a completed "SQL" topic, so no patch-notes are created."""
    previous_row = _previous_row(core_skills=[], emerging_skills=[])

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return _live_result(role_name)

    create_patch_note_mock = MagicMock()
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())
    monkeypatch.setattr(rr, "get_role", lambda session, role_name: previous_row)
    monkeypatch.setattr(
        rr, "get_completed_topics_matching_skill", MagicMock(return_value=[])
    )
    monkeypatch.setattr(rr, "create_patch_note", create_patch_note_mock)

    session = MagicMock()
    summary = await rr.refresh_roles_cache(
        session, ["Backend Engineer"], REFERENCE_TIME
    )

    assert summary.results[0].patch_notes_created == 0
    create_patch_note_mock.assert_not_called()


def test_get_stale_or_missing_roles_identifies_stale_fresh_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_role(session: object, role_name: str) -> dict | None:
        if role_name == "Backend Engineer":
            return {"last_updated": datetime(2026, 6, 1)}
        if role_name == "Frontend Engineer":
            return {"last_updated": datetime(2026, 7, 1)}
        return None

    monkeypatch.setattr(rr, "get_role", _fake_get_role)

    result = rr.get_stale_or_missing_roles(
        MagicMock(),
        ["Backend Engineer", "Frontend Engineer", "Data Analyst"],
        REFERENCE_TIME,
    )

    assert result == ["Backend Engineer", "Data Analyst"]


@pytest.mark.asyncio
async def test_check_and_refresh_stale_roles_refreshes_only_stale_or_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_role(session: object, role_name: str) -> dict | None:
        if role_name == "Frontend Engineer":
            return {"last_updated": datetime(2026, 7, 1)}
        return None

    ground_role_mock = MagicMock(side_effect=lambda role_name: _live_result(role_name))

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return cast(rr.LiveGroundingResult, ground_role_mock(role_name))

    monkeypatch.setattr(rr, "get_role", _fake_get_role)
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)
    monkeypatch.setattr(rr, "upsert_role", MagicMock())

    session = MagicMock()
    summary = await rr.check_and_refresh_stale_roles(
        session, rr.SEED_ROLES, REFERENCE_TIME
    )

    refreshed_roles = {result.role_name for result in summary.results}
    assert refreshed_roles == {"Backend Engineer", "Data Analyst", "DevOps Engineer"}
    assert "Frontend Engineer" not in refreshed_roles
    assert ground_role_mock.call_count == 3


@pytest.mark.asyncio
async def test_check_and_refresh_stale_roles_does_not_refresh_when_all_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_get_role(session: object, role_name: str) -> dict | None:
        return {"last_updated": datetime(2026, 7, 1)}

    ground_role_mock = MagicMock()

    async def _fake_ground_role(
        role_name: str, session: object, reference_time: datetime
    ) -> rr.LiveGroundingResult:
        return cast(rr.LiveGroundingResult, ground_role_mock(role_name))

    monkeypatch.setattr(rr, "get_role", _fake_get_role)
    monkeypatch.setattr(rr, "ground_role", _fake_ground_role)

    session = MagicMock()
    summary = await rr.check_and_refresh_stale_roles(
        session, rr.SEED_ROLES, REFERENCE_TIME
    )

    assert summary.results == []
    ground_role_mock.assert_not_called()
