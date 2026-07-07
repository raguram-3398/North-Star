"""End-to-end proof that a raw stated goal wires through the Clarify Gate, grounding, and outline creation."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import agents.research_outline_agent as roa
from data.grounding_fallback import CachedFallbackResult
from data.himalayas_parser import parse_search_jobs_response
from security.input_gate import ClarifyGateStage
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from tests.test_adk_runtime import _patch_adk_runtime
from tests.test_research_outline_agent import (
    _GENERIC_CONTENT,
    FIXTURES_DIR,
    NONSENSE_TEXT,
    _FakeHimalayasTool,
    _FakeTavilyClient,
    _himalayas_response,
    _patch_himalayas,
    _patch_tavily,
    _tavily_response,
    _tavily_result_dict,
)

REFERENCE_TIME = datetime(2026, 7, 5, 12, 0, 0)

DEVOPS_HIMALAYAS_TEXT = (
    FIXTURES_DIR / "himalayas_search_jobs_devops_engineer.txt"
).read_text()


def _expected_himalayas_skill_names(raw_text: str) -> list[str]:
    """Replicate ground_role's own casefold-dedup-first-listing-wins skill extraction exactly."""
    listings = parse_search_jobs_response(raw_text)
    seen: dict[str, str] = {}
    for listing in listings:
        if listing.source_url is None:
            continue
        for skill in listing.skills:
            seen.setdefault(skill.casefold(), skill)
    return list(seen.values())


def _hierarchy_response_covering(skill_names: list[str]) -> str:
    """Build a minimal mocked Gemini hierarchy response with one topic per skill, split across two groups."""
    midpoint = max(len(skill_names) // 2, 1)
    return json.dumps(
        {
            "groups": [
                {
                    "topic_group": "Fundamentals",
                    "topics": [
                        {"topic_name": f"{s} basics", "source_skill": s}
                        for s in skill_names[:midpoint]
                    ],
                },
                {
                    "topic_group": "Advanced",
                    "topics": [
                        {"topic_name": f"{s} in depth", "source_skill": s}
                        for s in skill_names[midpoint:]
                    ]
                    or [{"topic_name": "overflow", "source_skill": skill_names[-1]}],
                },
            ]
        }
    )


def _mock_session_with_anchor(role_name: str, skill_names: list[str]) -> MagicMock:
    """Build a mocked SQLAlchemy Session whose .get() returns a roles_cache-shaped row."""
    session = MagicMock()
    session.get.return_value = SimpleNamespace(
        role_name=role_name,
        core_skills=[
            {
                "skill": skill,
                "source_url": f"https://roles-cache.example/{i}",
                "confidence": "high",
            }
            for i, skill in enumerate(skill_names)
        ],
        emerging_skills=[],
        last_updated=REFERENCE_TIME,
    )
    return session


def _assert_valid_outline(
    topics: list[roa.InitialOutlineTopic],
    input_skills: list[ValidatedGroundedContent],
) -> None:
    """Assert every topic has contiguous hierarchy_position, real sourcing, and correct is_enrichment/status."""
    assert topics
    positions = [t.hierarchy_position for t in topics]
    assert positions == list(range(1, len(topics) + 1))

    valid_source_tuples = {
        (s.source_url, s.source_type, s.confidence) for s in input_skills
    }
    for topic in topics:
        assert topic.source_url
        assert topic.confidence
        assert (topic.source_url, topic.source_type, topic.confidence) in (
            valid_source_tuples
        )
        assert topic.is_enrichment is False
        assert topic.status == "not_started"


async def test_happy_path_real_role_through_to_valid_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clearly real stated goal resolves immediately and wires through grounding to a valid outline."""
    turn = await roa.begin_clarify_gate("DevOps Engineer")
    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    assert turn.resolved_role == "DevOps Engineer"
    resolved_role = turn.resolved_role
    assert resolved_role is not None

    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(DEVOPS_HIMALAYAS_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _GENERIC_CONTENT)])
        ),
    )
    session = _mock_session_with_anchor(resolved_role, ["Docker"])

    grounding_result = await roa.ground_role(resolved_role, session, REFERENCE_TIME)

    assert isinstance(grounding_result, roa.LiveGroundingResult)
    assert grounding_result.confidence in (ConfidenceTier.MEDIUM, ConfidenceTier.HIGH)
    assert grounding_result.skills

    expected_skills = _expected_himalayas_skill_names(DEVOPS_HIMALAYAS_TEXT)
    _patch_adk_runtime(
        monkeypatch, responses=[_hierarchy_response_covering(expected_skills)]
    )

    topics = await roa.create_initial_outline(
        resolved_role, grounding_result.skills, []
    )

    _assert_valid_outline(topics, grounding_result.skills)


async def test_vague_input_resolves_through_narrowing_then_valid_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A vague-but-genuine goal resolves within one narrowing round and wires through to a valid outline."""
    _patch_adk_runtime(
        monkeypatch,
        responses=[
            "What part of app development sounds most interesting to you?",
            json.dumps({"resolved": True, "role": "DevOps Engineer"}),
        ],
    )

    turn = await roa.begin_clarify_gate("I want to make apps")
    assert turn.gate_state.stage is ClarifyGateStage.NARROWING
    assert turn.gate_state.narrowing_rounds_used == 0

    conversation = [{"role": "agent", "content": turn.message}]
    turn = await roa.advance_clarify_gate(
        turn.gate_state,
        turn.context,
        conversation,
        "Automating deployments and infrastructure sounds interesting",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )

    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    assert turn.gate_state.narrowing_rounds_used == 0
    resolved_role = turn.resolved_role
    assert resolved_role == "DevOps Engineer"

    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(DEVOPS_HIMALAYAS_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _GENERIC_CONTENT)])
        ),
    )
    session = _mock_session_with_anchor(resolved_role, ["Docker"])

    grounding_result = await roa.ground_role(resolved_role, session, REFERENCE_TIME)
    assert isinstance(grounding_result, roa.LiveGroundingResult)
    assert grounding_result.confidence in (ConfidenceTier.MEDIUM, ConfidenceTier.HIGH)

    expected_skills = _expected_himalayas_skill_names(DEVOPS_HIMALAYAS_TEXT)
    _patch_adk_runtime(
        monkeypatch, responses=[_hierarchy_response_covering(expected_skills)]
    )

    topics = await roa.create_initial_outline(
        resolved_role, grounding_result.skills, []
    )

    _assert_valid_outline(topics, grounding_result.skills)


async def test_fallback_path_cached_result_wires_into_create_initial_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A role with zero live grounding signal but a roles_cache entry falls back and still wires to a valid outline."""
    turn = await roa.begin_clarify_gate("Mainframe Systems Engineer")
    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    resolved_role = turn.resolved_role
    assert resolved_role == "Mainframe Systems Engineer"

    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(NONSENSE_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _GENERIC_CONTENT)])
        ),
    )
    cached_skill_names = ["COBOL", "JCL"]
    session = _mock_session_with_anchor(resolved_role, cached_skill_names)

    grounding_result = await roa.ground_role(resolved_role, session, REFERENCE_TIME)

    assert isinstance(grounding_result, CachedFallbackResult)
    assert not isinstance(grounding_result, roa.LiveGroundingResult)
    assert [s.extra["skill"] for s in grounding_result.core_skills] == (
        cached_skill_names
    )
    assert grounding_result.emerging_skills == []

    _patch_adk_runtime(
        monkeypatch,
        responses=[_hierarchy_response_covering(cached_skill_names)],
    )

    topics = await roa.create_initial_outline(
        resolved_role,
        grounding_result.core_skills,
        grounding_result.emerging_skills,
    )

    _assert_valid_outline(topics, grounding_result.core_skills)
    for topic in topics:
        assert topic.confidence == ConfidenceTier.CACHED_LOW
        assert topic.source_type == "roles_cache-cached"
