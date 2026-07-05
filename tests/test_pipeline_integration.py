"""End-to-end wiring proof: raw stated goal -> Clarify Gate resolves a
role -> ground_role produces grounded skills -> create_initial_outline
produces a valid, schema-correct outline.

This is integration proof, not new feature work: it adds no behavior to
any existing module (see the reconciliation note in
specs/Architecture_North_Star.md §3 if any connecting fix turned out to
be necessary — none was; every call site here uses the real function
signatures exactly as already built). Mocking stays at the same
boundary the rest of the suite already uses: real Himalayas/Tavily
fixtures via a fake MCP tool/Tavily client, a mocked SQLAlchemy Session
(never a real Postgres/SQLite), and a fake Gemini client — no live API
calls anywhere in this file.

Fakes and fixtures are imported directly from
tests/test_research_outline_agent.py rather than duplicated, since that
module already builds and maintains them.
"""

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
from tests.test_research_outline_agent import (
    _GENERIC_CONTENT,
    FIXTURES_DIR,
    NONSENSE_TEXT,
    _FakeHimalayasTool,
    _FakeTavilyClient,
    _himalayas_response,
    _patch_gemini,
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
    """Replicate `ground_role`'s own casefold-dedup-first-listing-wins
    skill extraction exactly (see `agents/research_outline_agent.py`'s
    `himalayas_skill_map` construction), so the mocked outline-hierarchy
    response below covers precisely the skill names `ground_role` will
    actually produce — not a hand-guessed list that could silently drift
    from the real parser's output.
    """
    listings = parse_search_jobs_response(raw_text)
    seen: dict[str, str] = {}
    for listing in listings:
        if listing.source_url is None:
            continue
        for skill in listing.skills:
            seen.setdefault(skill.casefold(), skill)
    return list(seen.values())


def _hierarchy_response_covering(skill_names: list[str]) -> str:
    """Build a minimal but valid mocked Gemini hierarchy response: one
    topic per skill, split across two groups, covering every skill name
    exactly once. Satisfies `create_initial_outline`'s "every grounded
    skill must be covered by at least one topic" requirement without
    hand-typing dozens of entries — cross-group/within-group *ordering*
    correctness already has its own dedicated unit tests
    (tests/test_research_outline_agent.py); this file's job is proving
    the chain wires together, not re-verifying that logic.
    """
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
    """A mocked SQLAlchemy Session whose `.get()` returns a roles_cache-
    shaped row — the same "mock the Session, not the function" pattern
    tests/test_grounding_fallback.py already established. This one
    session object is shared by both `ground_role`'s anchor lookup *and*
    (when live grounding rejects) `get_cached_fallback`'s lookup, since
    both call the real `data.roles_cache.get_role`, which just calls
    `session.get(...)` — proving both call sites see consistent data
    without needing to patch two separate `get_role` import bindings.
    """
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
    """Shared assertions for every scenario below: non-empty, contiguous
    strictly-increasing hierarchy_position, every topic carries a
    non-empty source_url/confidence, every topic's sourcing traces back
    to a *specific* entry in the original grounded skill list (not just
    "truthy"), and is_enrichment/status are exactly as
    `create_initial_outline` promises.
    """
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
    """Clearly real stated goal -> begin_clarify_gate resolves
    immediately, no narrowing, no Gemini call -> ground_role against real
    Himalayas fixture data (Himalayas signal + a roles_cache anchor,
    Tavily deliberately generic/no-signal -> deterministic MEDIUM
    confidence) -> create_initial_outline via the degenerate
    core_skills=result.skills, emerging_skills=[] split.
    """
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
    _patch_gemini(
        monkeypatch, responses=[_hierarchy_response_covering(expected_skills)]
    )

    topics = await roa.create_initial_outline(
        resolved_role, grounding_result.skills, []
    )

    _assert_valid_outline(topics, grounding_result.skills)


async def test_vague_input_resolves_through_narrowing_then_valid_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vague-but-genuine goal -> begin_clarify_gate enters the narrowing
    loop -> advance_clarify_gate resolves within one round -> the
    resolved role feeds into the identical ground_role ->
    create_initial_outline chain as the happy path.
    """
    _patch_gemini(
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
    assert turn.gate_state.narrowing_rounds_used == 0  # resolved rounds don't increment
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
    _patch_gemini(
        monkeypatch, responses=[_hierarchy_response_covering(expected_skills)]
    )

    topics = await roa.create_initial_outline(
        resolved_role, grounding_result.skills, []
    )

    _assert_valid_outline(topics, grounding_result.skills)


async def test_fallback_path_cached_result_wires_into_create_initial_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stated goal that resolves to a role with zero live grounding
    signal (Himalayas real-but-irrelevant fallback text, Tavily generic
    content — neither clears its trust threshold) but a real roles_cache
    entry: confirms `ground_role` produces a genuine `CachedFallbackResult`
    (not a `LiveGroundingResult`), and — the actual point of this test —
    that `CachedFallbackResult.core_skills`/`emerging_skills` work at
    `create_initial_outline`'s real call site, not merely "look
    type-compatible."
    """
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

    _patch_gemini(
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
