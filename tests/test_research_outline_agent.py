"""Tests for agents/research_outline_agent.py's `ground_role` — the
cross-validation/grounding orchestrator (Architecture §3's "cross-
validation normalization judgment") — and its `begin_clarify_gate`/
`advance_clarify_gate` — the Clarify Gate's conversational half (PRD
§7.2).

Himalayas MCP and Tavily are both mocked with small fake clients rather
than a real McpToolset/TavilyClient, so these tests are fast, offline,
and deterministic — matching tests/test_roles_cache.py's and
tests/test_grounding_fallback.py's mocked-Session precedent. Where
realistic Himalayas response text is needed, the real fixtures gathered
for tests/test_himalayas_parser.py and tests/test_himalayas_relevance.py
are reused rather than inventing new synthetic text. Tavily content
strings are constructed to deliberately clear or miss
data/cross_validation.py's TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD (real
vocabulary terms vs. generic prose), since that's what now decides
Tavily's own signal, not `score`.

The clarify-gate tests mock Gemini with a small fake client (`_FakeGemini*`
below) rather than a real `google.genai.Client`, for the same reason —
fast, offline, deterministic, and focused on this module's own
orchestration logic (dispatch on ClarifyGateStage, context threading, the
ACCEPT_OWN_WORDS original-goal-not-latest-message rule) rather than on
Gemini's actual behavior. `ground_role` itself is mocked directly in
these tests (it already has its own dedicated tests above) so the
clarify-gate tests aren't re-exercising Himalayas/Tavily mocking too.

Patches target `agents.research_outline_agent.<name>` (where each name is
*used*), not where it's defined — CLAUDE.md's flagged
wrong-patch-target anti-pattern.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tavily.errors import InvalidAPIKeyError

import agents.research_outline_agent as roa
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from security.input_gate import (
    ClarifyGateStage,
    ClarifyGateState,
    OutlineConfirmationStage,
    OutlineConfirmationState,
)
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from utils.exceptions import GeminiCallError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _himalayas_response(raw_text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": raw_text}], "isError": is_error}


def _tavily_result_dict(
    score: float, content: str = "", url: str = "https://tavily.test/x"
) -> dict:
    return {"url": url, "title": "x", "content": content, "score": score}


def _tavily_response(results: list[dict]) -> dict:
    return {"results": results}


# Clears TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD (3) on its own: 4 distinct
# real vocabulary terms in one result.
_SKILL_BEARING_CONTENT = "You need Python, SQL, Excel, and Tableau for this role."
# Real prose with zero vocabulary hits.
_GENERIC_CONTENT = "Great communication and teamwork skills are essential."


class _FakeHimalayasTool:
    def __init__(self, name: str, response=None, raise_exc=None, sleep_seconds=0.0):
        self.name = name
        self._response = response
        self._raise_exc = raise_exc
        self._sleep_seconds = sleep_seconds

    async def run_async(self, args, tool_context):
        if self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_exc:
            raise self._raise_exc
        return self._response


class _FakeHimalayasToolset:
    def __init__(self, tools):
        self._tools = tools

    async def get_tools(self):
        return self._tools


class _FakeTavilyClient:
    def __init__(self, response=None, raise_exc=None, sleep_seconds=0.0):
        self._response = response
        self._raise_exc = raise_exc
        self._sleep_seconds = sleep_seconds

    def search(self, **kwargs):
        import time

        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        if self._raise_exc:
            raise self._raise_exc
        return self._response


def _patch_himalayas(monkeypatch: pytest.MonkeyPatch, tool: _FakeHimalayasTool) -> None:
    monkeypatch.setattr(
        roa, "_get_himalayas_toolset", lambda: _FakeHimalayasToolset([tool])
    )


def _patch_tavily(monkeypatch: pytest.MonkeyPatch, client: _FakeTavilyClient) -> None:
    monkeypatch.setattr(roa, "_get_tavily_client", lambda: client)


BACKEND_ENGINEER_TEXT = (
    FIXTURES_DIR / "himalayas_search_jobs_data_analyst.txt"
).read_text()
NONSENSE_TEXT = (
    FIXTURES_DIR / "himalayas_search_jobs_nonsense_keyword_fallback.txt"
).read_text()


# --- ground_role: full confidence-ladder branches ------------------------


async def test_high_confidence_both_sources_agree_with_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _SKILL_BEARING_CONTENT)])
        ),
    )
    monkeypatch.setattr(
        roa,
        "get_role",
        lambda session, role_name: {
            "role_name": role_name,
            "core_skills": [
                {"skill": "SQL", "source_url": "https://x", "confidence": "high"}
            ],
            "emerging_skills": [],
            "last_updated": datetime(2026, 1, 1),
        },
    )

    result = await roa.ground_role("Data Analyst", MagicMock(), datetime(2026, 7, 4))

    assert isinstance(result, roa.LiveGroundingResult)
    assert result.confidence == ConfidenceTier.HIGH
    assert result.has_conflict is False
    assert result.skills
    for skill in result.skills:
        assert isinstance(skill, ValidatedGroundedContent)
        assert skill.source_type == "job_listing"
        assert skill.confidence == ConfidenceTier.HIGH


async def test_medium_confidence_single_source_himalayas_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _GENERIC_CONTENT)])
        ),
    )
    monkeypatch.setattr(
        roa,
        "get_role",
        lambda session, role_name: {
            "role_name": role_name,
            "core_skills": [
                {"skill": "SQL", "source_url": "https://x", "confidence": "high"}
            ],
            "emerging_skills": [],
            "last_updated": datetime(2026, 1, 1),
        },
    )

    result = await roa.ground_role("Data Analyst", MagicMock(), datetime(2026, 7, 4))

    assert isinstance(result, roa.LiveGroundingResult)
    assert result.confidence == ConfidenceTier.MEDIUM
    assert result.has_conflict is False
    assert result.tavily_status == "no_signal"
    assert result.himalayas_status == "signal"
    assert all(skill.source_type == "job_listing" for skill in result.skills)


async def test_low_confidence_niche_no_anchor_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _SKILL_BEARING_CONTENT)])
        ),
    )
    monkeypatch.setattr(roa, "get_role", lambda session, role_name: None)

    result = await roa.ground_role("Data Analyst", MagicMock(), datetime(2026, 7, 4))

    assert isinstance(result, roa.LiveGroundingResult)
    assert result.confidence == ConfidenceTier.LOW
    assert result.has_conflict is False


async def test_medium_confidence_genuine_conflict_with_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _SKILL_BEARING_CONTENT)])
        ),
    )
    monkeypatch.setattr(
        roa,
        "get_role",
        lambda session, role_name: {
            "role_name": role_name,
            "core_skills": [
                {"skill": "Rust", "source_url": "https://x", "confidence": "high"}
            ],
            "emerging_skills": [],
            "last_updated": datetime(2026, 1, 1),
        },
    )

    result = await roa.ground_role("Data Analyst", MagicMock(), datetime(2026, 7, 4))

    assert isinstance(result, roa.LiveGroundingResult)
    assert result.confidence == ConfidenceTier.MEDIUM
    assert result.has_conflict is True


async def test_medium_confidence_tavily_only_himalayas_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario this task exists for: Himalayas has no usable signal
    (the real captured nonsense-keyword fallback text), but Tavily
    clears the distinct-skill trust threshold on its own — this must now
    reach medium confidence with Tavily-sourced, output_guard-validated
    skills, instead of always falling through to the fallback chain.
    """
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(NONSENSE_TEXT)),
    )
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response(
                [
                    _tavily_result_dict(
                        0.95, _GENERIC_CONTENT, url="https://junk.test"
                    ),
                    _tavily_result_dict(
                        0.3, _SKILL_BEARING_CONTENT, url="https://good.test"
                    ),
                ]
            )
        ),
    )
    monkeypatch.setattr(roa, "get_role", lambda session, role_name: None)

    result = await roa.ground_role(
        "zzznonexistentrolezzz123456", MagicMock(), datetime(2026, 7, 4)
    )

    assert isinstance(result, roa.LiveGroundingResult)
    assert result.confidence == ConfidenceTier.MEDIUM
    assert result.has_conflict is False
    assert result.himalayas_status == "no_signal"
    assert result.tavily_status == "signal"
    assert result.skills
    for skill in result.skills:
        assert isinstance(skill, ValidatedGroundedContent)
        assert skill.source_type == "web_search"
        # the high-score, skill-less result must never win citation
        assert skill.source_url == "https://good.test"
    assert {skill.extra["skill"] for skill in result.skills} == {
        "Python",
        "SQL",
        "Excel",
        "Tableau",
    }


async def test_cached_fallback_when_both_sources_have_no_usable_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Himalayas has no signal (real nonsense-keyword fallback text) and
    Tavily also fails its distinct-skill trust threshold — both sources
    contribute nothing, so this must still route to grounding_fallback,
    unchanged from before Tavily-only signal was possible.
    """
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
    monkeypatch.setattr(roa, "get_role", lambda session, role_name: None)

    sentinel = CachedFallbackResult(
        role_name="zzznonexistentrolezzz123456",
        core_skills=[],
        emerging_skills=[],
        last_updated=datetime(2026, 1, 1),
        is_stale=False,
    )
    get_cached_fallback_mock = MagicMock(return_value=sentinel)
    monkeypatch.setattr(roa, "get_cached_fallback", get_cached_fallback_mock)

    result = await roa.ground_role(
        "zzznonexistentrolezzz123456", MagicMock(), datetime(2026, 7, 4)
    )

    assert result is sentinel
    get_cached_fallback_mock.assert_called_once()


async def test_general_knowledge_floor_when_everything_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", raise_exc=ConnectionError("mcp down")),
    )
    _patch_tavily(
        monkeypatch, _FakeTavilyClient(raise_exc=InvalidAPIKeyError("bad key"))
    )
    monkeypatch.setattr(roa, "get_role", lambda session, role_name: None)
    monkeypatch.setattr(
        roa, "get_cached_fallback", lambda session, role_name, ref: None
    )

    sentinel = GeneralKnowledgeFloorResult(
        role_name="ghost_role",
        confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
        label="x",
    )
    get_floor_mock = MagicMock(return_value=sentinel)
    monkeypatch.setattr(roa, "get_general_knowledge_floor", get_floor_mock)

    result = await roa.ground_role("ghost_role", MagicMock(), datetime(2026, 7, 4))

    assert result is sentinel
    get_floor_mock.assert_called_once_with("ghost_role")


# --- source-level failure modes: distinguished, not accidental -----------


async def test_himalayas_call_failure_wraps_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", raise_exc=ConnectionError("boom")),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_himalayas_listings("Data Analyst")


async def test_himalayas_unparseable_response_wraps_as_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool(
            "search_jobs", _himalayas_response("not a recognizable response")
        ),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_himalayas_listings("Data Analyst")


async def test_himalayas_is_error_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool(
            "search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT, is_error=True)
        ),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_himalayas_listings("Data Analyst")


async def test_himalayas_missing_search_jobs_tool_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        roa, "_get_himalayas_toolset", lambda: _FakeHimalayasToolset([])
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_himalayas_listings("Data Analyst")


async def test_tavily_specific_error_wraps_as_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tavily(
        monkeypatch, _FakeTavilyClient(raise_exc=InvalidAPIKeyError("bad key"))
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_tavily_results("Data Analyst")


async def test_tavily_malformed_response_wraps_as_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result missing 'url'/'title' is a TavilyParseError inside
    data/tavily_parser.py — must be wrapped the same way a Himalayas
    HimalayasParseError is, not conflated with "no usable signal."
    """
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient({"results": [{"title": "x", "content": "y", "score": 0.5}]}),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_tavily_results("Data Analyst")


# --- timeout paths, tested explicitly, not just happy path ---------------


async def test_himalayas_timeout_raises_grounding_source_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(roa, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool(
            "search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT), sleep_seconds=0.2
        ),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_himalayas_listings("Data Analyst")


async def test_tavily_timeout_raises_grounding_source_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(roa, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _SKILL_BEARING_CONTENT)]),
            sleep_seconds=0.2,
        ),
    )
    with pytest.raises(roa.GroundingSourceCallError):
        await roa._fetch_tavily_results("Data Analyst")


# --- _safe_fetch_*: call_failed vs no_signal is intentional, not lost ----


async def test_safe_fetch_himalayas_distinguishes_call_failed_from_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", raise_exc=ConnectionError("boom")),
    )
    _, failed_status = await roa._safe_fetch_himalayas("Data Analyst")
    assert failed_status == "call_failed"

    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(NONSENSE_TEXT)),
    )
    _, no_signal_status = await roa._safe_fetch_himalayas("zzznonexistentrolezzz123456")
    assert no_signal_status == "no_signal"

    _patch_himalayas(
        monkeypatch,
        _FakeHimalayasTool("search_jobs", _himalayas_response(BACKEND_ENGINEER_TEXT)),
    )
    _, signal_status = await roa._safe_fetch_himalayas("Data Analyst")
    assert signal_status == "signal"


async def test_safe_fetch_tavily_distinguishes_call_failed_from_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tavily(
        monkeypatch, _FakeTavilyClient(raise_exc=InvalidAPIKeyError("bad key"))
    )
    _, failed_status = await roa._safe_fetch_tavily("Data Analyst")
    assert failed_status == "call_failed"

    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _GENERIC_CONTENT)])
        ),
    )
    _, no_signal_status = await roa._safe_fetch_tavily("Data Analyst")
    assert no_signal_status == "no_signal"

    _patch_tavily(
        monkeypatch,
        _FakeTavilyClient(
            _tavily_response([_tavily_result_dict(0.9, _SKILL_BEARING_CONTENT)])
        ),
    )
    _, signal_status = await roa._safe_fetch_tavily("Data Analyst")
    assert signal_status == "signal"


# --- Clarify Gate conversational content: begin_clarify_gate / advance_clarify_gate ---


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiModels:
    """Returns each of `responses` in order, one per call. Raises
    `AssertionError` (a test-authoring bug, not a code-under-test bug) if
    a test calls the fake client more times than it queued responses for.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        raise_exc: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._responses = list(responses or [])
        self._raise_exc = raise_exc
        self._sleep_seconds = sleep_seconds
        self.calls: list[dict] = []

    async def generate_content(self, *, model: str, contents: str, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_exc is not None:
            raise self._raise_exc
        if not self._responses:
            raise AssertionError(
                "_FakeGeminiModels ran out of queued responses — the test "
                "queued fewer canned responses than the code under test "
                "actually calls Gemini"
            )
        return _FakeGeminiResponse(self._responses.pop(0))


class _FakeGeminiAio:
    def __init__(self, models: _FakeGeminiModels) -> None:
        self.models = models


class _FakeGeminiClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        raise_exc: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.aio = _FakeGeminiAio(
            _FakeGeminiModels(responses, raise_exc, sleep_seconds)
        )


def _patch_gemini(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str] | None = None,
    raise_exc: Exception | None = None,
    sleep_seconds: float = 0.0,
) -> _FakeGeminiClient:
    client = _FakeGeminiClient(
        responses=responses, raise_exc=raise_exc, sleep_seconds=sleep_seconds
    )
    monkeypatch.setattr(roa, "_get_gemini_client", lambda: client)
    return client


def _agent_turn(content: str) -> dict:
    return {"role": "agent", "content": content}


def _user_turn(content: str) -> dict:
    return {"role": "user", "content": content}


REFERENCE_TIME = datetime(2026, 7, 5, 12, 0, 0)


async def test_begin_clarify_gate_real_role_resolves_without_any_gemini_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD §7.2: 'Clearly real role -> accept, proceed to Research' — no
    LLM call is needed at all, since classification is fully deterministic
    (security/input_gate.py).
    """
    client = _patch_gemini(monkeypatch, responses=[])

    turn = await roa.begin_clarify_gate("Data Analyst")

    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    assert turn.resolved_role == "Data Analyst"
    assert not turn.exited
    assert client.aio.models.calls == []


async def test_begin_clarify_gate_nonsense_loops_back_without_any_gemini_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD §7.2: nonsense routes to 'ask to clarify' — a loop, not an
    exit — and does not consume a narrowing round.
    """
    client = _patch_gemini(monkeypatch, responses=[])

    turn = await roa.begin_clarify_gate("asdkjfh")

    assert turn.gate_state.stage is ClarifyGateStage.NARROWING
    assert turn.gate_state.narrowing_rounds_used == 0
    assert turn.resolved_role is None
    assert not turn.exited
    assert turn.message == roa.CLARIFY_GATE_NONSENSE_REPROMPT
    assert client.aio.models.calls == []


async def test_begin_clarify_gate_vague_asks_one_narrowing_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD §7.2: vague-but-genuine enters the narrowing loop with one
    question at a time.
    """
    client = _patch_gemini(
        monkeypatch, responses=["What kind of apps do you want to build?"]
    )

    turn = await roa.begin_clarify_gate("I want to make apps")

    assert turn.gate_state.stage is ClarifyGateStage.NARROWING
    assert turn.gate_state.narrowing_rounds_used == 0
    assert turn.message == "What kind of apps do you want to build?"
    assert turn.context.original_stated_goal == "I want to make apps"
    assert len(client.aio.models.calls) == 1


async def test_advance_clarify_gate_narrowing_round_resolves_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A narrowing answer that resolves a concrete role moves straight to
    RESOLVED regardless of rounds used so far (mirrors
    tests/test_input_gate.py's `advance_after_narrowing_round` case).
    """
    _patch_gemini(
        monkeypatch,
        responses=[json.dumps({"resolved": True, "role": "Backend Engineer"})],
    )
    gate_state = roa.ClarifyGateState(
        stage=ClarifyGateStage.NARROWING, narrowing_rounds_used=0
    )
    context = roa.ClarifyGateContext(original_stated_goal="I want to make apps")
    conversation = [_agent_turn("What kind of apps?")]

    turn = await roa.advance_clarify_gate(
        gate_state,
        context,
        conversation,
        "Backend stuff, APIs and databases",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )

    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    assert turn.resolved_role == "Backend Engineer"


async def test_full_clarify_gate_sequence_rejects_proposal_and_explanation_then_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end Gherkin: vague goal -> two unresolved narrowing rounds
    (bound reached) -> best-guess proposal rejected -> explanation
    rejected -> grounding check on the user's ORIGINAL words -> zero
    market signal -> exit, no outline built.
    """
    client = _patch_gemini(
        monkeypatch,
        responses=[
            "What area of app development interests you?",  # narrowing Q1
            json.dumps({"resolved": False, "role": None}),  # eval answer 1
            "Do you prefer frontend or backend work?",  # narrowing Q2
            json.dumps({"resolved": False, "role": None}),  # eval answer 2
            json.dumps(
                {
                    "role": "Backend Engineer",
                    "message": "Sounds like Backend Engineer — right?",
                }
            ),  # best-guess proposal
            json.dumps({"accepted": False}),  # proposal rejected
            "Backend engineers build server-side logic and APIs.",  # explanation
            json.dumps({"accepted": False}),  # explanation rejected
        ],
    )
    ground_role_mock = MagicMock(
        return_value=roa.GeneralKnowledgeFloorResult(
            role_name="I want to make apps",
            confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
            label="No cached or live market data is available.",
        )
    )

    async def _fake_ground_role(role_name, session, reference_time):
        return ground_role_mock(role_name, session, reference_time)

    monkeypatch.setattr(roa, "ground_role", _fake_ground_role)

    turn = await roa.begin_clarify_gate("I want to make apps")
    assert turn.gate_state.stage is ClarifyGateStage.NARROWING
    assert turn.gate_state.narrowing_rounds_used == 0
    conversation = [_agent_turn(turn.message)]

    turn = await roa.advance_clarify_gate(
        turn.gate_state,
        turn.context,
        conversation,
        "I don't really know yet",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )
    assert turn.gate_state.stage is ClarifyGateStage.NARROWING
    assert turn.gate_state.narrowing_rounds_used == 1
    conversation += [_user_turn("I don't really know yet"), _agent_turn(turn.message)]

    turn = await roa.advance_clarify_gate(
        turn.gate_state,
        turn.context,
        conversation,
        "still not sure",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )
    assert turn.gate_state.stage is ClarifyGateStage.PROPOSE_BEST_GUESS
    assert turn.gate_state.narrowing_rounds_used == 2
    assert turn.context.proposed_role == "Backend Engineer"
    conversation += [_user_turn("still not sure"), _agent_turn(turn.message)]

    turn = await roa.advance_clarify_gate(
        turn.gate_state,
        turn.context,
        conversation,
        "no, that's not it",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )
    assert turn.gate_state.stage is ClarifyGateStage.EXPLAIN_ROLE
    conversation += [_user_turn("no, that's not it"), _agent_turn(turn.message)]

    turn = await roa.advance_clarify_gate(
        turn.gate_state,
        turn.context,
        conversation,
        "nope, still not right",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )

    assert turn.gate_state.stage is ClarifyGateStage.EXITED
    assert turn.exited is True
    assert turn.resolved_role is None
    assert "I want to make apps" in turn.message
    ground_role_mock.assert_called_once()
    assert ground_role_mock.call_args.args[0] == "I want to make apps"
    assert len(client.aio.models.calls) == 8


async def test_advance_clarify_gate_resolves_at_low_confidence_on_weak_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same rejection path, but any market signal (even weak) resolves at
    low confidence instead of exiting (PRD §7.2), grounding the user's
    ORIGINAL stated goal, never the most recent message."""
    _patch_gemini(monkeypatch, responses=[json.dumps({"accepted": False})])

    async def _fake_ground_role(role_name, session, reference_time):
        assert role_name == "something with computers"
        return roa.LiveGroundingResult(
            role_name=role_name,
            skills=[],
            confidence=ConfidenceTier.LOW,
            has_conflict=False,
            himalayas_status="no_signal",
            tavily_status="signal",
        )

    monkeypatch.setattr(roa, "ground_role", _fake_ground_role)

    gate_state = ClarifyGateState(
        stage=ClarifyGateStage.EXPLAIN_ROLE, narrowing_rounds_used=2
    )
    context = roa.ClarifyGateContext(
        original_stated_goal="something with computers",
        proposed_role="Computer Repair Technician",
    )
    conversation = [_agent_turn("Computer repair techs fix hardware. Sound right?")]

    turn = await roa.advance_clarify_gate(
        gate_state,
        context,
        conversation,
        "not really",
        session=MagicMock(),
        reference_time=REFERENCE_TIME,
    )

    assert turn.gate_state.stage is ClarifyGateStage.RESOLVED
    assert turn.resolved_role == "something with computers"
    assert not turn.exited


async def test_advance_clarify_gate_rejects_terminal_stage() -> None:
    resolved_state = ClarifyGateState(
        stage=ClarifyGateStage.RESOLVED, narrowing_rounds_used=1
    )
    context = roa.ClarifyGateContext(original_stated_goal="Data Analyst")

    with pytest.raises(ValueError):
        await roa.advance_clarify_gate(
            resolved_state,
            context,
            [],
            "anything",
            session=MagicMock(),
            reference_time=REFERENCE_TIME,
        )


async def test_last_agent_message_raises_without_a_prior_agent_turn() -> None:
    with pytest.raises(ValueError):
        roa._last_agent_message([_user_turn("hello")])


async def test_gemini_json_helper_raises_gemini_call_error_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=["not valid json at all"])

    with pytest.raises(GeminiCallError):
        await roa._evaluate_acceptance("some proposal", "yes")


async def test_gemini_json_helper_raises_gemini_call_error_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[json.dumps({"unexpected": "shape"})])

    with pytest.raises(GeminiCallError):
        await roa._evaluate_acceptance("some proposal", "yes")


async def test_gemini_timeout_raises_gemini_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit timeout-path test (CLAUDE.md testing expectations), mirroring
    test_himalayas_timeout_raises_grounding_source_call_error /
    test_tavily_timeout_raises_grounding_source_call_error above."""
    monkeypatch.setattr(roa, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_gemini(monkeypatch, responses=["irrelevant"], sleep_seconds=0.2)

    with pytest.raises(GeminiCallError):
        await roa._generate_narrowing_question("Data stuff", [])


# --- Initial Outline Creation: create_initial_outline ---------------------


def _grounded(
    skill: str,
    source_url: str,
    source_type: str = "job_listing",
    confidence: ConfidenceTier = ConfidenceTier.HIGH,
) -> ValidatedGroundedContent:
    return ValidatedGroundedContent(
        source_url=source_url,
        source_type=source_type,
        confidence=confidence,
        extra={"skill": skill},
    )


GIT_SKILL = _grounded("Git", "https://git.example/1")
PYTHON_SKILL = _grounded("Python", "https://python.example/1")
DJANGO_SKILL = _grounded(
    "Django",
    "https://django.example/1",
    source_type="web_search",
    confidence=ConfidenceTier.MEDIUM,
)

# One group per skill except Python, which fans out into 3 topics —
# exercises both cross-group ordering (Git, then Python*, then Django)
# and within-group ordering (Python syntax -> functions -> OOP).
_WELL_FORMED_HIERARCHY_RESPONSE = json.dumps(
    {
        "groups": [
            {
                "topic_group": "Git",
                "topics": [{"topic_name": "Git basics", "source_skill": "Git"}],
            },
            {
                "topic_group": "Python",
                "topics": [
                    {
                        "topic_name": "Python syntax and variables",
                        "source_skill": "Python",
                    },
                    {"topic_name": "Python functions", "source_skill": "Python"},
                    {
                        "topic_name": "Python object-oriented programming",
                        "source_skill": "Python",
                    },
                ],
            },
            {
                "topic_group": "Django",
                "topics": [
                    {
                        "topic_name": "Django framework basics",
                        "source_skill": "Django",
                    }
                ],
            },
        ]
    }
)


async def test_create_initial_outline_orders_groups_in_prerequisite_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global cross-group ordering: Git and all of Python's topics must
    precede Django (task's explicit example)."""
    _patch_gemini(monkeypatch, responses=[_WELL_FORMED_HIERARCHY_RESPONSE])

    topics = await roa.create_initial_outline(
        "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL]
    )

    by_name = {t.topic_name: t for t in topics}
    django_position = by_name["Django framework basics"].hierarchy_position
    assert by_name["Git basics"].hierarchy_position < django_position
    assert by_name["Python syntax and variables"].hierarchy_position < django_position
    assert by_name["Python functions"].hierarchy_position < django_position
    assert (
        by_name["Python object-oriented programming"].hierarchy_position
        < django_position
    )


async def test_create_initial_outline_orders_within_group_in_prerequisite_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within-group ordering: inside the 'Python' group, syntax must
    precede functions must precede OOP — both by position_in_group and by
    the global hierarchy_position."""
    _patch_gemini(monkeypatch, responses=[_WELL_FORMED_HIERARCHY_RESPONSE])

    topics = await roa.create_initial_outline(
        "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL]
    )

    python_topics = [t for t in topics if t.topic_group == "Python"]
    assert [t.topic_name for t in python_topics] == [
        "Python syntax and variables",
        "Python functions",
        "Python object-oriented programming",
    ]
    assert [t.position_in_group for t in python_topics] == [1, 2, 3]
    assert [t.hierarchy_position for t in python_topics] == sorted(
        t.hierarchy_position for t in python_topics
    )


async def test_create_initial_outline_carries_source_fields_through_unaltered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every output topic's source_url/source_type/confidence must equal
    its input skill's exactly — Gemini's JSON never even contains these
    fields, so this also proves they can't be silently altered."""
    _patch_gemini(monkeypatch, responses=[_WELL_FORMED_HIERARCHY_RESPONSE])

    topics = await roa.create_initial_outline(
        "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL]
    )
    by_name = {t.topic_name: t for t in topics}

    git_topic = by_name["Git basics"]
    assert git_topic.source_url == GIT_SKILL.source_url
    assert git_topic.source_type == GIT_SKILL.source_type
    assert git_topic.confidence == GIT_SKILL.confidence

    for name in (
        "Python syntax and variables",
        "Python functions",
        "Python object-oriented programming",
    ):
        assert by_name[name].source_url == PYTHON_SKILL.source_url
        assert by_name[name].source_type == PYTHON_SKILL.source_type
        assert by_name[name].confidence == PYTHON_SKILL.confidence

    django_topic = by_name["Django framework basics"]
    assert django_topic.source_url == DJANGO_SKILL.source_url
    assert django_topic.source_type == DJANGO_SKILL.source_type
    assert django_topic.confidence == DJANGO_SKILL.confidence


async def test_create_initial_outline_sets_is_enrichment_false_and_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[_WELL_FORMED_HIERARCHY_RESPONSE])

    topics = await roa.create_initial_outline(
        "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL]
    )

    assert topics
    for topic in topics:
        assert topic.is_enrichment is False
        assert topic.status == "not_started"


async def test_create_initial_outline_raises_on_missing_groups_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[json.dumps({"unexpected": "shape"})])

    with pytest.raises(GeminiCallError):
        await roa.create_initial_outline("Backend Engineer", [GIT_SKILL], [])


async def test_create_initial_outline_raises_on_fabricated_source_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LLM referencing a skill that isn't in the grounded input at all
    must raise, not silently pass through an unattributable topic."""
    response = json.dumps(
        {
            "groups": [
                {
                    "topic_group": "Git",
                    "topics": [
                        {
                            "topic_name": "Git basics",
                            "source_skill": "Some Made Up Skill",
                        }
                    ],
                }
            ]
        }
    )
    _patch_gemini(monkeypatch, responses=[response])

    with pytest.raises(GeminiCallError):
        await roa.create_initial_outline("Backend Engineer", [GIT_SKILL], [])


async def test_create_initial_outline_raises_when_a_skill_is_never_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping a grounded skill entirely (never referenced by any topic)
    must raise — CLAUDE.md guardrail #1's 'never drop' extended to mean
    every grounded skill must end up somewhere in the outline."""
    response = json.dumps(
        {
            "groups": [
                {
                    "topic_group": "Git",
                    "topics": [{"topic_name": "Git basics", "source_skill": "Git"}],
                }
            ]
        }
    )
    _patch_gemini(monkeypatch, responses=[response])

    with pytest.raises(GeminiCallError):
        await roa.create_initial_outline(
            "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], []
        )


async def test_create_initial_outline_raises_on_malformed_topic_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = json.dumps(
        {"groups": [{"topic_group": "Git", "topics": [{"topic_name": "Git basics"}]}]}
    )
    _patch_gemini(monkeypatch, responses=[response])

    with pytest.raises(GeminiCallError):
        await roa.create_initial_outline("Backend Engineer", [GIT_SKILL], [])


async def test_create_initial_outline_raises_on_empty_group_list() -> None:
    with pytest.raises(ValueError):
        await roa.create_initial_outline("Backend Engineer", [], [])


def test_build_grounded_skill_map_raises_on_duplicate_skill_name() -> None:
    duplicate = _grounded("Git", "https://git.example/2")
    with pytest.raises(ValueError):
        roa._build_grounded_skill_map([GIT_SKILL], [duplicate])


# --- Outline Confirmation: begin_outline_confirmation / handle_review_turn ---


async def _build_sample_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> list[roa.InitialOutlineTopic]:
    """5 real topics across 3 groups (Git, Python, Django), built via the
    already-tested create_initial_outline + the existing well-formed
    hierarchy fixture above — avoids hand-constructing InitialOutlineTopic
    instances that could drift from what the real function actually
    produces.
    """
    _patch_gemini(monkeypatch, responses=[_WELL_FORMED_HIERARCHY_RESPONSE])
    return await roa.create_initial_outline(
        "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL]
    )


def _topic_explanations_response(topics: list[roa.InitialOutlineTopic]) -> str:
    return json.dumps(
        {
            "topic_explanations": [
                {
                    "topic_name": t.topic_name,
                    "explanation": f"Because of {t.topic_name}.",
                }
                for t in topics
            ]
        }
    )


async def test_begin_outline_confirmation_presents_grounded_why_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)

    _patch_gemini(monkeypatch, responses=[_topic_explanations_response(topics)])
    turn = await roa.begin_outline_confirmation("Backend Engineer", topics)

    assert turn.state.stage is OutlineConfirmationStage.REVIEWING
    assert turn.state.rounds_used == 0
    assert turn.topics == topics
    assert not turn.concluded
    # every topic's real source_url must appear in the presentation —
    # never fabricated, never dropped (CLAUDE.md guardrail #1)
    for topic in topics:
        assert topic.topic_name in turn.message
        assert topic.source_url in turn.message


async def test_generate_topic_explanations_raises_on_uncovered_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    incomplete_response = json.dumps(
        {
            "topic_explanations": [
                {"topic_name": topics[0].topic_name, "explanation": "Because reasons."}
            ]
        }
    )
    _patch_gemini(monkeypatch, responses=[incomplete_response])

    with pytest.raises(GeminiCallError):
        await roa._generate_topic_explanations("Backend Engineer", topics)


async def test_generate_topic_explanations_raises_on_unknown_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    response_with_fabricated_topic = json.dumps(
        {
            "topic_explanations": [
                {"topic_name": t.topic_name, "explanation": "Because reasons."}
                for t in topics
            ]
            + [{"topic_name": "Not A Real Topic", "explanation": "Made up."}]
        }
    )
    _patch_gemini(monkeypatch, responses=[response_with_fabricated_topic])

    with pytest.raises(GeminiCallError):
        await roa._generate_topic_explanations("Backend Engineer", topics)


async def test_handle_review_turn_question_does_not_consume_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps({"action": "question"}),
            "Git is foundational for version control in this role.",
        ],
    )
    turn = await roa.handle_review_turn(
        state, "Backend Engineer", topics, "Why is Git in here?"
    )

    assert turn.state.stage is OutlineConfirmationStage.REVIEWING
    assert turn.state.rounds_used == 0
    assert turn.topics == topics
    assert not turn.concluded


async def test_handle_review_turn_question_never_consumes_a_round_even_repeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    for _ in range(4):
        _patch_gemini(
            monkeypatch,
            responses=[json.dumps({"action": "question"}), "Here's the answer."],
        )
        turn = await roa.handle_review_turn(
            state, "Backend Engineer", topics, "Another question?"
        )
        assert turn.state.rounds_used == 0
        assert turn.state.stage is OutlineConfirmationStage.REVIEWING
        state = turn.state


async def test_handle_review_turn_concern_consumes_a_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps({"action": "concern"}),
            "That's a fair point — here's some clarification.",
        ],
    )
    turn = await roa.handle_review_turn(
        state, "Backend Engineer", topics, "I'm worried this is too much Django."
    )

    assert turn.state.stage is OutlineConfirmationStage.REVIEWING
    assert turn.state.rounds_used == 1
    assert not turn.concluded


async def test_handle_review_turn_addition_request_consumes_round_without_regenerating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """handle_review_turn classifies and consumes the round for an
    addition request, but does NOT regenerate the outline itself — topics
    must come back unchanged; regeneration is a separate, explicit call
    (regenerate_outline_with_addition) once the addition is grounded.
    """
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    _patch_gemini(monkeypatch, responses=[json.dumps({"action": "addition_request"})])
    turn = await roa.handle_review_turn(
        state, "Backend Engineer", topics, "Can you add GraphQL?"
    )

    assert turn.state.stage is OutlineConfirmationStage.REVIEWING
    assert turn.state.rounds_used == 1
    assert turn.topics == topics
    assert not turn.concluded


async def test_handle_review_turn_confirm_ends_review_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
    )

    _patch_gemini(monkeypatch, responses=[json.dumps({"action": "confirm"})])
    turn = await roa.handle_review_turn(
        state, "Backend Engineer", topics, "Looks great, let's start!"
    )

    assert turn.state.stage is OutlineConfirmationStage.CONFIRMED
    assert turn.concluded
    assert turn.message == roa.OUTLINE_CONFIRMATION_CONFIRMED_MESSAGE


async def test_handle_review_turn_round_bound_reached_after_two_concerns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound-exhausted exit: after exactly 2 round-consuming actions, the
    loop concludes and the response is framed 'starting here, refine as
    we go' — never exceeding the 2-round bound.
    """
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    _patch_gemini(
        monkeypatch,
        responses=[json.dumps({"action": "concern"}), "First response."],
    )
    turn = await roa.handle_review_turn(
        state, "Backend Engineer", topics, "First concern"
    )
    assert turn.state.rounds_used == 1
    assert not turn.concluded

    _patch_gemini(
        monkeypatch,
        responses=[json.dumps({"action": "concern"}), "Second response."],
    )
    turn = await roa.handle_review_turn(
        turn.state, "Backend Engineer", topics, "Second concern"
    )

    assert turn.state.stage is OutlineConfirmationStage.BOUND_REACHED
    assert turn.state.rounds_used == 2
    assert turn.concluded
    assert roa.OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE in turn.message
    assert "Second response." in turn.message


async def test_handle_review_turn_rejects_wrong_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    confirmed_state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.CONFIRMED, rounds_used=1
    )

    with pytest.raises(ValueError):
        await roa.handle_review_turn(
            confirmed_state, "Backend Engineer", topics, "anything"
        )


async def test_classify_review_turn_raises_on_unrecognized_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = await _build_sample_topics(monkeypatch)
    _patch_gemini(monkeypatch, responses=[json.dumps({"action": "not_a_real_action"})])

    with pytest.raises(GeminiCallError):
        await roa._classify_review_turn("Backend Engineer", topics, "hello")


async def test_regenerate_outline_with_addition_produces_valid_sourced_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted addition: regenerate via create_initial_outline (never
    outline/hierarchy.py's insertion logic), producing a fully valid,
    fully sourced outline — strictly-increasing hierarchy positions,
    every topic's sourcing traceable to an input skill (reusing the same
    checks as tests/test_pipeline_integration.py's
    _assert_valid_outline)."""
    topics = await _build_sample_topics(monkeypatch)
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
    )
    new_skill = _grounded("GraphQL", "https://graphql.example/1")

    new_response = json.dumps(
        {
            "groups": [
                {
                    "topic_group": "Git",
                    "topics": [{"topic_name": "Git basics", "source_skill": "Git"}],
                },
                {
                    "topic_group": "Python",
                    "topics": [
                        {
                            "topic_name": "Python syntax and variables",
                            "source_skill": "Python",
                        },
                        {"topic_name": "Python functions", "source_skill": "Python"},
                        {
                            "topic_name": "Python object-oriented programming",
                            "source_skill": "Python",
                        },
                    ],
                },
                {
                    "topic_group": "Django",
                    "topics": [
                        {
                            "topic_name": "Django framework basics",
                            "source_skill": "Django",
                        }
                    ],
                },
                {
                    "topic_group": "GraphQL",
                    "topics": [
                        {"topic_name": "GraphQL basics", "source_skill": "GraphQL"}
                    ],
                },
            ]
        }
    )
    _patch_gemini(monkeypatch, responses=[new_response])

    turn = await roa.regenerate_outline_with_addition(
        state, "Backend Engineer", [GIT_SKILL, PYTHON_SKILL], [DJANGO_SKILL], new_skill
    )

    assert not turn.concluded
    assert "GraphQL" in turn.message
    new_topics = turn.topics
    assert len(new_topics) == len(topics) + 1

    positions = [t.hierarchy_position for t in new_topics]
    assert positions == list(range(1, len(new_topics) + 1))

    valid_source_tuples = {
        (s.source_url, s.source_type, s.confidence)
        for s in (GIT_SKILL, PYTHON_SKILL, DJANGO_SKILL, new_skill)
    }
    for topic in new_topics:
        assert topic.source_url
        assert topic.confidence
        assert (
            topic.source_url,
            topic.source_type,
            topic.confidence,
        ) in valid_source_tuples
        assert topic.is_enrichment is False
        assert topic.status == "not_started"

    # unchanged topics' sourcing must be untouched by regeneration
    old_by_name = {t.topic_name: t for t in topics}
    for topic in new_topics:
        if topic.topic_name in old_by_name:
            old_topic = old_by_name[topic.topic_name]
            assert topic.source_url == old_topic.source_url
            assert topic.source_type == old_topic.source_type
            assert topic.confidence == old_topic.confidence


async def test_regenerate_outline_with_addition_frames_bound_reached_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.BOUND_REACHED, rounds_used=2
    )
    new_skill = _grounded("GraphQL", "https://graphql.example/1")
    response = json.dumps(
        {
            "groups": [
                {
                    "topic_group": "Git",
                    "topics": [{"topic_name": "Git basics", "source_skill": "Git"}],
                },
                {
                    "topic_group": "GraphQL",
                    "topics": [
                        {"topic_name": "GraphQL basics", "source_skill": "GraphQL"}
                    ],
                },
            ]
        }
    )
    _patch_gemini(monkeypatch, responses=[response])

    turn = await roa.regenerate_outline_with_addition(
        state, "Backend Engineer", [GIT_SKILL], [], new_skill
    )

    assert turn.concluded
    assert roa.OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE in turn.message
