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
from security.input_gate import ClarifyGateStage, ClarifyGateState
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from utils.exceptions import ClarifyGateLLMError

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


async def test_gemini_json_helper_raises_clarify_gate_llm_error_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=["not valid json at all"])

    with pytest.raises(ClarifyGateLLMError):
        await roa._evaluate_acceptance("some proposal", "yes")


async def test_gemini_json_helper_raises_clarify_gate_llm_error_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[json.dumps({"unexpected": "shape"})])

    with pytest.raises(ClarifyGateLLMError):
        await roa._evaluate_acceptance("some proposal", "yes")


async def test_gemini_timeout_raises_clarify_gate_llm_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit timeout-path test (CLAUDE.md testing expectations), mirroring
    test_himalayas_timeout_raises_grounding_source_call_error /
    test_tavily_timeout_raises_grounding_source_call_error above."""
    monkeypatch.setattr(roa, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_gemini(monkeypatch, responses=["irrelevant"], sleep_seconds=0.2)

    with pytest.raises(ClarifyGateLLMError):
        await roa._generate_narrowing_question("Data stuff", [])
