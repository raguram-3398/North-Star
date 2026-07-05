"""Tests for agents/research_outline_agent.py's `ground_role` — the
cross-validation/grounding orchestrator (Architecture §3's "cross-
validation normalization judgment").

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

Patches target `agents.research_outline_agent.<name>` (where each name is
*used*), not where it's defined — CLAUDE.md's flagged
wrong-patch-target anti-pattern.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from tavily.errors import InvalidAPIKeyError

import agents.research_outline_agent as roa
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from security.output_guard import ConfidenceTier, ValidatedGroundedContent

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
