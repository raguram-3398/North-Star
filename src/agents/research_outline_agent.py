"""Research & Outline Agent — reasoning/generation only.

Owns (Architecture_North_Star.md §3): the *content* of clarify-gate
narrowing questions and best-guess role proposals/explanations,
cross-validation normalization judgment (anchored to roles_cache), and
initial full-outline hierarchy creation (sequencing sourced skills into
dependency order).

Calls as tools (deterministic, not owned — never reimplemented inline):
security/input_gate.py, security/output_guard.py, data/roles_cache.py,
data/himalayas_parser.py, data/himalayas_relevance.py, data/tavily_parser.py,
data/cross_validation.py, data/grounding_fallback.py,
outline/significant_event.py, outline/hierarchy.py, patches/patch_manager.py.

Tools: Himalayas MCP, Tavily search, Postgres (via gated write paths only
— never a raw insert).

`ground_role` below is `cross_validate_market_data`'s real implementation
(Architecture §3's "cross-validation normalization judgment"): PRD §7.3
frames this judgment as rule application "anchored to roles.json... not
open-ended LLM judgment," so the actual tier decision is delegated to
`data/cross_validation.py`'s pure function — this async function is the
orchestrator that calls the two live sources, runs Himalayas's response
through `data/himalayas_parser.py` + `data/himalayas_relevance.py` and
Tavily's through `data/tavily_parser.py`, reads the roles_cache anchor,
asks `data/cross_validation.py` for a tier (including a possible
Tavily-only medium result — see that module's docstring), and falls
through to `data/grounding_fallback.py` when live grounding produces no
usable signal, per PRD §7.3's confidence ladder.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from sqlalchemy.orm import Session
from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from data.cross_validation import decide_confidence_tier, tavily_has_usable_signal
from data.grounding_fallback import (
    CachedFallbackResult,
    GeneralKnowledgeFloorResult,
    get_cached_fallback,
    get_general_knowledge_floor,
)
from data.himalayas_parser import ParsedJobListing, parse_search_jobs_response
from data.himalayas_relevance import has_usable_himalayas_signal
from data.roles_cache import get_role
from data.tavily_parser import ParsedSearchResult, parse_tavily_response
from security.output_guard import (
    ConfidenceTier,
    ValidatedGroundedContent,
    validate_output_object,
)
from utils.exceptions import (
    GroundingSourceCallError,
    HimalayasParseError,
    TavilyParseError,
)

# CLAUDE.md guardrail #14: explicit timeout on every external call.
# Reuses the 10s convention already established in db/connection.py and
# tests/spike_grounding_connectivity.py.
EXTERNAL_CALL_TIMEOUT_SECONDS = 10

HIMALAYAS_MCP_URL = "https://mcp.himalayas.app/mcp"

# "job_listing" matches the source_type convention already used for
# Himalayas-origin ValidatedGroundedContent elsewhere in this codebase
# (tests/test_roles_cache.py, tests/test_output_guard.py).
HIMALAYAS_SOURCE_TYPE = "job_listing"

# source_type for skills built from data/cross_validation.py's
# TavilyCitation (the Tavily-only medium-confidence path) — distinct from
# Himalayas's, since it names a genuinely different kind of source.
TAVILY_SOURCE_TYPE = "web_search"

_SourceStatus = Literal["signal", "no_signal", "call_failed"]

# One client per module (CLAUDE.md coding conventions), lazy-but-memoized
# rather than instantiated at raw import time — mirrors db/connection.py's
# Engine singleton, avoiding a hard import-time requirement on
# TAVILY_API_KEY for modules that only transitively import this one (e.g.
# in CI/test contexts with no Tavily key configured).
_himalayas_toolset: McpToolset | None = None
_tavily_client: TavilyClient | None = None


def _get_himalayas_toolset() -> McpToolset:
    """Return the module-level Himalayas MCP toolset, creating it on
    first use. No auth is configured — Himalayas's public tools need
    none (confirmed in tests/spike_grounding_connectivity.py).
    """
    global _himalayas_toolset
    if _himalayas_toolset is None:
        _himalayas_toolset = McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=HIMALAYAS_MCP_URL,
                timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
            ),
        )
    return _himalayas_toolset


def _get_tavily_client() -> TavilyClient:
    """Return the module-level Tavily client, creating it on first use.

    Raises RuntimeError if TAVILY_API_KEY is not set — mirrors
    db/connection.py's `_normalized_connection_string`'s treatment of a
    missing required credential.
    """
    global _tavily_client
    if _tavily_client is None:
        import os

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


@dataclass(frozen=True)
class LiveGroundingResult:
    """A successful live-grounding outcome (PRD §7.3's `high`/`medium`/
    `low` rungs) — see `ground_role`. `skills` are already
    `ValidatedGroundedContent` (CLAUDE.md guardrail #12); every one is
    Himalayas-sourced (see module docstring's scope-driven
    simplification). `himalayas_status`/`tavily_status` are kept on the
    result (rather than discarded once the tier is decided) so a caller
    or test can distinguish "this source had no relevant results" from
    "this source's call itself failed" — the two are handled the same
    way for tier purposes, but that collapsing is intentional, not an
    accident of the code losing track of which happened.
    """

    role_name: str
    skills: list[ValidatedGroundedContent]
    confidence: ConfidenceTier
    has_conflict: bool
    himalayas_status: _SourceStatus
    tavily_status: _SourceStatus


async def _fetch_himalayas_listings(role_name: str) -> list[ParsedJobListing]:
    """Call Himalayas MCP's `search_jobs` tool for `role_name` and parse
    the response via `data/himalayas_parser.py`.

    Raises `GroundingSourceCallError` if the MCP call itself fails or
    exceeds `EXTERNAL_CALL_TIMEOUT_SECONDS`, the toolset has no
    `search_jobs` tool, the response reports `isError`, or the response
    can't be parsed at all (`HimalayasParseError`) — all of these mean
    "Himalayas is unusable this round," distinct from a successful call
    that simply returns no relevant listings (see
    `data/himalayas_relevance.py`).
    """
    toolset = _get_himalayas_toolset()
    try:
        tools = await asyncio.wait_for(
            toolset.get_tools(), timeout=EXTERNAL_CALL_TIMEOUT_SECONDS
        )
        by_name = {tool.name: tool for tool in tools}
        if "search_jobs" not in by_name:
            raise GroundingSourceCallError(
                "Himalayas MCP toolset has no 'search_jobs' tool"
            )
        result = await asyncio.wait_for(
            by_name["search_jobs"].run_async(
                args={"keyword": role_name}, tool_context=None
            ),
            timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
        )
    except (ConnectionError, TimeoutError) as exc:
        raise GroundingSourceCallError(f"Himalayas MCP call failed: {exc}") from exc

    if result.get("isError"):
        raise GroundingSourceCallError(
            f"Himalayas MCP call returned isError=True: {result!r}"
        )

    content = result.get("content") or []
    raw_text = content[0]["text"] if content else ""
    try:
        return parse_search_jobs_response(raw_text)
    except HimalayasParseError as exc:
        raise GroundingSourceCallError(
            f"Himalayas response could not be parsed: {exc}"
        ) from exc


async def _fetch_tavily_results(role_name: str) -> list[ParsedSearchResult]:
    """Call Tavily's search for `role_name` for the same query shape used
    in tests/spike_grounding_connectivity.py, in a worker thread
    (tavily-python's client is synchronous) with an explicit timeout —
    both on the call itself (`timeout=` kwarg) and defensively via the
    outer `asyncio.wait_for` — then parse the response via
    `data/tavily_parser.py`.

    Raises `GroundingSourceCallError` on any Tavily-specific API failure
    or timeout, or if the response can't be parsed at all
    (`TavilyParseError`) — mirrors `_fetch_himalayas_listings`'s
    contract. Never returns silently empty/wrong data.
    """
    client = _get_tavily_client()
    query = f"{role_name} job requirements and key skills"
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=query,
                search_depth="basic",
                max_results=10,
                timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
            ),
            timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
        )
    except (
        BadRequestError,
        InvalidAPIKeyError,
        ForbiddenError,
        UsageLimitExceededError,
        MissingAPIKeyError,
        TavilyTimeoutError,
        TimeoutError,
    ) as exc:
        raise GroundingSourceCallError(f"Tavily call failed: {exc}") from exc

    try:
        return parse_tavily_response(response)
    except TavilyParseError as exc:
        raise GroundingSourceCallError(
            f"Tavily response could not be parsed: {exc}"
        ) from exc


async def _safe_fetch_himalayas(
    role_name: str,
) -> tuple[list[ParsedJobListing], _SourceStatus]:
    """Fetch + relevance-check Himalayas, collapsing any
    `GroundingSourceCallError` into the `"call_failed"` status rather than
    letting it propagate — `ground_role` treats `"call_failed"` and
    `"no_signal"` identically (both mean "no usable signal from
    Himalayas"), but keeping them distinct here means that collapsing is
    a deliberate choice made once, in one place, not an accident of
    losing the information.
    """
    try:
        listings = await _fetch_himalayas_listings(role_name)
    except GroundingSourceCallError:
        return [], "call_failed"
    if has_usable_himalayas_signal(role_name, listings):
        return listings, "signal"
    return listings, "no_signal"


async def _safe_fetch_tavily(
    role_name: str,
) -> tuple[list[ParsedSearchResult], _SourceStatus]:
    """Fetch + trust-check Tavily, collapsing any `GroundingSourceCallError`
    into `"call_failed"` — see `_safe_fetch_himalayas`'s docstring for why
    this collapsing is deliberate rather than accidental. Trust is
    `data/cross_validation.py`'s `tavily_has_usable_signal` (a distinct-
    skill count, never `score` — see that module's docstring), not a
    score threshold.
    """
    try:
        results = await _fetch_tavily_results(role_name)
    except GroundingSourceCallError:
        return [], "call_failed"
    if tavily_has_usable_signal(results):
        return results, "signal"
    return results, "no_signal"


async def ground_role(
    role_name: str,
    session: Session,
    reference_time: datetime,
) -> LiveGroundingResult | CachedFallbackResult | GeneralKnowledgeFloorResult:
    """Ground `role_name` against Himalayas + Tavily in parallel, apply
    PRD §7.3's cross-validation rules (`data/cross_validation.py`), and
    fall through to `data/grounding_fallback.py`'s cached-fallback/
    general-knowledge-only rungs only if live grounding produces no
    usable signal — the fallback-only-on-failure ordering that was never
    previously enforced anywhere in the codebase.

    Returns a `LiveGroundingResult` for the `high`/`medium`/`low` rungs,
    or whatever `data/grounding_fallback.py` produces (`CachedFallbackResult`
    or `GeneralKnowledgeFloorResult`) for the `cached-low`/
    `general-knowledge-only` rungs. Every skill on a `LiveGroundingResult`
    has already passed `security/output_guard.py`'s `validate_output_object`
    (CLAUDE.md guardrail #12) — no raw dict escapes this function.

    Does not raise for "no usable signal" from either source (that's the
    expected, graceful-degradation case the whole ladder exists for —
    CLAUDE.md guardrail #6). Only propagates exceptions for genuine
    programming/data-integrity errors (e.g.
    `security.output_guard.ConfidenceValidationError`, which would
    indicate a bug in this function's own candidate construction, not an
    external failure).
    """
    (himalayas_listings, himalayas_status), (tavily_results, tavily_status) = (
        await asyncio.gather(
            _safe_fetch_himalayas(role_name), _safe_fetch_tavily(role_name)
        )
    )
    himalayas_has_signal = himalayas_status == "signal"

    anchor_row = get_role(session, role_name)
    anchor_skills: frozenset[str] = frozenset()
    if anchor_row is not None:
        anchor_skills = frozenset(
            entry["skill"].casefold()
            for entry in (*anchor_row["core_skills"], *anchor_row["emerging_skills"])
            if entry.get("skill")
        )

    # casefolded skill -> (original-cased skill, first listing's source_url)
    himalayas_skill_map: dict[str, tuple[str, str]] = {}
    for listing in himalayas_listings:
        if listing.source_url is None:
            continue
        for skill in listing.skills:
            himalayas_skill_map.setdefault(
                skill.casefold(), (skill, listing.source_url)
            )

    decision = decide_confidence_tier(
        himalayas_has_signal=himalayas_has_signal,
        tavily_results=tavily_results,
        anchor_skills=anchor_skills,
        himalayas_skills=frozenset(himalayas_skill_map),
    )

    if decision.confidence is ConfidenceTier.REJECT:
        cached = get_cached_fallback(session, role_name, reference_time)
        if cached is not None:
            return cached
        return get_general_knowledge_floor(role_name)

    # decision.tavily_citation is populated only on the Tavily-only
    # medium-confidence path (himalayas_has_signal was False) — every
    # other branch's skills come from Himalayas, which is the only
    # source with anything in himalayas_skill_map in that case.
    if decision.tavily_citation is not None:
        validated_skills = [
            validate_output_object(
                {
                    "source_url": decision.tavily_citation.source_url,
                    "source_type": TAVILY_SOURCE_TYPE,
                    "confidence": decision.confidence.value,
                    "skill": skill,
                }
            )
            for skill in sorted(decision.tavily_citation.skills)
        ]
    else:
        validated_skills = [
            validate_output_object(
                {
                    "source_url": source_url,
                    "source_type": HIMALAYAS_SOURCE_TYPE,
                    "confidence": decision.confidence.value,
                    "skill": original_skill,
                }
            )
            for original_skill, source_url in himalayas_skill_map.values()
        ]

    return LiveGroundingResult(
        role_name=role_name,
        skills=validated_skills,
        confidence=decision.confidence,
        has_conflict=decision.has_conflict,
        himalayas_status=himalayas_status,
        tavily_status=tavily_status,
    )


def generate_clarify_gate_response(
    conversation_so_far: list[dict[str, str]], current_round: int
) -> dict[str, Any]:
    """Generate the next clarify-gate turn: a narrowing question, a
    best-guess role proposal, or an explanation of a rejected proposal,
    per PRD §7.2. Round-counting and bound enforcement are delegated to
    security/input_gate.py, not decided here.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def create_initial_outline(
    resolved_role: str, grounded_skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sequence grounded skill data into a dependency hierarchy (basics ->
    full role requirements), per PRD §7.4.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
