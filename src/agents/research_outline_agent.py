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

`begin_clarify_gate`/`advance_clarify_gate` below are the Clarify Gate's
conversational half (PRD §7.2): they own the LLM-driven content (the
actual narrowing questions, best-guess proposals, and role explanations)
while `security/input_gate.py` owns the deterministic first-pass
real/vague/nonsense classification and all bounded-loop state/round
counting — never reimplemented inline here (CLAUDE.md guardrail #10).
Every prompt these functions use is versioned in `PROMPT_REGISTRY` per
CLAUDE.md's LLM Call Discipline.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from google import genai
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StreamableHTTPConnectionParams,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types as genai_types
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
from security.input_gate import (
    ClarifyGateStage,
    ClarifyGateState,
    GoalClassification,
    advance_after_explanation_response,
    advance_after_narrowing_round,
    advance_after_proposal_response,
    classify_stated_goal,
    resolve_after_grounding_check,
    start_clarify_gate,
)
from security.output_guard import (
    ConfidenceTier,
    ValidatedGroundedContent,
    validate_output_object,
)
from utils.exceptions import (
    ClarifyGateLLMError,
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


# --- Clarify Gate (PRD §7.2) conversational content --------------------

# Judgment call: "flash" tier chosen for short, low-latency conversational
# turns (a handful of sentences at most) rather than long grounded
# generation — distinct from any future model choice for outline/
# hierarchy sequencing, which may warrant a stronger tier.
CLARIFY_GATE_GEMINI_MODEL = "gemini-2.5-flash"

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    """Return the module-level Gemini client, creating it on first use
    (one client per module — CLAUDE.md coding conventions; same
    lazy-but-memoized pattern as `_get_tavily_client`).

    Raises RuntimeError if GEMINI_API_KEY is not set.
    """
    global _gemini_client
    if _gemini_client is None:
        import os

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# CLAUDE.md's LLM Call Discipline: every prompt used for grounded or
# safety-critical generation is versioned here, never deleted once its
# baseline regression test (tests/test_research_outline_agent.py) locks
# it in — a version is frozen prose, not something later tasks may edit
# in place; a changed prompt gets a new "_v2" key instead.
PROMPT_REGISTRY: dict[str, str] = {
    "clarify_gate_narrowing_question_v1": (
        "A user is describing what tech role or skill they want to learn, "
        "but has only given a vague answer so far. Ask exactly ONE short, "
        "friendly narrowing question to help pin down a specific, concrete "
        "role (e.g. 'Backend Engineer', 'Data Analyst') — do not ask more "
        "than one question, do not list options, do not explain why you're "
        "asking.\n\n"
        "Original stated goal: {original_goal!r}\n"
        "Conversation so far:\n{conversation}\n\n"
        "Respond with only the question text, nothing else."
    ),
    "clarify_gate_narrowing_answer_evaluation_v1": (
        "A user is being asked narrowing questions to resolve a vague "
        "career goal into one concrete, specific tech role. Given the "
        "conversation so far and their latest answer, decide whether a "
        "single concrete role can now be confidently named.\n\n"
        "Original stated goal: {original_goal!r}\n"
        "Conversation so far:\n{conversation}\n"
        "Latest answer: {answer!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"resolved": true or false, "role": "<specific role name>" or '
        "null if not resolved}}. Do not resolve to a role unless it is "
        "genuinely specific and concrete, not still vague."
    ),
    "clarify_gate_best_guess_proposal_v1": (
        "A user's career goal could not be narrowed to a specific role "
        "after a bounded round of clarifying questions. Based on "
        "everything said so far, propose your single best-guess "
        "interpretation of the most likely specific tech role they mean.\n\n"
        "Original stated goal: {original_goal!r}\n"
        "Conversation so far:\n{conversation}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"role": "<specific role name>", "message": "<a short, friendly '
        "message proposing this role and asking the user to confirm it "
        "fits, e.g. 'It sounds like you might be interested in becoming "
        "a ...  — does that sound right?'\"}}."
    ),
    "clarify_gate_role_explanation_v1": (
        "A user rejected a proposed role interpretation. Write a short "
        "(2-4 sentence), clear, friendly explanation of what the role "
        "below actually involves day to day, then ask again whether this "
        "is what they're looking for.\n\n"
        "Proposed role: {role!r}\n\n"
        "Respond with only the explanation-and-question text, nothing "
        "else."
    ),
    "clarify_gate_acceptance_evaluation_v1": (
        "An agent proposed or explained a specific career role to a user "
        "and asked them to confirm whether it fits. Given the agent's "
        "message and the user's reply, decide whether the user accepted "
        "it (a genuine yes/confirmation) or rejected it (a no, or any "
        "answer that does not affirmatively confirm).\n\n"
        "Agent's message: {agent_message!r}\n"
        "User's reply: {user_reply!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"accepted": true or false}}.'
    ),
}

# Fixed, non-LLM re-prompt for a nonsense-classified stated goal (PRD
# §7.2: "reject, ask to clarify" — a loop-back, not an exit). No LLM call
# is needed here: the classification is already fully deterministic
# (security/input_gate.py's classify_stated_goal), so a canned, friendly
# re-prompt is honest and avoids an unnecessary Gemini call/cost for
# content that doesn't need personalizing.
CLARIFY_GATE_NONSENSE_REPROMPT = (
    "I didn't quite catch a tech role or skill in that — could you tell "
    "me a bit about what kind of tech work you're interested in, or a "
    "role you might want to pursue?"
)

# Fixed, non-LLM exit message for the zero-market-signal path (PRD §7.2:
# "State plainly that no current hiring activity exists for this; no "
# outline is built"). Not LLM-generated for the same reason as the
# nonsense re-prompt above: the outcome is already fully decided by
# `ground_role`'s confidence-ladder result, so a canned, unambiguous
# message is more honest than an LLM paraphrase of a purely factual
# outcome.
CLARIFY_GATE_ZERO_SIGNAL_EXIT_MESSAGE_TEMPLATE = (
    "I couldn't find any current hiring activity for {role!r} — no "
    "learning plan will be built for this. Feel free to describe a "
    "different role or area you're interested in instead."
)


@dataclass(frozen=True)
class ClarifyGateContext:
    """Agent-owned conversational context threaded alongside
    `security.input_gate.ClarifyGateState` — the loop-state module
    deliberately doesn't track this since it's *content*, not bounded-loop
    mechanics (Architecture §3's ownership split).

    `original_stated_goal` is captured once, at the very first turn, and
    never overwritten by anything that happens later in the loop (CLAUDE.md's
    "capture the original input before a retry loop overwrites it" —
    the ACCEPT_OWN_WORDS rung must ground the user's *original* words, not
    whatever text happened to be exchanged most recently).
    `proposed_role` holds the most recently proposed best-guess role, needed
    at EXPLAIN_ROLE/ACCEPT_OWN_WORDS to know what was actually proposed and
    rejected.
    """

    original_stated_goal: str
    proposed_role: str | None = None


@dataclass(frozen=True)
class ClarifyGateTurn:
    """One turn of clarify-gate output: the caller (e.g. the Streamlit UI)
    renders `message`, persists `gate_state`/`context` to carry into the
    next turn, and — once `resolved_role` is populated — proceeds to
    Research (PRD §7.2's "a resolved role — not an outline yet").

    `exited` is True only for the zero-market-signal exit (PRD §7.2): no
    outline is ever built for that turn, regardless of `resolved_role`
    (which stays None in that case).
    """

    gate_state: ClarifyGateState
    context: ClarifyGateContext
    message: str
    resolved_role: str | None = None
    exited: bool = False


ConversationHistory = list[dict[str, str]]


def _format_conversation(conversation: ConversationHistory) -> str:
    """Render a `{"role": ..., "content": ...}` conversation history as
    plain text for embedding in a prompt. Empty history renders as an
    explicit marker rather than a blank line, so the prompt template
    never silently loses this section.
    """
    if not conversation:
        return "(no prior turns)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


def _last_agent_message(conversation: ConversationHistory) -> str:
    """Return the most recent `role == "agent"` message in `conversation`
    — the actual proposal/explanation text a user's reply is responding
    to. Raises `ValueError` if none exists: `advance_clarify_gate` must
    never guess what the user is replying to.
    """
    for turn in reversed(conversation):
        if turn.get("role") == "agent":
            return turn["content"]
    raise ValueError(
        "advance_clarify_gate needs the agent's last message in "
        "`conversation` (a turn with role='agent') to evaluate whether the "
        "user's reply accepted or rejected it"
    )


async def _call_gemini_text(prompt: str) -> str:
    """Call Gemini with a plain-text prompt and return the response text,
    stripped. Raises `ClarifyGateLLMError` if the call fails, times out, or
    returns no text at all.
    """
    client = _get_gemini_client()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=CLARIFY_GATE_GEMINI_MODEL,
                contents=prompt,
            ),
            timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — see module docstring: any
        # Gemini-side failure (connection, API error, or timeout) is a
        # ClarifyGateLLMError, never left as a bare SDK/asyncio exception
        # for the caller to guess at.
        raise ClarifyGateLLMError(f"Gemini call failed: {exc}") from exc

    text = response.text
    if not text or not text.strip():
        raise ClarifyGateLLMError("Gemini returned an empty response")
    return text.strip()


async def _call_gemini_json(prompt: str, required_keys: set[str]) -> dict[str, Any]:
    """Call Gemini requesting a JSON object response and return it parsed.

    Raises `ClarifyGateLLMError` if the call fails/times out, the response
    is not valid JSON, the parsed value is not a JSON object, or any of
    `required_keys` is missing — never returns a partially-valid dict for
    the caller to guess at.
    """
    client = _get_gemini_client()
    try:
        response = await asyncio.wait_for(
            client.aio.models.generate_content(
                model=CLARIFY_GATE_GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            ),
            timeout=EXTERNAL_CALL_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — see _call_gemini_text
        raise ClarifyGateLLMError(f"Gemini call failed: {exc}") from exc

    raw_text = response.text
    if not raw_text or not raw_text.strip():
        raise ClarifyGateLLMError("Gemini returned an empty response")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ClarifyGateLLMError(
            f"Gemini response was not valid JSON: {raw_text!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ClarifyGateLLMError(
            f"Gemini JSON response was not an object: {raw_text!r}"
        )
    missing = required_keys - parsed.keys()
    if missing:
        raise ClarifyGateLLMError(
            f"Gemini JSON response is missing required keys {missing}: {raw_text!r}"
        )
    return parsed


async def _generate_narrowing_question(
    original_goal: str, conversation: ConversationHistory
) -> str:
    """Generate the next narrowing question (PRD §7.2's "one narrowing
    question at a time")."""
    prompt = PROMPT_REGISTRY["clarify_gate_narrowing_question_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
    )
    return await _call_gemini_text(prompt)


async def _evaluate_narrowing_answer(
    original_goal: str, conversation: ConversationHistory, answer: str
) -> tuple[bool, str | None]:
    """Decide whether `answer` resolved a concrete role. Returns
    `(resolved, role)`; `role` is None whenever `resolved` is False.
    """
    prompt = PROMPT_REGISTRY["clarify_gate_narrowing_answer_evaluation_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
        answer=answer,
    )
    parsed = await _call_gemini_json(prompt, required_keys={"resolved", "role"})
    resolved = bool(parsed["resolved"])
    role = parsed["role"] if resolved else None
    if resolved and not isinstance(role, str):
        raise ClarifyGateLLMError(
            f"Gemini reported resolved=True but 'role' was not a string: {parsed!r}"
        )
    return resolved, role


async def _propose_best_guess_role(
    original_goal: str, conversation: ConversationHistory
) -> tuple[str, str]:
    """Propose a single best-guess role once the narrowing bound is
    reached (PRD §7.2). Returns `(role, user_facing_message)`.
    """
    prompt = PROMPT_REGISTRY["clarify_gate_best_guess_proposal_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
    )
    parsed = await _call_gemini_json(prompt, required_keys={"role", "message"})
    role, message = parsed["role"], parsed["message"]
    if not isinstance(role, str) or not role.strip():
        raise ClarifyGateLLMError(f"Gemini proposal had no usable 'role': {parsed!r}")
    if not isinstance(message, str) or not message.strip():
        raise ClarifyGateLLMError(
            f"Gemini proposal had no usable 'message': {parsed!r}"
        )
    return role, message


async def _explain_role(role: str) -> str:
    """Explain what `role` actually involves (PRD §7.2's second rung,
    after a rejected best-guess proposal)."""
    prompt = PROMPT_REGISTRY["clarify_gate_role_explanation_v1"].format(role=role)
    return await _call_gemini_text(prompt)


async def _evaluate_acceptance(agent_message: str, user_reply: str) -> bool:
    """Interpret whether `user_reply` affirmatively accepts `agent_message`
    (a proposal or explanation) — shared by both the proposal-response and
    explanation-response stages, since it's the same underlying judgment
    (PRD §7.2).
    """
    prompt = PROMPT_REGISTRY["clarify_gate_acceptance_evaluation_v1"].format(
        agent_message=agent_message,
        user_reply=user_reply,
    )
    parsed = await _call_gemini_json(prompt, required_keys={"accepted"})
    return bool(parsed["accepted"])


async def begin_clarify_gate(stated_goal: str) -> ClarifyGateTurn:
    """Handle the very first clarify-gate turn for a freshly stated goal.

    Runs `security/input_gate.py`'s `classify_stated_goal` on the raw
    input FIRST, before any LLM call touches it (CLAUDE.md's LLM Call
    Discipline: input validation always runs on raw input before any
    content-processing step).

    - `REAL` -> resolves immediately, no LLM call needed (PRD §7.2:
      "Clearly real role -> accept, proceed to Research").
    - `NONSENSE` -> loops back with a fixed re-prompt, not an exit (PRD
      §7.2: "reject, ask to clarify"); does not consume a narrowing round.
    - `VAGUE` -> enters the bounded narrowing loop with the first
      narrowing question.
    """
    classification = classify_stated_goal(stated_goal)
    context = ClarifyGateContext(original_stated_goal=stated_goal)

    if classification is GoalClassification.REAL:
        resolved_role = stated_goal.strip()
        return ClarifyGateTurn(
            gate_state=ClarifyGateState(
                stage=ClarifyGateStage.RESOLVED, narrowing_rounds_used=0
            ),
            context=context,
            message=f"Great — I'll build your plan around {resolved_role}.",
            resolved_role=resolved_role,
        )

    if classification is GoalClassification.NONSENSE:
        return ClarifyGateTurn(
            gate_state=start_clarify_gate(),
            context=context,
            message=CLARIFY_GATE_NONSENSE_REPROMPT,
        )

    # VAGUE
    question = await _generate_narrowing_question(stated_goal, [])
    return ClarifyGateTurn(
        gate_state=start_clarify_gate(),
        context=context,
        message=question,
    )


async def advance_clarify_gate(
    gate_state: ClarifyGateState,
    context: ClarifyGateContext,
    conversation: ConversationHistory,
    user_response: str,
    session: Session,
    reference_time: datetime,
) -> ClarifyGateTurn:
    """Advance the clarify gate by one turn given the user's latest
    response, dispatching on `gate_state.stage`.

    Raises `ValueError` if `gate_state.stage` is `RESOLVED`/`EXITED` — the
    caller must not advance a gate that has already terminated.
    """
    if gate_state.stage is ClarifyGateStage.NARROWING:
        resolved, role_guess = await _evaluate_narrowing_answer(
            context.original_stated_goal, conversation, user_response
        )
        next_state = advance_after_narrowing_round(gate_state, resolved=resolved)

        if next_state.stage is ClarifyGateStage.RESOLVED:
            assert role_guess is not None  # guaranteed by resolved=True above
            return ClarifyGateTurn(
                gate_state=next_state,
                context=context,
                message=f"Got it — {role_guess}.",
                resolved_role=role_guess,
            )

        if next_state.stage is ClarifyGateStage.NARROWING:
            question = await _generate_narrowing_question(
                context.original_stated_goal, conversation
            )
            return ClarifyGateTurn(
                gate_state=next_state, context=context, message=question
            )

        # PROPOSE_BEST_GUESS: the narrowing bound was just reached.
        proposed_role, proposal_message = await _propose_best_guess_role(
            context.original_stated_goal, conversation
        )
        return ClarifyGateTurn(
            gate_state=next_state,
            context=ClarifyGateContext(
                original_stated_goal=context.original_stated_goal,
                proposed_role=proposed_role,
            ),
            message=proposal_message,
        )

    if gate_state.stage is ClarifyGateStage.PROPOSE_BEST_GUESS:
        assert context.proposed_role is not None
        accepted = await _evaluate_acceptance(
            _last_agent_message(conversation), user_response
        )
        next_state = advance_after_proposal_response(gate_state, accepted=accepted)
        if next_state.stage is ClarifyGateStage.RESOLVED:
            return ClarifyGateTurn(
                gate_state=next_state,
                context=context,
                message=f"Great — let's go with {context.proposed_role}.",
                resolved_role=context.proposed_role,
            )
        explanation = await _explain_role(context.proposed_role)
        return ClarifyGateTurn(
            gate_state=next_state, context=context, message=explanation
        )

    if gate_state.stage is ClarifyGateStage.EXPLAIN_ROLE:
        assert context.proposed_role is not None
        accepted = await _evaluate_acceptance(
            _last_agent_message(conversation), user_response
        )
        next_state = advance_after_explanation_response(gate_state, accepted=accepted)
        if next_state.stage is ClarifyGateStage.RESOLVED:
            return ClarifyGateTurn(
                gate_state=next_state,
                context=context,
                message=f"Great — let's go with {context.proposed_role}.",
                resolved_role=context.proposed_role,
            )

        # ACCEPT_OWN_WORDS: ground the user's ORIGINAL words (never the
        # most recent message — CLAUDE.md's retry-loop-overwrite trap)
        # before committing, per PRD §7.2.
        grounding_result = await ground_role(
            context.original_stated_goal, session, reference_time
        )
        market_signal_found = not isinstance(
            grounding_result, GeneralKnowledgeFloorResult
        )
        final_state = resolve_after_grounding_check(
            next_state, market_signal_found=market_signal_found
        )
        if final_state.stage is ClarifyGateStage.RESOLVED:
            return ClarifyGateTurn(
                gate_state=final_state,
                context=context,
                message=(
                    f"Starting here with {context.original_stated_goal!r} — "
                    "we'll refine as we go."
                ),
                resolved_role=context.original_stated_goal,
            )
        return ClarifyGateTurn(
            gate_state=final_state,
            context=context,
            message=CLARIFY_GATE_ZERO_SIGNAL_EXIT_MESSAGE_TEMPLATE.format(
                role=context.original_stated_goal
            ),
            exited=True,
        )

    raise ValueError(
        f"advance_clarify_gate called with a terminal or unexpected stage: "
        f"{gate_state.stage}"
    )


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


def create_initial_outline(
    resolved_role: str, grounded_skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sequence grounded skill data into a dependency hierarchy (basics ->
    full role requirements), per PRD §7.4.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
