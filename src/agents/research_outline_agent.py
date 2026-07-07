"""Research & Outline Agent: owns clarify-gate conversational content, role grounding/cross-validation orchestration, and initial outline hierarchy generation, calling deterministic modules for everything else."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from google.adk.agents import LlmAgent
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
from security.input_gate import (
    ClarifyGateStage,
    ClarifyGateState,
    GoalClassification,
    OutlineConfirmationStage,
    OutlineConfirmationState,
    OutlineReviewAction,
    advance_after_explanation_response,
    advance_after_narrowing_round,
    advance_after_proposal_response,
    advance_after_review_turn,
    classify_stated_goal,
    resolve_after_grounding_check,
    start_clarify_gate,
    start_outline_confirmation,
)
from security.output_guard import (
    ConfidenceTier,
    ValidatedGroundedContent,
    validate_output_object,
)
from utils.adk_runtime import (
    EXTERNAL_CALL_TIMEOUT_SECONDS,
    HEAVY_GENERATION_TIMEOUT_SECONDS,
    SHORT_TURN_GEMINI_MODEL,
    build_retry_config,
    call_agent_json,
    call_agent_text,
    json_response_config,
)
from utils.exceptions import (
    GeminiCallError,
    GroundingSourceCallError,
    HimalayasParseError,
    TavilyParseError,
)

HIMALAYAS_MCP_URL = "https://mcp.himalayas.app/mcp"

HIMALAYAS_SOURCE_TYPE = "job_listing"

TAVILY_SOURCE_TYPE = "web_search"

_SourceStatus = Literal["signal", "no_signal", "call_failed"]

_himalayas_toolset: McpToolset | None = None
_tavily_client: TavilyClient | None = None


def _get_himalayas_toolset() -> McpToolset:
    """Return the module-level Himalayas MCP toolset, creating it on first use."""
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
    """Return the module-level Tavily client, creating it on first use, raising if the API key is missing."""
    global _tavily_client
    if _tavily_client is None:
        import os

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY environment variable is not set")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


OUTLINE_HIERARCHY_GEMINI_MODEL = "gemini-2.5-flash"


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
    "outline_hierarchy_sequencing_v1": (
        "You are designing the initial learning curriculum for someone "
        "pursuing the role of {role!r}. Below is a grounded list of "
        "skills this role actually requires (each labeled 'core' — "
        "consistently required — or 'emerging' — a real but newer/growing "
        "requirement). Sequence them into a dependency-ordered curriculum "
        "using real-world domain knowledge of how these subjects build on "
        "each other.\n\n"
        "Skills:\n{skill_list}\n\n"
        "Requirements:\n"
        "1. Organize the skills into named topic groups (e.g. 'Python', "
        "'SQL', 'Git') and order the GROUPS themselves in prerequisite "
        "order — foundational/basics groups before groups that build on "
        "them or are more advanced/role-specific.\n"
        "2. Within EACH group, order its topics in prerequisite order too "
        "(e.g. within a 'Python' group: syntax and variables, then "
        "functions, then object-oriented programming, before a framework "
        "built on Python appears as its own later group).\n"
        "3. Break a broad skill into multiple smaller topics where that "
        "reflects how it's actually learned (e.g. a broad skill like "
        "'Python' should become several topics, not one) — do not "
        "collapse a whole skill into a single topic just because it was "
        "one line in the list above.\n"
        "4. Every topic must set 'source_skill' to the EXACT text of "
        "exactly one skill from the list above. Every skill in the list "
        "above must be covered by at least one topic. Never invent a "
        "skill that isn't in the list above, and never omit one.\n\n"
        "Respond with ONLY a JSON object matching this shape:\n"
        '{{"groups": [{{"topic_group": "<group name>", "topics": '
        '[{{"topic_name": "<specific topic name>", "source_skill": '
        '"<exact skill text from the list above>"}}, ...]}}, ...]}}\n'
        "Groups must appear in prerequisite order; topics within each "
        "group must appear in prerequisite order."
    ),
    "outline_confirmation_topic_explanations_v1": (
        "You are presenting a learning outline to a user for role "
        "{role!r}, before they begin. For each topic below, write a "
        "short (1-2 sentence), friendly explanation of why it's included "
        "in their plan — grounded strictly in the source/confidence "
        "information given for that topic; do not state facts about the "
        "topic that aren't supported by that information.\n\n"
        "Topics:\n{topic_list}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"topic_explanations": [{{"topic_name": "<exact topic name '
        'from the list above>", "explanation": "<short why-explanation>"'
        "}}, ...]}}. Every topic listed above must appear exactly once; "
        "never invent a topic not in the list above."
    ),
    "outline_review_turn_classification_v1": (
        "A user is reviewing their learning outline for role {role!r} "
        "before starting it. Classify their latest message into exactly "
        "one of: 'question' (asking about the outline without requesting "
        "any change), 'concern' (expressing a worry or objection about "
        "something in the outline), 'addition_request' (asking for a new "
        "topic or skill to be added), or 'confirm' (explicitly satisfied "
        "and ready to proceed, e.g. 'looks good', 'let's start').\n\n"
        "Current outline topics: {topic_names}\n"
        "User's message: {user_message!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"action": "question" | "concern" | "addition_request" | '
        '"confirm"}}.'
    ),
    "outline_review_response_v1": (
        "A user asked a question or raised a concern about their "
        "learning outline for role {role!r}. Respond directly and "
        "helpfully, grounded strictly in the actual outline topics given "
        "— do not invent topics or facts not present below.\n\n"
        "Current outline topics: {topic_list}\n"
        "User's message: {user_message!r}\n\n"
        "Respond with only the response text, nothing else."
    ),
    "outline_addition_skill_name_extraction_v1": (
        "A user asked for a new topic or skill to be added to their "
        "learning outline. Extract just the specific skill or "
        "technology name they are asking to add, as a short, clean "
        "phrase suitable for a search engine query (e.g. 'Kubernetes', "
        "'Rust', 'GraphQL APIs') — not their full sentence.\n\n"
        "User's message: {user_message!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"skill_name": "<short skill/technology name>"}}.'
    ),
}


_narrowing_question_agent = LlmAgent(
    name="clarify_gate_narrowing_question_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Generate a single short, friendly narrowing question to help pin "
        "down a user's vague tech career goal into one specific, concrete "
        "role."
    ),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_narrowing_answer_evaluation_agent = LlmAgent(
    name="clarify_gate_narrowing_answer_evaluation_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Decide whether a user's latest answer during clarify-gate "
        "narrowing resolves their goal into one concrete, specific tech "
        "role."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_best_guess_proposal_agent = LlmAgent(
    name="clarify_gate_best_guess_proposal_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Propose a single best-guess specific tech role interpretation of "
        "a user's stated career goal, with a short friendly confirmation "
        "message."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_role_explanation_agent = LlmAgent(
    name="clarify_gate_role_explanation_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Explain what a proposed tech role actually involves day to day, "
        "then ask the user to confirm it fits."
    ),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_acceptance_evaluation_agent = LlmAgent(
    name="clarify_gate_acceptance_evaluation_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Decide whether a user's reply affirmatively accepts a proposed or "
        "explained tech role."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_outline_hierarchy_agent = LlmAgent(
    name="outline_hierarchy_sequencing_agent",
    model=OUTLINE_HIERARCHY_GEMINI_MODEL,
    instruction=(
        "Sequence a grounded list of skills for a tech role into a "
        "dependency-ordered, grouped learning curriculum."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_topic_explanations_agent = LlmAgent(
    name="outline_topic_explanations_agent",
    model=OUTLINE_HIERARCHY_GEMINI_MODEL,
    instruction=(
        "Write a short why-explanation for each topic in a learning "
        "outline, grounded strictly in its given source/confidence."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_review_turn_classification_agent = LlmAgent(
    name="outline_review_turn_classification_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Classify a user's outline-review message into exactly one of: "
        "question, concern, addition_request, confirm."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_review_response_agent = LlmAgent(
    name="outline_review_response_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Respond to a user's question or concern about their learning "
        "outline, grounded strictly in the given topics."
    ),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_addition_skill_name_extraction_agent = LlmAgent(
    name="outline_addition_skill_name_extraction_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Extract a short, search-ready skill/technology name from a "
        "user's addition request."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

RESEARCH_OUTLINE_AGENT = LlmAgent(
    name="research_outline_agent",
    model=SHORT_TURN_GEMINI_MODEL,
    instruction=(
        "Composite agent grouping this module's task agents for "
        "documentation/introspection. Never run via its own Runner — "
        "dispatch between sub-agents is deterministic Python, not "
        "auto-routing."
    ),
    sub_agents=[
        _narrowing_question_agent,
        _narrowing_answer_evaluation_agent,
        _best_guess_proposal_agent,
        _role_explanation_agent,
        _acceptance_evaluation_agent,
        _outline_hierarchy_agent,
        _topic_explanations_agent,
        _review_turn_classification_agent,
        _review_response_agent,
        _addition_skill_name_extraction_agent,
    ],
)

CLARIFY_GATE_NONSENSE_REPROMPT = (
    "I didn't quite catch a tech role or skill in that — could you tell "
    "me a bit about what kind of tech work you're interested in, or a "
    "role you might want to pursue?"
)

CLARIFY_GATE_ZERO_SIGNAL_EXIT_MESSAGE_TEMPLATE = (
    "I couldn't find any current hiring activity for {role!r} — no "
    "learning plan will be built for this. Feel free to describe a "
    "different role or area you're interested in instead."
)


@dataclass(frozen=True)
class ClarifyGateContext:
    """Agent-owned conversational context threaded alongside the clarify-gate loop state, holding the original stated goal and any proposed role."""

    original_stated_goal: str
    proposed_role: str | None = None


@dataclass(frozen=True)
class ClarifyGateTurn:
    """One turn of clarify-gate output: the message to show, updated state/context to persist, and the resolved role or exit flag once the gate concludes."""

    gate_state: ClarifyGateState
    context: ClarifyGateContext
    message: str
    resolved_role: str | None = None
    exited: bool = False


ConversationHistory = list[dict[str, str]]


def _format_conversation(conversation: ConversationHistory) -> str:
    """Render a conversation history as plain text for embedding in a prompt."""
    if not conversation:
        return "(no prior turns)"
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


def _last_agent_message(conversation: ConversationHistory) -> str:
    """Return the most recent agent message in the conversation, raising if none exists."""
    for turn in reversed(conversation):
        if turn.get("role") == "agent":
            return turn["content"]
    raise ValueError(
        "advance_clarify_gate needs the agent's last message in "
        "`conversation` (a turn with role='agent') to evaluate whether the "
        "user's reply accepted or rejected it"
    )


async def _generate_narrowing_question(
    original_goal: str, conversation: ConversationHistory
) -> str:
    """Generate the next narrowing question for the clarify gate."""
    prompt = PROMPT_REGISTRY["clarify_gate_narrowing_question_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
    )
    return await call_agent_text(_narrowing_question_agent, prompt)


async def _evaluate_narrowing_answer(
    original_goal: str, conversation: ConversationHistory, answer: str
) -> tuple[bool, str | None]:
    """Decide whether the answer resolved a concrete role, returning the role if so."""
    prompt = PROMPT_REGISTRY["clarify_gate_narrowing_answer_evaluation_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
        answer=answer,
    )
    parsed = await call_agent_json(
        _narrowing_answer_evaluation_agent, prompt, required_keys={"resolved", "role"}
    )
    resolved = bool(parsed["resolved"])
    role = parsed["role"] if resolved else None
    if resolved and not isinstance(role, str):
        raise GeminiCallError(
            f"Gemini reported resolved=True but 'role' was not a string: {parsed!r}"
        )
    return resolved, role


async def _propose_best_guess_role(
    original_goal: str, conversation: ConversationHistory
) -> tuple[str, str]:
    """Propose a single best-guess role once the narrowing bound is reached, returning the role and a confirmation message."""
    prompt = PROMPT_REGISTRY["clarify_gate_best_guess_proposal_v1"].format(
        original_goal=original_goal,
        conversation=_format_conversation(conversation),
    )
    parsed = await call_agent_json(
        _best_guess_proposal_agent, prompt, required_keys={"role", "message"}
    )
    role, message = parsed["role"], parsed["message"]
    if not isinstance(role, str) or not role.strip():
        raise GeminiCallError(f"Gemini proposal had no usable 'role': {parsed!r}")
    if not isinstance(message, str) or not message.strip():
        raise GeminiCallError(f"Gemini proposal had no usable 'message': {parsed!r}")
    return role, message


async def _explain_role(role: str) -> str:
    """Explain what a proposed role actually involves day to day."""
    prompt = PROMPT_REGISTRY["clarify_gate_role_explanation_v1"].format(role=role)
    return await call_agent_text(_role_explanation_agent, prompt)


async def _evaluate_acceptance(agent_message: str, user_reply: str) -> bool:
    """Interpret whether the user's reply affirmatively accepts a proposed or explained role."""
    prompt = PROMPT_REGISTRY["clarify_gate_acceptance_evaluation_v1"].format(
        agent_message=agent_message,
        user_reply=user_reply,
    )
    parsed = await call_agent_json(
        _acceptance_evaluation_agent, prompt, required_keys={"accepted"}
    )
    return bool(parsed["accepted"])


async def begin_clarify_gate(stated_goal: str) -> ClarifyGateTurn:
    """Handle the very first clarify-gate turn for a freshly stated goal, classifying it as real, nonsense, or vague before any LLM call."""
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
    """Advance the clarify gate by one turn given the user's latest response, dispatching on the current stage."""
    if gate_state.stage is ClarifyGateStage.NARROWING:
        resolved, role_guess = await _evaluate_narrowing_answer(
            context.original_stated_goal, conversation, user_response
        )
        next_state = advance_after_narrowing_round(gate_state, resolved=resolved)

        if next_state.stage is ClarifyGateStage.RESOLVED:
            assert role_guess is not None
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
    """A successful live-grounding outcome carrying validated skills, the decided confidence tier, and each source's status."""

    role_name: str
    skills: list[ValidatedGroundedContent]
    confidence: ConfidenceTier
    has_conflict: bool
    himalayas_status: _SourceStatus
    tavily_status: _SourceStatus


async def _fetch_himalayas_listings(role_name: str) -> list[ParsedJobListing]:
    """Call Himalayas MCP's search_jobs tool for a role and parse the response, raising if the call or parse fails."""
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


async def _fetch_tavily_results(query: str) -> list[ParsedSearchResult]:
    """Call Tavily search for a query in a worker thread with an explicit timeout, then parse the response."""
    client = _get_tavily_client()
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
    """Fetch and relevance-check Himalayas, collapsing any call failure into a no-signal status instead of raising."""
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
    """Fetch and trust-check Tavily, collapsing any call failure into a no-signal status instead of raising."""
    try:
        results = await _fetch_tavily_results(
            f"{role_name} job requirements and key skills"
        )
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
    """Ground a role against Himalayas and Tavily in parallel, decide a confidence tier, and fall back to cached or general-knowledge data if live grounding finds no usable signal."""
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


NOT_STARTED_STATUS = "not_started"


@dataclass(frozen=True)
class InitialOutlineTopic:
    """One outline_topics row produced by initial hierarchy creation, not yet persisted."""

    topic_name: str
    hierarchy_position: int
    topic_group: str
    position_in_group: int
    source_url: str
    source_type: str
    confidence: ConfidenceTier
    is_enrichment: bool
    status: str


def _build_grounded_skill_map(
    core_skills: list[ValidatedGroundedContent],
    emerging_skills: list[ValidatedGroundedContent],
) -> dict[str, ValidatedGroundedContent]:
    """Build a skill-name to validated-grounding lookup from grounded core and emerging skills, raising on missing names or duplicates."""
    skill_map: dict[str, ValidatedGroundedContent] = {}
    for grounded in (*core_skills, *emerging_skills):
        name = grounded.extra.get("skill")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"grounded skill entry has no usable 'skill' name: {grounded!r}"
            )
        if name in skill_map:
            raise ValueError(
                f"skill {name!r} appears more than once across core_skills/"
                "emerging_skills — each grounded skill must be listed exactly once"
            )
        skill_map[name] = grounded
    return skill_map


def _format_skill_list_for_prompt(
    core_skills: list[ValidatedGroundedContent],
    emerging_skills: list[ValidatedGroundedContent],
) -> str:
    lines = [f"- {skill.extra['skill']} (core)" for skill in core_skills]
    lines += [f"- {skill.extra['skill']} (emerging)" for skill in emerging_skills]
    return "\n".join(lines)


async def create_initial_outline(
    resolved_role: str,
    core_skills: list[ValidatedGroundedContent],
    emerging_skills: list[ValidatedGroundedContent],
) -> list[InitialOutlineTopic]:
    """Sequence already-grounded skill data into a dependency-ordered outline hierarchy, re-attaching each topic's sourcing from its matched input skill."""
    skill_map = _build_grounded_skill_map(core_skills, emerging_skills)
    if not skill_map:
        raise ValueError("create_initial_outline requires at least one grounded skill")

    prompt = PROMPT_REGISTRY["outline_hierarchy_sequencing_v1"].format(
        role=resolved_role,
        skill_list=_format_skill_list_for_prompt(core_skills, emerging_skills),
    )
    parsed = await call_agent_json(
        _outline_hierarchy_agent,
        prompt,
        required_keys={"groups"},
        timeout=HEAVY_GENERATION_TIMEOUT_SECONDS,
    )

    groups = parsed["groups"]
    if not isinstance(groups, list) or not groups:
        raise GeminiCallError(f"'groups' must be a non-empty list: {parsed!r}")

    topics: list[InitialOutlineTopic] = []
    referenced_skills: set[str] = set()
    hierarchy_position = 0

    for group in groups:
        if (
            not isinstance(group, dict)
            or "topic_group" not in group
            or "topics" not in group
        ):
            raise GeminiCallError(f"malformed group entry: {group!r}")
        topic_group = group["topic_group"]
        group_topics = group["topics"]
        if not isinstance(topic_group, str) or not topic_group.strip():
            raise GeminiCallError(f"group has no usable 'topic_group': {group!r}")
        if not isinstance(group_topics, list) or not group_topics:
            raise GeminiCallError(f"group {topic_group!r} has no topics: {group!r}")

        for position_in_group, topic in enumerate(group_topics, start=1):
            if (
                not isinstance(topic, dict)
                or "topic_name" not in topic
                or "source_skill" not in topic
            ):
                raise GeminiCallError(f"malformed topic entry: {topic!r}")
            topic_name = topic["topic_name"]
            source_skill = topic["source_skill"]
            if not isinstance(topic_name, str) or not topic_name.strip():
                raise GeminiCallError(f"topic has no usable 'topic_name': {topic!r}")
            if not isinstance(source_skill, str) or source_skill not in skill_map:
                raise GeminiCallError(
                    f"topic {topic_name!r} references 'source_skill' "
                    f"{source_skill!r}, which is not in the grounded skill list"
                )

            grounded = skill_map[source_skill]
            referenced_skills.add(source_skill)
            hierarchy_position += 1

            candidate = {
                "source_url": grounded.source_url,
                "source_type": grounded.source_type,
                "confidence": grounded.confidence.value,
                "topic_name": topic_name,
                "hierarchy_position": hierarchy_position,
                "topic_group": topic_group,
                "position_in_group": position_in_group,
                "is_enrichment": False,
                "status": NOT_STARTED_STATUS,
            }
            validated = validate_output_object(candidate)
            topics.append(
                InitialOutlineTopic(
                    topic_name=validated.extra["topic_name"],
                    hierarchy_position=validated.extra["hierarchy_position"],
                    topic_group=validated.extra["topic_group"],
                    position_in_group=validated.extra["position_in_group"],
                    source_url=validated.source_url,
                    source_type=validated.source_type,
                    confidence=validated.confidence,
                    is_enrichment=validated.extra["is_enrichment"],
                    status=validated.extra["status"],
                )
            )

    missing_skills = skill_map.keys() - referenced_skills
    if missing_skills:
        raise GeminiCallError(
            "the following grounded skills were never covered by any "
            f"topic: {sorted(missing_skills)}"
        )

    return topics


OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE = (
    "We've covered as much as we can before starting — starting here, "
    "we'll refine as we go."
)

OUTLINE_CONFIRMATION_CONFIRMED_MESSAGE = "Great — let's get started!"

OUTLINE_CONFIRMATION_ADDITION_ACK_MESSAGE_TEMPLATE = (
    "Got it — I'll add {addition!r} and update your outline."
)


@dataclass(frozen=True)
class OutlineConfirmationTurn:
    """One turn of outline-confirmation output: the message, updated state and topics to persist, and whether review has concluded."""

    state: OutlineConfirmationState
    message: str
    topics: list[InitialOutlineTopic]
    concluded: bool = False
    action: OutlineReviewAction | None = None


def _format_topic_list_for_prompt(topics: list[InitialOutlineTopic]) -> str:
    return "\n".join(
        f"- {t.topic_name} (group: {t.topic_group}, source: {t.source_url}, "
        f"confidence: {t.confidence.value})"
        for t in topics
    )


async def _generate_topic_explanations(
    resolved_role: str, topics: list[InitialOutlineTopic]
) -> dict[str, str]:
    """Generate a short why-explanation per topic, grounded in each topic's real source and confidence metadata."""
    prompt = PROMPT_REGISTRY["outline_confirmation_topic_explanations_v1"].format(
        role=resolved_role,
        topic_list=_format_topic_list_for_prompt(topics),
    )
    parsed = await call_agent_json(
        _topic_explanations_agent,
        prompt,
        required_keys={"topic_explanations"},
        timeout=HEAVY_GENERATION_TIMEOUT_SECONDS,
    )
    entries = parsed["topic_explanations"]
    if not isinstance(entries, list) or not entries:
        raise GeminiCallError(
            f"'topic_explanations' must be a non-empty list: {parsed!r}"
        )

    valid_topic_names = {t.topic_name for t in topics}
    explanations: dict[str, str] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or "topic_name" not in entry
            or "explanation" not in entry
        ):
            raise GeminiCallError(f"malformed topic_explanations entry: {entry!r}")
        topic_name, explanation = entry["topic_name"], entry["explanation"]
        if topic_name not in valid_topic_names:
            raise GeminiCallError(
                f"topic_explanations entry references unknown topic "
                f"{topic_name!r}, which is not in the current outline"
            )
        if not isinstance(explanation, str) or not explanation.strip():
            raise GeminiCallError(
                f"topic {topic_name!r} has no usable 'explanation': {entry!r}"
            )
        explanations[topic_name] = explanation

    missing = valid_topic_names - explanations.keys()
    if missing:
        raise GeminiCallError(
            f"the following topics were never explained: {sorted(missing)}"
        )
    return explanations


def _format_outline_presentation(
    resolved_role: str,
    topics: list[InitialOutlineTopic],
    explanations: dict[str, str],
) -> str:
    """Deterministically assemble the user-facing outline presentation from real topic data and Gemini-generated explanations."""
    lines = [f"Here's your learning plan for {resolved_role}:", ""]
    current_group: str | None = None
    for topic in topics:
        if topic.topic_group != current_group:
            current_group = topic.topic_group
            lines.append(f"## {current_group}")
        lines.append(
            f"- {topic.topic_name}: {explanations[topic.topic_name]} "
            f"(source: {topic.source_url}, confidence: {topic.confidence.value})"
        )
    return "\n".join(lines)


async def _classify_review_turn(
    resolved_role: str, topics: list[InitialOutlineTopic], user_message: str
) -> OutlineReviewAction:
    """Classify a user's outline-review message as a question, concern, addition request, or confirmation."""
    prompt = PROMPT_REGISTRY["outline_review_turn_classification_v1"].format(
        role=resolved_role,
        topic_names=", ".join(t.topic_name for t in topics),
        user_message=user_message,
    )
    parsed = await call_agent_json(
        _review_turn_classification_agent, prompt, required_keys={"action"}
    )
    raw_action = parsed["action"]
    try:
        return OutlineReviewAction(raw_action)
    except ValueError as exc:
        raise GeminiCallError(
            f"Gemini returned an unrecognized review action: {raw_action!r}"
        ) from exc


async def _respond_to_review_message(
    resolved_role: str, topics: list[InitialOutlineTopic], user_message: str
) -> str:
    """Respond to a question or concern raised about the outline."""
    prompt = PROMPT_REGISTRY["outline_review_response_v1"].format(
        role=resolved_role,
        topic_list=_format_topic_list_for_prompt(topics),
        user_message=user_message,
    )
    return await call_agent_text(_review_response_agent, prompt)


async def _extract_addition_skill_name(user_message: str) -> str:
    """Extract a short, search-ready skill or topic name from a raw addition-request message."""
    prompt = PROMPT_REGISTRY["outline_addition_skill_name_extraction_v1"].format(
        user_message=user_message
    )
    parsed = await call_agent_json(
        _addition_skill_name_extraction_agent, prompt, required_keys={"skill_name"}
    )
    skill_name = parsed["skill_name"]
    if not isinstance(skill_name, str) or not skill_name.strip():
        raise GeminiCallError(f"Gemini returned no usable skill_name: {parsed!r}")
    return skill_name.strip()


SINGLE_SKILL_GROUNDING_CONFIDENCE = ConfidenceTier.MEDIUM


async def ground_addition_request(user_message: str) -> ValidatedGroundedContent | None:
    """Ground a free-text outline addition request into a sourced grounded-content object via a Tavily lookup, or return None if nothing usable is found."""
    skill_name = await _extract_addition_skill_name(user_message)
    query = f"{skill_name} tutorial or official documentation"
    results = await _fetch_tavily_results(query)
    skill_bearing = [result for result in results if result.source_url]
    if not skill_bearing:
        return None
    top_result = max(skill_bearing, key=lambda result: result.score)
    return validate_output_object(
        {
            "source_url": top_result.source_url,
            "source_type": TAVILY_SOURCE_TYPE,
            "confidence": SINGLE_SKILL_GROUNDING_CONFIDENCE.value,
            "skill": skill_name,
        }
    )


async def begin_outline_confirmation(
    resolved_role: str, topics: list[InitialOutlineTopic]
) -> OutlineConfirmationTurn:
    """Show the outline for the first time, with grounded why-explanations per topic."""
    explanations = await _generate_topic_explanations(resolved_role, topics)
    message = _format_outline_presentation(resolved_role, topics, explanations)
    return OutlineConfirmationTurn(
        state=start_outline_confirmation(), message=message, topics=topics
    )


async def handle_review_turn(
    state: OutlineConfirmationState,
    resolved_role: str,
    topics: list[InitialOutlineTopic],
    user_message: str,
) -> OutlineConfirmationTurn:
    """Handle one user turn during outline confirmation by classifying the message and dispatching to the matching response."""
    if state.stage is not OutlineConfirmationStage.REVIEWING:
        raise ValueError(
            f"handle_review_turn called outside REVIEWING stage: {state.stage}"
        )

    action = await _classify_review_turn(resolved_role, topics, user_message)

    if action is OutlineReviewAction.CONFIRM:
        next_state = advance_after_review_turn(state, action)
        return OutlineConfirmationTurn(
            state=next_state,
            message=OUTLINE_CONFIRMATION_CONFIRMED_MESSAGE,
            topics=topics,
            concluded=True,
            action=action,
        )

    if action is OutlineReviewAction.QUESTION:
        response = await _respond_to_review_message(resolved_role, topics, user_message)
        next_state = advance_after_review_turn(state, action)
        return OutlineConfirmationTurn(
            state=next_state,
            message=response,
            topics=topics,
            concluded=False,
            action=action,
        )

    if action is OutlineReviewAction.CONCERN:
        response = await _respond_to_review_message(resolved_role, topics, user_message)
        next_state = advance_after_review_turn(state, action)
        concluded = next_state.stage is OutlineConfirmationStage.BOUND_REACHED
        if concluded:
            response = f"{response} {OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE}"
        return OutlineConfirmationTurn(
            state=next_state,
            message=response,
            topics=topics,
            concluded=concluded,
            action=action,
        )

    next_state = advance_after_review_turn(state, action)
    concluded = next_state.stage is OutlineConfirmationStage.BOUND_REACHED
    message = OUTLINE_CONFIRMATION_ADDITION_ACK_MESSAGE_TEMPLATE.format(
        addition=user_message
    )
    if concluded:
        message = f"{message} {OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE}"
    return OutlineConfirmationTurn(
        state=next_state,
        message=message,
        topics=topics,
        concluded=concluded,
        action=action,
    )


async def regenerate_outline_with_addition(
    state: OutlineConfirmationState,
    resolved_role: str,
    core_skills: list[ValidatedGroundedContent],
    emerging_skills: list[ValidatedGroundedContent],
    new_addition: ValidatedGroundedContent,
) -> OutlineConfirmationTurn:
    """Regenerate the full outline via create_initial_outline with the new addition folded into the emerging skills."""
    updated_emerging_skills = [*emerging_skills, new_addition]
    new_topics = await create_initial_outline(
        resolved_role, core_skills, updated_emerging_skills
    )

    addition_name = new_addition.extra.get("skill", "the requested topic")
    message = f"Updated your outline to include {addition_name}."
    if state.stage is OutlineConfirmationStage.BOUND_REACHED:
        message = f"{message} {OUTLINE_CONFIRMATION_BOUND_REACHED_MESSAGE}"

    return OutlineConfirmationTurn(
        state=state,
        message=message,
        topics=new_topics,
        concluded=state.stage is OutlineConfirmationStage.BOUND_REACHED,
    )
