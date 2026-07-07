"""Coaching & Pace Agent: generates day-by-day lesson content and closing notes, and orchestrates verification retries, pace-signal computation, enrichment/pacing-extension triggers, and patch-note delivery."""

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from google.adk.agents import LlmAgent
from sqlalchemy.orm import Session
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from agents.research_outline_agent import _get_tavily_client
from data.grounding_fallback import CACHED_SOURCE_TYPE
from data.outline_topics import (
    COMPLETED_STATUS,
    COMPLETED_TEST_OUT_STATUS,
    augment_outline_topic,
    get_all_topics_for_user,
    get_topic,
    has_pending_enrichment_topic,
    insert_new_outline_topic,
    mark_topic_completed,
)
from data.pace_snapshots import get_pace_snapshot_history, write_pace_snapshot
from data.patch_notes import (
    get_deferred_patch_notes,
    get_pending_patch_notes,
    update_patch_note_status,
)
from data.progress_log import log_progress_step
from data.roles_cache import get_role
from data.users import extend_pacing, get_user
from data.verification_log import get_attempts_for_topic, write_verification_attempt
from pace.calculator import (
    calculate_combined_pace_signal,
    calculate_timing_ratio,
    calculate_topic_score,
    detect_sustained_drift,
)
from patches.patch_manager import (
    PatchDecisionState,
    PatchStatus,
    decide_patch_delivery,
    resolve_patch_decision,
)
from security.output_guard import ConfidenceTier, validate_output_object
from skills.verification_question_generator import (
    VerificationQuestion,
    generate_questions,
    grade_answer,
)
from utils.adk_runtime import (
    EXTERNAL_CALL_TIMEOUT_SECONDS,
    HEAVY_GENERATION_TIMEOUT_SECONDS,
    build_retry_config,
    call_agent_json,
    json_response_config,
)
from utils.exceptions import GeminiCallError, GroundingSourceCallError

MAX_VERIFICATION_ATTEMPTS = 3
FULL_CREDIT = 1.0
HALF_CREDIT = 0.5
NOT_YET_RESOLVED_CREDIT = 0.0

DAY_CONTENT_GEMINI_MODEL = "gemini-2.5-flash"

CLOSING_NOTE_GEMINI_MODEL = "gemini-2.5-flash"

STUDY_DAYS_PER_WEEK = 5


def compute_hands_on_intensity(position_in_group: int, group_size: int) -> float:
    """Compute how much hands-on depth today's content should have, from 0.0 (none) to 1.0 (full depth), ramping linearly across the topic group."""
    if group_size < 1:
        raise ValueError(f"group_size must be at least 1, got {group_size}")
    if not (1 <= position_in_group <= group_size):
        raise ValueError(
            f"position_in_group ({position_in_group}) must be within "
            f"[1, group_size={group_size}]"
        )
    if group_size == 1:
        return 1.0
    return (position_in_group - 1) / (group_size - 1)


def is_conceptual_only_day(position_in_group: int, group_size: int) -> bool:
    """Whether today is a conceptual-only day, i.e. hands-on intensity is exactly 0.0."""
    return compute_hands_on_intensity(position_in_group, group_size) == 0.0


def convert_weekly_hours_to_daily_minutes(available_time_per_week_hours: int) -> int:
    """Convert a user's weekly available hours into today's study time budget in minutes."""
    if available_time_per_week_hours <= 0:
        raise ValueError(
            "available_time_per_week_hours must be positive, got "
            f"{available_time_per_week_hours}"
        )
    return round(available_time_per_week_hours * 60 / STUDY_DAYS_PER_WEEK)


ESTIMATED_MINUTES_PER_TOPIC = 120


def calculate_days_expected(available_time_per_week_hours: int) -> int:
    """Compute the baseline number of days a topic is expected to take given the user's daily time budget."""
    minutes_per_day = convert_weekly_hours_to_daily_minutes(
        available_time_per_week_hours
    )
    return max(1, math.ceil(ESTIMATED_MINUTES_PER_TOPIC / minutes_per_day))


PROMPT_REGISTRY: dict[str, str] = {
    "day_content_generation_hands_on_v1": (
        "Generate today's lesson content for a learner studying "
        "{topic_name!r} (part of the {topic_group!r} topic group). They "
        "have about {minutes_available} minutes available today, and "
        "today's hands-on exercise should be scaled to intensity "
        "{hands_on_intensity:.2f} on a 0 (very light) to 1 (full depth) "
        "scale.\n\n"
        "{carried_over_instruction}"
        "Theory material to build the lesson around — these are real, "
        "existing sources; cite them by number, never invent any other "
        "source:\n{theory_sources}\n\n"
        "Respond with ONLY a JSON object matching this shape:\n"
        '{{"summary": "<1-2 sentence summary of today\'s topic>", '
        '"theory_framing": "<prose introducing/framing the numbered '
        'theory sources above>", '
        '"hands_on_exercise": "<a hands-on exercise scaled to the given '
        'intensity>", '
        '"review_prompt": "<a prompt for reviewing/refactoring the '
        'hands-on work>", '
        '"reflection_prompt": "<a short reflection question about '
        "today's material>\", "
        '"preview": "<a short preview of tomorrow and how it connects>", '
        '"remaining_content": "<anything from today\'s intended material '
        "that did not fit in the time budget and should carry over to "
        'tomorrow, or an empty string if everything fit>"}}\n'
        "Size the depth/length of every field to genuinely fit within "
        "{minutes_available} minutes total for the whole lesson."
    ),
    "day_content_generation_conceptual_v1": (
        "Generate today's lesson content for a learner studying "
        "{topic_name!r} (part of the {topic_group!r} topic group). This "
        "is a conceptual-only day (no hands-on exercise yet) — they have "
        "about {minutes_available} minutes available today.\n\n"
        "{carried_over_instruction}"
        "Theory material to build the lesson around — these are real, "
        "existing sources; cite them by number, never invent any other "
        "source:\n{theory_sources}\n\n"
        "Respond with ONLY a JSON object matching this shape:\n"
        '{{"summary": "<1-2 sentence summary of today\'s topic>", '
        '"theory_framing": "<prose introducing/framing the numbered '
        'theory sources above>", '
        '"reflection_prompt": "<a short reflection question about '
        "today's material>\", "
        '"preview": "<a short preview of tomorrow and how it connects>", '
        '"remaining_content": "<anything from today\'s intended material '
        "that did not fit in the time budget and should carry over to "
        'tomorrow, or an empty string if everything fit>"}}\n'
        "Size the depth/length of every field to genuinely fit within "
        "{minutes_available} minutes total for the whole lesson."
    ),
    "day_content_generation_hands_on_v2": (
        "Generate today's lesson content for a learner studying "
        "{topic_name!r} (part of the {topic_group!r} topic group). They "
        "have about {minutes_available} minutes available today, and "
        "today's hands-on exercise should be scaled to intensity "
        "{hands_on_intensity:.2f} on a 0 (very light) to 1 (full depth) "
        "scale.\n\n"
        "{carried_over_instruction}"
        "Theory material to build the lesson around — these are real, "
        "existing sources; cite them by number, never invent any other "
        "source:\n{theory_sources}\n\n"
        "Respond with ONLY a JSON object matching this shape:\n"
        '{{"summary": "<1-2 sentence summary of today\'s topic>", '
        '"theory_framing": "<prose introducing/framing the numbered '
        'theory sources above>", '
        '"hands_on_exercise": "<a hands-on exercise scaled to the given '
        'intensity>", '
        '"review_prompt": "<a prompt for reviewing/refactoring the '
        'hands-on work>", '
        '"reflection_prompt": "<a short prompt instructing the learner '
        "to recall and restate, in their own words, what they just "
        "studied today — an instruction to recall, never phrased as a "
        'question>", '
        '"preview": "<a short preview of tomorrow and how it connects>", '
        '"remaining_content": "<anything from today\'s intended material '
        "that did not fit in the time budget and should carry over to "
        'tomorrow, or an empty string if everything fit>"}}\n'
        "Size the depth/length of every field to genuinely fit within "
        "{minutes_available} minutes total for the whole lesson."
    ),
    "day_content_generation_conceptual_v2": (
        "Generate today's lesson content for a learner studying "
        "{topic_name!r} (part of the {topic_group!r} topic group). This "
        "is a conceptual-only day (no hands-on exercise yet) — they have "
        "about {minutes_available} minutes available today.\n\n"
        "{carried_over_instruction}"
        "Theory material to build the lesson around — these are real, "
        "existing sources; cite them by number, never invent any other "
        "source:\n{theory_sources}\n\n"
        "Respond with ONLY a JSON object matching this shape:\n"
        '{{"summary": "<1-2 sentence summary of today\'s topic>", '
        '"theory_framing": "<prose introducing/framing the numbered '
        'theory sources above>", '
        '"reflection_prompt": "<a short prompt instructing the learner '
        "to recall and restate, in their own words, what they just "
        "studied today — an instruction to recall, never phrased as a "
        'question>", '
        '"preview": "<a short preview of tomorrow and how it connects>", '
        '"remaining_content": "<anything from today\'s intended material '
        "that did not fit in the time budget and should carry over to "
        'tomorrow, or an empty string if everything fit>"}}\n'
        "Size the depth/length of every field to genuinely fit within "
        "{minutes_available} minutes total for the whole lesson."
    ),
    "goal_completion_closing_note_v1": (
        "Compose a warm, encouraging goal-completion closing note for a "
        "learner who has just finished their core learning plan for the "
        "{resolved_role!r} role.\n\n"
        "Demonstrated strengths (extra-credit/enrichment topics they "
        "completed beyond the core plan — list as genuine accomplishments "
        "if any are given below; if none are given, do not mention "
        "strengths at all): {demonstrated_strengths}\n"
        "Suggested next steps (only relevant when there are no "
        "demonstrated strengths above — frame positively, as an exciting "
        "opportunity, never as something missing, lacking, or a "
        "deficiency): {suggested_next_steps}\n"
        "Number of still-pending market-update notes to mention exist for "
        "later review, if greater than zero (do not mention if zero): "
        "{deferred_patch_count}\n\n"
        "Hard rule, non-negotiable: never use seniority, grading, or "
        "leveling language of any kind — no words like 'junior', "
        "'senior', 'beginner', 'expert', 'novice', 'grade', 'score', or "
        "similar, and no comparison to any level or standard. Describe "
        "accomplishments and next steps in plain, encouraging terms "
        "only.\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"note_text": "<the composed closing note prose>"}}.'
    ),
}


_day_content_agent = LlmAgent(
    name="day_content_generation_agent",
    model=DAY_CONTENT_GEMINI_MODEL,
    instruction=(
        "Generate one day's lesson content for a learner studying a "
        "specific topic, grounded strictly in the given numbered theory "
        "sources."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_closing_note_agent = LlmAgent(
    name="goal_completion_closing_note_agent",
    model=CLOSING_NOTE_GEMINI_MODEL,
    instruction=(
        "Compose a warm, encouraging goal-completion closing note for a "
        "learner, never using seniority/grading/leveling language."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

COACHING_PACE_AGENT = LlmAgent(
    name="coaching_pace_agent",
    model=DAY_CONTENT_GEMINI_MODEL,
    instruction=(
        "Composite agent grouping this module's task agents for "
        "documentation/introspection. Never run via its own Runner — "
        "dispatch between sub-agents is deterministic Python, not "
        "auto-routing."
    ),
    sub_agents=[_day_content_agent, _closing_note_agent],
)


@dataclass(frozen=True)
class DayContent:
    """One day's generated lesson content, including real theory links and any content that spilled over to the next day."""

    summary: str
    theory_framing: str
    theory_links: list[dict[str, str]]
    hands_on_exercise: str | None
    review_prompt: str | None
    reflection_prompt: str
    preview: str
    remaining_content: str | None


async def fetch_theory_material_links(topic_name: str) -> list[dict[str, str]]:
    """Run a live Tavily search for real educational material (docs, tutorials, videos) on a topic."""
    client = _get_tavily_client()
    query = f"{topic_name} tutorial or official documentation"
    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.search,
                query=query,
                search_depth="basic",
                max_results=5,
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

    results = response.get("results") or []
    return [
        {
            "url": result["url"],
            "title": result.get("title", ""),
            "content": result.get("content", ""),
        }
        for result in results
        if result.get("url")
    ][:5]


def _format_theory_sources(theory_links: list[dict[str, str]]) -> str:
    return "\n".join(
        f"[{i}] {link['title']} — {link['url']}\n{link['content'][:300]}"
        for i, link in enumerate(theory_links, start=1)
    )


def _format_carried_over_instruction(carried_over_content: str | None) -> str:
    if not carried_over_content:
        return ""
    return (
        "The following content carried over from a previous day and "
        f"must be worked in today, time permitting:\n{carried_over_content}\n\n"
    )


async def generate_day_content(
    topic_name: str,
    topic_group: str,
    position_in_group: int,
    group_size: int,
    available_time_per_week_hours: int,
    carried_over_content: str | None = None,
) -> DayContent:
    """Generate one day's lesson content, following the hands-on or conceptual-only structure depending on where the topic sits in its group."""
    minutes_available = convert_weekly_hours_to_daily_minutes(
        available_time_per_week_hours
    )
    theory_links = await fetch_theory_material_links(topic_name)
    theory_sources = _format_theory_sources(theory_links)
    carried_over_instruction = _format_carried_over_instruction(carried_over_content)

    hands_on = not is_conceptual_only_day(position_in_group, group_size)

    if hands_on:
        intensity = compute_hands_on_intensity(position_in_group, group_size)
        prompt = PROMPT_REGISTRY["day_content_generation_hands_on_v2"].format(
            topic_name=topic_name,
            topic_group=topic_group,
            minutes_available=minutes_available,
            hands_on_intensity=intensity,
            carried_over_instruction=carried_over_instruction,
            theory_sources=theory_sources,
        )
        required_keys = {
            "summary",
            "theory_framing",
            "hands_on_exercise",
            "review_prompt",
            "reflection_prompt",
            "preview",
            "remaining_content",
        }
    else:
        prompt = PROMPT_REGISTRY["day_content_generation_conceptual_v2"].format(
            topic_name=topic_name,
            topic_group=topic_group,
            minutes_available=minutes_available,
            carried_over_instruction=carried_over_instruction,
            theory_sources=theory_sources,
        )
        required_keys = {
            "summary",
            "theory_framing",
            "reflection_prompt",
            "preview",
            "remaining_content",
        }

    parsed = await call_agent_json(
        _day_content_agent,
        prompt,
        required_keys=required_keys,
        timeout=HEAVY_GENERATION_TIMEOUT_SECONDS,
    )

    return DayContent(
        summary=parsed["summary"],
        theory_framing=parsed["theory_framing"],
        theory_links=theory_links,
        hands_on_exercise=parsed.get("hands_on_exercise") if hands_on else None,
        review_prompt=parsed.get("review_prompt") if hands_on else None,
        reflection_prompt=parsed["reflection_prompt"],
        preview=parsed["preview"],
        remaining_content=parsed.get("remaining_content") or None,
    )


def record_day_content(
    session: Session,
    user_id: str,
    topic_id: str,
    day_number: int,
    content: DayContent,
) -> None:
    """Log each generated content step (summary, theory, hands-on/review, reflection, preview) to the progress log."""
    log_progress_step(session, user_id, topic_id, day_number, "summary")
    log_progress_step(session, user_id, topic_id, day_number, "theory")
    if content.hands_on_exercise is not None:
        log_progress_step(session, user_id, topic_id, day_number, "hands_on")
        log_progress_step(session, user_id, topic_id, day_number, "review")
    log_progress_step(
        session,
        user_id,
        topic_id,
        day_number,
        "reflection",
        reflection_text=content.reflection_prompt,
    )
    log_progress_step(session, user_id, topic_id, day_number, "preview")


@dataclass(frozen=True)
class VerificationSlotState:
    """One verification question slot's in-progress state, including the current attempt number and question, resolved once a pass or the retry cap is reached."""

    topic_id: str
    question_number: int
    attempt_number: int
    current_question: VerificationQuestion
    previous_question_texts: tuple[str, ...]
    resolved: bool = False
    credit: float | None = None
    taught_answer_message: str | None = None


def _build_taught_answer_message(question: VerificationQuestion) -> str:
    """Build the deterministic message that teaches the correct answer once a question slot exhausts its retry attempts."""
    return (
        f"Here's what a correct answer needed: {question.grading_criteria} "
        f"See {question.source_url} for the full explanation."
    )


async def begin_verification_question(
    topic_id: str,
    question_number: int,
    topic_source_material: str,
    source_url: str,
) -> VerificationSlotState:
    """Generate the first verification question (attempt 1) for one question slot of a topic."""
    questions = await generate_questions(
        topic_source_material, source_url, num_questions=1
    )
    question = questions[0]
    return VerificationSlotState(
        topic_id=topic_id,
        question_number=question_number,
        attempt_number=1,
        current_question=question,
        previous_question_texts=(question.question_text,),
    )


async def submit_verification_answer(
    state: VerificationSlotState,
    user_answer: str,
    session: Session,
    topic_source_material: str,
    source_url: str,
    is_test_out: bool = False,
) -> VerificationSlotState:
    """Grade a user's answer for the current attempt and advance the slot's state, whether passed, retried, or capped at half credit."""
    passed = await grade_answer(state.current_question, user_answer)
    at_cap = state.attempt_number >= MAX_VERIFICATION_ATTEMPTS
    credit = (
        FULL_CREDIT if passed else (HALF_CREDIT if at_cap else NOT_YET_RESOLVED_CREDIT)
    )

    write_verification_attempt(
        session,
        state.topic_id,
        state.question_number,
        state.attempt_number,
        state.current_question.question_text,
        state.current_question.grading_criteria,
        user_answer,
        passed=passed,
        credit=credit,
        is_test_out=is_test_out,
    )

    if passed:
        return VerificationSlotState(
            topic_id=state.topic_id,
            question_number=state.question_number,
            attempt_number=state.attempt_number,
            current_question=state.current_question,
            previous_question_texts=state.previous_question_texts,
            resolved=True,
            credit=FULL_CREDIT,
        )

    if at_cap:
        return VerificationSlotState(
            topic_id=state.topic_id,
            question_number=state.question_number,
            attempt_number=state.attempt_number,
            current_question=state.current_question,
            previous_question_texts=state.previous_question_texts,
            resolved=True,
            credit=HALF_CREDIT,
            taught_answer_message=_build_taught_answer_message(state.current_question),
        )

    next_questions = await generate_questions(
        topic_source_material,
        source_url,
        num_questions=1,
        previous_question_texts=list(state.previous_question_texts),
    )
    next_question = next_questions[0]
    return VerificationSlotState(
        topic_id=state.topic_id,
        question_number=state.question_number,
        attempt_number=state.attempt_number + 1,
        current_question=next_question,
        previous_question_texts=(
            *state.previous_question_texts,
            next_question.question_text,
        ),
    )


ENRICHMENT_TOPIC_GROUP_SUFFIX = " (Enrichment)"
ENRICHMENT_POSITION_IN_GROUP = 1

PACE_EXTENSION_DAYS_PER_TRIGGER = 2

COLD_START_CALIBRATION_DAYS = 14


@dataclass(frozen=True)
class TopicCompletionResult:
    """The pace signal computed once a topic's verification questions all resolve, plus whatever enrichment, pacing, or patch-delivery action was taken as a result."""

    topic_score: float
    timing_ratio: float
    combined_pace_signal: float
    drift: Literal["ahead", "behind", "on_track"] | None = None
    enrichment_topic: dict[str, Any] | None = None
    pace_extension_applied: int | None = None
    delivered_patch_topic: dict[str, Any] | None = None
    pending_patch_decision: "PendingPatchDecision | None" = None


def _select_enrichment_skill(
    emerging_skills: list[dict[str, Any]], existing_topic_names: frozenset[str]
) -> dict[str, Any] | None:
    """Pick the first emerging skill from a role's cache entry that isn't already an existing outline topic for the user."""
    for skill_entry in emerging_skills:
        if skill_entry["skill"].casefold() not in existing_topic_names:
            return skill_entry
    return None


def maybe_trigger_enrichment(
    session: Session,
    user_id: str,
    resolved_role: str,
    origin_topic_id: str,
) -> dict[str, Any] | None:
    """Insert an unused emerging skill as a new enrichment topic after the just-completed topic, when the user is on a sustained-ahead pace."""
    if has_pending_enrichment_topic(session, user_id):
        return None

    role = get_role(session, resolved_role)
    if role is None or not role["emerging_skills"]:
        return None

    existing_topics = get_all_topics_for_user(session, user_id)
    existing_topic_names = frozenset(
        topic["topic_name"].casefold() for topic in existing_topics
    )

    selected = _select_enrichment_skill(role["emerging_skills"], existing_topic_names)
    if selected is None:
        return None

    grounded = validate_output_object(
        {
            "source_url": selected["source_url"],
            "source_type": CACHED_SOURCE_TYPE,
            "confidence": selected["confidence"],
            "skill": selected["skill"],
        }
    )

    return insert_new_outline_topic(
        session,
        user_id=user_id,
        topic_name=selected["skill"],
        topic_group=f"{selected['skill']}{ENRICHMENT_TOPIC_GROUP_SUFFIX}",
        position_in_group=ENRICHMENT_POSITION_IN_GROUP,
        source_url=grounded.source_url,
        source_type=grounded.source_type,
        confidence=grounded.confidence,
        is_enrichment=True,
        prerequisite_topic_ids=frozenset({origin_topic_id}),
    )


PATCH_NOTE_SOURCE_TYPE = "patch-note"


@dataclass(frozen=True)
class PendingPatchDecision:
    """A low-confidence patch-note awaiting the user's choice to learn it now or defer it."""

    patch_note_id: str
    origin_topic_id: str
    new_content: str
    source_url: str
    confidence: ConfidenceTier


def maybe_deliver_patch(
    session: Session,
    user_id: str,
    origin_topic_id: str,
) -> dict[str, Any] | PendingPatchDecision | None:
    """Decide and apply delivery of a user's pending patch-notes, either augmenting the relevant topic immediately, asking the user to decide, or doing nothing."""
    pending = get_pending_patch_notes(session, user_id)
    if not pending:
        return None

    current_topic = get_topic(session, origin_topic_id)
    if current_topic is None:
        return None
    current_hierarchy_position = current_topic["hierarchy_position"]

    assembled: list[dict[str, Any]] = []
    for patch in pending:
        origin = get_topic(session, patch["origin_topic_id"])
        if origin is None:
            continue
        assembled.append(
            {
                "id": patch["id"],
                "confidence": ConfidenceTier(patch["confidence"]),
                "hierarchy_position": origin["hierarchy_position"],
            }
        )
    if not assembled:
        return None

    decision = decide_patch_delivery(assembled, current_hierarchy_position)
    if decision.action == "none":
        return None

    chosen = next(patch for patch in pending if patch["id"] == decision.patch_note_id)

    if decision.action == "ask_user":
        return PendingPatchDecision(
            patch_note_id=chosen["id"],
            origin_topic_id=chosen["origin_topic_id"],
            new_content=chosen["new_content"],
            source_url=chosen["source_url"],
            confidence=ConfidenceTier(chosen["confidence"]),
        )

    origin_topic = get_topic(session, chosen["origin_topic_id"])
    assert origin_topic is not None

    grounded = validate_output_object(
        {
            "source_url": chosen["source_url"],
            "source_type": PATCH_NOTE_SOURCE_TYPE,
            "confidence": chosen["confidence"],
        }
    )

    augmented = augment_outline_topic(
        session,
        origin_topic["id"],
        source_url=grounded.source_url,
        source_type=grounded.source_type,
        confidence=grounded.confidence,
    )
    update_patch_note_status(
        session,
        chosen["id"],
        PatchStatus.DELIVERED,
        datetime.now(UTC).replace(tzinfo=None, microsecond=0),
    )
    return augmented


def resolve_pending_patch_decision(
    session: Session,
    decision: PendingPatchDecision,
    user_choice: Literal["learn_now", "defer"],
    resolved_at: datetime,
) -> dict[str, Any] | None:
    """Apply the user's learn-now-or-defer choice for a pending patch-note and update its status accordingly."""
    state = PatchDecisionState(patch_note_id=decision.patch_note_id)
    resolved_state = resolve_patch_decision(state, user_choice, resolved_at)

    augmented: dict[str, Any] | None = None
    if user_choice == "learn_now":
        grounded = validate_output_object(
            {
                "source_url": decision.source_url,
                "source_type": PATCH_NOTE_SOURCE_TYPE,
                "confidence": decision.confidence,
            }
        )
        augmented = augment_outline_topic(
            session,
            decision.origin_topic_id,
            source_url=grounded.source_url,
            source_type=grounded.source_type,
            confidence=grounded.confidence,
        )

    assert resolved_state.status is not None
    update_patch_note_status(
        session, resolved_state.patch_note_id, resolved_state.status, resolved_at
    )
    return augmented


def _get_latest_attempt_per_question(
    session: Session, topic_id: str
) -> dict[int, dict[str, Any]]:
    """Read back each verification question slot's most recent attempt for a topic, requiring all 5 slots to be resolved."""
    attempts = get_attempts_for_topic(session, topic_id)
    latest_by_question: dict[int, dict[str, Any]] = {}
    for attempt in attempts:
        latest_by_question[attempt["question_number"]] = attempt

    if sorted(latest_by_question.keys()) != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"topic {topic_id!r} does not have all 5 question slots "
            f"attempted yet: {sorted(latest_by_question.keys())}"
        )

    unresolved = [
        question_number
        for question_number, attempt in latest_by_question.items()
        if not attempt["passed"]
        and attempt["attempt_number"] < MAX_VERIFICATION_ATTEMPTS
    ]
    if unresolved:
        raise ValueError(
            f"topic {topic_id!r} has question slot(s) not yet resolved "
            f"(failed, but not at the retry cap): {sorted(unresolved)}"
        )

    return latest_by_question


def _get_final_credits_per_question(session: Session, topic_id: str) -> list[float]:
    """Read back the final credit awarded for each of a topic's 5 verification question slots."""
    latest_by_question = _get_latest_attempt_per_question(session, topic_id)
    return [latest_by_question[q]["credit"] for q in range(1, 6)]


def complete_topic_verification(
    session: Session,
    user_id: str,
    topic_id: str,
    days_taken: int,
    days_expected: int,
    is_test_out: bool = False,
    is_enrichment: bool = False,
    reference_time: datetime | None = None,
) -> TopicCompletionResult:
    """Finalize a topic's verification: compute and persist its pace signal, act on sustained drift or pending patch-notes, and mark the topic completed."""
    credits = _get_final_credits_per_question(session, topic_id)
    topic_score = calculate_topic_score(credits)
    timing_ratio = calculate_timing_ratio(days_taken, days_expected)
    combined_signal = calculate_combined_pace_signal(topic_score, timing_ratio)

    drift: Literal["ahead", "behind", "on_track"] | None = None
    enrichment_topic: dict[str, Any] | None = None
    pace_extension_applied: int | None = None
    delivered_patch_topic: dict[str, Any] | None = None
    pending_patch_decision: PendingPatchDecision | None = None

    if not is_enrichment:
        write_pace_snapshot(
            session,
            user_id,
            topic_id,
            topic_score,
            timing_ratio,
            days_taken,
            days_expected,
        )
        user = get_user(session, user_id)
        resolved_reference_time = reference_time or datetime.now(UTC).replace(
            tzinfo=None, microsecond=0
        )
        in_cold_start = (
            user is not None
            and user["created_at"] is not None
            and (resolved_reference_time - user["created_at"]).days
            < COLD_START_CALIBRATION_DAYS
        )

        if not in_cold_start:
            history = get_pace_snapshot_history(session, user_id)
            pace_signals = [
                calculate_combined_pace_signal(row["topic_score"], row["timing_ratio"])
                for row in history
            ]
            drift = detect_sustained_drift(pace_signals)
        else:
            drift = "on_track"

        if drift == "ahead":
            if user is not None and user["resolved_role"]:
                enrichment_topic = maybe_trigger_enrichment(
                    session, user_id, user["resolved_role"], topic_id
                )
        elif drift == "behind":
            pace_extension_applied = extend_pacing(
                session, user_id, PACE_EXTENSION_DAYS_PER_TRIGGER
            )

        patch_result = maybe_deliver_patch(session, user_id, topic_id)
        if isinstance(patch_result, PendingPatchDecision):
            pending_patch_decision = patch_result
        else:
            delivered_patch_topic = patch_result

    mark_topic_completed(
        session,
        topic_id,
        status=COMPLETED_TEST_OUT_STATUS if is_test_out else COMPLETED_STATUS,
    )

    return TopicCompletionResult(
        topic_score=topic_score,
        timing_ratio=timing_ratio,
        combined_pace_signal=combined_signal,
        drift=drift,
        enrichment_topic=enrichment_topic,
        pace_extension_applied=pace_extension_applied,
        delivered_patch_topic=delivered_patch_topic,
        pending_patch_decision=pending_patch_decision,
    )


@dataclass(frozen=True)
class TestOutResult:
    """Result of a topic's test-out attempt, including whether every question slot was passed in full."""

    completion: TopicCompletionResult
    full_pass: bool


async def complete_topic_test_out(
    session: Session,
    user_id: str,
    topic_id: str,
    days_taken: int,
    days_expected: int,
    is_enrichment: bool = False,
) -> TestOutResult:
    """Resolve a topic via test-out, where the user answers all verification questions before any study content is generated."""
    credits = _get_final_credits_per_question(session, topic_id)
    full_pass = all(credit == FULL_CREDIT for credit in credits)

    completion = complete_topic_verification(
        session,
        user_id,
        topic_id,
        days_taken,
        days_expected,
        is_test_out=True,
        is_enrichment=is_enrichment,
    )

    return TestOutResult(completion=completion, full_pass=full_pass)


def is_goal_complete(session: Session, user_id: str) -> bool:
    """Whether every core (non-enrichment) outline topic for a user is completed."""
    topics = get_all_topics_for_user(session, user_id)
    core_topics = [topic for topic in topics if not topic["is_enrichment"]]
    if not core_topics:
        return False
    return all(
        topic["status"] in {COMPLETED_STATUS, COMPLETED_TEST_OUT_STATUS}
        for topic in core_topics
    )


_BANNED_LEVELING_TERMS = frozenset(
    {
        "junior",
        "senior",
        "beginner",
        "entry-level",
        "entry level",
        "mid-level",
        "novice",
        "expert",
        "grade",
        "graded",
        "grading",
        "score of",
        "level up",
        "leveled up",
    }
)


def _contains_banned_leveling_language(text: str) -> bool:
    """Whether text contains any banned seniority, grading, or leveling term."""
    lowered = text.casefold()
    return any(term in lowered for term in _BANNED_LEVELING_TERMS)


@dataclass(frozen=True)
class ClosingNote:
    """The goal-completion closing note, along with the deterministic facts it was composed from."""

    resolved_role: str
    note_text: str
    demonstrated_strengths: list[str]
    suggested_next_steps: list[str]
    deferred_patch_notes: list[dict[str, Any]]


async def generate_closing_note(session: Session, user_id: str) -> ClosingNote:
    """Compose the goal-completion closing note from the user's role, completed enrichment topics, and deferred patch-notes, rejecting output that uses banned leveling language."""
    user = get_user(session, user_id)
    resolved_role = user["resolved_role"] if user is not None else None
    if not resolved_role:
        raise ValueError(
            f"user {user_id!r} has no resolved_role set — cannot compose a "
            "closing note without knowing which role's roles_cache entry to reuse"
        )

    role = get_role(session, resolved_role)
    topics = get_all_topics_for_user(session, user_id)
    completed_enrichment_topics = [
        topic["topic_name"]
        for topic in topics
        if topic["is_enrichment"]
        and topic["status"] in {COMPLETED_STATUS, COMPLETED_TEST_OUT_STATUS}
    ]
    deferred_patches = get_deferred_patch_notes(session, user_id)

    demonstrated_strengths: list[str] = []
    suggested_next_steps: list[str] = []
    if completed_enrichment_topics:
        demonstrated_strengths = completed_enrichment_topics
    elif role is not None:
        suggested_next_steps = [entry["skill"] for entry in role["emerging_skills"]]

    prompt = PROMPT_REGISTRY["goal_completion_closing_note_v1"].format(
        resolved_role=resolved_role,
        demonstrated_strengths=", ".join(demonstrated_strengths) or "none",
        suggested_next_steps=", ".join(suggested_next_steps) or "none",
        deferred_patch_count=len(deferred_patches),
    )
    parsed = await call_agent_json(
        _closing_note_agent,
        prompt,
        required_keys={"note_text"},
        timeout=HEAVY_GENERATION_TIMEOUT_SECONDS,
    )
    note_text = parsed["note_text"]
    if not isinstance(note_text, str) or not note_text.strip():
        raise GeminiCallError(f"Gemini returned no usable 'note_text': {parsed!r}")
    if _contains_banned_leveling_language(note_text):
        raise GeminiCallError(
            "closing note text contains banned seniority/grading/leveling "
            f"language (PRD §7.11 hard constraint): {note_text!r}"
        )

    return ClosingNote(
        resolved_role=resolved_role,
        note_text=note_text,
        demonstrated_strengths=demonstrated_strengths,
        suggested_next_steps=suggested_next_steps,
        deferred_patch_notes=deferred_patches,
    )
