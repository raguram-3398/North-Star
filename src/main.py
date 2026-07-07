"""Streamlit entry point that wires the intake-through-goal-completion pipeline stages together using session state, calling out to already-tested agent and data modules for all decision logic."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from agents.coaching_pace_agent import (
    FULL_CREDIT,
    DayContent,
    PendingPatchDecision,
    TestOutResult,
    TopicCompletionResult,
    VerificationSlotState,
    begin_verification_question,
    calculate_days_expected,
    complete_topic_test_out,
    complete_topic_verification,
    fetch_theory_material_links,
    generate_closing_note,
    generate_day_content,
    is_goal_complete,
    record_day_content,
    resolve_pending_patch_decision,
    submit_verification_answer,
)
from agents.research_outline_agent import (
    ClarifyGateTurn,
    LiveGroundingResult,
    OutlineConfirmationTurn,
    ValidatedGroundedContent,
    advance_clarify_gate,
    begin_clarify_gate,
    begin_outline_confirmation,
    create_initial_outline,
    ground_addition_request,
    ground_role,
    handle_review_turn,
    regenerate_outline_with_addition,
)
from cron.refresh_roles import SEED_ROLES, check_and_refresh_stale_roles
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from data.outline_topics import (
    get_all_topics_for_user,
    get_topic,
    get_topics_in_group,
    insert_outline_topics,
)
from data.users import create_user, get_user, set_resolved_role
from db.connection import get_session
from security.input_gate import ClarifyGateStage, OutlineReviewAction
from security.output_guard import ConfidenceTier
from utils.exceptions import (
    ConfidenceValidationError,
    GeminiCallError,
    GroundingSourceCallError,
    HimalayasParseError,
    TavilyParseError,
)

load_dotenv()

NOT_STARTED_STATUS = "not_started"


class PipelineStage(Enum):
    """The pipeline stages a user navigates through, from the landing page to goal completion."""

    LANDING = "landing"
    INTAKE = "intake"
    CLARIFY_GATE = "clarify_gate"
    RESEARCH_GROUNDING = "research_grounding"
    OUTLINE_CREATION = "outline_creation"
    OUTLINE_CONFIRMATION = "outline_confirmation"
    DAY_BY_DAY_COACHING = "day_by_day_coaching"
    VERIFICATION = "verification"
    GOAL_COMPLETION = "goal_completion"


_STAGE_EXCEPTIONS = (
    GeminiCallError,
    GroundingSourceCallError,
    ConfidenceValidationError,
    HimalayasParseError,
    TavilyParseError,
    ValueError,
    TypeError,
)


def _run_async(coro: Any) -> Any:
    """Run one async call to completion using a fresh event loop for this Streamlit rerun."""
    return asyncio.run(coro)


def _init_session_state() -> None:
    """Populate every session-state key this module uses with its default value, if not already set."""
    defaults: dict[str, Any] = {
        "db_session": None,
        "startup_staleness_checked": False,
        "current_stage": PipelineStage.LANDING.value,
        "user_id": None,
        "stated_goal": None,
        "clarify_turn": None,
        "clarify_conversation": None,
        "resolved_role": None,
        "grounding_result": None,
        "outline_topics": None,
        "persisted_topics": None,
        "outline_confirmation_turn": None,
        "outline_confirmation_conversation": None,
        "current_topic_id": None,
        "day_number_for_topic": 1,
        "carried_over_content": None,
        "day_content": None,
        "day_coaching_step_index": 0,
        "test_out_prompt_dismissed": False,
        "is_test_out": False,
        "verification_source_material": None,
        "verification_source_url": None,
        "current_question_number": 1,
        "verification_slot_state": None,
        "last_completion_result": None,
        "last_test_out_full_pass": None,
        "pending_patch_decision": None,
        "closing_note": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_db_session() -> Session:
    previous_session: Session | None = st.session_state.db_session
    if previous_session is not None:
        try:
            previous_session.close()
        except SQLAlchemyError:
            pass
    session = get_session()
    st.session_state.db_session = session
    return session


def _maybe_check_stale_roles() -> None:
    """Refresh any stale or missing seed-role cache entries once per browser session, warning rather than crashing on failure."""
    if st.session_state.startup_staleness_checked:
        return
    st.session_state.startup_staleness_checked = True
    session = _get_db_session()
    reference_time = datetime.now(UTC).replace(tzinfo=None)
    try:
        _run_async(check_and_refresh_stale_roles(session, SEED_ROLES, reference_time))
    except _STAGE_EXCEPTIONS as exc:
        st.warning(f"Startup roles_cache staleness check failed: {exc}")


def _domain_from_url(url: str) -> str:
    """Return the host portion of a URL for compact citation display, falling back to the full URL if parsing fails."""
    return urlparse(url).netloc or url


def _render_kicker(label: str) -> None:
    """Render a small muted section label such as "Summary" or "Theory"."""
    st.caption(f"**{label.upper()}**")


def _render_stage_header(label: str) -> None:
    """Render a center-aligned heading for a stage that auto-advances without a Continue button."""
    st.markdown(f"<h2 style='text-align: center;'>{label}</h2>", unsafe_allow_html=True)


def _render_citations(links: list[dict[str, str]]) -> None:
    """Render one caption per citation as a clickable markdown link to its full URL, alongside the bare domain."""
    for link in links:
        url = link["url"]
        domain = _domain_from_url(url)
        title = (link.get("title") or "").strip()
        label = title if title else domain
        st.caption(f"Source: [{label}]({url}) — {domain}")


def _render_progress_dots(current_question_number: int, total: int = 5) -> None:
    """Render the verification step tracker, striking through completed questions and bolding the current one."""
    parts = []
    for n in range(1, total + 1):
        if n < current_question_number:
            parts.append(f"~~{n}~~")
        elif n == current_question_number:
            parts.append(f"**{n}**")
        else:
            parts.append(str(n))
    st.caption("Question " + "  ·  ".join(parts) + f" of {total}")


def _render_landing() -> None:
    st.markdown(
        "<h3 style='text-align: center;'>Adaptive career learning with "
        "every topic grounded in real hiring data and verified "
        "sources.</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    pillars = (
        (col1, "GROUNDED", "Every topic comes from live job-market data, not a guess."),
        (
            col2,
            "VERIFIED",
            "Five source-anchored questions confirm real understanding.",
        ),
        (
            col3,
            "ADAPTIVE",
            "Pace adjusts to you — content is never removed, only extended.",
        ),
    )
    for col, kicker, body in pillars:
        with col.container(border=True, key=f"ns-card-landing-{kicker.lower()}"):
            _render_kicker(kicker)
            st.write(body)

    st.write("")
    if st.button("Begin", type="primary"):
        st.session_state.current_stage = PipelineStage.INTAKE.value
        st.rerun()


def _render_intake() -> None:
    """Render the Intake form and create the user record once submitted."""
    st.header("Intake")
    _render_kicker("Step 1 — tell us where you're starting from")
    with st.form("intake_form"):
        job_col, years_col = st.columns(2)
        current_job = job_col.text_input("Current job")
        years_experience = years_col.number_input(
            "Years of experience", min_value=0, step=1, value=0
        )
        background = st.text_area("Background")
        prior_self_study = st.text_area("Prior self-study (specific, not yes/no)")

        with st.container(border=True, key="ns-card-intake-destination"):
            _render_kicker("Your destination")
            goal = st.text_input("What tech role or skill do you want to learn?")
            available_time_per_week = st.number_input(
                "Available time per week (hours)", min_value=1, step=1, value=10
            )

        submitted = st.form_submit_button("Submit", type="primary")

    if not submitted:
        return
    if not goal.strip():
        st.error("Please describe a goal before continuing.")
        return

    session = _get_db_session()
    user = create_user(
        session,
        background=background.strip() or None,
        current_job=current_job.strip() or None,
        years_experience=years_experience or None,
        prior_self_study=prior_self_study.strip() or None,
        available_time_per_week=available_time_per_week,
    )
    st.session_state.user_id = str(user["id"])
    st.session_state.stated_goal = goal.strip()
    st.session_state.current_stage = PipelineStage.CLARIFY_GATE.value
    st.rerun()


def _render_clarify_gate() -> None:
    """Render the bounded Clarify Gate chat loop, dispatching on its current stage until resolved or exited."""
    _render_stage_header("Clarify Gate")

    if st.session_state.clarify_turn is None:
        try:
            first_turn = _run_async(begin_clarify_gate(st.session_state.stated_goal))
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Clarify Gate failed: {exc}")
            return
        st.session_state.clarify_turn = first_turn
        st.session_state.clarify_conversation = [
            {"role": "agent", "content": first_turn.message}
        ]

    turn: ClarifyGateTurn = st.session_state.clarify_turn
    for entry in st.session_state.clarify_conversation:
        with st.chat_message("assistant" if entry["role"] == "agent" else "user"):
            st.write(entry["content"])

    stage = turn.gate_state.stage

    if stage is ClarifyGateStage.RESOLVED:
        assert turn.resolved_role is not None
        st.session_state.resolved_role = turn.resolved_role
        st.session_state.current_stage = PipelineStage.RESEARCH_GROUNDING.value
        st.rerun()
        return

    if stage is ClarifyGateStage.EXITED:
        st.session_state.resolved_role = turn.context.original_stated_goal
        st.session_state.current_stage = PipelineStage.RESEARCH_GROUNDING.value
        st.rerun()
        return

    if stage not in (
        ClarifyGateStage.NARROWING,
        ClarifyGateStage.PROPOSE_BEST_GUESS,
        ClarifyGateStage.EXPLAIN_ROLE,
    ):
        raise AssertionError(f"unexpected clarify gate stage reached the UI: {stage}")

    user_response = st.chat_input("Your response")
    if not user_response:
        return

    conversation_so_far = list(st.session_state.clarify_conversation)
    session = _get_db_session()
    reference_time = datetime.now(UTC).replace(tzinfo=None)
    try:
        next_turn = _run_async(
            advance_clarify_gate(
                turn.gate_state,
                turn.context,
                conversation_so_far,
                user_response,
                session,
                reference_time,
            )
        )
    except _STAGE_EXCEPTIONS as exc:
        st.error(f"Clarify Gate failed: {exc}")
        return

    st.session_state.clarify_conversation.append(
        {"role": "user", "content": user_response}
    )
    st.session_state.clarify_conversation.append(
        {"role": "agent", "content": next_turn.message}
    )
    st.session_state.clarify_turn = next_turn
    st.rerun()


def _render_research_grounding() -> None:
    """Run role grounding and auto-advance to Outline Creation once it resolves to a usable result."""
    _render_stage_header("Research & Market Grounding")
    session = _get_db_session()

    if st.session_state.grounding_result is None:
        reference_time = datetime.now(UTC).replace(tzinfo=None)
        try:
            result = _run_async(
                ground_role(st.session_state.resolved_role, session, reference_time)
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Research/Grounding failed: {exc}")
            return
        st.session_state.grounding_result = result

    result = st.session_state.grounding_result
    st.write(f"Resolved role: **{st.session_state.resolved_role}**")

    if isinstance(result, GeneralKnowledgeFloorResult):
        st.warning(result.label)
        st.info(
            "No outline can be built from this floor rung — per PRD §7.3 "
            "and specs/scenarios/high_risk_flows.feature's \"No source "
            'returns usable data" scenario, no outline item is ever '
            "created here."
        )
        return

    if isinstance(result, CachedFallbackResult):
        role_confidence = ConfidenceTier.CACHED_LOW
        st.caption(
            f"(cached fallback, last updated {result.last_updated}, "
            f"stale={result.is_stale})"
        )
    else:
        assert isinstance(result, LiveGroundingResult)
        role_confidence = result.confidence

    set_resolved_role(
        session,
        st.session_state.user_id,
        st.session_state.resolved_role,
        role_confidence.value,
    )
    st.session_state.current_stage = PipelineStage.OUTLINE_CREATION.value
    st.rerun()


def _core_and_emerging_skills(
    result: object,
) -> tuple[list[ValidatedGroundedContent], list[ValidatedGroundedContent]] | None:
    """Split a grounding result into core and emerging skill lists, or return None if it carries no grounded skills."""
    if isinstance(result, LiveGroundingResult):
        return result.skills, []
    if isinstance(result, CachedFallbackResult):
        return result.core_skills, result.emerging_skills
    return None


def _render_outline_creation() -> None:
    """Generate and persist the initial outline, then render it with a Continue button to Outline Confirmation."""
    _render_stage_header("Outline Creation")
    session = _get_db_session()
    result = st.session_state.grounding_result

    if st.session_state.outline_topics is None:
        split = _core_and_emerging_skills(result)
        if split is None:
            st.error("Cannot create an outline without a grounded skill list.")
            return
        core_skills, emerging_skills = split

        try:
            topics = _run_async(
                create_initial_outline(
                    st.session_state.resolved_role, core_skills, emerging_skills
                )
            )
            persisted = insert_outline_topics(session, st.session_state.user_id, topics)
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Outline Creation failed: {exc}")
            return
        st.session_state.outline_topics = topics
        st.session_state.persisted_topics = persisted

    persisted_topics = st.session_state.persisted_topics
    st.write(f"{len(persisted_topics)} topics created and persisted.")
    for topic in sorted(persisted_topics, key=lambda t: t["hierarchy_position"]):
        st.write(
            f"{topic['hierarchy_position']}. {topic['topic_name']} "
            f"({topic['topic_group']})"
        )

    if st.button("Continue to Outline Confirmation"):
        st.session_state.current_stage = PipelineStage.OUTLINE_CONFIRMATION.value
        st.rerun()


def _advance_to_day_one() -> None:
    """Pick the lowest-position not-started topic and move to Day-by-Day Coaching."""
    session = _get_db_session()
    all_topics = get_all_topics_for_user(session, st.session_state.user_id)
    not_started = [t for t in all_topics if t["status"] == NOT_STARTED_STATUS]
    if not not_started:
        st.error("No topics were persisted — cannot start Day 1.")
        return
    first_topic = min(not_started, key=lambda t: t["hierarchy_position"])
    st.session_state.current_topic_id = str(first_topic["id"])
    st.session_state.current_stage = PipelineStage.DAY_BY_DAY_COACHING.value
    st.rerun()


def _regenerate_outline_for_addition(
    session: Session, turn: OutlineConfirmationTurn, user_message: str
) -> OutlineConfirmationTurn:
    """Ground the requested addition, fold it into the outline, and persist the regenerated hierarchy, degrading gracefully to an unchanged turn if grounding fails."""
    try:
        new_addition = _run_async(ground_addition_request(user_message))
    except _STAGE_EXCEPTIONS as exc:
        return OutlineConfirmationTurn(
            state=turn.state,
            message=f"Couldn't process that addition request: {exc}",
            topics=turn.topics,
            concluded=turn.concluded,
        )

    if new_addition is None:
        return OutlineConfirmationTurn(
            state=turn.state,
            message=(
                "I couldn't find a reliable source for that addition, so "
                "your outline wasn't changed."
            ),
            topics=turn.topics,
            concluded=turn.concluded,
        )

    split = _core_and_emerging_skills(st.session_state.grounding_result)
    if split is None:
        return OutlineConfirmationTurn(
            state=turn.state,
            message="Cannot add to an outline without a grounded skill list.",
            topics=turn.topics,
            concluded=turn.concluded,
        )
    core_skills, emerging_skills = split

    try:
        regenerated_turn: OutlineConfirmationTurn = _run_async(
            regenerate_outline_with_addition(
                turn.state,
                st.session_state.resolved_role,
                core_skills,
                emerging_skills,
                new_addition,
            )
        )
        persisted = insert_outline_topics(
            session, st.session_state.user_id, regenerated_turn.topics
        )
    except _STAGE_EXCEPTIONS as exc:
        return OutlineConfirmationTurn(
            state=turn.state,
            message=f"Couldn't regenerate the outline: {exc}",
            topics=turn.topics,
            concluded=turn.concluded,
        )

    st.session_state.outline_topics = regenerated_turn.topics
    st.session_state.persisted_topics = persisted
    return regenerated_turn


def _render_outline_confirmation() -> None:
    """Render the outline alongside the bounded Outline Confirmation review chat, regenerating the outline on addition requests."""
    st.header("Outline Confirmation")
    session = _get_db_session()

    if st.session_state.outline_confirmation_turn is None:
        try:
            first_turn = _run_async(
                begin_outline_confirmation(
                    st.session_state.resolved_role, st.session_state.outline_topics
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Outline Confirmation failed: {exc}")
            return
        st.session_state.outline_confirmation_turn = first_turn
        st.session_state.outline_confirmation_conversation = [
            {"role": "agent", "content": first_turn.message}
        ]

    turn: OutlineConfirmationTurn = st.session_state.outline_confirmation_turn

    outline_col, chat_col = st.columns([1, 2])
    with outline_col:
        st.subheader("Current outline")
        current_group: str | None = None
        for topic in turn.topics:
            if topic.topic_group != current_group:
                current_group = topic.topic_group
                st.markdown(f"**{current_group}**")
            st.write(f"- {topic.topic_name}")

    with chat_col:
        for entry in st.session_state.outline_confirmation_conversation:
            with st.chat_message("assistant" if entry["role"] == "agent" else "user"):
                st.write(entry["content"])

        if turn.concluded:
            if st.button("Continue to Day 1"):
                _advance_to_day_one()
            return

        user_message = st.chat_input(
            "Ask a question, raise a concern, request an addition, or confirm"
        )
        if not user_message:
            return

        try:
            next_turn = _run_async(
                handle_review_turn(
                    turn.state,
                    st.session_state.resolved_role,
                    turn.topics,
                    user_message,
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Outline review failed: {exc}")
            return

        st.session_state.outline_confirmation_conversation.append(
            {"role": "user", "content": user_message}
        )
        st.session_state.outline_confirmation_conversation.append(
            {"role": "agent", "content": next_turn.message}
        )

        if next_turn.action is OutlineReviewAction.ADDITION_REQUEST:
            next_turn = _regenerate_outline_for_addition(
                session, next_turn, user_message
            )
            st.session_state.outline_confirmation_conversation.append(
                {"role": "agent", "content": next_turn.message}
            )

        st.session_state.outline_confirmation_turn = next_turn
        st.rerun()


def _build_verification_source(
    content: DayContent, topic: dict[str, Any]
) -> tuple[str, str]:
    """Build the verification source material and URL from the day's teaching content, falling back to the topic's own source URL if no theory links exist."""
    material_parts = [content.theory_framing]
    material_parts.extend(
        link["content"] for link in content.theory_links if link.get("content")
    )
    topic_source_material = "\n\n".join(material_parts)
    source_url = (
        content.theory_links[0]["url"] if content.theory_links else topic["source_url"]
    )
    return topic_source_material, source_url


def _build_test_out_verification_source(
    theory_links: list[dict[str, str]],
) -> tuple[str, str] | None:
    """Build the test-out verification source material and URL from fetched theory links, or return None if none were found."""
    if not theory_links:
        return None
    material_parts = [link["content"] for link in theory_links if link.get("content")]
    if not material_parts:
        return None
    topic_source_material = "\n\n".join(material_parts)
    source_url = theory_links[0]["url"]
    return topic_source_material, source_url


def _day_coaching_steps(
    content: DayContent,
) -> list[tuple[str, str, Callable[[], None]]]:
    """Return the ordered list of Day-by-Day Coaching sections to reveal one at a time as (container key, label, render function)."""

    def _render_theory() -> None:
        st.write(content.theory_framing)
        if content.theory_links:
            _render_citations(content.theory_links)

    steps: list[tuple[str, str, Callable[[], None]]] = [
        ("day-summary", "Summary", lambda: st.write(content.summary)),
        ("day-theory", "Theory", _render_theory),
    ]
    if content.hands_on_exercise:
        steps.append(
            (
                "day-hands-on",
                "Hands-on exercise",
                lambda: st.write(content.hands_on_exercise),
            )
        )
    if content.review_prompt:
        steps.append(("day-review", "Review", lambda: st.write(content.review_prompt)))
    steps.append(
        ("day-reflection", "Reflection", lambda: st.write(content.reflection_prompt))
    )
    steps.append(("day-preview", "Preview", lambda: st.write(content.preview)))
    return steps


def _render_test_out_choice(topic: dict[str, Any]) -> None:
    """Render the once-per-topic test-out choice screen, letting the user skip straight to verification or decline it."""
    st.write(
        "You can test out of this topic by answering 5 verification "
        "questions first — a full pass skips today's study content "
        "entirely."
    )
    col_test_out, col_decline = st.columns(2)
    with col_test_out:
        test_out_clicked = st.button("Test out of this topic", type="primary")
    with col_decline:
        decline_clicked = st.button("No, teach me the material")

    if decline_clicked:
        st.session_state.test_out_prompt_dismissed = True
        st.rerun()

    if test_out_clicked:
        try:
            theory_links = _run_async(fetch_theory_material_links(topic["topic_name"]))
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Fetching teaching material failed: {exc}")
            return
        source = _build_test_out_verification_source(theory_links)
        if source is None:
            st.error(
                "No grounded teaching material was found for this topic "
                "yet — test-out isn't available right now."
            )
            return
        st.session_state.verification_source_material = source[0]
        st.session_state.verification_source_url = source[1]
        st.session_state.current_question_number = 1
        st.session_state.verification_slot_state = None
        st.session_state.is_test_out = True
        st.session_state.current_stage = PipelineStage.VERIFICATION.value
        st.rerun()


def _render_day_by_day_coaching() -> None:
    st.header("Day-by-Day Coaching")
    session = _get_db_session()
    topic = get_topic(session, st.session_state.current_topic_id)
    if topic is None:
        st.error("Current topic not found.")
        return

    st.subheader(topic["topic_name"])

    if (
        st.session_state.day_content is None
        and not st.session_state.test_out_prompt_dismissed
    ):
        _render_test_out_choice(topic)
        return

    if st.session_state.day_content is None:
        group_topics = get_topics_in_group(
            session, st.session_state.user_id, topic["topic_group"]
        )
        user = get_user(session, st.session_state.user_id)
        assert user is not None
        try:
            generated_content = _run_async(
                generate_day_content(
                    topic_name=topic["topic_name"],
                    topic_group=topic["topic_group"],
                    position_in_group=topic["position_in_group"],
                    group_size=len(group_topics),
                    available_time_per_week_hours=user["available_time_per_week"],
                    carried_over_content=st.session_state.carried_over_content,
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Day content generation failed: {exc}")
            return
        record_day_content(
            session,
            st.session_state.user_id,
            st.session_state.current_topic_id,
            st.session_state.day_number_for_topic,
            generated_content,
        )
        st.session_state.day_content = generated_content
        st.session_state.day_coaching_step_index = 0

    content: DayContent = st.session_state.day_content

    _render_kicker(f"Day {st.session_state.day_number_for_topic} of this topic")

    steps = _day_coaching_steps(content)
    step_index = min(st.session_state.day_coaching_step_index, len(steps) - 1)
    for key, label, render_step in steps[: step_index + 1]:
        with st.container(border=True, key=f"ns-card-{key}"):
            _render_kicker(label)
            render_step()

    if step_index < len(steps) - 1:
        if st.button("Next"):
            st.session_state.day_coaching_step_index += 1
            st.rerun()
        return

    if content.remaining_content:
        st.info(
            "Some content didn't fit today's time budget and will carry "
            "over to the next day."
        )
        if st.button("Continue to next day (same topic)"):
            st.session_state.day_number_for_topic += 1
            st.session_state.carried_over_content = content.remaining_content
            st.session_state.day_content = None
            st.session_state.day_coaching_step_index = 0
            st.rerun()
        return

    if st.button("Start Quiz", type="primary"):
        source_material, source_url = _build_verification_source(content, topic)
        st.session_state.verification_source_material = source_material
        st.session_state.verification_source_url = source_url
        st.session_state.current_question_number = 1
        st.session_state.verification_slot_state = None
        st.session_state.current_stage = PipelineStage.VERIFICATION.value
        st.rerun()


def _render_patch_decision_banner(session: Session) -> None:
    """Render a non-blocking banner letting the user learn now or defer a pending low-confidence patch-note."""
    decision: PendingPatchDecision = st.session_state.pending_patch_decision
    with st.container(border=True, key="ns-card-patch-decision"):
        _render_kicker("Market update awaiting your decision")
        st.write(decision.new_content)
        _render_citations([{"title": "Source", "url": decision.source_url}])
        learn_col, defer_col = st.columns(2)
        with learn_col:
            learn_now_clicked = st.button("Learn now", type="primary")
        with defer_col:
            defer_clicked = st.button("Defer")

    if not (learn_now_clicked or defer_clicked):
        return

    user_choice: Literal["learn_now", "defer"] = (
        "learn_now" if learn_now_clicked else "defer"
    )
    resolve_pending_patch_decision(
        session,
        decision,
        user_choice,
        datetime.now(UTC).replace(tzinfo=None, microsecond=0),
    )
    st.session_state.pending_patch_decision = None
    st.rerun()


def _advance_after_topic_completion(session: Session) -> None:
    result: TopicCompletionResult = st.session_state.last_completion_result
    st.subheader("Topic completed")
    st.write(
        f"topic_score={result.topic_score:.2f}  "
        f"timing_ratio={result.timing_ratio:.2f}  "
        f"combined_pace_signal={result.combined_pace_signal:.2f}"
    )
    if result.drift is not None:
        st.write(f"Pace drift this check-in: **{result.drift}**")
    if result.enrichment_topic is not None:
        st.info(
            "New enrichment topic added: " f"{result.enrichment_topic['topic_name']}"
        )
    if result.delivered_patch_topic is not None:
        st.info(
            "Market data refreshed for a completed topic: "
            f"{result.delivered_patch_topic['topic_name']}"
        )
    if st.session_state.last_test_out_full_pass is not None:
        if st.session_state.last_test_out_full_pass:
            st.success("Tested out — full pass, no study content needed.")
        else:
            st.info(
                "Tested out — partial pass. You've already been taught the "
                "questions you missed above."
            )
    if st.session_state.pending_patch_decision is not None:
        _render_patch_decision_banner(session)

    if is_goal_complete(session, st.session_state.user_id):
        if st.button("View closing note"):
            st.session_state.current_stage = PipelineStage.GOAL_COMPLETION.value
            st.rerun()
        return

    if st.button("Continue to next topic"):
        all_topics = get_all_topics_for_user(session, st.session_state.user_id)
        not_started = [t for t in all_topics if t["status"] == NOT_STARTED_STATUS]
        if not not_started:
            st.error("No further topics found, but the goal is not yet complete.")
            return
        next_topic = min(not_started, key=lambda t: t["hierarchy_position"])
        st.session_state.current_topic_id = str(next_topic["id"])
        st.session_state.day_number_for_topic = 1
        st.session_state.carried_over_content = None
        st.session_state.day_content = None
        st.session_state.day_coaching_step_index = 0
        st.session_state.test_out_prompt_dismissed = False
        st.session_state.is_test_out = False
        st.session_state.last_completion_result = None
        st.session_state.last_test_out_full_pass = None
        st.session_state.pending_patch_decision = None
        st.session_state.current_stage = PipelineStage.DAY_BY_DAY_COACHING.value
        st.rerun()


def _render_verification() -> None:
    st.header("Verification")
    session = _get_db_session()
    topic = get_topic(session, st.session_state.current_topic_id)
    if topic is None:
        st.error("Current topic not found.")
        return

    if st.session_state.current_question_number > 5:
        if st.session_state.last_completion_result is None:
            user = get_user(session, st.session_state.user_id)
            assert user is not None
            days_expected = calculate_days_expected(user["available_time_per_week"])
            try:
                if st.session_state.is_test_out:
                    test_out_result: TestOutResult = _run_async(
                        complete_topic_test_out(
                            session,
                            st.session_state.user_id,
                            st.session_state.current_topic_id,
                            days_taken=st.session_state.day_number_for_topic,
                            days_expected=days_expected,
                            is_enrichment=topic["is_enrichment"],
                        )
                    )
                    result = test_out_result.completion
                    st.session_state.last_test_out_full_pass = test_out_result.full_pass
                else:
                    result = complete_topic_verification(
                        session,
                        st.session_state.user_id,
                        st.session_state.current_topic_id,
                        days_taken=st.session_state.day_number_for_topic,
                        days_expected=days_expected,
                        is_test_out=False,
                        is_enrichment=topic["is_enrichment"],
                    )
            except _STAGE_EXCEPTIONS as exc:
                st.error(f"Completing verification failed: {exc}")
                return
            st.session_state.last_completion_result = result
            st.session_state.pending_patch_decision = result.pending_patch_decision
        _advance_after_topic_completion(session)
        return

    question_number = st.session_state.current_question_number
    _render_progress_dots(question_number)

    if st.session_state.verification_slot_state is None:
        try:
            first_slot_state = _run_async(
                begin_verification_question(
                    st.session_state.current_topic_id,
                    question_number,
                    st.session_state.verification_source_material,
                    st.session_state.verification_source_url,
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Question generation failed: {exc}")
            return
        st.session_state.verification_slot_state = first_slot_state

    state: VerificationSlotState = st.session_state.verification_slot_state

    with st.container(border=True, key="ns-card-verification-question"):
        _render_kicker(f"Question {question_number} of 5")
        st.write(state.current_question.question_text)
        _render_citations(
            [{"title": "Source material", "url": state.current_question.source_url}]
        )

    if state.resolved:
        if state.credit == FULL_CREDIT:
            st.success("✓ Correct — full credit")
        else:
            st.warning("◐ Partial credit — here's what you missed:")
            if state.taught_answer_message:
                st.write(state.taught_answer_message)
        if st.button("Next question", type="primary"):
            st.session_state.current_question_number += 1
            st.session_state.verification_slot_state = None
            st.rerun()
        return

    answer = st.text_input(
        "Your answer",
        key=f"verification_answer_{question_number}_{state.attempt_number}",
    )
    if st.button("Submit answer", type="primary"):
        try:
            new_state = _run_async(
                submit_verification_answer(
                    state,
                    answer,
                    session,
                    st.session_state.verification_source_material,
                    st.session_state.verification_source_url,
                    is_test_out=st.session_state.is_test_out,
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Answer submission failed: {exc}")
            return
        st.session_state.verification_slot_state = new_state
        st.rerun()


def _render_goal_completion() -> None:
    st.header("Goal Completion")
    session = _get_db_session()

    if st.session_state.closing_note is None:
        try:
            note = _run_async(generate_closing_note(session, st.session_state.user_id))
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Closing note generation failed: {exc}")
            return
        st.session_state.closing_note = note

    note = st.session_state.closing_note
    st.write(note.note_text)
    if note.demonstrated_strengths:
        st.write("Demonstrated strengths: " + ", ".join(note.demonstrated_strengths))
    if note.suggested_next_steps:
        st.write("Suggested next steps: " + ", ".join(note.suggested_next_steps))
    if note.deferred_patch_notes:
        st.write(
            f"{len(note.deferred_patch_notes)} market-update note(s) still "
            "pending review."
        )


_STAGE_RENDERERS = {
    PipelineStage.LANDING.value: _render_landing,
    PipelineStage.INTAKE.value: _render_intake,
    PipelineStage.CLARIFY_GATE.value: _render_clarify_gate,
    PipelineStage.RESEARCH_GROUNDING.value: _render_research_grounding,
    PipelineStage.OUTLINE_CREATION.value: _render_outline_creation,
    PipelineStage.OUTLINE_CONFIRMATION.value: _render_outline_confirmation,
    PipelineStage.DAY_BY_DAY_COACHING.value: _render_day_by_day_coaching,
    PipelineStage.VERIFICATION.value: _render_verification,
    PipelineStage.GOAL_COMPLETION.value: _render_goal_completion,
}


def main() -> None:
    """Start Project North Star's Streamlit application."""
    _init_session_state()
    _maybe_check_stale_roles()
    st.markdown(
        "<h1 style='text-align: center;'>North Star</h1>", unsafe_allow_html=True
    )
    _STAGE_RENDERERS[st.session_state.current_stage]()


if __name__ == "__main__":
    main()
