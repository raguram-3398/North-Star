"""Application entry point — Streamlit orchestration skeleton wiring
Project North Star's full pipeline (PRD §7.1) end to end.

This module wires already-built, already-tested functions from `agents/`,
`security/`, `pace/`, `outline/`, `patches/`, and `data/` together in
sequence. It owns no decision logic of its own — every confidence
branch, bound-count, drift check, and patch/enrichment trigger already
lives in those modules; this file only calls them in the right order and
holds `st.session_state` between Streamlit reruns.

**PipelineStage** (PRD §7.1's 8 numbered stages the user actually
navigates through). Pace tracking, patch-notes, and enrichment are
deliberately NOT separate stages here — they fire as side effects inside
`complete_topic_verification` and are only read back for display during
the Day-by-Day Coaching / Verification stages, never triggered directly by
this module (CLAUDE.md guardrail #10: never reimplement a decision
already made elsewhere).

**st.session_state shape** (initialized by `_init_session_state`, the only
place these keys are declared):

- `db_session`: the one SQLAlchemy `Session` for this browser session,
  created lazily via `db.connection.get_session()` and cached here rather
  than recreated every rerun (Streamlit reruns the whole script on every
  interaction) — every `data/*` function still commits its own
  transaction, so a single long-lived `Session` object is safe to reuse
  across reruns the same way a real request-scoped session would be.
- `current_stage`: the *string* `.value` of the `PipelineStage` currently
  being rendered — the single source of truth for which stage function
  `main()` dispatches to. Stored as a plain string, never the `Enum`
  member object itself: Streamlit re-executes this entire script's
  top-level code (including the `class PipelineStage(Enum)` statement)
  from scratch on every single rerun when `main.py` is the literal
  `streamlit run` target, which mints a brand-new, distinct `PipelineStage`
  class object each time — an enum member persisted in `session_state`
  from a *previous* rerun's class would then fail every equality/dict-key
  check against the *current* rerun's freshly-defined class, a real
  `KeyError` crash confirmed live via `streamlit.testing.v1.AppTest`
  during this task (see this task's report). Plain strings have no such
  identity problem, so every stage transition below assigns
  `PipelineStage.X.value`, never `PipelineStage.X` itself.
- `user_id`: the persisted `users.id` (str), set once Intake completes.
- `stated_goal`: the raw Intake goal text, threaded into the Clarify Gate.
- `clarify_turn`: the `ClarifyGateTurn` from the one `begin_clarify_gate`
  call (Clarify Gate is stubbed this pass — see that stage's own comment).
- `resolved_role`: the role name Research/Grounding and everything
  downstream operates on.
- `grounding_result`: `ground_role`'s return value (`LiveGroundingResult`
  / `CachedFallbackResult` / `GeneralKnowledgeFloorResult`).
- `outline_topics`: the pre-persistence `list[InitialOutlineTopic]` from
  `create_initial_outline` — kept only because `begin_outline_confirmation`
  needs that exact dataclass shape, not the persisted dict rows.
  `persisted_topics`: the same outline after `insert_outline_topics`
  (real `id`s/`hierarchy_position`s).
- `outline_confirmation_turn`: the `OutlineConfirmationTurn` from the one
  `begin_outline_confirmation` call (also stubbed this pass).
- `current_topic_id`: the outline topic currently being taught/verified —
  looked up fresh from the DB each time (never a stale index), since
  `maybe_trigger_enrichment`/`maybe_deliver_patch` can insert new topics
  mid-flow and shift `hierarchy_position`s.
- `day_number_for_topic`: how many days of content have been generated for
  the *current* topic so far (starts at 1, increments once per spillover
  day — see `DayContent.remaining_content`) — this is also `days_taken`
  passed to `complete_topic_verification` once verification starts.
- `carried_over_content`: `generate_day_content`'s spillover input for the
  next day of the same topic, or `None`.
- `day_content`: the current day's generated `DayContent`, or `None`
  before it's been generated for this day.
- `verification_source_material` / `verification_source_url`: the single
  source pair (computed once per topic, when its content is generated —
  see `_build_verification_source`) that all 5 verification question
  slots for this topic are anchored to.
- `current_question_number`: which of the 5 verification slots (1-5) is
  active; `6` signals all slots resolved.
- `verification_slot_state`: the current slot's `VerificationSlotState`,
  or `None` before its first question has been generated.
- `last_completion_result`: the `TopicCompletionResult` from the most
  recent `complete_topic_verification` call, read back for display only.
- `closing_note`: the `ClosingNote` from the one `generate_closing_note`
  call, once Goal Completion is reached.
"""

import asyncio
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from agents.coaching_pace_agent import (
    DayContent,
    TopicCompletionResult,
    VerificationSlotState,
    begin_verification_question,
    complete_topic_verification,
    generate_closing_note,
    generate_day_content,
    is_goal_complete,
    record_day_content,
    submit_verification_answer,
)
from agents.research_outline_agent import (
    ClarifyGateTurn,
    LiveGroundingResult,
    begin_clarify_gate,
    begin_outline_confirmation,
    create_initial_outline,
    ground_role,
)
from data.grounding_fallback import CachedFallbackResult, GeneralKnowledgeFloorResult
from data.outline_topics import (
    get_all_topics_for_user,
    get_topic,
    get_topics_in_group,
    insert_outline_topics,
)
from data.users import create_user, get_user, set_resolved_role
from db.connection import get_session
from security.output_guard import ConfidenceTier
from utils.exceptions import (
    ConfidenceValidationError,
    GeminiCallError,
    GroundingSourceCallError,
    HimalayasParseError,
    TavilyParseError,
)

# Loaded here (not left to the operator to `source .env` manually, unlike
# src/cron/refresh_roles.py's __main__ block, which relies on GH Actions
# already injecting secrets as real env vars) — this is the first real
# end-user-facing entry point in this codebase, not a CI/cron context.
load_dotenv()

NOT_STARTED_STATUS = "not_started"


class PipelineStage(Enum):
    """The 8 pipeline stages a user navigates through (PRD §7.1). Pace
    tracking / patch-notes / enrichment are intentionally absent — see
    module docstring.
    """

    INTAKE = "intake"
    CLARIFY_GATE = "clarify_gate"
    RESEARCH_GROUNDING = "research_grounding"
    OUTLINE_CREATION = "outline_creation"
    OUTLINE_CONFIRMATION = "outline_confirmation"
    DAY_BY_DAY_COACHING = "day_by_day_coaching"
    VERIFICATION = "verification"
    GOAL_COMPLETION = "goal_completion"


# Judgment call, flagged loudly — a genuine spec gap discovered while
# wiring this task, not a pre-existing constant: no function anywhere in
# this codebase computes `days_expected` (pace/calculator.py's own
# docstring: "supplied by the caller as already derived from the user's
# own established baseline" — PRD §7.8 explicitly defers that baseline
# calculation to "a future days_expected calculation," not built by any
# prior task). Rather than inventing an unbaked formula, this orchestration
# skeleton uses a flat 1-day-per-topic baseline: `days_taken`
# (`day_number_for_topic`) is derived honestly from the existing spillover
# mechanism (one real day per generate_day_content call for the same
# topic), so a topic that spills over 2+ days already registers as
# genuinely "behind" via timing_ratio; a topic resolved same-day reads as
# exactly on-baseline (ratio 1.0, timing contributes nothing — see
# pace/calculator.py's TIMING_OUTLIER_THRESHOLD). See this task's report
# for why this was decided rather than asked about, and why it's flagged
# as revisable, not settled.
DAYS_EXPECTED_PER_TOPIC = 1

# Broad but named (not bare `except:`) — every exception a stage's real
# calls can raise, per each function's own documented contract. Caught at
# the orchestration boundary only, exactly the way a UI must degrade
# gracefully on a live-call failure without crashing the whole app.
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
    """Run one async call to completion from Streamlit's synchronous
    script execution. Each Streamlit rerun is its own fresh top-to-bottom
    script run, so a fresh event loop per call is correct here, not a
    workaround.
    """
    return asyncio.run(coro)


def _init_session_state() -> None:
    """Populate every `st.session_state` key this module uses, if not
    already present — see module docstring for the full shape. Runs at
    the top of every rerun; a no-op for keys already set.
    """
    defaults: dict[str, Any] = {
        "db_session": None,
        "current_stage": PipelineStage.INTAKE.value,
        "user_id": None,
        "stated_goal": None,
        "clarify_turn": None,
        "resolved_role": None,
        "grounding_result": None,
        "outline_topics": None,
        "persisted_topics": None,
        "outline_confirmation_turn": None,
        "current_topic_id": None,
        "day_number_for_topic": 1,
        "carried_over_content": None,
        "day_content": None,
        "verification_source_material": None,
        "verification_source_url": None,
        "current_question_number": 1,
        "verification_slot_state": None,
        "last_completion_result": None,
        "closing_note": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_db_session() -> Session:
    if st.session_state.db_session is None:
        st.session_state.db_session = get_session()
    session: Session = st.session_state.db_session
    return session


# --- Intake (PRD §7.1 stage 1 / §7.2) ------------------------------------


def _render_intake() -> None:
    st.header("Intake")
    with st.form("intake_form"):
        background = st.text_area("Background")
        current_job = st.text_input("Current job")
        years_experience = st.number_input(
            "Years of experience", min_value=0, step=1, value=0
        )
        prior_self_study = st.text_area("Prior self-study (specific, not yes/no)")
        goal = st.text_input("What tech role or skill do you want to learn?")
        available_time_per_week = st.number_input(
            "Available time per week (hours)", min_value=1, step=1, value=10
        )
        submitted = st.form_submit_button("Submit")

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


# --- Clarify Gate (PRD §7.1 stage 2 / §7.2) ------------------------------


def _render_clarify_gate() -> None:
    st.header("Clarify Gate")
    st.caption(
        "Stand-in for this pass — the real bounded narrowing/proposal/"
        "explanation loop (security/input_gate.py's ClarifyGateState, "
        "advance_clarify_gate) is a later UI task. begin_clarify_gate is "
        "called once for real and its output is shown below, but "
        "'Accept and continue' always proceeds with the stated goal "
        "exactly as typed, regardless of what the gate classified it as."
    )

    if st.session_state.clarify_turn is None:
        try:
            first_turn = _run_async(begin_clarify_gate(st.session_state.stated_goal))
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Clarify Gate failed: {exc}")
            return
        st.session_state.clarify_turn = first_turn

    turn: ClarifyGateTurn = st.session_state.clarify_turn
    st.write(turn.message)

    if st.button("Accept and continue"):
        st.session_state.resolved_role = st.session_state.stated_goal
        st.session_state.current_stage = PipelineStage.RESEARCH_GROUNDING.value
        st.rerun()


# --- Research & Market Grounding (PRD §7.1 stage 3 / §7.3) --------------


def _render_research_grounding() -> None:
    st.header("Research & Market Grounding")
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
        st.write(f"Confidence: {result.confidence.value}")
        st.info(
            "No outline can be built from this floor rung — per PRD §7.3 "
            "and specs/scenarios/high_risk_flows.feature's \"No source "
            'returns usable data" scenario, no outline item is ever '
            "created here."
        )
        return

    if isinstance(result, CachedFallbackResult):
        role_confidence = ConfidenceTier.CACHED_LOW
        st.write(
            f"Confidence: {role_confidence.value} (cached fallback, "
            f"last updated {result.last_updated}, stale={result.is_stale})"
        )
    else:
        assert isinstance(result, LiveGroundingResult)
        role_confidence = result.confidence
        st.write(f"Confidence: {role_confidence.value}")

    if st.button("Continue to Outline Creation"):
        set_resolved_role(
            session,
            st.session_state.user_id,
            st.session_state.resolved_role,
            role_confidence.value,
        )
        st.session_state.current_stage = PipelineStage.OUTLINE_CREATION.value
        st.rerun()


# --- Outline Creation (PRD §7.1 stage 4 / §7.4) -------------------------


def _render_outline_creation() -> None:
    st.header("Outline Creation")
    session = _get_db_session()
    result = st.session_state.grounding_result

    if st.session_state.outline_topics is None:
        if isinstance(result, LiveGroundingResult):
            # Degenerate core/emerging split — matches the established
            # workaround already used at src/cron/refresh_roles.py's
            # upsert_role call: LiveGroundingResult has no real split.
            core_skills, emerging_skills = result.skills, []
        elif isinstance(result, CachedFallbackResult):
            core_skills, emerging_skills = result.core_skills, result.emerging_skills
        else:
            st.error("Cannot create an outline without a grounded skill list.")
            return

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


# --- Outline Confirmation (PRD §7.1 stage 5 / §7.5) ---------------------


def _render_outline_confirmation() -> None:
    st.header("Outline Confirmation")
    st.caption(
        "Stand-in for this pass — the real bounded review loop "
        "(security/input_gate.py's OutlineConfirmationState, "
        "handle_review_turn/regenerate_outline_with_addition) is a later "
        "UI task. begin_outline_confirmation is called once for real."
    )

    if st.session_state.outline_confirmation_turn is None:
        try:
            turn = _run_async(
                begin_outline_confirmation(
                    st.session_state.resolved_role, st.session_state.outline_topics
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Outline Confirmation failed: {exc}")
            return
        st.session_state.outline_confirmation_turn = turn

    st.write(st.session_state.outline_confirmation_turn.message)

    if st.button("Confirm and start Day 1"):
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


# --- Day-by-Day Coaching (PRD §7.1 stage 6 / §7.6) ----------------------


def _build_verification_source(
    content: DayContent, topic: dict[str, Any]
) -> tuple[str, str]:
    """Judgment call, flagged — no real caller of `begin_verification_question`
    existed before this task to establish this convention: verification
    questions must be anchored to the actual teaching material shown
    today (`content.theory_framing` + each real `theory_links` entry's
    content), never the topic's own `source_url` (market-grounding
    provenance, per `generate_day_content`'s own docstring — "a fresh,
    distinct search for genuine teaching material", not a learning
    resource).

    `source_url` is the first (highest-relevance) theory link's URL,
    since `generate_questions` accepts exactly one `source_url` per call.
    Falls back to the topic's own market-grounding `source_url` only if
    Tavily's theory search returned nothing at all (a real, documented
    possibility — data/tavily_parser.py's own finding that Tavily can
    legitimately return zero usable results) — still a real, valid URL
    for this topic, just a different provenance layer, rather than
    hard-blocking the demo.
    """
    material_parts = [content.theory_framing]
    material_parts.extend(
        link["content"] for link in content.theory_links if link.get("content")
    )
    topic_source_material = "\n\n".join(material_parts)
    source_url = (
        content.theory_links[0]["url"] if content.theory_links else topic["source_url"]
    )
    return topic_source_material, source_url


def _render_day_by_day_coaching() -> None:
    st.header("Day-by-Day Coaching")
    session = _get_db_session()
    topic = get_topic(session, st.session_state.current_topic_id)
    if topic is None:
        st.error("Current topic not found.")
        return

    st.subheader(topic["topic_name"])

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

    content: DayContent = st.session_state.day_content
    st.caption(f"Day {st.session_state.day_number_for_topic} of this topic")
    st.write("**Summary**")
    st.write(content.summary)
    st.write("**Theory**")
    st.write(content.theory_framing)
    for link in content.theory_links:
        st.write(f"- [{link['title'] or link['url']}]({link['url']})")
    if content.hands_on_exercise:
        st.write("**Hands-on exercise**")
        st.write(content.hands_on_exercise)
    if content.review_prompt:
        st.write("**Review**")
        st.write(content.review_prompt)
    st.write("**Reflection**")
    st.write(content.reflection_prompt)
    st.write("**Preview**")
    st.write(content.preview)

    if content.remaining_content:
        st.info(
            "Some content didn't fit today's time budget and will carry "
            "over to the next day."
        )
        if st.button("Continue to next day (same topic)"):
            st.session_state.day_number_for_topic += 1
            st.session_state.carried_over_content = content.remaining_content
            st.session_state.day_content = None
            st.rerun()
        return

    if st.button("Start Verification"):
        source_material, source_url = _build_verification_source(content, topic)
        st.session_state.verification_source_material = source_material
        st.session_state.verification_source_url = source_url
        st.session_state.current_question_number = 1
        st.session_state.verification_slot_state = None
        st.session_state.current_stage = PipelineStage.VERIFICATION.value
        st.rerun()


# --- Verification (PRD §7.1 stage 7 / §7.7) -----------------------------


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
            "Market-update topic delivered: "
            f"{result.delivered_patch_topic['topic_name']}"
        )

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
        st.session_state.last_completion_result = None
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
            try:
                result = complete_topic_verification(
                    session,
                    st.session_state.user_id,
                    st.session_state.current_topic_id,
                    days_taken=st.session_state.day_number_for_topic,
                    days_expected=DAYS_EXPECTED_PER_TOPIC,
                    is_test_out=False,
                    is_enrichment=topic["is_enrichment"],
                )
            except _STAGE_EXCEPTIONS as exc:
                st.error(f"Completing verification failed: {exc}")
                return
            st.session_state.last_completion_result = result
        _advance_after_topic_completion(session)
        return

    question_number = st.session_state.current_question_number
    st.write(f"Question {question_number} of 5")

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
    st.write(state.current_question.question_text)

    if state.resolved:
        st.success(f"Resolved — credit: {state.credit}")
        if state.taught_answer_message:
            st.info(state.taught_answer_message)
        if st.button("Next question"):
            st.session_state.current_question_number += 1
            st.session_state.verification_slot_state = None
            st.rerun()
        return

    answer = st.text_input(
        f"Your answer (attempt {state.attempt_number} of 3)",
        key=f"verification_answer_{question_number}_{state.attempt_number}",
    )
    if st.button("Submit answer"):
        try:
            new_state = _run_async(
                submit_verification_answer(
                    state,
                    answer,
                    session,
                    st.session_state.verification_source_material,
                    st.session_state.verification_source_url,
                    is_test_out=False,
                )
            )
        except _STAGE_EXCEPTIONS as exc:
            st.error(f"Answer submission failed: {exc}")
            return
        st.session_state.verification_slot_state = new_state
        st.rerun()


# --- Goal Completion (PRD §7.1 stage 8 / §7.11) -------------------------


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
    st.title("North Star")
    _STAGE_RENDERERS[st.session_state.current_stage]()


if __name__ == "__main__":
    main()
