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
- `clarify_turn`: the most recent `ClarifyGateTurn` returned by
  `begin_clarify_gate`/`advance_clarify_gate` — carries the loop state
  (`gate_state`/`context`) needed to advance the next real user reply.
- `clarify_conversation`: the Clarify Gate's displayed chat history, a
  `list[{"role": "agent" | "user", "content": str}]` — also the exact
  shape `advance_clarify_gate`'s `conversation` parameter expects
  (`role="agent"` is load-bearing: `_last_agent_message` in
  `agents/research_outline_agent.py` looks for that literal string).
- `resolved_role`: the role name Research/Grounding and everything
  downstream operates on.
- `grounding_result`: `ground_role`'s return value (`LiveGroundingResult`
  / `CachedFallbackResult` / `GeneralKnowledgeFloorResult`).
- `outline_topics`: the pre-persistence `list[InitialOutlineTopic]` from
  `create_initial_outline` — kept only because `begin_outline_confirmation`
  needs that exact dataclass shape, not the persisted dict rows.
  `persisted_topics`: the same outline after `insert_outline_topics`
  (real `id`s/`hierarchy_position`s).
- `outline_confirmation_turn`: the most recent `OutlineConfirmationTurn`
  from `begin_outline_confirmation`/`handle_review_turn` — carries
  `state`/`topics` needed to advance the next real review turn; `topics`
  is also the single source of truth for the outline list rendered
  alongside the chat.
- `outline_confirmation_conversation`: the Outline Confirmation chat
  history, same shape as `clarify_conversation`.
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
- `day_coaching_step_index`: which Day-by-Day Coaching section (Summary,
  Theory, ...) is the furthest one revealed so far for the current day —
  a stepped, "Next"-button reveal rather than showing every section at
  once. Reset to 0 whenever `day_content` is reset to `None` (a new day
  or a new topic).
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
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from agents.coaching_pace_agent import (
    FULL_CREDIT,
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
    reset_gemini_client_for_new_event_loop,
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
from security.input_gate import ClarifyGateStage, OutlineReviewAction
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
    """The 8 pipeline stages a user navigates through (PRD §7.1), plus
    `LANDING` — a new, purely presentational entry point ahead of Intake
    (the visual-pass task), not a PRD-numbered pipeline stage of its own.
    Pace tracking / patch-notes / enrichment are intentionally absent —
    see module docstring.
    """

    LANDING = "landing"
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

    Always resets the memoized Gemini client afterward (success or
    failure) — see `reset_gemini_client_for_new_event_loop`'s own
    docstring for the live-reproduced "Event loop is closed" bug this
    closes: the client's cached async transport is bound to *this* call's
    loop, which is about to close, so it must not be reused by the next
    `_run_async` call's own fresh loop.
    """
    try:
        return asyncio.run(coro)
    finally:
        reset_gemini_client_for_new_event_loop()


def _init_session_state() -> None:
    """Populate every `st.session_state` key this module uses, if not
    already present — see module docstring for the full shape. Runs at
    the top of every rerun; a no-op for keys already set.
    """
    defaults: dict[str, Any] = {
        "db_session": None,
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


# --- Shared presentation helpers (visual-pass task, native-Streamlit) ---
#
# Pure presentation, no session-state, no business logic. A prior version
# of this module injected a custom CSS color-token system (a from-scratch
# palette as raw `<style>`). That caused a real contrast bug: it forced
# every page's *background* to a fixed light color without also forcing
# every element's *text* color, so plain text left on Streamlit's own
# theme-dependent default color could render near-white-on-near-white
# whenever the active Streamlit theme was dark. Per product decision,
# this reverts to Streamlit's default theme entirely rather than patching
# the custom palette in place — every helper below renders through a
# native Streamlit component (`st.caption`/`st.info`/`st.success`/
# `st.warning`), each of which Streamlit itself keeps legible against
# whatever theme is active, not a fixed hex value this module owns.


def _domain_from_url(url: str) -> str:
    """Presentation-only: the host portion of `url`, for a compact
    citation display. Falls back to the full URL if it can't be parsed —
    never raises, since a citation must always be shown, not dropped.
    """
    return urlparse(url).netloc or url


def _render_kicker(label: str) -> None:
    """A small section label (e.g. "Summary", "Theory") — native
    `st.caption`, Streamlit's own muted small-text component.
    """
    st.caption(f"**{label.upper()}**")


def _render_stamp(confidence: str, source_url: str, *, sample: bool = False) -> None:
    """Render the Confidence Stamp — this product's signature element —
    via native `st.info`, so it always matches Streamlit's own theme
    rather than a fixed custom color. Pure presentation over an
    already-computed/fetched confidence tier and source_url — never
    computes or validates either itself (that's `security/output_guard.py`
    and the confidence ladder, upstream of this function).

    `sample=True` prefixes the message with an explicit "SAMPLE" label on
    the same line as the stamp itself, so a demo viewer can never mistake
    the Landing page's illustrative mockup for a real grounded result —
    this product's whole pitch is honesty about real confidence, so the
    one place this stamp isn't real data must say so unambiguously, not
    just be implied by page context.
    """
    domain = _domain_from_url(source_url)
    prefix = "SAMPLE — " if sample else ""
    st.info(f"{prefix}CONFIDENCE: {confidence.upper()} · source: {domain}")


def _render_citations(links: list[dict[str, str]]) -> None:
    """One caption line per citation — title (or domain, if untitled) as
    an explicit markdown link to the citation's real, full `url`, plus
    the bare domain for a quick trust glance — native `st.caption`, no
    custom color.

    A real, reported bug: this previously rendered only the bare domain
    as plain text (`_domain_from_url(url)` discards the real path/query),
    relying on whichever text happened to look link-shaped to Streamlit's
    markdown renderer to become clickable at all — inconsistent (some
    domains autolink, some don't) and, even when it did autolink, always
    pointed at that site's homepage rather than the actual article/video,
    since the real path was already thrown away before rendering. An
    explicit `[label](url)` link to the real `url` fixes both: always
    clickable, always the correct page (e.g. the actual YouTube video,
    not youtube.com's homepage).
    """
    for link in links:
        url = link["url"]
        domain = _domain_from_url(url)
        title = (link.get("title") or "").strip()
        label = title if title else domain
        st.caption(f"Source: [{label}]({url}) — {domain}")


def _render_progress_dots(current_question_number: int, total: int = 5) -> None:
    """The Verification screen's step tracker — plain markdown text (done
    questions struck through, the current one bolded) via `st.caption`,
    no custom color needed to distinguish state.
    """
    parts = []
    for n in range(1, total + 1):
        if n < current_question_number:
            parts.append(f"~~{n}~~")
        elif n == current_question_number:
            parts.append(f"**{n}**")
        else:
            parts.append(str(n))
    st.caption("Question " + "  ·  ".join(parts) + f" of {total}")


def _render_attempt_chips(attempt_number: int, max_attempts: int = 3) -> None:
    """Attempt-number markers (1/2/3) for the current verification
    question — same plain-markdown treatment as `_render_progress_dots`.
    """
    parts = []
    for n in range(1, max_attempts + 1):
        if n < attempt_number:
            parts.append(f"~~attempt {n}~~")
        elif n == attempt_number:
            parts.append(f"**attempt {n}**")
        else:
            parts.append(f"attempt {n}")
    st.caption("  ·  ".join(parts))


# --- Landing (visual-pass task — new entry point ahead of Intake) -------


def _render_landing() -> None:
    st.subheader(
        "Grounded, verified, adaptive career learning coaching — for "
        "people without a bootcamp or a mentor."
    )
    st.write(
        "No bootcamp. No mentor. Just a cited, adaptive study plan built "
        "from what companies are actually hiring for right now — every "
        "topic and every check-in traces back to a real source, "
        "labeled with how confident we actually are in it."
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
    _render_kicker("What a real citation looks like")
    _render_stamp("high", "https://www.example-job-board.com/listing", sample=True)

    st.write("")
    if st.button("Begin", type="primary"):
        st.session_state.current_stage = PipelineStage.INTAKE.value
        st.rerun()


# --- Intake (PRD §7.1 stage 1 / §7.2) ------------------------------------


def _render_intake() -> None:
    """Restyled per the approved visual-pass design plan — same fields,
    same validation, same `create_user` call below, only their layout
    changed (columns for the paired fields, a bordered "Destination" card
    around the goal/time fields since everything downstream depends on
    them). No widget was added, removed, or changed in type.
    """
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


# --- Clarify Gate (PRD §7.1 stage 2 / §7.2) ------------------------------


def _render_clarify_gate() -> None:
    """The real bounded Clarify Gate loop (PRD §7.2): a `st.chat_message`
    history plus `st.chat_input` for the next reply, with every turn
    computed by `begin_clarify_gate`/`advance_clarify_gate` — this
    function owns no narrowing/proposal/acceptance logic of its own, only
    rendering and round-tripping `st.session_state`.

    Dispatches on `turn.gate_state.stage` exactly as
    `security/input_gate.py`'s `ClarifyGateState` defines it:
    NARROWING/PROPOSE_BEST_GUESS/EXPLAIN_ROLE all collect the next free-text
    reply (they differ only in what `advance_clarify_gate` does with it
    internally, not in what this UI needs to do); RESOLVED and EXITED are
    both terminal and stop offering `st.chat_input` (CLAUDE.md guardrail
    #8 — the round bound must never be bypassable from the UI, so once a
    terminal stage is reached, `advance_clarify_gate` is never called
    again). ACCEPT_OWN_WORDS is asserted unreachable here: `advance_clarify_gate`
    always resolves it (to RESOLVED or EXITED) within the same call that
    entered it, so a turn's returned stage can never actually be it.
    """
    st.header("Clarify Gate")

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
        if st.button("Continue to Research & Market Grounding"):
            assert turn.resolved_role is not None  # guaranteed by RESOLVED
            st.session_state.resolved_role = turn.resolved_role
            st.session_state.current_stage = PipelineStage.RESEARCH_GROUNDING.value
            st.rerun()
        return

    if stage is ClarifyGateStage.EXITED:
        if st.button("Continue to Research & Market Grounding"):
            # PRD §7.2's zero-market-signal exit: resolved_role stays None
            # on `turn` (no role was accepted), but the gate already
            # confirmed there's no live signal for the user's own original
            # words specifically (that's how EXITED was reached). Routing
            # to Research & Market Grounding on that same goal is
            # deliberate, not a bug: `_render_research_grounding` already
            # renders the "no outline can be built" floor-rung outcome and
            # offers no way past it, so every confidence-ladder result
            # still surfaces through the one stage built to display it,
            # rather than this function duplicating that rendering. The
            # cost is one redundant live grounding call for an outcome
            # already known.
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


def _core_and_emerging_skills(
    result: object,
) -> tuple[list[ValidatedGroundedContent], list[ValidatedGroundedContent]] | None:
    """The core/emerging skill split `create_initial_outline`/
    `regenerate_outline_with_addition` both need, from whichever grounding
    rung `st.session_state.grounding_result` actually landed on. Returns
    `None` if `result` carries no grounded skill list at all (the
    general-knowledge-only floor) — shared by Outline Creation and the
    Outline Confirmation addition-regeneration path, so the isinstance
    split lives in exactly one place, not duplicated between them.
    """
    if isinstance(result, LiveGroundingResult):
        # Degenerate core/emerging split — matches the established
        # workaround already used at src/cron/refresh_roles.py's
        # upsert_role call: LiveGroundingResult has no real split.
        return result.skills, []
    if isinstance(result, CachedFallbackResult):
        return result.core_skills, result.emerging_skills
    return None


def _render_outline_creation() -> None:
    st.header("Outline Creation")
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


# --- Outline Confirmation (PRD §7.1 stage 5 / §7.5) ---------------------


def _advance_to_day_one() -> None:
    """Shared terminal action for Outline Confirmation (PRD §7.5): pick
    the lowest-`hierarchy_position` not-started persisted topic and move
    to Day-by-Day Coaching — identical target/lookup this stage's stub
    always used, now reached from either `CONFIRM` or bound exhaustion.
    """
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
    """Ground the raw addition request in `user_message`
    (`ground_addition_request`) and fold it into the outline
    (`regenerate_outline_with_addition`), persisting the regenerated
    hierarchy exactly as Outline Creation does (`insert_outline_topics`
    already supports replacing a not-yet-started outline wholesale — see
    that function's own docstring). Closes the gap `handle_review_turn`
    used to flag as unaddressed (Architecture §10/PRD §11 item 6).

    Never raises past this function — a grounding failure (Tavily has
    nothing usable for the extracted skill name, or any `_STAGE_
    EXCEPTIONS`) degrades to `turn` unchanged plus a message explaining
    the addition could not be grounded, rather than crashing the whole
    review turn or silently dropping the request.
    """
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
    """The real bounded Outline Confirmation loop (PRD §7.5): the current
    outline shown alongside a `st.chat_message`/`st.chat_input` review
    conversation, every turn computed by `begin_outline_confirmation`/
    `handle_review_turn` — this function renders what those return, plus
    one addition-specific follow-up: an `ADDITION_REQUEST` turn is passed
    to `_regenerate_outline_for_addition`, which grounds the raw request
    and regenerates the outline (see that function's own docstring). This
    function still owns no classification/round-bound logic of its own —
    that follow-up is a mechanical consequence of a round already
    consumed by `handle_review_turn`, not a second review round.
    """
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


def _day_coaching_steps(
    content: DayContent,
) -> list[tuple[str, str, Callable[[], None]]]:
    """The Day-by-Day Coaching sections to reveal, in order, as
    (container key, kicker label, render fn) — `_render_day_by_day_
    coaching` shows one more of these each time "Next" is clicked
    (`day_coaching_step_index`), rather than all at once. Hands-on/Review
    are conditionally present, exactly as the old all-at-once rendering
    already conditioned them on `content.hands_on_exercise`/
    `content.review_prompt`.
    """

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
        st.session_state.day_coaching_step_index = 0

    content: DayContent = st.session_state.day_content

    badge_col, stamp_col = st.columns([1, 2])
    with badge_col:
        _render_kicker(f"Day {st.session_state.day_number_for_topic} of this topic")
    with stamp_col:
        _render_stamp(topic["confidence"], topic["source_url"])

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
        st.session_state.day_coaching_step_index = 0
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
        # Deliberately a plain citation, not `_render_stamp` — a
        # verification question carries a `source_url` but no confidence
        # tier of its own (that lives on the topic, not the question), so
        # labeling it "CONFIDENCE: ..." here would fabricate a value this
        # product's whole premise is to never fabricate.
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

    _render_attempt_chips(state.attempt_number)
    answer = st.text_input(
        f"Your answer (attempt {state.attempt_number} of 3)",
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
    st.title("North Star")
    _STAGE_RENDERERS[st.session_state.current_stage]()


if __name__ == "__main__":
    main()
