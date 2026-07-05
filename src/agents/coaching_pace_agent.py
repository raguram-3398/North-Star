"""Coaching & Pace Agent — reasoning/generation only.

Owns (Architecture_North_Star.md §3): day-by-day content generation
(summary, theory framing, hands-on exercise design, reflection prompts)
and goal-completion closing-note composition.

Calls as tools (deterministic, not owned — never reimplemented inline):
the Verification Question Generator Skill, pace/calculator.py,
data/progress_log.py, data/verification_log.py, data/pace_snapshots.py,
data/outline_topics.py.

Tools: Verification Skill, Tavily search (theory-material grounding
only — never Himalayas, never job-market grounding; that's Agent 1's
job), Postgres (progress log, verification attempts, pace snapshots,
outline status — via gated write paths), roles_cache (read-only, for
closing note + enrichment source).

**Scope for this task**: the 7-step hands-on-eligible day structure (and
its conceptual-only, steps-3-4-omitted variant), verification retry-cap
orchestration (exactly 3 attempts, half-credit teach-and-de-escalate at
cap), pace-signal computation + `pace_snapshots` persistence once a
topic's 5 questions resolve, and test-out (verification-first — PRD's
day-by-day coaching section's exception). Explicitly NOT built here
(deliberately deferred to a later task, not stubbed with a guessed
shape): patch-note delivery/surfacing, goal-completion/closing-note
content.

**Acting on the pace signal (the sustained-drift-wiring task — closes
this module's own previously-flagged "detect_sustained_drift is not
called here at all" gap):** `complete_topic_verification`, immediately
after writing a topic's `pace_snapshots` entry (and only when
`is_enrichment=False` — enrichment topics never feed pace at all, PRD
§7.10's isolation rule), reads the user's full pace-snapshot history
(`data/pace_snapshots.py`'s `get_pace_snapshot_history`), recomputes each
row's combined pace signal (`calculate_combined_pace_signal` — not
reimplemented), and calls `pace/calculator.py`'s `detect_sustained_drift`
unmodified. `"ahead"` -> `maybe_trigger_enrichment` (PRD §7.10, selects an
unused `roles_cache` emerging skill and inserts it via
`outline/hierarchy.py`'s existing `insert_new_topic`). `"behind"` ->
`data/users.py`'s `extend_pacing` (a new `users.pace_extension_days`
column — see Architecture §5's "Resolved" block for why this schema
addition was needed). `"on_track"` -> no action. See the dedicated
"Resolved" block near `complete_topic_verification` below for the full
set of judgment calls.

**Test-out** (`complete_topic_test_out`, `generate_gap_study_content`)
reuses the identical `begin_verification_question`/
`submit_verification_answer` retry-cap machinery regular verification
uses — no separate retry cap or attempt-counting shape — and
`complete_topic_verification`'s existing pace-signal/completion path,
extended with an `is_test_out` flag so a full pass writes the schema's
distinct `completed_test_out` status rather than `completed`. Only a
partial pass (>=1 question resolved via the retry-cap teach-and-
de-escalate path) generates any content at all, and only
`generate_gap_study_content` — a new, purpose-built path scoped strictly
to the failed question(s), never `generate_day_content`'s 7-step
structure.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sqlalchemy.orm import Session
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    UsageLimitExceededError,
)
from tavily.errors import TimeoutError as TavilyTimeoutError

from agents.research_outline_agent import (
    EXTERNAL_CALL_TIMEOUT_SECONDS,
    _call_gemini_json,
    _get_tavily_client,
)
from data.grounding_fallback import CACHED_SOURCE_TYPE
from data.outline_topics import (
    COMPLETED_STATUS,
    COMPLETED_TEST_OUT_STATUS,
    get_all_topics_for_user,
    has_pending_enrichment_topic,
    insert_new_outline_topic,
    mark_topic_completed,
)
from data.pace_snapshots import get_pace_snapshot_history, write_pace_snapshot
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
from security.output_guard import validate_output_object
from utils.exceptions import GeminiCallError, GroundingSourceCallError

# Reuses agents/research_outline_agent.py's private Gemini-call helper
# directly, same flagged architectural seam as
# .agent/skills/verification_question_generator/generator.py (see
# Architecture §4's "Resolved" block and PRD §11 item #7) — this is now
# the *third* consumer of that not-yet-extracted shared infrastructure
# (the other two being research_outline_agent.py's own clarify-gate/
# outline functions and the Verification Skill), strengthening the case
# for eventually promoting it to a shared src/utils/ module. Still not
# attempted here, for the same reason as before: out of this task's
# scope, and it would mean touching already-tested, already-committed
# code as a side effect.

# The Verification Skill lives in .agent/skills/, outside the src/
# package (required for Antigravity workspace-manager recognition, per
# CLAUDE.md) — not importable via the normal editable-install path the
# way src/ is. Adding it to sys.path here (matching
# tests/test_verification_skill.py's identical pattern) is required for
# *any* real caller of the Skill, not just tests — a second data point
# (beyond the tests file) that this Skill's location creates real
# friction for legitimate importers, worth flagging alongside the
# existing seam note above.
_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / ".agent" / "skills"
if str(_SKILLS_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILLS_DIR))

from verification_question_generator.generator import (  # noqa: E402
    VerificationQuestion,
    generate_questions,
    grade_answer,
)

# --- Constants -------------------------------------------------------

MAX_VERIFICATION_ATTEMPTS = 3
FULL_CREDIT = 1.0
HALF_CREDIT = 0.5
NOT_YET_RESOLVED_CREDIT = 0.0

# Judgment call: "flash" tier — day-content generation is a bounded, once-
# per-day task (a handful of short fields), closer in shape to the
# clarify gate's per-turn calls than to outline-hierarchy sequencing's
# whole-curriculum reasoning. Independent constant, not imported from
# research_outline_agent.py or the Verification Skill, matching the
# precedent that each feature's model choice is its own explicit
# decision even when the value happens to match.
DAY_CONTENT_GEMINI_MODEL = "gemini-2.5-flash"

# Judgment call, flagged for review — not specified anywhere in PRD/
# Architecture: converting `users.available_time_per_week` (hours) into
# a per-day time budget needs an assumed study cadence. 5 (a
# Monday-through-Friday-shaped week) is the simplest, most common
# default; PRD never states whether weekends are included. Revisable if
# a different cadence is intended.
STUDY_DAYS_PER_WEEK = 5

# Judgment call, flagged for review — PRD §7.6 states the *principle*
# ("hands-on intensity ramps progressively within a topic-group as days
# progress, scaled to that group's size") but not a formula. Resolved
# directly for this task (see Architecture §3's "Resolved" block): the
# first day of any topic-group is conceptual-only (steps 3-4 omitted);
# hands-on intensity then scales linearly from the second day through
# the last day of the group, reaching full intensity (1.0) on the
# group's final day. A single-topic group (group_size == 1) is the one
# exception — see `compute_hands_on_intensity`'s docstring for why it's
# hands-on-eligible at full intensity from its only day, rather than
# conceptual-only forever.


def compute_hands_on_intensity(position_in_group: int, group_size: int) -> float:
    """Compute how much hands-on depth today's content should have, on a
    0.0 (none) to 1.0 (full depth) scale, per this task's resolved
    ramping rule (see module-level judgment-call note above):
    `(position_in_group - 1) / (group_size - 1)`, linear across the
    group.

    Edge case: a single-topic group (`group_size == 1`) has no "later
    days" to ramp into — returning 0.0 (permanently conceptual-only)
    would mean that skill never gets any hands-on practice at all.
    Returns 1.0 (full intensity, hands-on-eligible immediately) instead
    — a deliberate, flagged judgment call, not an oversight.

    Raises `ValueError` if `position_in_group` is not in
    `[1, group_size]`.
    """
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
    """Whether today (steps 3-4 omitted) is conceptual-only, per PRD
    §7.6 — true exactly when `compute_hands_on_intensity` is 0.0 (the
    first day of a multi-day topic-group).
    """
    return compute_hands_on_intensity(position_in_group, group_size) == 0.0


def convert_weekly_hours_to_daily_minutes(available_time_per_week_hours: int) -> int:
    """Convert `users.available_time_per_week` (hours) into today's time
    budget in minutes, per this task's resolved cadence assumption
    (`STUDY_DAYS_PER_WEEK`, flagged above). Raises `ValueError` if
    `available_time_per_week_hours` is not positive.
    """
    if available_time_per_week_hours <= 0:
        raise ValueError(
            "available_time_per_week_hours must be positive, got "
            f"{available_time_per_week_hours}"
        )
    return round(available_time_per_week_hours * 60 / STUDY_DAYS_PER_WEEK)


# --- Day content generation (PRD §7.6) --------------------------------

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
    "gap_study_content_v1": (
        "A learner tried to test out of a topic by answering verification "
        "questions before any study material was shown. They did not pass "
        "the question(s) below outright — they only passed after being "
        "taught the answer at the retry cap. Generate focused study "
        "material that teaches ONLY the specific gap(s) below — do not "
        "cover the whole topic, only what these question(s) actually "
        "test.\n\n"
        "{gaps}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"study_content": "<prose study material covering only these '
        'specific gaps, citing the given source(s)>"}}.'
    ),
}


@dataclass(frozen=True)
class DayContent:
    """One day's generated content (PRD §7.6). `hands_on_exercise`/
    `review_prompt` are `None` exactly on conceptual-only days (steps 3-4
    omitted) — never empty strings standing in for "not applicable".

    `theory_links` are real, already-existing URLs from a live Tavily
    search, attached by this module directly — never something Gemini
    produces or could alter (CLAUDE.md guardrail #1), the same
    structural sourcing-safety split
    `agents/research_outline_agent.py`'s `create_initial_outline` and
    `.agent/skills/verification_question_generator/generator.py` already
    use, applied here to a third kind of content.

    `remaining_content` is this task's spillover mechanism (PRD §7.6's
    "dynamic sizing... anything that doesn't fit... spills to the next
    day") — a single generic "content still pending" string, deliberately
    not specific to *why* something is pending. A patch-note's content
    could plug into the *same* `carried_over_content` input parameter
    `generate_day_content` accepts (see its docstring) without any
    rework — this task doesn't build that wiring, but the mechanism
    itself doesn't need to change to support it later.
    """

    summary: str
    theory_framing: str
    theory_links: list[dict[str, str]]
    hands_on_exercise: str | None
    review_prompt: str | None
    reflection_prompt: str
    preview: str
    remaining_content: str | None


async def _fetch_theory_material_links(topic_name: str) -> list[dict[str, str]]:
    """Live Tavily search for real, existing educational content (docs,
    tutorials, videos) for `topic_name` — never fabricated.

    The outline topic's own `source_url` (from `ground_role`) is market-
    grounding provenance (why this skill matters to employers), not a
    learning resource — this is a fresh, distinct search for genuine
    teaching material, per this task's explicit grounding-rule
    requirement. Reuses `agents/research_outline_agent.py`'s Tavily
    client and timeout convention directly, mirroring
    `_fetch_tavily_results`'s pattern exactly.

    Raises `GroundingSourceCallError` on any Tavily-specific API failure
    or timeout. Returns up to 5 real `{url, title, content}` candidates,
    ranked by Tavily's own relevance score.
    """
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
    """Generate one day's content (PRD §7.6). Hands-on-eligible days
    follow the 7-step structure (summary, theory, hands-on, review,
    reflection, verification, preview); conceptual-only days omit steps
    3-4. Verification (step 6) is not generated here — resolving it is
    `begin_verification_question`/`submit_verification_answer`'s job,
    called once per question after this content is shown.

    Whether today is conceptual-only, and how intense a hands-on-eligible
    day's exercise should be, is computed from `position_in_group`/
    `group_size` (`is_conceptual_only_day`/`compute_hands_on_intensity`)
    — never a caller-supplied flag, so the ramping rule can't drift out
    of sync with the outline's actual shape.

    `carried_over_content`, if given, is content that didn't fit a
    previous day and must be worked in today (this task's generic
    spillover mechanism — see `DayContent`'s docstring for why a future
    patch-note could reuse this same parameter). The returned
    `DayContent.remaining_content` is what — if anything — still didn't
    fit today and should be passed as `carried_over_content` to
    tomorrow's call.

    Raises `GroundingSourceCallError` if the live Tavily search for
    theory material fails, or `GeminiCallError` if Gemini's response is
    malformed.
    """
    minutes_available = convert_weekly_hours_to_daily_minutes(
        available_time_per_week_hours
    )
    theory_links = await _fetch_theory_material_links(topic_name)
    theory_sources = _format_theory_sources(theory_links)
    carried_over_instruction = _format_carried_over_instruction(carried_over_content)

    hands_on = not is_conceptual_only_day(position_in_group, group_size)

    if hands_on:
        intensity = compute_hands_on_intensity(position_in_group, group_size)
        prompt = PROMPT_REGISTRY["day_content_generation_hands_on_v1"].format(
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
        prompt = PROMPT_REGISTRY["day_content_generation_conceptual_v1"].format(
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

    parsed = await _call_gemini_json(
        prompt, required_keys=required_keys, model=DAY_CONTENT_GEMINI_MODEL
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
    """Log each generated step to `progress_log` (Architecture §5) —
    summary, theory, hands_on/review (only when hands-on-eligible), the
    reflection prompt (stored as generated; capturing the user's own
    reflection *response* as a distinct event is not addressed by this
    task), and preview. Verification's own `progress_log` entry is
    written separately, once the topic's questions actually resolve —
    not at content-generation time.
    """
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


# --- Verification retry-cap orchestration (PRD §7.7) ------------------


@dataclass(frozen=True)
class VerificationSlotState:
    """One question slot's (1-5) in-progress verification state.

    `attempt_number` is whichever attempt is currently pending an answer
    (1, 2, or 3) — `submit_verification_answer` is called identically
    for every attempt number, no special-cased "first attempt" code path
    (CLAUDE.md's named anti-pattern: the first attempt must live inside
    the same counter as retries 2/3, never handled outside it).
    `resolved`/`credit` are populated only once the slot is done (either
    a pass, or the retry cap was reached).
    """

    topic_id: str
    question_number: int
    attempt_number: int
    current_question: VerificationQuestion
    previous_question_texts: tuple[str, ...]
    resolved: bool = False
    credit: float | None = None
    taught_answer_message: str | None = None


def _build_taught_answer_message(question: VerificationQuestion) -> str:
    """Deterministic (non-LLM) "teach the answer inline" message for the
    3rd-attempt-failure de-escalation (PRD §7.7). `grading_criteria` is
    already "the specific rubric... what must be present for correctness"
    (per the Verification Skill's own contract) — genuinely LLM-free
    prose generation here would risk restating or subtly contradicting
    that rubric; deriving the teaching message directly from it and the
    real `source_url` is both simpler and safer (no fabrication risk),
    consistent with this task's other structural sourcing-safety
    choices. Flagged as a judgment call, not specified in PRD §7.7.
    """
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
    """Generate the first question (attempt 1) for one question slot
    (1-5) of a topic's verification (PRD §7.7). Takes no `session` — it
    does not write to `verification_attempts` yet; a row is written once
    the user's answer for this attempt is known
    (`submit_verification_answer`, which does take `session`).
    """
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
    """Grade the user's answer for `state`'s current attempt, and advance
    the slot — the exact same code path regardless of whether this is
    attempt 1, 2, or 3 (CLAUDE.md's named anti-pattern: never special-
    case the first attempt outside the retry-counting logic), and
    regardless of whether this is regular verification or test-out
    (PRD's day-by-day coaching section's "verification-first" exception)
    — test-out gets no separate retry cap or attempt-counting shape.

    - Pass (any attempt) -> full credit, `resolved=True`, no further
      generation call.
    - Fail, `attempt_number < MAX_VERIFICATION_ATTEMPTS` -> writes this
      attempt at 0.0 credit (not yet resolved — only the slot's *final*
      attempt carries meaningful credit), regenerates a fresh question
      for the next attempt (passing forward every question text asked so
      far, so the Skill's freshness mechanism can enforce non-repetition),
      `resolved=False`.
    - Fail, `attempt_number == MAX_VERIFICATION_ATTEMPTS` (the cap) ->
      half credit, `resolved=True`, a deterministic taught-answer message
      — no 4th generation call.

    Every attempt (all 3, whether or not reached) is written to
    `verification_attempts` as it happens. `is_test_out` is threaded
    straight through to `write_verification_attempt`'s existing parameter
    of the same name (Architecture §5's schema column, previously always
    written `False` regardless of caller since nothing before this task
    ever passed `True`) — the only thing test-out changes about this
    function; the retry-cap mechanics themselves are identical either way.
    Default `False` preserves this function's existing behavior for every
    regular (non-test-out) caller.
    """
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


# --- Pace signal computation + persistence (PRD §7.8) ------------------


# Judgment call, flagged: enrichment topics get their own singleton
# topic_group — the selected skill's own name plus this suffix, not
# folded into any existing topic-group — so
# `compute_hands_on_intensity`'s existing `group_size == 1` special case
# (full intensity immediately, not a permanently-conceptual-only day)
# applies naturally: an enrichment topic is always exactly one day, never
# part of a multi-day ramping sequence. The suffix (not just the bare
# skill name) avoids colliding with a core topic-group that happens to
# already use the same string.
ENRICHMENT_TOPIC_GROUP_SUFFIX = " (Enrichment)"
ENRICHMENT_POSITION_IN_GROUP = 1

# Judgment call, flagged for tuning — not specified anywhere in PRD/
# Architecture: how many days a single sustained-behind trigger adds to
# a user's effective pacing baseline. 2 is a small, conservative bump
# (roughly matching a week's worth of one extra half-day of slack every
# few topics) rather than a large jump; unvalidated against real usage,
# same status as pace/calculator.py's own threshold constants.
PACE_EXTENSION_DAYS_PER_TRIGGER = 2


@dataclass(frozen=True)
class TopicCompletionResult:
    """The pace signal computed once a topic's 5 verification question
    slots have all resolved (PRD §7.8), plus what (if anything) acting on
    it did this call.

    `drift` is `None` exactly when `is_enrichment=True` — enrichment
    completions never write a `pace_snapshots` row at all (PRD §7.10's
    isolation rule), so drift is never even evaluated for them, not just
    evaluated-and-ignored. For a non-enrichment completion, `drift` is
    always one of `detect_sustained_drift`'s three literal values.

    `enrichment_topic` is populated only when `drift == "ahead"` **and**
    `maybe_trigger_enrichment` actually inserted a topic (it can return
    `None` on "ahead" too — see that function's docstring for the cases
    where nothing gets inserted despite sustained-ahead drift).

    `pace_extension_applied` is populated only when `drift == "behind"` —
    the new total `users.pace_extension_days` after this trigger.
    """

    topic_score: float
    timing_ratio: float
    combined_pace_signal: float
    drift: Literal["ahead", "behind", "on_track"] | None = None
    enrichment_topic: dict[str, Any] | None = None
    pace_extension_applied: int | None = None


def _select_enrichment_skill(
    emerging_skills: list[dict[str, Any]], existing_topic_names: frozenset[str]
) -> dict[str, Any] | None:
    """Judgment call, flagged: pick the first `roles_cache` `emerging_skills`
    entry (in the order already stored there) whose skill name doesn't
    already match an existing outline topic for this user (case-
    insensitive) — "first not-yet-used", not a weighted/ranked pick,
    since `emerging_skills`' stored order carries no other selection
    signal (demand strength, recency) to prefer one entry over another.

    Returns `None` if every entry is already used, or the list is empty.
    """
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
    """Sustained-ahead branch (PRD §7.10): select an unused emerging skill
    from `resolved_role`'s `roles_cache` entry and insert it as a new,
    `is_enrichment=True` outline topic immediately after `origin_topic_id`
    (the topic whose completion triggered this check) — additive,
    hierarchy-positioned, via `outline/hierarchy.py`'s existing
    `insert_new_topic` (`data/outline_topics.py`'s `insert_new_outline_topic`,
    not reimplemented here), never reducing existing content.

    Source fields (`source_url`/`confidence`) are carried through from
    the selected `roles_cache` entry, re-validated via
    `security/output_guard.py`'s `validate_output_object` (CLAUDE.md
    guardrail #12) rather than trusted as an already-safe dict — the same
    structural-sourcing-safety pattern `data/grounding_fallback.py`'s
    `_rehydrate_skill_entry` already uses for cached-fallback skills.
    Unlike that function, the persisted `confidence` is **not** overridden
    to `cached-low`: this isn't standing in for a failed live lookup, it's
    a legitimate reference to an already-graded emerging skill for
    enrichment selection, so the original tier stays meaningful.
    `source_type` is stamped `CACHED_SOURCE_TYPE` (re-imported from
    `data/grounding_fallback.py`) for the same reason that module already
    established: `roles_cache` never persists a per-skill `source_type`.

    Returns the inserted topic dict, or `None` if nothing was inserted:
    the user already has a pending (unresolved) enrichment topic
    (`has_pending_enrichment_topic`), the role has no `roles_cache` entry
    or an empty `emerging_skills` list, or every emerging skill already
    matches an existing outline topic for this user.
    """
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


def _get_latest_attempt_per_question(
    session: Session, topic_id: str
) -> dict[int, dict[str, Any]]:
    """Read back each of the 5 question slots' final (most recent)
    attempt, from `verification_attempts` — the single source of truth,
    rather than trusting a caller-tracked list that could drift from what
    was actually persisted. Shared by `_get_final_credits_per_question`
    (regular completion) and `_get_failed_questions_for_topic` (test-out's
    gap detection) — both need "the final attempt per slot", just
    different fields off of it.

    Raises `ValueError` if fewer than all 5 question slots have been
    attempted yet, or if any slot's most recent attempt is a failure
    that hasn't reached the retry cap (i.e. genuinely still in progress,
    not yet resolved) — a topic requires all 5 slots *resolved* (full or
    half credit), not merely attempted, before it can complete (PRD
    §7.7).
    """
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
    """Read back the final (most recent) attempt's credit for each of
    the 5 question slots — see `_get_latest_attempt_per_question` for the
    shared read/validation this delegates to.
    """
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
) -> TopicCompletionResult:
    """Called once all 5 verification question slots for `topic_id` have
    resolved: computes `topic_score` from their final credits
    (`pace/calculator.py`), blends it with `timing_ratio` into the
    combined pace signal, persists a `pace_snapshots` row, acts on
    sustained drift if any, and marks the topic completed in
    `outline_topics` (PRD §7.7: "topic requires all 5 questions passed...
    to complete").

    `is_test_out`, when `True`, marks the topic `completed_test_out`
    instead of `completed` (Architecture §5's schema: a distinct status
    value, not a synonym) — `complete_topic_test_out` is the only caller
    that passes `True`; every regular (non-test-out) caller is unaffected
    by this parameter's default of `False`.

    `is_enrichment`, when `True`, is a **structural, unconditional skip**
    of the entire pace-snapshot-write-and-act block below (PRD §7.10:
    "isolated from pace/verification consequences... never feeds the
    pace formula") — not a comment or a caller-side convention; the write
    call itself is inside the `if not is_enrichment:` guard, so there is
    no code path by which an enrichment completion can write a
    `pace_snapshots` row or influence `detect_sustained_drift`'s window
    for any other topic. The topic still gets marked
    completed/completed_test_out either way, purely for future
    closing-note credit (goal-completion closing-note content itself is a
    separate, later task).

    When a `pace_snapshots` row *is* written (i.e. `is_enrichment=False`),
    this function also reads the user's full pace-snapshot history
    (`data/pace_snapshots.py`'s `get_pace_snapshot_history` — which now
    includes the row just written), recomputes each entry's combined pace
    signal (`calculate_combined_pace_signal`, not reimplemented), and
    calls `pace/calculator.py`'s `detect_sustained_drift` unmodified
    (cold-start/window-size gating is entirely that function's own
    responsibility — not duplicated here). `"ahead"` calls
    `maybe_trigger_enrichment` (skipped, logged as `enrichment_topic=None`,
    if the user's `users.resolved_role` is missing — a data-integrity gap
    upstream, not something this function raises over). `"behind"` calls
    `data/users.py`'s `extend_pacing` by `PACE_EXTENSION_DAYS_PER_TRIGGER`.
    `"on_track"` (including every cold-start call, fewer than
    `DRIFT_WINDOW_SIZE` snapshots) does nothing further.

    Raises `ValueError` (via `_get_final_credits_per_question` or
    `pace/calculator.py`) if the topic's 5 question slots aren't all
    genuinely resolved yet, or if `days_expected` isn't positive. Marking
    the topic completed only happens after the score/signal are
    successfully computed and the snapshot is written — never marked
    complete on a path that could still raise.
    """
    credits = _get_final_credits_per_question(session, topic_id)
    topic_score = calculate_topic_score(credits)
    timing_ratio = calculate_timing_ratio(days_taken, days_expected)
    combined_signal = calculate_combined_pace_signal(topic_score, timing_ratio)

    drift: Literal["ahead", "behind", "on_track"] | None = None
    enrichment_topic: dict[str, Any] | None = None
    pace_extension_applied: int | None = None

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
        history = get_pace_snapshot_history(session, user_id)
        pace_signals = [
            calculate_combined_pace_signal(row["topic_score"], row["timing_ratio"])
            for row in history
        ]
        drift = detect_sustained_drift(pace_signals)

        if drift == "ahead":
            user = get_user(session, user_id)
            if user is not None and user["resolved_role"]:
                enrichment_topic = maybe_trigger_enrichment(
                    session, user_id, user["resolved_role"], topic_id
                )
        elif drift == "behind":
            pace_extension_applied = extend_pacing(
                session, user_id, PACE_EXTENSION_DAYS_PER_TRIGGER
            )

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
    )


# --- Test-out: verification-first (PRD §7.6's exception) ---------------

# Judgment call, flagged for review — resolved directly for this task,
# not derivable from PRD/Architecture as written: "full pass" vs. "partial
# pass" for test-out purposes is defined in terms of the *existing*
# credit scale (`FULL_CREDIT`/`HALF_CREDIT`), not a new concept. A slot
# that ultimately passed within the retry cap (on attempt 1, 2, or 3)
# counts as a full pass for that question; a slot that only resolved via
# the retry-cap teach-and-de-escalate path (`HALF_CREDIT`) counts as a
# partial pass. "Full pass" for the whole topic means every one of the 5
# slots individually full-passed. This mirrors PRD §7.7's own completion
# rule ("all 5 questions passed, full or half credit, to complete") —
# test-out does not introduce a second, competing definition of "passed."
#
# Reconsidered and corrected: a partial pass's `HALF_CREDIT` slot(s) are,
# by construction, exactly the slot(s) that already fired
# `_build_taught_answer_message` during `submit_verification_answer`'s own
# 3rd-attempt de-escalation — there is no other way to reach `HALF_CREDIT`.
# `generate_gap_study_content` was built, then deliberately NOT wired into
# `complete_topic_test_out` below, because doing so would re-teach the
# identical rubric (`grading_criteria`) a second time, worded differently,
# in the same session, moments after the user already saw it — a genuine
# double-remediation bug, not a richer second pass. `complete_topic_test_out`
# relies entirely on that already-delivered teach-in; it generates nothing
# further. `generate_gap_study_content` is kept, unwired, as a possible
# building block for a future, non-test-out remediation flow (e.g. a
# dedicated "review what you missed" feature outside the test-out path)
# — not deleted, since it is not itself wrong, only wrong to call here.


def _format_failed_questions_for_prompt(
    failed_questions: list[VerificationQuestion],
) -> str:
    return "\n\n".join(
        f"Gap {i}:\nQuestion: {q.question_text}\n"
        f"What a correct answer needed: {q.grading_criteria}\n"
        f"Source: {q.source_url}"
        for i, q in enumerate(failed_questions, start=1)
    )


async def generate_gap_study_content(
    failed_questions: list[VerificationQuestion],
) -> str:
    """Generate remedial study material scoped ONLY to specific
    verification question(s) a user did not pass outright.

    A new, purpose-built content path — distinct from
    `generate_day_content`'s 7-step structure, which this function does
    not call into or reuse in any way.

    **Not currently called anywhere in this module.** It was built for
    test-out's partial-pass path (PRD §7.6), then deliberately *not*
    wired into `complete_topic_test_out` — see the module-level note
    above this section for why: every question this function could be
    given during test-out already received
    `submit_verification_answer`'s inline teach-in
    (`_build_taught_answer_message`) built from the exact same
    `grading_criteria`, in the same session, moments earlier; calling
    this function too would re-teach the identical rubric a second time
    in different words, not add anything. Retained, unwired, as a
    possible building block for a future *non*-test-out remediation flow
    (e.g. a dedicated "review what you missed" feature) — not deleted,
    since the function itself is sound, only wrong to call from
    test-out.

    Raises `ValueError` if `failed_questions` is empty. Raises
    `GeminiCallError` if Gemini's response is malformed or has no usable
    `study_content`.
    """
    if not failed_questions:
        raise ValueError(
            "generate_gap_study_content requires at least one failed question"
        )
    prompt = PROMPT_REGISTRY["gap_study_content_v1"].format(
        gaps=_format_failed_questions_for_prompt(failed_questions)
    )
    parsed = await _call_gemini_json(
        prompt, required_keys={"study_content"}, model=DAY_CONTENT_GEMINI_MODEL
    )
    content = parsed["study_content"]
    if not isinstance(content, str) or not content.strip():
        raise GeminiCallError(f"Gemini returned no usable 'study_content': {parsed!r}")
    return content


@dataclass(frozen=True)
class TestOutResult:
    """Result of a topic's test-out attempt (PRD §7.6's "verification-
    first" exception: "for any topic, the user may trigger verification
    first, before study content is generated").

    `full_pass` is True exactly when every one of the 5 question slots
    resolved at `FULL_CREDIT` (passed within the retry cap, on any
    attempt) — see the module-level judgment-call note above this
    section for why this reuses the existing credit scale rather than
    introducing a new concept. When `full_pass` is False, the user has
    already been taught the answer for the relevant slot(s) inline,
    during the retry-cap attempts themselves (PRD §7.7) — this module
    generates no further content for that case; see the module-level
    note above for why a separate gap-study step was considered and
    rejected as redundant.
    """

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
    """Resolve a topic via test-out (PRD §7.6's "verification-first"
    exception): the user answers all 5 verification questions before any
    study content is generated, using the identical
    `begin_verification_question`/`submit_verification_answer` turn-based
    path (same exactly-3-attempt retry cap, same attempt-counting shape)
    regular verification uses — every attempt recorded with
    `is_test_out=True`, not a separate mechanism. Call this once all 5
    question slots have resolved (mirrors `complete_topic_verification`'s
    own precondition, enforced the same way).

    `is_enrichment`, threaded straight through to
    `complete_topic_verification`, structurally skips the
    `pace_snapshots` write/drift-detection block exactly as it does for
    regular verification (PRD §7.10) — nothing in this codebase's test-out
    logic restricts it to non-enrichment topics, so this combination
    (test-out an enrichment topic) is treated as genuinely possible, not
    assumed impossible.

    Full pass (every slot resolved at `FULL_CREDIT`) -> topic marked
    `completed_test_out`, no study content generated at all (PRD §7.6).
    Partial pass (>=1 slot resolved at `HALF_CREDIT`) -> topic still
    marked `completed_test_out` (PRD §7.7's completion rule doesn't
    distinguish full/half credit for completion purposes — test-out
    doesn't either), and **this function generates nothing further**:
    every `HALF_CREDIT` slot already received
    `submit_verification_answer`'s inline teach-in
    (`_build_taught_answer_message`) during the retry-cap attempt itself,
    moments before this function is ever called. A separate gap-study
    generation step (`generate_gap_study_content`, built but deliberately
    left unwired here) was considered and rejected: it would have
    re-derived prose from the identical `grading_criteria` the teach-in
    already used, in the same session, teaching the same fact twice in
    different words. See the module-level note above this section.

    Raises `ValueError` (via `_get_final_credits_per_question`) if the
    topic's 5 question slots aren't all genuinely resolved yet, or if
    `days_expected` isn't positive.
    """
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


def generate_closing_note(user_id: str) -> str:
    """Compose the goal-completion closing note, reusing roles_cache
    infrastructure for current hiring signal and in-demand skills, per PRD
    §7.11. Never makes a seniority, grading, or leveling claim.

    Not yet implemented — scaffolding only. Explicitly out of scope for
    this task; deliberately left untouched, not stubbed with a guessed
    shape.
    """
    raise NotImplementedError
