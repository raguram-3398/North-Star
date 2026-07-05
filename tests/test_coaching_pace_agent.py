"""Tests for agents/coaching_pace_agent.py — day-by-day content
generation (7-step hands-on-eligible structure and its conceptual-only
variant), verification retry-cap orchestration (exactly 3 attempts,
half-credit teach-and-de-escalate), and pace-signal computation +
persistence.

Gemini is mocked by reusing tests/test_research_outline_agent.py's
`_patch_gemini` fake unchanged — `agents.coaching_pace_agent._call_gemini_json`
is the *same function object* as `agents.research_outline_agent._call_gemini_json`
(a direct import, not a copy), and that function's own internal call to
`_get_gemini_client()` resolves via `research_outline_agent`'s namespace
regardless of which module calls it — so patching
`agents.research_outline_agent._get_gemini_client` (what `_patch_gemini`
already does) is the correct target here too.

Tavily is different: `agents.coaching_pace_agent._fetch_theory_material_links`
calls `_get_tavily_client()` as a bare name *defined in this module's own
namespace* (a separate import binding from research_outline_agent.py's),
so it must be patched at `agents.coaching_pace_agent._get_tavily_client`
specifically — reusing `research_outline_agent`'s `_patch_tavily` would
silently patch the wrong module and not take effect.

The Verification Skill's `generate_questions`/`grade_answer` are mocked
directly at `agents.coaching_pace_agent.generate_questions`/`grade_answer`
(again, "patch where used" — these are direct imports into this module's
own namespace) rather than exercising the Skill's own Gemini calls, since
the Skill already has its own dedicated test suite
(tests/test_verification_skill.py) — these tests focus on this module's
own retry-cap orchestration logic.
"""

import json
from unittest.mock import MagicMock

import pytest

import agents.coaching_pace_agent as cpa
from security.output_guard import ConfidenceTier
from tests.test_research_outline_agent import (
    _FakeTavilyClient,
    _patch_gemini,
    _tavily_response,
    _tavily_result_dict,
)
from utils.exceptions import GeminiCallError, GroundingSourceCallError

TOPIC_SOURCE_MATERIAL = (
    "Git branching lets you create, merge, and delete lightweight "
    "pointers to commits."
)
SOURCE_URL = "https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell"


def _patch_tavily_theory(
    monkeypatch: pytest.MonkeyPatch, client: _FakeTavilyClient
) -> None:
    """Patch Tavily at *this module's own* import binding — see module
    docstring for why `research_outline_agent`'s `_patch_tavily` wouldn't
    take effect here."""
    monkeypatch.setattr(cpa, "_get_tavily_client", lambda: client)


def _theory_tavily_client() -> _FakeTavilyClient:
    return _FakeTavilyClient(
        _tavily_response(
            [
                _tavily_result_dict(
                    0.9,
                    "Branches let you develop features in isolation.",
                    url="https://git-scm.com/book/branching",
                )
            ]
        )
    )


class _FakeGenerateQuestions:
    """Fake `generate_questions` returning one canned question per call,
    in order, with call-count/argument tracking."""

    def __init__(self, question_texts: list[str]) -> None:
        self._question_texts = list(question_texts)
        self.call_count = 0
        self.calls: list[dict] = []

    async def __call__(
        self,
        topic_source_material: str,
        source_url: str,
        num_questions: int = 1,
        previous_question_texts: list[str] | None = None,
    ):
        self.calls.append(
            {
                "topic_source_material": topic_source_material,
                "source_url": source_url,
                "num_questions": num_questions,
                "previous_question_texts": list(previous_question_texts or []),
            }
        )
        text = self._question_texts[self.call_count]
        self.call_count += 1
        return [
            cpa.VerificationQuestion(
                question_text=text,
                grading_criteria=f"Must correctly explain: {text}",
                source_url=source_url,
            )
        ]


class _FakeGradeAnswer:
    """Fake `grade_answer` returning canned pass/fail results in order."""

    def __init__(self, results: list[bool]) -> None:
        self._results = list(results)
        self.call_count = 0

    async def __call__(self, question, user_answer: str) -> bool:
        result = self._results[self.call_count]
        self.call_count += 1
        return result


def _patch_verification_skill(
    monkeypatch: pytest.MonkeyPatch,
    question_texts: list[str],
    grade_results: list[bool],
) -> tuple[_FakeGenerateQuestions, _FakeGradeAnswer]:
    fake_generate = _FakeGenerateQuestions(question_texts)
    fake_grade = _FakeGradeAnswer(grade_results)
    monkeypatch.setattr(cpa, "generate_questions", fake_generate)
    monkeypatch.setattr(cpa, "grade_answer", fake_grade)
    return fake_generate, fake_grade


def _no_op_write_attempt(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    written: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "write_verification_attempt",
        lambda *args, **kwargs: written.append((args, kwargs)),
    )
    return written


# --- Day content generation: 7-step hands-on-eligible structure ----------


async def test_generate_day_content_hands_on_eligible_is_fully_populated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tavily_theory(monkeypatch, _theory_tavily_client())
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "summary": "Today you'll learn Git branching.",
                    "theory_framing": "Branches let you isolate work; see [1].",
                    "hands_on_exercise": "Create a branch and make a commit.",
                    "review_prompt": "Review your branch's commit history.",
                    "reflection_prompt": "Why did branching help here?",
                    "preview": "Tomorrow: merging branches back together.",
                    "remaining_content": "",
                }
            )
        ],
    )

    content = await cpa.generate_day_content(
        topic_name="Git Branching",
        topic_group="Git",
        position_in_group=2,
        group_size=5,
        available_time_per_week_hours=10,
    )

    assert content.summary
    assert content.theory_framing
    assert content.theory_links
    assert content.hands_on_exercise is not None
    assert content.review_prompt is not None
    assert content.reflection_prompt
    assert content.preview
    assert content.remaining_content is None


async def test_generate_day_content_conceptual_only_omits_hands_on_and_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tavily_theory(monkeypatch, _theory_tavily_client())
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "summary": "Today you'll learn what Git branches are.",
                    "theory_framing": "Branches let you isolate work; see [1].",
                    "reflection_prompt": "What problem do branches solve?",
                    "preview": "Tomorrow: creating your first branch hands-on.",
                    "remaining_content": "",
                }
            )
        ],
    )

    content = await cpa.generate_day_content(
        topic_name="Git Branching",
        topic_group="Git",
        position_in_group=1,
        group_size=5,
        available_time_per_week_hours=10,
    )

    assert content.summary
    assert content.theory_framing
    assert content.reflection_prompt
    assert content.preview
    assert content.hands_on_exercise is None
    assert content.review_prompt is None


async def test_generate_day_content_carries_remaining_content_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tavily_theory(monkeypatch, _theory_tavily_client())
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "summary": "Summary.",
                    "theory_framing": "Framing.",
                    "reflection_prompt": "Reflect.",
                    "preview": "Preview.",
                    "remaining_content": "Merging strategies didn't fit today.",
                }
            )
        ],
    )

    content = await cpa.generate_day_content(
        topic_name="Git Branching",
        topic_group="Git",
        position_in_group=1,
        group_size=5,
        available_time_per_week_hours=1,
    )

    assert content.remaining_content == "Merging strategies didn't fit today."


async def test_fetch_theory_material_links_raises_on_tavily_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tavily.errors import InvalidAPIKeyError

    _patch_tavily_theory(
        monkeypatch, _FakeTavilyClient(raise_exc=InvalidAPIKeyError("bad key"))
    )

    with pytest.raises(GroundingSourceCallError):
        await cpa._fetch_theory_material_links("Git Branching")


# --- Hands-on ramping rule (position_in_group / group_size driven) -------


def test_compute_hands_on_intensity_first_day_is_zero() -> None:
    assert cpa.compute_hands_on_intensity(1, 5) == 0.0


def test_compute_hands_on_intensity_last_day_is_full() -> None:
    assert cpa.compute_hands_on_intensity(5, 5) == 1.0


def test_compute_hands_on_intensity_scales_linearly() -> None:
    assert cpa.compute_hands_on_intensity(3, 5) == pytest.approx(0.5)


def test_compute_hands_on_intensity_single_day_group_is_full_not_zero() -> None:
    """Edge case: a single-topic group has no room to ramp — full
    intensity immediately rather than permanently conceptual-only."""
    assert cpa.compute_hands_on_intensity(1, 1) == 1.0


def test_compute_hands_on_intensity_rejects_out_of_range_position() -> None:
    with pytest.raises(ValueError):
        cpa.compute_hands_on_intensity(6, 5)
    with pytest.raises(ValueError):
        cpa.compute_hands_on_intensity(0, 5)


def test_is_conceptual_only_day_matches_zero_intensity() -> None:
    assert cpa.is_conceptual_only_day(1, 5) is True
    assert cpa.is_conceptual_only_day(2, 5) is False
    assert cpa.is_conceptual_only_day(1, 1) is False


def test_convert_weekly_hours_to_daily_minutes() -> None:
    # 10 hours/week over a 5-day cadence = 2 hours = 120 minutes/day
    assert cpa.convert_weekly_hours_to_daily_minutes(10) == 120


def test_convert_weekly_hours_to_daily_minutes_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        cpa.convert_weekly_hours_to_daily_minutes(0)


# --- PROMPT_REGISTRY: baseline regression asserts on the prompt string ----


def test_day_content_hands_on_prompt_v1_is_frozen() -> None:
    assert cpa.PROMPT_REGISTRY["day_content_generation_hands_on_v1"] == (
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
    )


def test_day_content_conceptual_prompt_v1_is_frozen() -> None:
    assert cpa.PROMPT_REGISTRY["day_content_generation_conceptual_v1"] == (
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
    )


# --- Verification retry-cap orchestration ---------------------------------


async def test_retry_cap_is_exactly_three_first_attempt_inside_the_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAUDE.md's named anti-pattern: the first attempt must live inside
    the same counter as retries 2/3, not run once outside the loop
    before it starts. Verifies actual generate_questions call count is
    exactly 3 (1 for attempt 1, 1 regeneration each for attempts 2 and
    3) — not 2 (which would mean attempt 1 wasn't really counted) and
    not 4 (a call after the cap)."""
    fake_generate, fake_grade = _patch_verification_skill(
        monkeypatch, ["Q1", "Q2", "Q3"], [False, False, False]
    )
    written = _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        topic_id="t1",
        question_number=1,
        topic_source_material=TOPIC_SOURCE_MATERIAL,
        source_url=SOURCE_URL,
    )
    assert state.attempt_number == 1
    assert fake_generate.call_count == 1

    state = await cpa.submit_verification_answer(
        state, "wrong 1", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    assert not state.resolved
    assert state.attempt_number == 2
    assert fake_generate.call_count == 2

    state = await cpa.submit_verification_answer(
        state, "wrong 2", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    assert not state.resolved
    assert state.attempt_number == 3
    assert fake_generate.call_count == 3

    state = await cpa.submit_verification_answer(
        state, "wrong 3", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )

    assert state.resolved
    assert fake_generate.call_count == 3  # NOT 4 — no generation call past the cap
    assert fake_grade.call_count == 3
    assert len(written) == 3


async def test_third_attempt_failure_gives_half_credit_and_teaches_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verification_skill(monkeypatch, ["Q1", "Q2", "Q3"], [False, False, False])
    _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        "t1", 1, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    state = await cpa.submit_verification_answer(
        state, "wrong", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    state = await cpa.submit_verification_answer(
        state, "wrong", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    state = await cpa.submit_verification_answer(
        state, "wrong", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )

    assert state.resolved
    assert state.credit == cpa.HALF_CREDIT
    assert state.taught_answer_message is not None
    assert "Q3" in state.taught_answer_message or SOURCE_URL in (
        state.taught_answer_message
    )


@pytest.mark.parametrize("success_attempt", [1, 2, 3])
async def test_success_on_any_attempt_stops_retrying_with_full_credit(
    monkeypatch: pytest.MonkeyPatch, success_attempt: int
) -> None:
    grade_results = [False] * (success_attempt - 1) + [True]
    fake_generate, fake_grade = _patch_verification_skill(
        monkeypatch, ["Q1", "Q2", "Q3"][:success_attempt], grade_results
    )
    _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        "t1", 1, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    for _ in range(success_attempt - 1):
        state = await cpa.submit_verification_answer(
            state, "wrong", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
        )
        assert not state.resolved
    state = await cpa.submit_verification_answer(
        state, "correct", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )

    assert state.resolved
    assert state.credit == cpa.FULL_CREDIT
    assert state.taught_answer_message is None
    # no generation call after a resolving success
    assert fake_generate.call_count == success_attempt
    assert fake_grade.call_count == success_attempt


async def test_intermediate_failed_attempts_are_written_with_zero_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verification_skill(monkeypatch, ["Q1", "Q2"], [False, True])
    written = _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        "t1", 1, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    await cpa.submit_verification_answer(
        state, "wrong", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )

    first_attempt_kwargs = written[0][1]
    assert first_attempt_kwargs["credit"] == cpa.NOT_YET_RESOLVED_CREDIT
    assert first_attempt_kwargs["passed"] is False


# --- Topic completion: requires all 5 slots resolved, calls pace calc ----


def test_complete_topic_verification_requires_all_five_slots_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa,
        "get_attempts_for_topic",
        lambda session, topic_id: [
            {
                "question_number": q,
                "attempt_number": 1,
                "passed": True,
                "credit": 1.0,
            }
            for q in range(1, 4)
        ],
    )

    with pytest.raises(ValueError):
        cpa.complete_topic_verification(
            MagicMock(), "u1", "t1", days_taken=3, days_expected=3
        )


def test_complete_topic_verification_requires_slots_resolved_not_just_attempted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = [
        {"question_number": q, "attempt_number": 1, "passed": True, "credit": 1.0}
        for q in range(1, 5)
    ]
    # Question 5 has been attempted once, failed, and is NOT at the cap
    # yet — genuinely still in progress, not resolved.
    attempts.append(
        {"question_number": 5, "attempt_number": 1, "passed": False, "credit": 0.0}
    )
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )

    with pytest.raises(ValueError):
        cpa.complete_topic_verification(
            MagicMock(), "u1", "t1", days_taken=3, days_expected=3
        )


def _all_full_credit_attempts() -> list[dict]:
    return [
        {"question_number": q, "attempt_number": 1, "passed": True, "credit": 1.0}
        for q in range(1, 6)
    ]


def _patch_completion_writes(
    monkeypatch: pytest.MonkeyPatch, attempts: list[dict]
) -> tuple[list[tuple], list[tuple]]:
    """Shared setup for complete_topic_verification tests: fake
    get_attempts_for_topic/write_pace_snapshot/mark_topic_completed,
    returning the snapshot_calls/completed_calls lists so callers can
    assert on them. Also fakes get_pending_patch_notes to return an empty
    list — these tests aren't about patch delivery (see the dedicated
    maybe_deliver_patch tests below), so this keeps that call a
    deliberate, explicit no-op rather than relying on an unmocked
    MagicMock session's query chain happening to iterate empty.
    """
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    snapshot_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "write_pace_snapshot",
        lambda *args, **kwargs: snapshot_calls.append((args, kwargs)),
    )
    completed_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "mark_topic_completed",
        lambda *args, **kwargs: completed_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(cpa, "get_pending_patch_notes", lambda session, uid: [])
    return snapshot_calls, completed_calls


def test_complete_topic_verification_calls_pace_calculator_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = [
        {"question_number": 1, "attempt_number": 1, "passed": True, "credit": 1.0},
        {"question_number": 2, "attempt_number": 1, "passed": True, "credit": 1.0},
        {"question_number": 3, "attempt_number": 3, "passed": False, "credit": 0.5},
        {"question_number": 4, "attempt_number": 1, "passed": True, "credit": 1.0},
        {"question_number": 5, "attempt_number": 2, "passed": True, "credit": 1.0},
    ]
    snapshot_calls, completed_calls = _patch_completion_writes(monkeypatch, attempts)
    # Cold start (no prior history) — this test is about the pace-calc/
    # persistence wiring, not drift detection (see the dedicated drift
    # tests below).
    monkeypatch.setattr(cpa, "get_pace_snapshot_history", lambda session, uid: [])
    session = MagicMock()

    result = cpa.complete_topic_verification(
        session, "u1", "t1", days_taken=4, days_expected=5
    )

    expected_topic_score = (1.0 + 1.0 + 0.5 + 1.0 + 1.0) / 5
    assert result.topic_score == pytest.approx(expected_topic_score)
    assert result.timing_ratio == pytest.approx(4 / 5)
    assert result.drift == "on_track"
    assert len(snapshot_calls) == 1
    snapshot_args = snapshot_calls[0][0]
    assert snapshot_args[0] is session
    assert snapshot_args[1] == "u1"
    assert snapshot_args[2] == "t1"
    assert snapshot_args[3] == pytest.approx(expected_topic_score)
    assert snapshot_args[4] == pytest.approx(4 / 5)
    assert snapshot_args[5] == 4
    assert snapshot_args[6] == 5
    assert len(completed_calls) == 1
    assert completed_calls[0][0] == (session, "t1")


# --- Sustained-drift wiring (Part 1) ---------------------------------------


def test_complete_topic_verification_cold_start_triggers_neither_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than DRIFT_WINDOW_SIZE (3) total snapshots — detect_sustained_drift's
    own cold-start gating, not a second check duplicated in this module.
    """
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    # Only 2 total snapshots (including the one this call is about to add
    # would make 3, but get_pace_snapshot_history is faked to represent
    # "before this call's own write took effect on a mocked session" —
    # i.e. genuinely below the window).
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [
            {"topic_score": 1.0, "timing_ratio": 1.0},
            {"topic_score": 1.0, "timing_ratio": 1.0},
        ],
    )
    maybe_trigger_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "maybe_trigger_enrichment",
        lambda *a, **k: maybe_trigger_calls.append((a, k)),
    )
    extend_pacing_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "extend_pacing", lambda *a, **k: extend_pacing_calls.append((a, k))
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "on_track"
    assert maybe_trigger_calls == []
    assert extend_pacing_calls == []
    assert result.enrichment_topic is None
    assert result.pace_extension_applied is None


def test_complete_topic_verification_ordinary_variation_triggers_neither_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full window of signals that land strictly between the behind/
    ahead thresholds must trigger neither branch."""
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [
            {"topic_score": 0.8, "timing_ratio": 1.0},
            {"topic_score": 0.85, "timing_ratio": 1.0},
            {"topic_score": 0.8, "timing_ratio": 1.0},
        ],
    )
    maybe_trigger_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "maybe_trigger_enrichment",
        lambda *a, **k: maybe_trigger_calls.append((a, k)),
    )
    extend_pacing_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "extend_pacing", lambda *a, **k: extend_pacing_calls.append((a, k))
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "on_track"
    assert maybe_trigger_calls == []
    assert extend_pacing_calls == []


def test_complete_topic_verification_sustained_ahead_triggers_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [
            {"topic_score": 1.0, "timing_ratio": 1.0},
            {"topic_score": 1.0, "timing_ratio": 1.0},
            {"topic_score": 1.0, "timing_ratio": 1.0},
        ],
    )
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    trigger_calls: list[tuple] = []

    def _fake_trigger(session, user_id, resolved_role, origin_topic_id):
        trigger_calls.append((user_id, resolved_role, origin_topic_id))
        return {"id": "enrichment-topic-1", "topic_name": "GraphQL"}

    monkeypatch.setattr(cpa, "maybe_trigger_enrichment", _fake_trigger)
    extend_pacing_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "extend_pacing", lambda *a, **k: extend_pacing_calls.append((a, k))
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "ahead"
    assert trigger_calls == [("u1", "Backend Engineer", "t1")]
    assert result.enrichment_topic == {
        "id": "enrichment-topic-1",
        "topic_name": "GraphQL",
    }
    assert extend_pacing_calls == []


def test_complete_topic_verification_sustained_behind_triggers_pacing_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [
            {"topic_score": 0.5, "timing_ratio": 1.0},
            {"topic_score": 0.5, "timing_ratio": 1.0},
            {"topic_score": 0.5, "timing_ratio": 1.0},
        ],
    )
    trigger_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "maybe_trigger_enrichment", lambda *a, **k: trigger_calls.append((a, k))
    )
    extend_calls: list[tuple] = []

    def _fake_extend(session, user_id, extension_days):
        extend_calls.append((user_id, extension_days))
        return 2

    monkeypatch.setattr(cpa, "extend_pacing", _fake_extend)

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "behind"
    assert extend_calls == [("u1", cpa.PACE_EXTENSION_DAYS_PER_TRIGGER)]
    assert result.pace_extension_applied == 2
    assert trigger_calls == []


def test_complete_topic_verification_sustained_behind_never_touches_outline_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pacing extension must never touch outline_topics content (no
    deletion, no is_enrichment tagging, nothing structural) — proven here
    by asserting the only three functions that could modify
    outline_topics are never called at all on the behind path.
    """
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [{"topic_score": 0.5, "timing_ratio": 1.0}] * 3,
    )
    monkeypatch.setattr(cpa, "extend_pacing", lambda *a, **k: 2)
    insert_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "insert_new_outline_topic", lambda *a, **k: insert_calls.append((a, k))
    )
    get_all_topics_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "get_all_topics_for_user",
        lambda *a, **k: get_all_topics_calls.append((a, k)) or [],
    )
    has_pending_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "has_pending_enrichment_topic",
        lambda *a, **k: has_pending_calls.append((a, k)) or False,
    )

    cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert insert_calls == []
    assert get_all_topics_calls == []
    assert has_pending_calls == []


# --- Enrichment selection/insertion (Part 2) -------------------------------


def test_maybe_trigger_enrichment_inserts_unused_emerging_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "has_pending_enrichment_topic", lambda session, uid: False)
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {
            "role_name": role,
            "core_skills": [],
            "emerging_skills": [
                {
                    "skill": "GraphQL",
                    "source_url": "https://example.com/graphql",
                    "confidence": "medium",
                },
                {
                    "skill": "gRPC",
                    "source_url": "https://example.com/grpc",
                    "confidence": "low",
                },
            ],
            "last_updated": None,
        },
    )
    monkeypatch.setattr(
        cpa,
        "get_all_topics_for_user",
        lambda session, uid: [{"topic_name": "SQL"}, {"topic_name": "Docker"}],
    )
    insert_calls: list[dict] = []

    def _fake_insert(session, **kwargs):
        insert_calls.append(kwargs)
        return {"id": "new-topic-1", **kwargs}

    monkeypatch.setattr(cpa, "insert_new_outline_topic", _fake_insert)

    result = cpa.maybe_trigger_enrichment(
        MagicMock(), "u1", "Backend Engineer", "origin-topic-1"
    )

    assert result is not None
    assert len(insert_calls) == 1
    call = insert_calls[0]
    assert call["user_id"] == "u1"
    assert call["topic_name"] == "GraphQL"
    assert call["topic_group"] == "GraphQL (Enrichment)"
    assert call["is_enrichment"] is True
    assert call["source_url"] == "https://example.com/graphql"
    assert call["source_type"] == "roles_cache-cached"
    assert call["confidence"] == ConfidenceTier.MEDIUM
    assert call["prerequisite_topic_ids"] == frozenset({"origin-topic-1"})


def test_maybe_trigger_enrichment_skips_an_already_used_emerging_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first emerging skill (GraphQL) already matches an existing
    outline topic (case-insensitively) — selection must move on to the
    next one (gRPC), not stop or error.
    """
    monkeypatch.setattr(cpa, "has_pending_enrichment_topic", lambda session, uid: False)
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {
            "core_skills": [],
            "emerging_skills": [
                {
                    "skill": "GraphQL",
                    "source_url": "https://x/graphql",
                    "confidence": "medium",
                },
                {"skill": "gRPC", "source_url": "https://x/grpc", "confidence": "low"},
            ],
        },
    )
    monkeypatch.setattr(
        cpa,
        "get_all_topics_for_user",
        lambda session, uid: [{"topic_name": "graphql"}],
    )
    insert_calls: list[dict] = []

    def _fake_insert(session, **kwargs):
        insert_calls.append(kwargs)
        return {"id": "t2", **kwargs}

    monkeypatch.setattr(cpa, "insert_new_outline_topic", _fake_insert)

    result = cpa.maybe_trigger_enrichment(MagicMock(), "u1", "Backend Engineer", "t1")

    assert result is not None
    assert insert_calls[0]["topic_name"] == "gRPC"


def test_maybe_trigger_enrichment_skips_if_pending_enrichment_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "has_pending_enrichment_topic", lambda session, uid: True)
    get_role_calls: list[str] = []
    monkeypatch.setattr(
        cpa, "get_role", lambda session, role: get_role_calls.append(role)
    )
    insert_calls: list[dict] = []
    monkeypatch.setattr(
        cpa, "insert_new_outline_topic", lambda *a, **k: insert_calls.append(k)
    )

    result = cpa.maybe_trigger_enrichment(MagicMock(), "u1", "Backend Engineer", "t1")

    assert result is None
    assert get_role_calls == []
    assert insert_calls == []


def test_maybe_trigger_enrichment_does_nothing_when_role_has_no_emerging_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "has_pending_enrichment_topic", lambda session, uid: False)
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {"core_skills": [], "emerging_skills": []},
    )
    insert_calls: list[dict] = []
    monkeypatch.setattr(
        cpa, "insert_new_outline_topic", lambda *a, **k: insert_calls.append(k)
    )

    result = cpa.maybe_trigger_enrichment(MagicMock(), "u1", "Backend Engineer", "t1")

    assert result is None
    assert insert_calls == []


def test_maybe_trigger_enrichment_does_nothing_when_role_not_in_roles_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "has_pending_enrichment_topic", lambda session, uid: False)
    monkeypatch.setattr(cpa, "get_role", lambda session, role: None)

    result = cpa.maybe_trigger_enrichment(MagicMock(), "u1", "Obscure Role", "t1")

    assert result is None


# --- Structural pace isolation for enrichment verification (Part 3) -------


def test_complete_topic_verification_enrichment_never_writes_pace_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _all_full_credit_attempts()  # would be "ahead"-eligible if evaluated
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    snapshot_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "write_pace_snapshot", lambda *a, **k: snapshot_calls.append((a, k))
    )
    history_calls: list[str] = []
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: history_calls.append(uid) or [],
    )
    completed_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "mark_topic_completed",
        lambda *args, **kwargs: completed_calls.append((args, kwargs)),
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=1, days_expected=1, is_enrichment=True
    )

    assert snapshot_calls == []
    assert history_calls == []  # drift isn't even evaluated for enrichment
    assert result.drift is None
    assert result.enrichment_topic is None
    assert result.pace_extension_applied is None
    assert len(completed_calls) == 1  # still marks completed, for closing-note credit


def test_complete_topic_verification_enrichment_half_credit_still_skips_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even when full/half credit was earned (a topic can still complete
    with HALF_CREDIT slots via the retry-cap teach-in) — is_enrichment=True
    must still skip the pace_snapshots write unconditionally.
    """
    attempts = [
        {
            "question_number": q,
            "attempt_number": 1 if q <= 3 else cpa.MAX_VERIFICATION_ATTEMPTS,
            "passed": q <= 3,
            "credit": 1.0 if q <= 3 else 0.5,
        }
        for q in range(1, 6)
    ]
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    snapshot_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "write_pace_snapshot", lambda *a, **k: snapshot_calls.append((a, k))
    )
    monkeypatch.setattr(cpa, "mark_topic_completed", lambda *a, **k: None)

    cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=1, days_expected=1, is_enrichment=True
    )

    assert snapshot_calls == []


def test_complete_topic_verification_non_enrichment_still_writes_pace_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: adding `is_enrichment` must not change the
    default (`is_enrichment=False`) behavior — pace_snapshots still gets
    written exactly as before.
    """
    attempts = _all_full_credit_attempts()
    snapshot_calls, completed_calls = _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(cpa, "get_pace_snapshot_history", lambda session, uid: [])

    cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=1, days_expected=1
    )

    assert len(snapshot_calls) == 1
    assert len(completed_calls) == 1


async def test_complete_topic_test_out_with_is_enrichment_skips_pace_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test-out AND enrichment together: nothing in this codebase's
    test-out logic restricts it to non-enrichment topics (both regular
    verification and test-out drive the identical
    complete_topic_verification completion path), so this combination is
    treated as genuinely possible and tested here, not assumed impossible.
    """
    attempts = [_resolved_attempt(q, cpa.FULL_CREDIT) for q in range(1, 6)]
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    snapshot_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "write_pace_snapshot", lambda *a, **k: snapshot_calls.append((a, k))
    )
    completed_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "mark_topic_completed",
        lambda *args, **kwargs: completed_calls.append((args, kwargs)),
    )

    result = await cpa.complete_topic_test_out(
        MagicMock(), "u1", "t1", days_taken=1, days_expected=1, is_enrichment=True
    )

    assert snapshot_calls == []
    assert result.completion.drift is None
    assert completed_calls[0][1] == {"status": cpa.COMPLETED_TEST_OUT_STATUS}


# --- Patch-note delivery wiring (patch-delivery task, Part 1) -------------


def test_maybe_deliver_patch_inserts_high_confidence_patch_and_marks_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pending_patch = {
        "id": "patch-1",
        "user_id": "u1",
        "origin_topic_id": "topic-sql",
        "new_content": "SQL is now more in-demand.",
        "source_url": "https://example.com/sql-update",
        "confidence": "high",
        "status": "pending",
        "created_at": None,
        "resolved_at": None,
    }
    monkeypatch.setattr(
        cpa, "get_pending_patch_notes", lambda session, uid: [pending_patch]
    )

    def _fake_get_topic(session: object, topic_id: str) -> dict | None:
        if topic_id == "t1":
            return {"id": "t1", "topic_name": "Docker", "hierarchy_position": 5}
        if topic_id == "topic-sql":
            return {"id": "topic-sql", "topic_name": "SQL", "hierarchy_position": 2}
        return None

    monkeypatch.setattr(cpa, "get_topic", _fake_get_topic)
    insert_calls: list[dict] = []

    def _fake_insert(session: object, **kwargs: object) -> dict:
        insert_calls.append(kwargs)
        return {"id": "new-patch-topic-1", **kwargs}

    monkeypatch.setattr(cpa, "insert_new_outline_topic", _fake_insert)
    update_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "update_patch_note_status",
        lambda session, patch_id, status, resolved_at: update_calls.append(
            (patch_id, status, resolved_at)
        ),
    )

    result = cpa.maybe_deliver_patch(MagicMock(), "u1", "t1")

    assert result is not None
    assert len(insert_calls) == 1
    call = insert_calls[0]
    assert call["user_id"] == "u1"
    assert call["topic_name"] == "SQL (Update)"
    assert call["topic_group"] == "SQL (Update)"
    assert call["is_enrichment"] is False
    assert call["source_url"] == "https://example.com/sql-update"
    assert call["source_type"] == "patch-note"
    assert call["confidence"] == ConfidenceTier.HIGH
    assert call["prerequisite_topic_ids"] == frozenset({"t1"})
    assert len(update_calls) == 1
    assert update_calls[0][0] == "patch-1"
    assert update_calls[0][1] == cpa.PatchStatus.DELIVERED


def test_maybe_deliver_patch_does_not_insert_low_confidence_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A low/uncertain-confidence patch routes to "ask_user", not
    "insert_now" — nothing gets inserted, and the patch-note stays
    pending (no status update at all).
    """
    pending_patch = {
        "id": "patch-2",
        "user_id": "u1",
        "origin_topic_id": "topic-sql",
        "new_content": "content",
        "source_url": "https://example.com/x",
        "confidence": "low",
        "status": "pending",
        "created_at": None,
        "resolved_at": None,
    }
    monkeypatch.setattr(
        cpa, "get_pending_patch_notes", lambda session, uid: [pending_patch]
    )
    monkeypatch.setattr(
        cpa,
        "get_topic",
        lambda session, topic_id: {
            "id": topic_id,
            "topic_name": "X",
            "hierarchy_position": 1,
        },
    )
    insert_calls: list[dict] = []
    monkeypatch.setattr(
        cpa, "insert_new_outline_topic", lambda *a, **k: insert_calls.append(k)
    )
    update_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa, "update_patch_note_status", lambda *a, **k: update_calls.append((a, k))
    )

    result = cpa.maybe_deliver_patch(MagicMock(), "u1", "t1")

    assert result is None
    assert insert_calls == []
    assert update_calls == []


def test_maybe_deliver_patch_no_pending_patches_does_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "get_pending_patch_notes", lambda session, uid: [])
    insert_calls: list[dict] = []
    monkeypatch.setattr(
        cpa, "insert_new_outline_topic", lambda *a, **k: insert_calls.append(k)
    )

    result = cpa.maybe_deliver_patch(MagicMock(), "u1", "t1")

    assert result is None
    assert insert_calls == []


def test_decide_patch_delivery_ask_user_can_construct_patch_decision_state() -> None:
    """Confirms the interface a future caller would use for the
    "needs_user_decision" outcome — this task does not build or wire the
    actual resolution (no UI exists yet), only confirms a
    `PatchDecisionState` constructs correctly from the decision; no
    resolution is fabricated.
    """
    from patches.patch_manager import PatchDecisionState, decide_patch_delivery

    pending = [
        {"id": "patch-low", "confidence": ConfidenceTier.LOW, "hierarchy_position": 1}
    ]

    decision = decide_patch_delivery(pending, current_hierarchy_position=5)

    assert decision.action == "ask_user"
    assert decision.patch_note_id is not None
    state = PatchDecisionState(patch_note_id=decision.patch_note_id)
    assert state.patch_note_id == "patch-low"
    assert state.resolved is False


def test_complete_topic_verification_calls_maybe_deliver_patch_regardless_of_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch-note delivery is not pace-gated — unlike enrichment, it must
    run on every non-enrichment completion regardless of drift's value
    (here: cold-start "on_track").
    """
    attempts = _all_full_credit_attempts()
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    monkeypatch.setattr(cpa, "write_pace_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(cpa, "mark_topic_completed", lambda *a, **k: None)
    monkeypatch.setattr(cpa, "get_pace_snapshot_history", lambda session, uid: [])
    deliver_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "maybe_deliver_patch",
        lambda *a, **k: deliver_calls.append(a) or None,
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "on_track"
    assert len(deliver_calls) == 1
    assert deliver_calls[0][1] == "u1"
    assert deliver_calls[0][2] == "t1"


def test_complete_topic_verification_ahead_and_pending_patch_both_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Co-occurrence judgment call: sustained-ahead drift and a pending
    high-confidence patch-note are independent conditions and can both
    fire in the same call — neither suppresses the other.
    """
    attempts = _all_full_credit_attempts()
    _patch_completion_writes(monkeypatch, attempts)
    monkeypatch.setattr(
        cpa,
        "get_pace_snapshot_history",
        lambda session, uid: [{"topic_score": 1.0, "timing_ratio": 1.0}] * 3,
    )
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    enrichment_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "maybe_trigger_enrichment",
        lambda *a, **k: enrichment_calls.append(a) or {"id": "enrichment-1"},
    )
    patch_deliver_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "maybe_deliver_patch",
        lambda *a, **k: patch_deliver_calls.append(a) or {"id": "patch-topic-1"},
    )

    result = cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert result.drift == "ahead"
    assert result.enrichment_topic == {"id": "enrichment-1"}
    assert result.delivered_patch_topic == {"id": "patch-topic-1"}
    assert len(enrichment_calls) == 1
    assert len(patch_deliver_calls) == 1


# --- Test-out: verification-first (PRD §7.6's exception) ------------------


def _resolved_attempt(
    question_number: int,
    credit: float,
    question_text: str | None = None,
    grading_criteria: str | None = None,
) -> dict:
    """Build one 'final attempt' fixture row, shaped like
    `get_attempts_for_topic`'s real return value. `credit` drives
    `passed`/`attempt_number` consistently with how
    `submit_verification_answer` actually writes them: a full-credit slot
    passed on some attempt (attempt_number doesn't matter for these
    tests, so attempt 1 is used); a half-credit slot only resolved by
    failing all the way to the retry cap.
    """
    passed = credit == cpa.FULL_CREDIT
    attempt_number = 1 if passed else cpa.MAX_VERIFICATION_ATTEMPTS
    return {
        "question_number": question_number,
        "attempt_number": attempt_number,
        "question_text": question_text or f"Question {question_number}?",
        "grading_criteria": grading_criteria
        or f"Must explain concept {question_number}.",
        "passed": passed,
        "credit": credit,
    }


async def test_generate_gap_study_content_calls_gemini_and_returns_study_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps({"study_content": "Rebase moves commits onto a new base."})
        ],
    )
    failed_questions = [
        cpa.VerificationQuestion(
            question_text="What is a rebase?",
            grading_criteria="Must define a rebase.",
            source_url=SOURCE_URL,
        )
    ]

    content = await cpa.generate_gap_study_content(failed_questions)

    assert content == "Rebase moves commits onto a new base."


async def test_generate_gap_study_content_rejects_empty_list() -> None:
    with pytest.raises(ValueError):
        await cpa.generate_gap_study_content([])


async def test_test_out_full_pass_writes_completed_test_out_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full pass (PRD §7.6): every slot resolved at FULL_CREDIT -> the
    topic is marked the schema's distinct `completed_test_out` value
    (not `completed`), and no study content is generated at all.
    """
    attempts = [_resolved_attempt(q, cpa.FULL_CREDIT) for q in range(1, 6)]
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    monkeypatch.setattr(cpa, "write_pace_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(cpa, "get_pace_snapshot_history", lambda session, uid: [])
    monkeypatch.setattr(cpa, "get_pending_patch_notes", lambda session, uid: [])
    completed_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "mark_topic_completed",
        lambda *args, **kwargs: completed_calls.append((args, kwargs)),
    )
    gap_content_calls: list[list] = []

    async def fake_generate_gap_study_content(failed_questions):
        gap_content_calls.append(failed_questions)
        return "should never be reached"

    monkeypatch.setattr(
        cpa, "generate_gap_study_content", fake_generate_gap_study_content
    )
    session = MagicMock()

    result = await cpa.complete_topic_test_out(
        session, "u1", "t1", days_taken=3, days_expected=3
    )

    assert result.full_pass is True
    assert gap_content_calls == []
    assert len(completed_calls) == 1
    assert completed_calls[0][0] == (session, "t1")
    assert completed_calls[0][1] == {"status": cpa.COMPLETED_TEST_OUT_STATUS}


async def test_test_out_partial_pass_relies_on_the_inline_teach_in_not_a_second_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is the key regression guard for a real interaction this task
    got wrong on the first pass and corrected after review: a partial
    pass's HALF_CREDIT slot(s) already received
    `submit_verification_answer`'s inline teach-in
    (`_build_taught_answer_message`), built from the identical
    `grading_criteria`, during the retry-cap attempt itself — moments
    before `complete_topic_test_out` is ever called. Calling
    `generate_gap_study_content` here too would re-teach the same rubric
    a second time in different words. `complete_topic_test_out` must
    still mark the topic complete (`completed_test_out`, PRD §7.7's
    completion rule doesn't distinguish full/half credit), but it must
    call `generate_gap_study_content` exactly zero times, regardless of
    how many slots are half-credit.
    """
    attempts = [
        _resolved_attempt(1, cpa.FULL_CREDIT, "What is a commit?"),
        _resolved_attempt(2, cpa.FULL_CREDIT, "What is a branch?"),
        _resolved_attempt(3, cpa.HALF_CREDIT, "What is a rebase?"),
        _resolved_attempt(4, cpa.FULL_CREDIT, "What is a merge?"),
        _resolved_attempt(5, cpa.HALF_CREDIT, "What is a cherry-pick?"),
    ]
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    monkeypatch.setattr(cpa, "write_pace_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(cpa, "get_pace_snapshot_history", lambda session, uid: [])
    monkeypatch.setattr(cpa, "get_pending_patch_notes", lambda session, uid: [])
    completed_calls: list[tuple] = []
    monkeypatch.setattr(
        cpa,
        "mark_topic_completed",
        lambda *args, **kwargs: completed_calls.append((args, kwargs)),
    )
    gap_content_calls: list[list] = []

    async def fake_generate_gap_study_content(failed_questions):
        gap_content_calls.append(failed_questions)
        return "should never be called from complete_topic_test_out"

    monkeypatch.setattr(
        cpa, "generate_gap_study_content", fake_generate_gap_study_content
    )
    session = MagicMock()

    result = await cpa.complete_topic_test_out(
        session, "u1", "t1", days_taken=3, days_expected=3
    )

    assert result.full_pass is False
    assert gap_content_calls == []  # the inline teach-in already covered this
    assert len(completed_calls) == 1
    assert completed_calls[0][1] == {"status": cpa.COMPLETED_TEST_OUT_STATUS}


async def test_submit_verification_answer_defaults_to_is_test_out_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: adding the `is_test_out` parameter must not
    change regular (non-test-out) verification's persisted behavior."""
    _patch_verification_skill(monkeypatch, ["Q1"], [True])
    written = _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        "t1", 1, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )
    await cpa.submit_verification_answer(
        state, "correct", session, TOPIC_SOURCE_MATERIAL, SOURCE_URL
    )

    assert written[0][1]["is_test_out"] is False


async def test_test_out_reuses_the_identical_retry_cap_machinery_not_a_second_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test-out drives the exact same `begin_verification_question`/
    `submit_verification_answer` functions and the exact same
    `MAX_VERIFICATION_ATTEMPTS == 3` cap as regular verification —
    `is_test_out` only changes what gets persisted per attempt, never the
    retry-counting mechanism. If test-out had grown a second, parallel
    retry-cap implementation instead of reusing this one, the
    attempt/write counts below (identical to
    `test_retry_cap_is_exactly_three_first_attempt_inside_the_counter`)
    would diverge.
    """
    fake_generate, fake_grade = _patch_verification_skill(
        monkeypatch, ["Q1", "Q2", "Q3"], [False, False, False]
    )
    written = _no_op_write_attempt(monkeypatch)
    session = MagicMock()

    state = await cpa.begin_verification_question(
        topic_id="t1",
        question_number=1,
        topic_source_material=TOPIC_SOURCE_MATERIAL,
        source_url=SOURCE_URL,
    )
    for _ in range(3):
        state = await cpa.submit_verification_answer(
            state,
            "wrong",
            session,
            TOPIC_SOURCE_MATERIAL,
            SOURCE_URL,
            is_test_out=True,
        )

    assert state.resolved
    assert state.credit == cpa.HALF_CREDIT
    assert fake_generate.call_count == 3  # identical shape to regular verification
    assert fake_grade.call_count == 3
    assert len(written) == 3
    assert all(kwargs["is_test_out"] is True for _, kwargs in written)


# --- Goal completion / closing note (patch-delivery task, Part 2) ---------


def test_is_goal_complete_true_when_all_core_topics_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [
        {"topic_name": "SQL", "is_enrichment": False, "status": cpa.COMPLETED_STATUS},
        {
            "topic_name": "Python",
            "is_enrichment": False,
            "status": cpa.COMPLETED_TEST_OUT_STATUS,
        },
        {"topic_name": "GraphQL", "is_enrichment": True, "status": "not_started"},
    ]
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: topics)

    assert cpa.is_goal_complete(MagicMock(), "u1") is True


def test_is_goal_complete_false_when_a_core_topic_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [
        {"topic_name": "SQL", "is_enrichment": False, "status": cpa.COMPLETED_STATUS},
        {"topic_name": "Python", "is_enrichment": False, "status": "in_progress"},
    ]
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: topics)

    assert cpa.is_goal_complete(MagicMock(), "u1") is False


def test_is_goal_complete_ignores_incomplete_enrichment_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topics = [
        {"topic_name": "SQL", "is_enrichment": False, "status": cpa.COMPLETED_STATUS},
        {"topic_name": "GraphQL", "is_enrichment": True, "status": "not_started"},
    ]
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: topics)

    assert cpa.is_goal_complete(MagicMock(), "u1") is True


def test_is_goal_complete_false_when_no_core_topics_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: [])

    assert cpa.is_goal_complete(MagicMock(), "u1") is False


@pytest.mark.parametrize(
    "text",
    [
        "You've grown from a junior developer into a senior one!",
        "This puts you at an expert level in the field.",
        "You started as a total beginner.",
        "Your score of 95% shows real mastery.",
    ],
)
def test_contains_banned_leveling_language_detects_banned_terms(text: str) -> None:
    assert cpa._contains_banned_leveling_language(text) is True


def test_contains_banned_leveling_language_allows_clean_text() -> None:
    text = (
        "Congratulations on completing your learning plan! You built real, "
        "demonstrable skills employers are looking for right now."
    )
    assert cpa._contains_banned_leveling_language(text) is False


async def test_generate_closing_note_fast_learner_lists_enrichment_strengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {
            "core_skills": [],
            "emerging_skills": [
                {"skill": "gRPC", "source_url": "https://x", "confidence": "medium"}
            ],
        },
    )
    topics = [
        {"topic_name": "SQL", "is_enrichment": False, "status": cpa.COMPLETED_STATUS},
        {
            "topic_name": "GraphQL",
            "is_enrichment": True,
            "status": cpa.COMPLETED_STATUS,
        },
    ]
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: topics)
    monkeypatch.setattr(cpa, "get_deferred_patch_notes", lambda session, uid: [])
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps({"note_text": "Congratulations! You built strong skills."})
        ],
    )

    result = await cpa.generate_closing_note(MagicMock(), "u1")

    assert result.resolved_role == "Backend Engineer"
    assert result.demonstrated_strengths == ["GraphQL"]
    assert result.suggested_next_steps == []
    assert result.note_text == "Congratulations! You built strong skills."
    assert result.deferred_patch_notes == []


async def test_generate_closing_note_core_only_learner_suggests_emerging_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {
            "core_skills": [],
            "emerging_skills": [
                {"skill": "gRPC", "source_url": "https://x", "confidence": "medium"},
                {"skill": "GraphQL", "source_url": "https://y", "confidence": "low"},
            ],
        },
    )
    topics = [
        {"topic_name": "SQL", "is_enrichment": False, "status": cpa.COMPLETED_STATUS}
    ]
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: topics)
    monkeypatch.setattr(cpa, "get_deferred_patch_notes", lambda session, uid: [])
    _patch_gemini(
        monkeypatch,
        responses=[json.dumps({"note_text": "Great work finishing the plan!"})],
    )

    result = await cpa.generate_closing_note(MagicMock(), "u1")

    assert result.demonstrated_strengths == []
    assert result.suggested_next_steps == ["gRPC", "GraphQL"]


async def test_generate_closing_note_surfaces_deferred_patch_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {"core_skills": [], "emerging_skills": []},
    )
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: [])
    deferred = [{"id": "patch-1", "new_content": "x", "status": "deferred"}]
    monkeypatch.setattr(cpa, "get_deferred_patch_notes", lambda session, uid: deferred)
    _patch_gemini(monkeypatch, responses=[json.dumps({"note_text": "text"})])

    result = await cpa.generate_closing_note(MagicMock(), "u1")

    assert result.deferred_patch_notes == deferred


async def test_generate_closing_note_unaffected_when_no_deferred_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {"core_skills": [], "emerging_skills": []},
    )
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: [])
    monkeypatch.setattr(cpa, "get_deferred_patch_notes", lambda session, uid: [])
    _patch_gemini(monkeypatch, responses=[json.dumps({"note_text": "text"})])

    result = await cpa.generate_closing_note(MagicMock(), "u1")

    assert result.deferred_patch_notes == []


async def test_generate_closing_note_raises_on_banned_leveling_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cpa, "get_user", lambda session, uid: {"resolved_role": "Backend Engineer"}
    )
    monkeypatch.setattr(
        cpa,
        "get_role",
        lambda session, role: {"core_skills": [], "emerging_skills": []},
    )
    monkeypatch.setattr(cpa, "get_all_topics_for_user", lambda session, uid: [])
    monkeypatch.setattr(cpa, "get_deferred_patch_notes", lambda session, uid: [])
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {"note_text": "You've grown from a junior to a senior developer!"}
            )
        ],
    )

    with pytest.raises(GeminiCallError):
        await cpa.generate_closing_note(MagicMock(), "u1")


async def test_generate_closing_note_raises_if_no_resolved_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpa, "get_user", lambda session, uid: {"resolved_role": None})

    with pytest.raises(ValueError):
        await cpa.generate_closing_note(MagicMock(), "u1")
