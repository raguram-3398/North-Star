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
from tests.test_research_outline_agent import (
    _FakeTavilyClient,
    _patch_gemini,
    _tavily_response,
    _tavily_result_dict,
)
from utils.exceptions import GroundingSourceCallError

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
    session = MagicMock()

    result = cpa.complete_topic_verification(
        session, "u1", "t1", days_taken=4, days_expected=5
    )

    expected_topic_score = (1.0 + 1.0 + 0.5 + 1.0 + 1.0) / 5
    assert result.topic_score == pytest.approx(expected_topic_score)
    assert result.timing_ratio == pytest.approx(4 / 5)
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


def test_complete_topic_verification_does_not_call_detect_sustained_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit scope-boundary test: acting on the pace signal (sustained
    drift) is the next task's job — this task only computes/persists it."""
    attempts = [
        {"question_number": q, "attempt_number": 1, "passed": True, "credit": 1.0}
        for q in range(1, 6)
    ]
    monkeypatch.setattr(
        cpa, "get_attempts_for_topic", lambda session, topic_id: attempts
    )
    monkeypatch.setattr(cpa, "write_pace_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(cpa, "mark_topic_completed", lambda *a, **k: None)

    called = []
    import pace.calculator as calculator_module

    monkeypatch.setattr(
        calculator_module,
        "detect_sustained_drift",
        lambda *a, **k: called.append((a, k)),
    )

    cpa.complete_topic_verification(
        MagicMock(), "u1", "t1", days_taken=5, days_expected=5
    )

    assert called == []
