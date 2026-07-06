"""Tests for main.py — the Streamlit orchestration skeleton.

Uses `streamlit.testing.v1.AppTest`, which runs the real script (widgets,
`st.session_state`, reruns) rather than calling the render functions
directly — those functions are not callable outside a live Streamlit
script-run context (they call `st.write`/`st.button`/etc.), so this is the
only real way to exercise them, matching Streamlit's own recommended
testing approach.

Every underlying agent/data function is mocked at the module that
*defines* it (`agents.research_outline_agent.ground_role`, etc.) — this
still counts as "patch where it's used" for this specific harness: `AppTest`
re-executes main.py's entire top-level code (including every
`from x import y` statement) from scratch on every single `.run()` call,
exactly like a real Streamlit rerun, so main.py's own name binding is
re-resolved from the (now-patched) source module on every run. Confirmed
directly against the real harness before writing this suite, not assumed.

`db.connection.get_session` is mocked everywhere (a `MagicMock` session) —
consistent with this codebase's no-SQLite-substitute, mocked-Session
convention; no test here touches a real database.

Rather than replaying the whole pipeline from Intake for every test,
most tests seed `at.session_state` directly before the first `.run()` —
confirmed this works (session_state set before the first run survives
`_init_session_state`'s "only set defaults for missing keys" guard).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from streamlit.testing.v1 import AppTest

import agents.coaching_pace_agent as cpa
import agents.research_outline_agent as roa
import data.outline_topics as outline_topics_module
import data.users as users_module
import db.connection as connection
from agents.research_outline_agent import (
    ClarifyGateContext,
    ClarifyGateStage,
    ClarifyGateState,
    ClarifyGateTurn,
    InitialOutlineTopic,
    LiveGroundingResult,
    OutlineConfirmationTurn,
)
from data.grounding_fallback import GeneralKnowledgeFloorResult
from main import _build_verification_source
from security.input_gate import OutlineConfirmationStage, OutlineConfirmationState
from security.output_guard import ConfidenceTier, ValidatedGroundedContent

MAIN_PATH = "src/main.py"


@pytest.fixture(autouse=True)
def _mock_db_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_session = MagicMock()
    monkeypatch.setattr(connection, "get_session", lambda: fake_session)
    return fake_session


def _make_at() -> AppTest:
    return AppTest.from_file(MAIN_PATH, default_timeout=20)


def test_intake_creates_user_and_advances_to_clarify_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_create_user = MagicMock(return_value={"id": "user-123"})
    monkeypatch.setattr(users_module, "create_user", fake_create_user)

    at = _make_at()
    at.run()
    assert at.session_state["current_stage"] == "landing"
    at.button[0].click()  # "Begin" on the new Landing stage
    at.run()

    at.text_input[1].input("Backend Engineer")
    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.session_state["current_stage"] == "clarify_gate"
    assert at.session_state["user_id"] == "user-123"
    assert at.session_state["stated_goal"] == "Backend Engineer"
    fake_create_user.assert_called_once()
    _, kwargs = fake_create_user.call_args
    assert kwargs["available_time_per_week"] == 10


def test_clarify_gate_real_answer_resolves_and_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A REAL-classified stated goal resolves immediately (no narrowing
    round needed) — the resulting `Continue` button uses the gate's real
    `resolved_role`, not a placeholder pass-through of the raw stated goal.
    """
    fake_turn = ClarifyGateTurn(
        gate_state=ClarifyGateState(stage=ClarifyGateStage.RESOLVED),
        context=ClarifyGateContext(original_stated_goal="Backend Engineer"),
        message="Great — I'll build your plan around Backend Engineer.",
        resolved_role="Backend Engineer",
    )
    fake_begin_clarify_gate = AsyncMock(return_value=fake_turn)
    monkeypatch.setattr(roa, "begin_clarify_gate", fake_begin_clarify_gate)

    at = _make_at()
    at.session_state["current_stage"] = "clarify_gate"
    at.session_state["stated_goal"] = "Backend Engineer"
    at.session_state["user_id"] = "user-1"
    at.run()

    fake_begin_clarify_gate.assert_called_once_with("Backend Engineer")
    # No chat_input is offered once RESOLVED — the round bound must never
    # be bypassable, and there's nothing left to advance.
    assert len(at.chat_input) == 0

    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.session_state["resolved_role"] == "Backend Engineer"
    assert at.session_state["current_stage"] == "research_grounding"


def test_clarify_gate_narrowing_round_calls_advance_clarify_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A VAGUE stated goal enters NARROWING; a chat reply calls
    `advance_clarify_gate` with the real gate_state/context/conversation,
    and a second narrowing round (bound not yet reached) keeps
    `st.chat_input` available rather than exposing a bypass.
    """
    first_turn = ClarifyGateTurn(
        gate_state=ClarifyGateState(
            stage=ClarifyGateStage.NARROWING, narrowing_rounds_used=0
        ),
        context=ClarifyGateContext(original_stated_goal="coding"),
        message="What kind of coding interests you most?",
    )
    monkeypatch.setattr(roa, "begin_clarify_gate", AsyncMock(return_value=first_turn))

    at = _make_at()
    at.session_state["current_stage"] = "clarify_gate"
    at.session_state["stated_goal"] = "coding"
    at.session_state["user_id"] = "user-1"
    at.run()

    assert len(at.chat_input) == 1

    second_turn = ClarifyGateTurn(
        gate_state=ClarifyGateState(
            stage=ClarifyGateStage.NARROWING, narrowing_rounds_used=1
        ),
        context=ClarifyGateContext(original_stated_goal="coding"),
        message="Backend, frontend, or something else?",
    )
    fake_advance = AsyncMock(return_value=second_turn)
    monkeypatch.setattr(roa, "advance_clarify_gate", fake_advance)

    at.chat_input[0].set_value("Backend stuff").run()

    assert not at.exception
    fake_advance.assert_called_once()
    args = fake_advance.call_args[0]
    assert args[0] == first_turn.gate_state
    assert args[1] == first_turn.context
    assert args[2] == [{"role": "agent", "content": first_turn.message}]
    assert args[3] == "Backend stuff"
    assert at.session_state["clarify_turn"] is second_turn
    assert at.session_state["clarify_conversation"] == [
        {"role": "agent", "content": first_turn.message},
        {"role": "user", "content": "Backend stuff"},
        {"role": "agent", "content": second_turn.message},
    ]
    # Bound not yet reached — still offers a way to reply, still no way to
    # skip past the gate.
    assert len(at.chat_input) == 1
    assert len(at.button) == 0


def test_clarify_gate_exited_uses_original_stated_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The zero-market-signal exit (PRD §7.2) routes to Research & Market
    Grounding using the gate's `context.original_stated_goal` — the real
    resolved goal for this path, since `resolved_role` stays None on an
    EXITED turn by contract.
    """
    exited_turn = ClarifyGateTurn(
        gate_state=ClarifyGateState(stage=ClarifyGateStage.EXITED),
        context=ClarifyGateContext(original_stated_goal="Dragon Whisperer II"),
        message="I couldn't find any current hiring activity for this.",
        exited=True,
    )
    monkeypatch.setattr(roa, "begin_clarify_gate", AsyncMock(return_value=exited_turn))

    at = _make_at()
    at.session_state["current_stage"] = "clarify_gate"
    at.session_state["stated_goal"] = "Dragon Whisperer II"
    at.session_state["user_id"] = "user-1"
    at.run()

    assert len(at.chat_input) == 0
    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.session_state["resolved_role"] == "Dragon Whisperer II"
    assert at.session_state["current_stage"] == "research_grounding"


def test_research_grounding_live_result_persists_role_and_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_result = LiveGroundingResult(
        role_name="Backend Engineer",
        skills=[
            ValidatedGroundedContent(
                source_url="https://example.com/a",
                source_type="job_listing",
                confidence=ConfidenceTier.HIGH,
                extra={"skill": "Docker"},
            )
        ],
        confidence=ConfidenceTier.HIGH,
        has_conflict=False,
        himalayas_status="signal",
        tavily_status="signal",
    )
    fake_ground_role = AsyncMock(return_value=fake_result)
    monkeypatch.setattr(roa, "ground_role", fake_ground_role)
    fake_set_resolved_role = MagicMock()
    monkeypatch.setattr(users_module, "set_resolved_role", fake_set_resolved_role)

    at = _make_at()
    at.session_state["current_stage"] = "research_grounding"
    at.session_state["user_id"] = "user-1"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.run()

    fake_ground_role.assert_called_once()
    assert fake_ground_role.call_args[0][0] == "Backend Engineer"

    at.button[0].click()
    at.run()

    assert not at.exception
    fake_set_resolved_role.assert_called_once_with(
        fake_ground_role.call_args[0][1], "user-1", "Backend Engineer", "high"
    )
    assert at.session_state["current_stage"] == "outline_creation"


def test_research_grounding_general_knowledge_floor_is_a_dead_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_result = GeneralKnowledgeFloorResult(
        role_name="Obscure Role",
        confidence=ConfidenceTier.GENERAL_KNOWLEDGE_ONLY,
        label="No cached or live market data is available.",
    )
    monkeypatch.setattr(roa, "ground_role", AsyncMock(return_value=fake_result))

    at = _make_at()
    at.session_state["current_stage"] = "research_grounding"
    at.session_state["user_id"] = "user-1"
    at.session_state["resolved_role"] = "Obscure Role"
    at.run()

    assert not at.exception
    # No outline can be built — no "continue" button is ever rendered.
    assert len(at.button) == 0
    assert at.session_state["current_stage"] == "research_grounding"


def test_outline_creation_calls_insert_outline_topics_with_its_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grounded_skill = ValidatedGroundedContent(
        source_url="https://example.com/a",
        source_type="job_listing",
        confidence=ConfidenceTier.HIGH,
        extra={"skill": "Docker"},
    )
    fake_grounding_result = LiveGroundingResult(
        role_name="Backend Engineer",
        skills=[grounded_skill],
        confidence=ConfidenceTier.HIGH,
        has_conflict=False,
        himalayas_status="signal",
        tavily_status="signal",
    )
    fake_topics = [
        InitialOutlineTopic(
            topic_name="Docker basics",
            hierarchy_position=1,
            topic_group="Docker",
            position_in_group=1,
            source_url="https://example.com/a",
            source_type="job_listing",
            confidence=ConfidenceTier.HIGH,
            is_enrichment=False,
            status="not_started",
        )
    ]
    fake_create_initial_outline = AsyncMock(return_value=fake_topics)
    monkeypatch.setattr(roa, "create_initial_outline", fake_create_initial_outline)
    fake_insert_outline_topics = MagicMock(
        return_value=[
            {
                "id": "topic-1",
                "hierarchy_position": 1,
                "topic_name": "Docker basics",
                "topic_group": "Docker",
            }
        ]
    )
    monkeypatch.setattr(
        outline_topics_module, "insert_outline_topics", fake_insert_outline_topics
    )

    at = _make_at()
    at.session_state["current_stage"] = "outline_creation"
    at.session_state["user_id"] = "user-1"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["grounding_result"] = fake_grounding_result
    at.run()

    assert not at.exception
    fake_create_initial_outline.assert_called_once()
    fake_insert_outline_topics.assert_called_once()
    call_args = fake_insert_outline_topics.call_args[0]
    assert call_args[1] == "user-1"
    # The actual point of this test: the exact object create_initial_outline
    # returned is what gets passed to insert_outline_topics, unmodified.
    assert call_args[2] is fake_topics

    at.button[0].click()
    at.run()
    assert at.session_state["current_stage"] == "outline_confirmation"


def _fake_outline_topics() -> list[InitialOutlineTopic]:
    return [
        InitialOutlineTopic(
            topic_name="Docker basics",
            hierarchy_position=1,
            topic_group="Docker",
            position_in_group=1,
            source_url="https://example.com/a",
            source_type="job_listing",
            confidence=ConfidenceTier.HIGH,
            is_enrichment=False,
            status="not_started",
        )
    ]


def test_outline_confirmation_initial_render_shows_outline_and_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_topics = _fake_outline_topics()
    fake_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(stage=OutlineConfirmationStage.REVIEWING),
        message="Here's your learning plan.",
        topics=fake_topics,
    )
    fake_begin = AsyncMock(return_value=fake_turn)
    monkeypatch.setattr(roa, "begin_outline_confirmation", fake_begin)

    at = _make_at()
    at.session_state["current_stage"] = "outline_confirmation"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["outline_topics"] = fake_topics
    at.session_state["user_id"] = "user-1"
    at.run()

    assert not at.exception
    fake_begin.assert_called_once_with("Backend Engineer", fake_topics)
    assert at.session_state["outline_confirmation_conversation"] == [
        {"role": "agent", "content": "Here's your learning plan."}
    ]
    # REVIEWING is not concluded — a chat_input is offered, not a button.
    assert len(at.chat_input) == 1
    assert len(at.button) == 0


def test_outline_confirmation_review_turn_calls_handle_review_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concern consumes a round (per `advance_after_review_turn`, not
    yet concluded) — the reply is rendered in-chat and the outline stays
    on screen, still awaiting further review.
    """
    fake_topics = _fake_outline_topics()
    first_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(stage=OutlineConfirmationStage.REVIEWING),
        message="Here's your learning plan.",
        topics=fake_topics,
    )
    monkeypatch.setattr(
        roa, "begin_outline_confirmation", AsyncMock(return_value=first_turn)
    )

    at = _make_at()
    at.session_state["current_stage"] = "outline_confirmation"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["outline_topics"] = fake_topics
    at.session_state["user_id"] = "user-1"
    at.run()

    next_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(
            stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
        ),
        message="Docker is included because it's core to this role.",
        topics=fake_topics,
        concluded=False,
    )
    fake_handle_review_turn = AsyncMock(return_value=next_turn)
    monkeypatch.setattr(roa, "handle_review_turn", fake_handle_review_turn)

    at.chat_input[0].set_value("Why is Docker in here?").run()

    assert not at.exception
    fake_handle_review_turn.assert_called_once_with(
        first_turn.state, "Backend Engineer", fake_topics, "Why is Docker in here?"
    )
    assert at.session_state["outline_confirmation_turn"] is next_turn
    assert at.session_state["outline_confirmation_conversation"] == [
        {"role": "agent", "content": "Here's your learning plan."},
        {"role": "user", "content": "Why is Docker in here?"},
        {"role": "agent", "content": next_turn.message},
    ]
    assert at.session_state["current_stage"] == "outline_confirmation"
    assert len(at.chat_input) == 1
    assert len(at.button) == 0


def test_outline_confirmation_concluded_advances_to_lowest_hierarchy_not_started_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_topics = _fake_outline_topics()
    concluded_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(stage=OutlineConfirmationStage.CONFIRMED),
        message="Great — let's get started!",
        topics=fake_topics,
        concluded=True,
    )
    monkeypatch.setattr(
        roa, "begin_outline_confirmation", AsyncMock(return_value=concluded_turn)
    )
    all_topics = [
        {"id": "topic-completed", "hierarchy_position": 1, "status": "completed"},
        {"id": "topic-first", "hierarchy_position": 2, "status": "not_started"},
        {"id": "topic-second", "hierarchy_position": 3, "status": "not_started"},
    ]
    monkeypatch.setattr(
        outline_topics_module, "get_all_topics_for_user", lambda *a, **k: all_topics
    )

    at = _make_at()
    at.session_state["current_stage"] = "outline_confirmation"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["outline_topics"] = fake_topics
    at.session_state["user_id"] = "user-1"
    at.run()

    # Concluded — no chat_input offered, only the continue button.
    assert len(at.chat_input) == 0
    at.button[0].click()
    at.run()

    assert not at.exception
    assert at.session_state["current_topic_id"] == "topic-first"
    assert at.session_state["current_stage"] == "day_by_day_coaching"


def _fake_topic(**overrides: object) -> dict:
    topic = {
        "id": "topic-1",
        "topic_name": "Docker basics",
        "topic_group": "Docker",
        "position_in_group": 1,
        "hierarchy_position": 1,
        "source_url": "https://example.com/market",
        "source_type": "job_listing",
        "confidence": "high",
        "is_enrichment": False,
        "status": "not_started",
    }
    topic.update(overrides)
    return topic


def test_day_by_day_coaching_start_verification_builds_source_and_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        outline_topics_module, "get_topics_in_group", lambda *a, **k: [topic]
    )
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    docker_docs_link = {
        "url": "https://docs.docker.com/",
        "title": "Docker docs",
        "content": "...",
    }
    fake_content = cpa.DayContent(
        summary="Summary",
        theory_framing="Docker isolates processes using namespaces.",
        theory_links=[docker_docs_link],
        hands_on_exercise="Build an image",
        review_prompt="Review it",
        reflection_prompt="What surprised you?",
        preview="Tomorrow: volumes",
        remaining_content="",
    )
    monkeypatch.setattr(
        cpa, "generate_day_content", AsyncMock(return_value=fake_content)
    )
    monkeypatch.setattr(cpa, "record_day_content", MagicMock())

    at = _make_at()
    at.session_state["current_stage"] = "day_by_day_coaching"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.run()
    at.button[0].click()  # "Start Verification"
    at.run()

    assert not at.exception
    assert at.session_state["current_stage"] == "verification"
    assert at.session_state["verification_source_url"] == "https://docs.docker.com/"
    material = at.session_state["verification_source_material"]
    assert "Docker isolates processes" in material


def test_day_by_day_coaching_spillover_stays_and_increments_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        outline_topics_module, "get_topics_in_group", lambda *a, **k: [topic]
    )
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    fake_content = cpa.DayContent(
        summary="Summary",
        theory_framing="Theory",
        theory_links=[],
        hands_on_exercise=None,
        review_prompt=None,
        reflection_prompt="Reflect",
        preview="Preview",
        remaining_content="leftover material",
    )
    fake_generate_day_content = AsyncMock(return_value=fake_content)
    monkeypatch.setattr(cpa, "generate_day_content", fake_generate_day_content)
    monkeypatch.setattr(cpa, "record_day_content", MagicMock())

    at = _make_at()
    at.session_state["current_stage"] = "day_by_day_coaching"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.run()
    at.button[0].click()  # "Continue to next day (same topic)"
    at.run()

    assert not at.exception
    assert at.session_state["current_stage"] == "day_by_day_coaching"
    assert at.session_state["day_number_for_topic"] == 2
    assert at.session_state["carried_over_content"] == "leftover material"
    # AppTest's .run() follows through the internal st.rerun() automatically,
    # so content is genuinely regenerated for the new day (correct behavior,
    # not a bug) — the real point of this test is that the *second* call
    # actually threads yesterday's remaining_content forward as today's
    # carried_over_content.
    assert fake_generate_day_content.call_count == 2
    _, second_call_kwargs = fake_generate_day_content.call_args_list[1]
    assert second_call_kwargs["carried_over_content"] == "leftover material"


def test_verification_single_slot_wiring_begin_and_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = cpa.VerificationQuestion(
        question_text="What does Docker use to isolate processes?",
        grading_criteria="Mentions namespaces/cgroups.",
        source_url="https://docs.docker.com/",
    )
    fake_slot_state = cpa.VerificationSlotState(
        topic_id="topic-1",
        question_number=1,
        attempt_number=1,
        current_question=question,
        previous_question_texts=(question.question_text,),
    )
    fake_begin = AsyncMock(return_value=fake_slot_state)
    monkeypatch.setattr(cpa, "begin_verification_question", fake_begin)
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 1
    at.session_state["verification_source_material"] = "material"
    at.session_state["verification_source_url"] = "https://docs.docker.com/"
    at.run()

    fake_begin.assert_called_once_with(
        "topic-1", 1, "material", "https://docs.docker.com/"
    )

    resolved_state = cpa.VerificationSlotState(
        topic_id="topic-1",
        question_number=1,
        attempt_number=1,
        current_question=question,
        previous_question_texts=(question.question_text,),
        resolved=True,
        credit=1.0,
    )
    fake_submit = AsyncMock(return_value=resolved_state)
    monkeypatch.setattr(cpa, "submit_verification_answer", fake_submit)

    at.text_input[0].input("namespaces and cgroups")
    at.button[0].click()  # "Submit answer"
    at.run()

    assert not at.exception
    fake_submit.assert_called_once()
    args = fake_submit.call_args[0]
    assert args[1] == "namespaces and cgroups"
    assert at.session_state["verification_slot_state"].resolved is True

    at.button[0].click()  # "Next question"
    at.run()
    assert at.session_state["current_question_number"] == 2


def test_verification_completion_calls_complete_topic_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    fake_result = cpa.TopicCompletionResult(
        topic_score=0.9,
        timing_ratio=1.0,
        combined_pace_signal=0.9,
        drift="ahead",
        enrichment_topic={"topic_name": "Kubernetes"},
        delivered_patch_topic=None,
    )
    fake_complete = MagicMock(return_value=fake_result)
    monkeypatch.setattr(cpa, "complete_topic_verification", fake_complete)
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: False)
    monkeypatch.setattr(
        outline_topics_module,
        "get_all_topics_for_user",
        lambda *a, **k: [
            {"id": "topic-1", "hierarchy_position": 1, "status": "completed"},
            {"id": "topic-2", "hierarchy_position": 2, "status": "not_started"},
        ],
    )

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 2
    at.run()

    assert not at.exception
    fake_complete.assert_called_once()
    _, kwargs = fake_complete.call_args
    assert fake_complete.call_args[0][1] == "user-1"
    assert fake_complete.call_args[0][2] == "topic-1"
    assert kwargs == {
        "days_taken": 2,
        "days_expected": 1,
        "is_test_out": False,
        "is_enrichment": False,
    }
    assert at.session_state["last_completion_result"] is fake_result

    at.button[0].click()  # "Continue to next topic"
    at.run()
    assert at.session_state["current_topic_id"] == "topic-2"
    assert at.session_state["day_number_for_topic"] == 1
    assert at.session_state["current_stage"] == "day_by_day_coaching"


def test_verification_completion_goal_complete_advances_to_goal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    fake_result = cpa.TopicCompletionResult(
        topic_score=1.0, timing_ratio=1.0, combined_pace_signal=1.0
    )
    monkeypatch.setattr(
        cpa, "complete_topic_verification", MagicMock(return_value=fake_result)
    )
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: True)

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 1
    at.run()
    at.button[0].click()  # "View closing note"
    at.run()

    assert not at.exception
    assert at.session_state["current_stage"] == "goal_completion"


def test_goal_completion_composes_generate_closing_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_note = cpa.ClosingNote(
        resolved_role="Backend Engineer",
        note_text="You've completed your plan!",
        demonstrated_strengths=["Kubernetes"],
        suggested_next_steps=[],
        deferred_patch_notes=[],
    )
    fake_generate_closing_note = AsyncMock(return_value=fake_note)
    monkeypatch.setattr(cpa, "generate_closing_note", fake_generate_closing_note)

    at = _make_at()
    at.session_state["current_stage"] = "goal_completion"
    at.session_state["user_id"] = "user-1"
    at.run()

    assert not at.exception
    fake_generate_closing_note.assert_called_once()
    assert fake_generate_closing_note.call_args[0][1] == "user-1"
    assert at.session_state["closing_note"] is fake_note


# --- _build_verification_source (pure function, no Streamlit needed) ----


def test_build_verification_source_uses_first_theory_link() -> None:
    content = cpa.DayContent(
        summary="s",
        theory_framing="Docker isolates processes.",
        theory_links=[
            {"url": "https://a.example/", "title": "A", "content": "content A"},
            {"url": "https://b.example/", "title": "B", "content": "content B"},
        ],
        hands_on_exercise=None,
        review_prompt=None,
        reflection_prompt="r",
        preview="p",
        remaining_content=None,
    )
    topic = _fake_topic(source_url="https://market.example/")

    material, source_url = _build_verification_source(content, topic)

    assert source_url == "https://a.example/"
    assert "Docker isolates processes." in material
    assert "content A" in material
    assert "content B" in material


def test_build_verification_source_falls_back_without_theory_links() -> None:
    content = cpa.DayContent(
        summary="s",
        theory_framing="Theory only.",
        theory_links=[],
        hands_on_exercise=None,
        review_prompt=None,
        reflection_prompt="r",
        preview="p",
        remaining_content=None,
    )
    topic = _fake_topic(source_url="https://market.example/")

    _, source_url = _build_verification_source(content, topic)

    assert source_url == "https://market.example/"
