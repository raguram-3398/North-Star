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
import cron.refresh_roles as refresh_roles_module
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
from cron.refresh_roles import SEED_ROLES
from data.grounding_fallback import GeneralKnowledgeFloorResult
from main import _build_verification_source, _run_async
from security.input_gate import (
    OutlineConfirmationStage,
    OutlineConfirmationState,
    OutlineReviewAction,
)
from security.output_guard import ConfidenceTier, ValidatedGroundedContent
from utils.exceptions import GeminiCallError, GroundingSourceCallError

MAIN_PATH = "src/main.py"


@pytest.fixture(autouse=True)
def _mock_db_session(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake_session = MagicMock()
    monkeypatch.setattr(connection, "get_session", lambda: fake_session)
    return fake_session


def _make_at() -> AppTest:
    """Every test gets a fresh `AppTest` with the startup staleness check
    (`_maybe_check_stale_roles`) pre-disabled — that check calls real
    `cron.refresh_roles` machinery which, against this file's bare
    `MagicMock` session (`_mock_db_session`), doesn't raise but instead
    treats every `SEED_ROLES` entry as stale (a `MagicMock` compares
    truthy against `is_stale`'s `datetime` arithmetic) and would attempt
    real `ground_role` grounding calls once per test — violating this
    suite's fast/offline/deterministic convention by accident, not
    design. Dedicated tests below explicitly re-enable it (`at.session_
    state["startup_staleness_checked"] = False`) to exercise the wiring
    itself, with `check_and_refresh_stale_roles` mocked.
    """
    at = AppTest.from_file(MAIN_PATH, default_timeout=20)
    at.session_state["startup_staleness_checked"] = True
    return at


def test_run_async_resets_gemini_client_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring regression test for the real, live-reproduced 'Outline
    Confirmation failed: Gemini call failed: Event loop is closed'
    incident: `_run_async` must reset the memoized Gemini client after
    every `asyncio.run(...)` completes, since that client's cached async
    transport is bound to the loop this call just closed — reused as-is
    by the next call's own fresh loop, it raises exactly that error (see
    `reset_gemini_client_for_new_event_loop`'s own docstring for the live
    repro). This test only proves the reset is *called* every time, not
    the underlying transport/event-loop behavior itself, which isn't
    reachable through a mock.
    """
    calls: list[bool] = []
    monkeypatch.setattr(
        "main.reset_gemini_client_for_new_event_loop", lambda: calls.append(True)
    )

    async def _ok() -> str:
        return "result"

    assert _run_async(_ok()) == "result"
    assert calls == [True]


def test_run_async_resets_gemini_client_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reset must fire in a `finally`, not only on the happy path — a
    failed Gemini call still closes its loop, so the next call would hit
    the same stale-transport bug if the reset were skipped on error."""
    calls: list[bool] = []
    monkeypatch.setattr(
        "main.reset_gemini_client_for_new_event_loop", lambda: calls.append(True)
    )

    async def _boom() -> str:
        raise GeminiCallError("boom")

    with pytest.raises(GeminiCallError):
        _run_async(_boom())
    assert calls == [True]


def test_maybe_check_stale_roles_runs_once_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Architecture §3's startup staleness check must run exactly once per
    browser session (`startup_staleness_checked`), not on every rerun.

    Patched at `cron.refresh_roles.check_and_refresh_stale_roles` (the
    defining module), not a `"main.X"` string target: `AppTest` re-executes
    main.py's `from cron.refresh_roles import ... check_and_refresh_stale_
    roles` on every `.run()`, which would silently re-bind main's own name
    back to the real function and clobber a patch made at that string
    target before the first run.
    """
    fake_check = AsyncMock(return_value=None)
    monkeypatch.setattr(
        refresh_roles_module, "check_and_refresh_stale_roles", fake_check
    )

    at = _make_at()
    at.session_state["startup_staleness_checked"] = False
    at.run()

    assert not at.exception
    fake_check.assert_called_once()
    assert fake_check.call_args[0][1] == SEED_ROLES
    assert at.session_state["startup_staleness_checked"] is True

    at.run()  # a second rerun must not call it again
    fake_check.assert_called_once()


def test_maybe_check_stale_roles_failure_shows_warning_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resilience layer must never become a hard dependency — a live-call
    failure here must degrade to a dismissable warning, never crash the
    rest of the app."""
    monkeypatch.setattr(
        refresh_roles_module,
        "check_and_refresh_stale_roles",
        AsyncMock(side_effect=GroundingSourceCallError("Tavily call failed")),
    )

    at = _make_at()
    at.session_state["startup_staleness_checked"] = False
    at.run()

    assert not at.exception
    assert at.session_state["startup_staleness_checked"] is True
    assert any("staleness check failed" in w.value for w in at.warning)


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


def test_outline_confirmation_addition_request_grounds_and_regenerates_outline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the real, user-reported gap: an addition
    request on Outline Confirmation previously always said grounding a
    raw addition wasn't wired up (Architecture §10/PRD §11 item 6). Now
    it grounds the request and regenerates the outline for real.
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

    addition_ack_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(
            stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
        ),
        message="Got it — I'll add 'can we add kubernetes?' and update your outline.",
        topics=fake_topics,
        concluded=False,
        action=OutlineReviewAction.ADDITION_REQUEST,
    )
    monkeypatch.setattr(
        roa, "handle_review_turn", AsyncMock(return_value=addition_ack_turn)
    )

    grounded_addition = ValidatedGroundedContent(
        source_url="https://kubernetes.io/docs/",
        source_type="web_search",
        confidence=ConfidenceTier.MEDIUM,
        extra={"skill": "Kubernetes"},
    )
    monkeypatch.setattr(
        roa, "ground_addition_request", AsyncMock(return_value=grounded_addition)
    )

    new_topics = [
        *fake_topics,
        InitialOutlineTopic(
            topic_name="Kubernetes basics",
            hierarchy_position=2,
            topic_group="Kubernetes",
            position_in_group=1,
            source_url="https://kubernetes.io/docs/",
            source_type="web_search",
            confidence=ConfidenceTier.MEDIUM,
            is_enrichment=False,
            status="not_started",
        ),
    ]
    regenerated_turn = OutlineConfirmationTurn(
        state=addition_ack_turn.state,
        message="Updated your outline to include Kubernetes.",
        topics=new_topics,
        concluded=False,
    )
    monkeypatch.setattr(
        roa,
        "regenerate_outline_with_addition",
        AsyncMock(return_value=regenerated_turn),
    )

    fake_insert_outline_topics = MagicMock(
        return_value=[
            {
                "id": "topic-2",
                "hierarchy_position": 2,
                "topic_name": "Kubernetes basics",
                "topic_group": "Kubernetes",
            }
        ]
    )
    monkeypatch.setattr(
        outline_topics_module, "insert_outline_topics", fake_insert_outline_topics
    )

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

    at = _make_at()
    at.session_state["current_stage"] = "outline_confirmation"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["outline_topics"] = fake_topics
    at.session_state["user_id"] = "user-1"
    at.session_state["grounding_result"] = fake_grounding_result
    at.run()

    at.chat_input[0].set_value("can we add kubernetes?").run()

    assert not at.exception
    fake_insert_outline_topics.assert_called_once()
    assert at.session_state["outline_topics"] is new_topics
    assert (
        at.session_state["persisted_topics"] == fake_insert_outline_topics.return_value
    )
    assert at.session_state["outline_confirmation_turn"] is regenerated_turn
    assert at.session_state["outline_confirmation_conversation"] == [
        {"role": "agent", "content": "Here's your learning plan."},
        {"role": "user", "content": "can we add kubernetes?"},
        {"role": "agent", "content": addition_ack_turn.message},
        {"role": "agent", "content": regenerated_turn.message},
    ]


def test_outline_confirmation_addition_request_ungroundable_leaves_outline_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never a fabricated source_url — if Tavily has nothing for the
    extracted skill name, the outline must stay exactly as it was, with a
    plain explanation appended to the conversation, not a crash or a
    silently invented topic."""
    fake_topics = _fake_outline_topics()
    first_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(stage=OutlineConfirmationStage.REVIEWING),
        message="Here's your learning plan.",
        topics=fake_topics,
    )
    monkeypatch.setattr(
        roa, "begin_outline_confirmation", AsyncMock(return_value=first_turn)
    )

    addition_ack_turn = OutlineConfirmationTurn(
        state=OutlineConfirmationState(
            stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
        ),
        message="Got it — I'll add 'zzz nonsense' and update your outline.",
        topics=fake_topics,
        concluded=False,
        action=OutlineReviewAction.ADDITION_REQUEST,
    )
    monkeypatch.setattr(
        roa, "handle_review_turn", AsyncMock(return_value=addition_ack_turn)
    )
    monkeypatch.setattr(roa, "ground_addition_request", AsyncMock(return_value=None))
    fake_regenerate = AsyncMock()
    monkeypatch.setattr(roa, "regenerate_outline_with_addition", fake_regenerate)
    fake_insert_outline_topics = MagicMock()
    monkeypatch.setattr(
        outline_topics_module, "insert_outline_topics", fake_insert_outline_topics
    )

    at = _make_at()
    at.session_state["current_stage"] = "outline_confirmation"
    at.session_state["resolved_role"] = "Backend Engineer"
    at.session_state["outline_topics"] = fake_topics
    at.session_state["user_id"] = "user-1"
    at.run()

    at.chat_input[0].set_value("add zzz nonsense").run()

    assert not at.exception
    fake_regenerate.assert_not_called()
    fake_insert_outline_topics.assert_not_called()
    assert at.session_state["outline_topics"] is fake_topics
    assert at.session_state["outline_confirmation_conversation"][-1] == {
        "role": "agent",
        "content": (
            "I couldn't find a reliable source for that addition, so "
            "your outline wasn't changed."
        ),
    }


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
    at.session_state["test_out_prompt_dismissed"] = True
    at.run()
    # Stepped reveal: Summary, Theory, Hands-on, Review, Reflection,
    # Preview — 5 "Next" clicks before the final "Start Quiz" button.
    for _ in range(5):
        assert at.button[0].label == "Next"
        at.button[0].click()
        at.run()
    assert at.button[0].label == "Start Quiz"
    at.button[0].click()
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
    at.session_state["test_out_prompt_dismissed"] = True
    at.run()
    # Stepped reveal: no hands-on/review this time (both None), so only
    # Summary, Theory, Reflection, Preview — 3 "Next" clicks before the
    # spillover button.
    for _ in range(3):
        assert at.button[0].label == "Next"
        at.button[0].click()
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


def test_day_by_day_coaching_steps_reveal_progressively_via_next_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real, requested UX change: each Day-by-Day Coaching section must
    appear one at a time as "Next" is clicked, not all at once on a
    single page load."""
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        outline_topics_module, "get_topics_in_group", lambda *a, **k: [topic]
    )
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    fake_content = cpa.DayContent(
        summary="SUMMARY_TEXT",
        theory_framing="THEORY_TEXT",
        theory_links=[],
        hands_on_exercise="HANDSON_TEXT",
        review_prompt="REVIEW_TEXT",
        reflection_prompt="REFLECTION_TEXT",
        preview="PREVIEW_TEXT",
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
    at.session_state["test_out_prompt_dismissed"] = True
    at.run()

    def _visible_texts() -> set[str]:
        return {m.value for m in at.markdown}

    assert "SUMMARY_TEXT" in _visible_texts()
    assert "THEORY_TEXT" not in _visible_texts()
    assert at.button[0].label == "Next"

    at.button[0].click()
    at.run()
    assert "THEORY_TEXT" in _visible_texts()
    assert "HANDSON_TEXT" not in _visible_texts()

    at.button[0].click()
    at.run()
    assert "HANDSON_TEXT" in _visible_texts()
    assert "REVIEW_TEXT" not in _visible_texts()

    at.button[0].click()
    at.run()
    assert "REVIEW_TEXT" in _visible_texts()
    assert "REFLECTION_TEXT" not in _visible_texts()

    at.button[0].click()
    at.run()
    assert "REFLECTION_TEXT" in _visible_texts()
    assert "PREVIEW_TEXT" not in _visible_texts()

    at.button[0].click()
    at.run()
    assert "PREVIEW_TEXT" in _visible_texts()
    assert not at.exception
    assert at.button[0].label == "Start Quiz"


def test_day_by_day_coaching_citation_links_to_the_real_full_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real, reported bug: citations previously
    rendered only the bare domain as plain text, so whatever became
    clickable (inconsistently, depending on the markdown renderer's own
    autolink heuristics) pointed at that site's homepage, not the actual
    article/video — e.g. a YouTube citation went to a broken YouTube
    landing page instead of the real video. The caption must now be an
    explicit markdown link to the citation's real, full URL.
    """
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        outline_topics_module, "get_topics_in_group", lambda *a, **k: [topic]
    )
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    real_video_url = "https://www.youtube.com/watch?v=abc123XYZ"
    fake_content = cpa.DayContent(
        summary="Summary",
        theory_framing="Theory",
        theory_links=[
            {
                "url": real_video_url,
                "title": "Linux For Beginners - Full Course [NEW]",
                "content": "...",
            }
        ],
        hands_on_exercise=None,
        review_prompt=None,
        reflection_prompt="Reflect",
        preview="Preview",
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
    at.session_state["test_out_prompt_dismissed"] = True
    at.run()
    at.button[0].click()  # reveal the Theory step
    at.run()

    assert not at.exception
    citation_captions = [c.value for c in at.caption if c.value.startswith("Source:")]
    assert len(citation_captions) == 1
    assert citation_captions[0] == (
        f"Source: [Linux For Beginners - Full Course [NEW]]({real_video_url}) "
        "— www.youtube.com"
    )


def test_day_by_day_coaching_shows_test_out_choice_before_generating_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD §7.6's test-out exception: "the user may trigger verification
    first, before study content is generated" — the choice screen must
    appear, and `generate_day_content` must NOT be called, until the user
    has explicitly declined it."""
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    fake_generate_day_content = AsyncMock()
    monkeypatch.setattr(cpa, "generate_day_content", fake_generate_day_content)

    at = _make_at()
    at.session_state["current_stage"] = "day_by_day_coaching"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.run()

    assert not at.exception
    fake_generate_day_content.assert_not_called()
    button_labels = {b.label for b in at.button}
    assert "Test out of this topic" in button_labels
    assert "No, teach me the material" in button_labels


def test_day_by_day_coaching_test_out_declined_falls_through_to_generation(
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

    decline_button = next(
        b for b in at.button if b.label == "No, teach me the material"
    )
    decline_button.click()
    at.run()

    assert not at.exception
    assert at.session_state["test_out_prompt_dismissed"] is True
    assert at.session_state["day_content"] is not None


def test_day_by_day_coaching_test_out_fetches_theory_links_and_advances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choosing "Test out" must fetch real teaching material directly
    (never call `generate_day_content` — that's the whole point of "before
    study content is generated") and jump straight to Verification with
    `is_test_out=True`."""
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    fake_generate_day_content = AsyncMock()
    monkeypatch.setattr(cpa, "generate_day_content", fake_generate_day_content)
    docker_docs_link = {
        "url": "https://docs.docker.com/",
        "title": "Docker docs",
        "content": "Docker isolates processes using namespaces.",
    }
    fake_fetch = AsyncMock(return_value=[docker_docs_link])
    monkeypatch.setattr(cpa, "fetch_theory_material_links", fake_fetch)

    at = _make_at()
    at.session_state["current_stage"] = "day_by_day_coaching"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.run()

    test_out_button = next(b for b in at.button if b.label == "Test out of this topic")
    test_out_button.click()
    at.run()

    assert not at.exception
    fake_generate_day_content.assert_not_called()
    fake_fetch.assert_called_once_with("Docker basics")
    assert at.session_state["current_stage"] == "verification"
    assert at.session_state["is_test_out"] is True
    assert at.session_state["verification_source_url"] == "https://docs.docker.com/"
    assert (
        "Docker isolates processes" in at.session_state["verification_source_material"]
    )


def test_day_by_day_coaching_test_out_refused_without_theory_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAUDE.md guardrail #6: never silently fabricate a source. If
    Tavily's theory search comes back empty, test-out must be refused
    (an error, staying on Day-by-Day Coaching) rather than generating
    verification questions with no real grounding material."""
    topic = _fake_topic()
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(cpa, "fetch_theory_material_links", AsyncMock(return_value=[]))

    at = _make_at()
    at.session_state["current_stage"] = "day_by_day_coaching"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.run()

    test_out_button = next(b for b in at.button if b.label == "Test out of this topic")
    test_out_button.click()
    at.run()

    assert not at.exception
    assert at.session_state["current_stage"] == "day_by_day_coaching"
    assert any("test-out isn't available" in e.value for e in at.error)


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
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
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


def test_verification_completion_shows_non_blocking_patch_decision_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRD §7.9's "ask_user" branch: a low/uncertain-confidence patch-note
    must render a banner, but "Continue to next topic" must remain
    clickable alongside it — ignoring the banner is always safe."""
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    pending_decision = cpa.PendingPatchDecision(
        patch_note_id="patch-1",
        origin_topic_id="topic-1",
        new_content="SQL is now more in-demand.",
        source_url="https://example.com/sql-update",
        confidence=ConfidenceTier.LOW,
    )
    fake_result = cpa.TopicCompletionResult(
        topic_score=0.9,
        timing_ratio=1.0,
        combined_pace_signal=0.9,
        pending_patch_decision=pending_decision,
    )
    monkeypatch.setattr(
        cpa, "complete_topic_verification", MagicMock(return_value=fake_result)
    )
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
    at.session_state["day_number_for_topic"] = 1
    at.run()

    assert not at.exception
    assert at.session_state["pending_patch_decision"] == pending_decision
    button_labels = {b.label for b in at.button}
    assert "Learn now" in button_labels
    assert "Defer" in button_labels
    assert "Continue to next topic" in button_labels


def test_patch_decision_banner_learn_now_resolves_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    pending_decision = cpa.PendingPatchDecision(
        patch_note_id="patch-1",
        origin_topic_id="topic-1",
        new_content="SQL is now more in-demand.",
        source_url="https://example.com/sql-update",
        confidence=ConfidenceTier.LOW,
    )
    fake_result = cpa.TopicCompletionResult(
        topic_score=0.9,
        timing_ratio=1.0,
        combined_pace_signal=0.9,
        pending_patch_decision=pending_decision,
    )
    monkeypatch.setattr(
        cpa, "complete_topic_verification", MagicMock(return_value=fake_result)
    )
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: False)
    fake_resolve = MagicMock(return_value=None)
    monkeypatch.setattr(cpa, "resolve_pending_patch_decision", fake_resolve)

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 1
    at.run()

    learn_now_button = next(b for b in at.button if b.label == "Learn now")
    learn_now_button.click()
    at.run()

    assert not at.exception
    fake_resolve.assert_called_once()
    args = fake_resolve.call_args[0]
    assert args[1] == pending_decision
    assert args[2] == "learn_now"
    assert at.session_state["pending_patch_decision"] is None


def test_patch_decision_banner_defer_resolves_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    pending_decision = cpa.PendingPatchDecision(
        patch_note_id="patch-1",
        origin_topic_id="topic-1",
        new_content="content",
        source_url="https://example.com/x",
        confidence=ConfidenceTier.LOW,
    )
    fake_result = cpa.TopicCompletionResult(
        topic_score=0.9,
        timing_ratio=1.0,
        combined_pace_signal=0.9,
        pending_patch_decision=pending_decision,
    )
    monkeypatch.setattr(
        cpa, "complete_topic_verification", MagicMock(return_value=fake_result)
    )
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: False)
    fake_resolve = MagicMock(return_value=None)
    monkeypatch.setattr(cpa, "resolve_pending_patch_decision", fake_resolve)

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 1
    at.run()

    defer_button = next(b for b in at.button if b.label == "Defer")
    defer_button.click()
    at.run()

    assert not at.exception
    fake_resolve.assert_called_once()
    args = fake_resolve.call_args[0]
    assert args[1] == pending_decision
    assert args[2] == "defer"
    assert at.session_state["pending_patch_decision"] is None


def test_verification_completion_test_out_full_pass_calls_complete_topic_test_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`is_test_out=True` (set by the Day-by-Day Coaching test-out choice)
    must route completion through `complete_topic_test_out`, not
    `complete_topic_verification` directly — and a full pass shows the
    "no study content needed" message (PRD §7.6)."""
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    fake_completion = cpa.TopicCompletionResult(
        topic_score=1.0, timing_ratio=1.0, combined_pace_signal=1.0
    )
    fake_test_out_result = cpa.TestOutResult(completion=fake_completion, full_pass=True)
    fake_complete_test_out = AsyncMock(return_value=fake_test_out_result)
    monkeypatch.setattr(cpa, "complete_topic_test_out", fake_complete_test_out)
    fake_complete_verification = MagicMock()
    monkeypatch.setattr(cpa, "complete_topic_verification", fake_complete_verification)
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: False)
    monkeypatch.setattr(
        outline_topics_module,
        "get_all_topics_for_user",
        lambda *a, **k: [
            {"id": "topic-1", "hierarchy_position": 1, "status": "completed"}
        ],
    )

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 1
    at.session_state["is_test_out"] = True
    at.run()

    assert not at.exception
    fake_complete_test_out.assert_called_once()
    fake_complete_verification.assert_not_called()
    _, kwargs = fake_complete_test_out.call_args
    assert fake_complete_test_out.call_args[0][1] == "user-1"
    assert fake_complete_test_out.call_args[0][2] == "topic-1"
    assert kwargs == {
        "days_taken": 1,
        "days_expected": 1,
        "is_enrichment": False,
    }
    assert at.session_state["last_completion_result"] is fake_completion
    assert at.session_state["last_test_out_full_pass"] is True
    assert any("full pass, no study content needed" in s.value for s in at.success)


def test_verification_completion_test_out_partial_pass_shows_partial_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
    fake_completion = cpa.TopicCompletionResult(
        topic_score=0.8, timing_ratio=1.0, combined_pace_signal=0.8
    )
    fake_test_out_result = cpa.TestOutResult(
        completion=fake_completion, full_pass=False
    )
    monkeypatch.setattr(
        cpa, "complete_topic_test_out", AsyncMock(return_value=fake_test_out_result)
    )
    monkeypatch.setattr(cpa, "is_goal_complete", lambda *a, **k: False)
    monkeypatch.setattr(
        outline_topics_module,
        "get_all_topics_for_user",
        lambda *a, **k: [
            {"id": "topic-1", "hierarchy_position": 1, "status": "completed"}
        ],
    )

    at = _make_at()
    at.session_state["current_stage"] = "verification"
    at.session_state["current_topic_id"] = "topic-1"
    at.session_state["user_id"] = "user-1"
    at.session_state["current_question_number"] = 6
    at.session_state["day_number_for_topic"] = 1
    at.session_state["is_test_out"] = True
    at.run()

    assert not at.exception
    assert at.session_state["last_test_out_full_pass"] is False
    assert any("Tested out — partial pass" in i.value for i in at.info)


def test_verification_completion_goal_complete_advances_to_goal_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topic = _fake_topic(is_enrichment=False)
    monkeypatch.setattr(outline_topics_module, "get_topic", lambda *a, **k: topic)
    monkeypatch.setattr(
        users_module, "get_user", lambda *a, **k: {"available_time_per_week": 10}
    )
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
