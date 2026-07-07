"""Tests for security/input_gate.py: clarify-gate and outline-confirmation bound/loop state, goal classification, and reject detection."""

import pytest

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
    detect_reject,
    has_reached_outline_confirmation_bound,
    has_reached_round_bound,
    resolve_after_grounding_check,
    start_clarify_gate,
    start_outline_confirmation,
)


def test_detect_reject_flags_blank_input() -> None:
    assert detect_reject("") is True
    assert detect_reject("   ") is True
    assert detect_reject("\n\t  ") is True


def test_detect_reject_passes_non_blank_input() -> None:
    assert detect_reject("I want to make apps") is False


def test_start_clarify_gate_begins_in_narrowing_with_zero_rounds() -> None:
    state = start_clarify_gate()

    assert state.stage is ClarifyGateStage.NARROWING
    assert state.narrowing_rounds_used == 0


@pytest.mark.parametrize(
    ("rounds_used", "expected"),
    [(0, False), (1, False), (2, True), (3, True)],
)
def test_has_reached_round_bound_default_max(rounds_used: int, expected: bool) -> None:
    assert has_reached_round_bound(rounds_used) is expected


def test_has_reached_round_bound_respects_custom_max() -> None:
    assert has_reached_round_bound(1, max_rounds=1) is True
    assert has_reached_round_bound(0, max_rounds=1) is False


def test_advance_after_narrowing_round_resolved_goes_straight_to_resolved() -> None:
    state = start_clarify_gate()

    next_state = advance_after_narrowing_round(state, resolved=True)

    assert next_state.stage is ClarifyGateStage.RESOLVED
    assert next_state.narrowing_rounds_used == 0


def test_advance_after_narrowing_round_unresolved_stays_within_bound() -> None:
    """The gate either accepts a resolved role or asks one more narrowing question, never exceeding the round bound."""
    state = start_clarify_gate()

    state = advance_after_narrowing_round(state, resolved=False)
    assert state.stage is ClarifyGateStage.NARROWING
    assert state.narrowing_rounds_used == 1

    state = advance_after_narrowing_round(state, resolved=False)
    assert state.stage is ClarifyGateStage.PROPOSE_BEST_GUESS
    assert state.narrowing_rounds_used == 2


def test_advance_after_narrowing_round_rejects_wrong_stage() -> None:
    state = start_clarify_gate()
    resolved_state = advance_after_narrowing_round(state, resolved=True)

    with pytest.raises(ValueError):
        advance_after_narrowing_round(resolved_state, resolved=False)


def test_advance_after_proposal_response_accepted() -> None:
    propose_state = ClarifyGateState(
        stage=ClarifyGateStage.PROPOSE_BEST_GUESS, narrowing_rounds_used=2
    )

    next_state = advance_after_proposal_response(propose_state, accepted=True)

    assert next_state.stage is ClarifyGateStage.RESOLVED


def test_advance_after_proposal_response_rejected() -> None:
    propose_state = ClarifyGateState(
        stage=ClarifyGateStage.PROPOSE_BEST_GUESS, narrowing_rounds_used=2
    )

    next_state = advance_after_proposal_response(propose_state, accepted=False)

    assert next_state.stage is ClarifyGateStage.EXPLAIN_ROLE


def test_advance_after_proposal_response_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError):
        advance_after_proposal_response(start_clarify_gate(), accepted=True)


def test_advance_after_explanation_response_accepted() -> None:
    explain_state = ClarifyGateState(
        stage=ClarifyGateStage.EXPLAIN_ROLE, narrowing_rounds_used=2
    )

    next_state = advance_after_explanation_response(explain_state, accepted=True)

    assert next_state.stage is ClarifyGateStage.RESOLVED


def test_advance_after_explanation_response_rejected_moves_to_accept_own_words() -> (
    None
):
    explain_state = ClarifyGateState(
        stage=ClarifyGateStage.EXPLAIN_ROLE, narrowing_rounds_used=2
    )

    next_state = advance_after_explanation_response(explain_state, accepted=False)

    assert next_state.stage is ClarifyGateStage.ACCEPT_OWN_WORDS


def test_advance_after_explanation_response_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError):
        advance_after_explanation_response(start_clarify_gate(), accepted=True)


def test_resolve_after_grounding_check_signal_found_resolves() -> None:
    """If any market signal is found, the system proceeds at low confidence."""
    accept_own_words_state = ClarifyGateState(
        stage=ClarifyGateStage.ACCEPT_OWN_WORDS, narrowing_rounds_used=2
    )

    next_state = resolve_after_grounding_check(
        accept_own_words_state, market_signal_found=True
    )

    assert next_state.stage is ClarifyGateStage.RESOLVED


def test_resolve_after_grounding_check_zero_signal_exits() -> None:
    """If zero market signal is found, the system exits and builds no outline."""
    accept_own_words_state = ClarifyGateState(
        stage=ClarifyGateStage.ACCEPT_OWN_WORDS, narrowing_rounds_used=2
    )

    next_state = resolve_after_grounding_check(
        accept_own_words_state, market_signal_found=False
    )

    assert next_state.stage is ClarifyGateStage.EXITED


def test_resolve_after_grounding_check_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError):
        resolve_after_grounding_check(start_clarify_gate(), market_signal_found=True)


def test_full_sequence_rejects_proposal_and_explanation_then_exits_on_zero_signal() -> (
    None
):
    """End-to-end: user rejects the proposed interpretation twice, then zero market signal leads to exit."""
    state = start_clarify_gate()
    state = advance_after_narrowing_round(state, resolved=False)
    state = advance_after_narrowing_round(state, resolved=False)
    assert state.stage is ClarifyGateStage.PROPOSE_BEST_GUESS
    assert state.narrowing_rounds_used == 2

    state = advance_after_proposal_response(state, accepted=False)
    assert state.stage is ClarifyGateStage.EXPLAIN_ROLE

    state = advance_after_explanation_response(state, accepted=False)
    assert state.stage is ClarifyGateStage.ACCEPT_OWN_WORDS

    state = resolve_after_grounding_check(state, market_signal_found=False)
    assert state.stage is ClarifyGateStage.EXITED


def test_full_sequence_rejects_proposal_and_explanation_resolves_on_weak_signal() -> (
    None
):
    """Same rejection path, but any market signal, even weak, proceeds at low confidence instead of exiting."""
    state = start_clarify_gate()
    state = advance_after_narrowing_round(state, resolved=False)
    state = advance_after_narrowing_round(state, resolved=False)
    state = advance_after_proposal_response(state, accepted=False)
    state = advance_after_explanation_response(state, accepted=False)

    state = resolve_after_grounding_check(state, market_signal_found=True)

    assert state.stage is ClarifyGateStage.RESOLVED


@pytest.mark.parametrize(
    "stated_goal",
    [
        "",
        "   ",
        "\n\t",
        "asdkjfh",
        "kjfhskjd",
        "1234!!!",
        "🚀🔥",
        "!!!???",
        "banana",
        "purple",
        "Dragon Whisperer",
        "Chief Vibes Officer",
    ],
)
def test_classify_stated_goal_nonsense_examples(stated_goal: str) -> None:
    """Blank, keyboard-mash, emoji/symbols, an unrelated word, or a fabricated job title all classify as nonsense."""
    assert classify_stated_goal(stated_goal) is GoalClassification.NONSENSE


@pytest.mark.parametrize(
    "stated_goal",
    [
        "I want to make apps",
        "something with computers",
        "I like fixing things on my laptop",
        "data stuff",
        "AI",
        "coding",
        "something in tech",
    ],
)
def test_classify_stated_goal_vague_examples(stated_goal: str) -> None:
    """Vague-but-genuine goal statements are accepted into the narrowing loop, never rejected."""
    assert classify_stated_goal(stated_goal) is GoalClassification.VAGUE


@pytest.mark.parametrize(
    "stated_goal",
    [
        "Data Analyst",
        "Backend Engineer",
        "Frontend Engineer",
        "DevOps Engineer",
        "AI/ML Engineer",
    ],
)
def test_classify_stated_goal_clearly_real_role_examples(stated_goal: str) -> None:
    """A clearly real, well-formed role name accepts straight through to research."""
    assert classify_stated_goal(stated_goal) is GoalClassification.REAL


def test_classify_stated_goal_niche_real_role_is_not_gate_rejected() -> None:
    """A real-but-obscure or niche job title must classify as REAL, not NONSENSE or VAGUE, even if never seen before."""
    assert classify_stated_goal("Site Reliability Engineer") is GoalClassification.REAL
    assert (
        classify_stated_goal("Platform Reliability Architect")
        is GoalClassification.REAL
    )


def test_classify_stated_goal_fabricated_title_is_rejected_not_niche_real() -> None:
    """A fabricated or absurd title must be rejected as nonsense based on vocabulary, not on whether the role really exists."""
    assert classify_stated_goal("Dragon Whisperer") is GoalClassification.NONSENSE
    assert classify_stated_goal("Chief Vibes Officer") is GoalClassification.NONSENSE


def test_classify_stated_goal_never_performs_a_market_existence_check() -> None:
    """A title never seen before must still classify as REAL if lexically title-shaped, proving the check is lexical, not a lookup."""
    assert classify_stated_goal("Quantum Firmware Engineer") is GoalClassification.REAL


def test_narrowing_loop_never_exceeds_round_bound_under_repeated_non_resolution() -> (
    None
):
    """Hammering the loop with unresolved rounds well past the bound must never exceed MAX_NARROWING_ROUNDS or linger in NARROWING."""
    state = start_clarify_gate()
    for _ in range(10):
        if state.stage is not ClarifyGateStage.NARROWING:
            break
        state = advance_after_narrowing_round(state, resolved=False)
        assert state.narrowing_rounds_used <= 2

    assert state.stage is ClarifyGateStage.PROPOSE_BEST_GUESS
    assert state.narrowing_rounds_used == 2


def test_start_outline_confirmation_begins_reviewing_with_zero_rounds() -> None:
    state = start_outline_confirmation()

    assert state.stage is OutlineConfirmationStage.REVIEWING
    assert state.rounds_used == 0


@pytest.mark.parametrize(
    ("rounds_used", "expected"),
    [(0, False), (1, False), (2, True), (3, True)],
)
def test_has_reached_outline_confirmation_bound_default_max(
    rounds_used: int, expected: bool
) -> None:
    assert has_reached_outline_confirmation_bound(rounds_used) is expected


def test_question_never_consumes_a_round() -> None:
    """A free question must leave stage and round count completely unchanged, even when asked repeatedly."""
    state = start_outline_confirmation()

    for _ in range(5):
        state = advance_after_review_turn(state, OutlineReviewAction.QUESTION)
        assert state.stage is OutlineConfirmationStage.REVIEWING
        assert state.rounds_used == 0


def test_concern_consumes_a_round() -> None:
    state = start_outline_confirmation()

    state = advance_after_review_turn(state, OutlineReviewAction.CONCERN)

    assert state.stage is OutlineConfirmationStage.REVIEWING
    assert state.rounds_used == 1


def test_addition_request_consumes_a_round() -> None:
    state = start_outline_confirmation()

    state = advance_after_review_turn(state, OutlineReviewAction.ADDITION_REQUEST)

    assert state.stage is OutlineConfirmationStage.REVIEWING
    assert state.rounds_used == 1


def test_confirm_ends_review_immediately_regardless_of_rounds_used() -> None:
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )

    state = advance_after_review_turn(state, OutlineReviewAction.CONFIRM)

    assert state.stage is OutlineConfirmationStage.CONFIRMED


def test_confirm_ends_review_even_after_a_round_already_used() -> None:
    state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=1
    )

    state = advance_after_review_turn(state, OutlineReviewAction.CONFIRM)

    assert state.stage is OutlineConfirmationStage.CONFIRMED


def test_round_bound_reached_after_exactly_two_consuming_actions() -> None:
    """Mixing concerns and addition requests, both round-consuming, must reach the bound at exactly 2, never before or after."""
    state = start_outline_confirmation()

    state = advance_after_review_turn(state, OutlineReviewAction.CONCERN)
    assert state.stage is OutlineConfirmationStage.REVIEWING
    assert state.rounds_used == 1

    state = advance_after_review_turn(state, OutlineReviewAction.ADDITION_REQUEST)
    assert state.stage is OutlineConfirmationStage.BOUND_REACHED
    assert state.rounds_used == 2


def test_questions_interleaved_with_consuming_actions_never_advance_bound_early() -> (
    None
):
    """Free questions interleaved between round-consuming actions must not affect when the bound is reached."""
    state = start_outline_confirmation()

    for _ in range(3):
        state = advance_after_review_turn(state, OutlineReviewAction.QUESTION)
    state = advance_after_review_turn(state, OutlineReviewAction.CONCERN)
    for _ in range(3):
        state = advance_after_review_turn(state, OutlineReviewAction.QUESTION)
    assert state.stage is OutlineConfirmationStage.REVIEWING
    assert state.rounds_used == 1

    state = advance_after_review_turn(state, OutlineReviewAction.ADDITION_REQUEST)

    assert state.stage is OutlineConfirmationStage.BOUND_REACHED
    assert state.rounds_used == 2


def test_advance_after_review_turn_rejects_wrong_stage() -> None:
    confirmed_state = OutlineConfirmationState(
        stage=OutlineConfirmationStage.CONFIRMED, rounds_used=1
    )

    with pytest.raises(ValueError):
        advance_after_review_turn(confirmed_state, OutlineReviewAction.QUESTION)


def test_outline_confirmation_loop_never_exceeds_round_bound_under_stress() -> None:
    """Hammering the loop with round-consuming actions well past the bound must never advance beyond MAX_OUTLINE_CONFIRMATION_ROUNDS."""
    state = start_outline_confirmation()
    for _ in range(10):
        if state.stage is not OutlineConfirmationStage.REVIEWING:
            break
        state = advance_after_review_turn(state, OutlineReviewAction.CONCERN)
        assert state.rounds_used <= 2

    assert state.stage is OutlineConfirmationStage.BOUND_REACHED
    assert state.rounds_used == 2
