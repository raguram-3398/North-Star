"""Tests for security/input_gate.py — clarify-gate bound/loop state and
structural reject detection.
"""

import pytest

from security.input_gate import (
    ClarifyGateStage,
    ClarifyGateState,
    advance_after_explanation_response,
    advance_after_narrowing_round,
    advance_after_proposal_response,
    detect_reject,
    has_reached_round_bound,
    resolve_after_grounding_check,
    start_clarify_gate,
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
    """Gherkin: 'the gate either accepts a resolved role or asks one more
    narrowing question' and 'the total narrowing rounds never exceed 2'.
    """
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
    """Gherkin: 'if any market signal is found, the system proceeds at low
    confidence'.
    """
    accept_own_words_state = ClarifyGateState(
        stage=ClarifyGateStage.ACCEPT_OWN_WORDS, narrowing_rounds_used=2
    )

    next_state = resolve_after_grounding_check(
        accept_own_words_state, market_signal_found=True
    )

    assert next_state.stage is ClarifyGateStage.RESOLVED


def test_resolve_after_grounding_check_zero_signal_exits() -> None:
    """Gherkin: 'if zero market signal is found, the system exits and
    builds no outline'.
    """
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
    """End-to-end Gherkin: 'User rejects the proposed interpretation
    twice' -> zero market signal -> exit.
    """
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
    """Same rejection path, but any market signal (even weak) proceeds at
    low confidence instead of exiting.
    """
    state = start_clarify_gate()
    state = advance_after_narrowing_round(state, resolved=False)
    state = advance_after_narrowing_round(state, resolved=False)
    state = advance_after_proposal_response(state, accepted=False)
    state = advance_after_explanation_response(state, accepted=False)

    state = resolve_after_grounding_check(state, market_signal_found=True)

    assert state.stage is ClarifyGateStage.RESOLVED
