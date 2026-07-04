"""Clarify-gate bound/loop state and structural reject detection.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per LLM Call Discipline, this runs on raw user
input before it reaches any clarify-gate LLM call — never after any
content-processing step that could corrupt the pattern it needs to catch.

Tracks only the bounded-loop *mechanics* of PRD §7.2's Clarify Gate:
narrowing rounds (bounded to ~2), then propose-best-guess -> explain-role
-> accept-own-words -> zero-signal-exit. The *content* of each narrowing
question, proposal, and explanation is Agent 1's reasoning job
(Architecture §3) — this module only knows what stage the loop is in, how
many rounds have been used, and when it must terminate. Semantic
classification of a stated goal (real / vague / nonsense) and of a user's
free-text response (accepted / rejected) is likewise Agent 1's job; the
booleans this module's functions take as input are assumed already
decided by that reasoning step.
"""

from dataclasses import dataclass
from enum import Enum

MAX_NARROWING_ROUNDS = 2


class ClarifyGateStage(Enum):
    """A stage in the PRD §7.2 bounded-loop sequence."""

    NARROWING = "narrowing"
    PROPOSE_BEST_GUESS = "propose_best_guess"
    EXPLAIN_ROLE = "explain_role"
    ACCEPT_OWN_WORDS = "accept_own_words"
    RESOLVED = "resolved"
    EXITED = "exited"


@dataclass(frozen=True)
class ClarifyGateState:
    """Immutable snapshot of the clarify gate's loop state."""

    stage: ClarifyGateStage
    narrowing_rounds_used: int = 0


def detect_reject(raw_input: str) -> bool:
    """Detect a structural reject condition in raw user input — blank or
    whitespace-only text with no content to narrow on — per PRD §7.2.

    This is deliberately narrow: semantic judgment of whether a
    *non-blank* stated goal is a real role, vague-but-genuine, or clearly
    nonsense is Agent 1's reasoning job, not this module's.
    """
    return not raw_input.strip()


def start_clarify_gate() -> ClarifyGateState:
    """Return the initial clarify-gate state: the narrowing stage, zero
    rounds used.
    """
    return ClarifyGateState(stage=ClarifyGateStage.NARROWING, narrowing_rounds_used=0)


def has_reached_round_bound(
    narrowing_rounds_used: int, max_rounds: int = MAX_NARROWING_ROUNDS
) -> bool:
    """Determine whether the clarify-gate narrowing-question bound (~2
    rounds, PRD §7.2) has been reached.
    """
    return narrowing_rounds_used >= max_rounds


def advance_after_narrowing_round(
    state: ClarifyGateState, resolved: bool
) -> ClarifyGateState:
    """Record the outcome of one narrowing round.

    If Agent 1 resolved a concrete role from the user's answer, move
    straight to RESOLVED regardless of rounds used so far. Otherwise
    increment the round count and move to PROPOSE_BEST_GUESS once the
    ~2-round bound is reached, or stay in NARROWING for another round
    (PRD §7.2; Gherkin "resolves within the round bound").

    Raises ValueError if called outside the NARROWING stage.
    """
    if state.stage is not ClarifyGateStage.NARROWING:
        raise ValueError(
            f"advance_after_narrowing_round called outside NARROWING stage: "
            f"{state.stage}"
        )
    if resolved:
        return ClarifyGateState(
            stage=ClarifyGateStage.RESOLVED,
            narrowing_rounds_used=state.narrowing_rounds_used,
        )
    rounds_used = state.narrowing_rounds_used + 1
    next_stage = (
        ClarifyGateStage.PROPOSE_BEST_GUESS
        if has_reached_round_bound(rounds_used)
        else ClarifyGateStage.NARROWING
    )
    return ClarifyGateState(stage=next_stage, narrowing_rounds_used=rounds_used)


def advance_after_proposal_response(
    state: ClarifyGateState, accepted: bool
) -> ClarifyGateState:
    """Advance from PROPOSE_BEST_GUESS: RESOLVED if the user accepted the
    best-guess role, else EXPLAIN_ROLE (PRD §7.2).

    Raises ValueError if called outside the PROPOSE_BEST_GUESS stage.
    """
    if state.stage is not ClarifyGateStage.PROPOSE_BEST_GUESS:
        raise ValueError(
            "advance_after_proposal_response called outside "
            f"PROPOSE_BEST_GUESS stage: {state.stage}"
        )
    next_stage = (
        ClarifyGateStage.RESOLVED if accepted else ClarifyGateStage.EXPLAIN_ROLE
    )
    return ClarifyGateState(
        stage=next_stage, narrowing_rounds_used=state.narrowing_rounds_used
    )


def advance_after_explanation_response(
    state: ClarifyGateState, accepted: bool
) -> ClarifyGateState:
    """Advance from EXPLAIN_ROLE: RESOLVED if the user accepted the
    explained role, else ACCEPT_OWN_WORDS — PRD §7.2's second rejection,
    where the system accepts the user's own words verbatim.

    Raises ValueError if called outside the EXPLAIN_ROLE stage.
    """
    if state.stage is not ClarifyGateStage.EXPLAIN_ROLE:
        raise ValueError(
            f"advance_after_explanation_response called outside EXPLAIN_ROLE stage: "
            f"{state.stage}"
        )
    next_stage = (
        ClarifyGateStage.RESOLVED if accepted else ClarifyGateStage.ACCEPT_OWN_WORDS
    )
    return ClarifyGateState(
        stage=next_stage, narrowing_rounds_used=state.narrowing_rounds_used
    )


def resolve_after_grounding_check(
    state: ClarifyGateState, market_signal_found: bool
) -> ClarifyGateState:
    """Advance from ACCEPT_OWN_WORDS: RESOLVED if any market signal was
    found (confidence is assigned downstream by security/output_guard.py,
    at the "low" tier per PRD §7.2), else EXITED — no outline is built.

    Raises ValueError if called outside the ACCEPT_OWN_WORDS stage.
    """
    if state.stage is not ClarifyGateStage.ACCEPT_OWN_WORDS:
        raise ValueError(
            f"resolve_after_grounding_check called outside ACCEPT_OWN_WORDS stage: "
            f"{state.stage}"
        )
    next_stage = (
        ClarifyGateStage.RESOLVED if market_signal_found else ClarifyGateStage.EXITED
    )
    return ClarifyGateState(
        stage=next_stage, narrowing_rounds_used=state.narrowing_rounds_used
    )
