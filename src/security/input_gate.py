"""Bounded-loop state tracking for the clarify gate and outline confirmation, plus first-pass goal classification."""

import re
from dataclasses import dataclass
from enum import Enum

MAX_NARROWING_ROUNDS = 2
MAX_OUTLINE_CONFIRMATION_ROUNDS = 2


class ClarifyGateStage(Enum):
    """A stage in the clarify gate's bounded-loop sequence."""

    NARROWING = "narrowing"
    PROPOSE_BEST_GUESS = "propose_best_guess"
    EXPLAIN_ROLE = "explain_role"
    ACCEPT_OWN_WORDS = "accept_own_words"
    RESOLVED = "resolved"
    EXITED = "exited"


class GoalClassification(Enum):
    """The first-pass, deterministic classification of a raw stated goal."""

    REAL = "real"
    VAGUE = "vague"
    NONSENSE = "nonsense"


@dataclass(frozen=True)
class ClarifyGateState:
    """Immutable snapshot of the clarify gate's loop state."""

    stage: ClarifyGateStage
    narrowing_rounds_used: int = 0


class OutlineConfirmationStage(Enum):
    """A stage in the outline confirmation bounded loop."""

    REVIEWING = "reviewing"
    CONFIRMED = "confirmed"
    BOUND_REACHED = "bound_reached"


class OutlineReviewAction(Enum):
    """How a user's outline-confirmation turn was classified."""

    QUESTION = "question"
    CONCERN = "concern"
    ADDITION_REQUEST = "addition_request"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class OutlineConfirmationState:
    """Immutable snapshot of the outline confirmation loop's state."""

    stage: OutlineConfirmationStage
    rounds_used: int = 0


def detect_reject(raw_input: str) -> bool:
    """Return True if raw_input is blank or whitespace-only."""
    return not raw_input.strip()


_VOWELS = frozenset("aeiou")

_VOWELLESS_REAL_WORDS = frozenset(
    {
        "ux",
        "ui",
        "it",
        "hr",
        "qa",
        "pr",
        "db",
        "os",
        "vr",
        "ar",
        "ci",
        "cd",
        "sdk",
        "api",
        "css",
        "sql",
        "http",
        "url",
        "vp",
    }
)

_MAX_PLAUSIBLE_CONSONANT_RUN = 4

_ROLE_NOUN_SUFFIXES = frozenset(
    {
        "engineer",
        "developer",
        "analyst",
        "manager",
        "officer",
        "specialist",
        "designer",
        "architect",
        "scientist",
        "administrator",
        "consultant",
        "coordinator",
        "director",
        "technician",
        "strategist",
        "lead",
        "advocate",
        "programmer",
    }
)

_VAGUE_TECH_SINGLE_WORDS = frozenset(
    {
        "ai",
        "ml",
        "coding",
        "code",
        "coder",
        "programming",
        "software",
        "hardware",
        "tech",
        "technology",
        "data",
        "it",
        "computers",
        "computer",
        "computing",
        "web",
        "apps",
        "app",
        "development",
        "developer",
        "dev",
        "devops",
        "cybersecurity",
        "cyber",
        "security",
        "networking",
        "networks",
        "robotics",
        "automation",
        "cloud",
        "database",
        "databases",
        "hacking",
        "engineering",
        "python",
        "java",
        "javascript",
        "html",
        "css",
    }
    | _ROLE_NOUN_SUFFIXES
)

_VAGUE_TECH_PHRASE_KEYWORDS = _VAGUE_TECH_SINGLE_WORDS | frozenset(
    {
        "laptop",
        "laptops",
        "phone",
        "phones",
        "gadget",
        "gadgets",
        "internet",
        "website",
        "websites",
        "program",
        "programs",
        "electronics",
        "circuits",
        "machine",
        "machines",
        "algorithm",
        "algorithms",
    }
)

_FABRICATED_TITLE_MARKERS = frozenset(
    {
        "vibes",
        "vibe",
        "dragon",
        "whisperer",
        "wizard",
        "wizardry",
        "ninja",
        "guru",
        "rockstar",
        "unicorn",
        "jedi",
        "sorcerer",
        "overlord",
        "superhero",
        "magic",
        "magician",
        "sparkle",
        "fairy",
        "genie",
    }
)


def _looks_like_a_word(token: str) -> bool:
    """Return True if token reads as a plausible word rather than keyboard mash."""
    folded = token.casefold()
    if folded in _VOWELLESS_REAL_WORDS:
        return True
    if not any(c in _VOWELS for c in folded):
        return False
    consonant_run = 0
    for char in folded:
        if char.isalpha() and char not in _VOWELS:
            consonant_run += 1
            if consonant_run >= _MAX_PLAUSIBLE_CONSONANT_RUN:
                return False
        else:
            consonant_run = 0
    return True


def classify_stated_goal(raw_input: str) -> GoalClassification:
    """Classify a raw stated career goal as REAL, VAGUE, or NONSENSE using lexical/plausibility rules."""
    if detect_reject(raw_input):
        return GoalClassification.NONSENSE

    text = raw_input.strip()
    if not any(c.isalpha() for c in text):
        return GoalClassification.NONSENSE

    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return GoalClassification.NONSENSE

    folded_words = [w.casefold() for w in words]

    if not any(_looks_like_a_word(w) for w in folded_words):
        return GoalClassification.NONSENSE

    if len(folded_words) == 1:
        single = folded_words[0]
        if single in _VAGUE_TECH_SINGLE_WORDS:
            return GoalClassification.VAGUE
        return GoalClassification.NONSENSE

    ends_with_role_noun = folded_words[-1] in _ROLE_NOUN_SUFFIXES
    has_fabricated_marker = any(w in _FABRICATED_TITLE_MARKERS for w in folded_words)

    if ends_with_role_noun:
        return (
            GoalClassification.NONSENSE
            if has_fabricated_marker
            else GoalClassification.REAL
        )

    if has_fabricated_marker:
        return GoalClassification.NONSENSE

    if any(w in _VAGUE_TECH_PHRASE_KEYWORDS for w in folded_words):
        return GoalClassification.VAGUE

    return GoalClassification.NONSENSE


def start_clarify_gate() -> ClarifyGateState:
    """Return the initial clarify-gate state: narrowing stage, zero rounds used."""
    return ClarifyGateState(stage=ClarifyGateStage.NARROWING, narrowing_rounds_used=0)


def has_reached_round_bound(
    narrowing_rounds_used: int, max_rounds: int = MAX_NARROWING_ROUNDS
) -> bool:
    """Return True if the clarify-gate narrowing-question round bound has been reached."""
    return narrowing_rounds_used >= max_rounds


def advance_after_narrowing_round(
    state: ClarifyGateState, resolved: bool
) -> ClarifyGateState:
    """Advance the clarify-gate state after one narrowing round, given whether the role was resolved."""
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
    """Advance from PROPOSE_BEST_GUESS to RESOLVED if accepted, else EXPLAIN_ROLE."""
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
    """Advance from EXPLAIN_ROLE to RESOLVED if accepted, else ACCEPT_OWN_WORDS."""
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
    """Advance from ACCEPT_OWN_WORDS to RESOLVED if a market signal was found, else EXITED."""
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


def start_outline_confirmation() -> OutlineConfirmationState:
    """Return the initial outline-confirmation state: reviewing stage, zero rounds used."""
    return OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )


def has_reached_outline_confirmation_bound(
    rounds_used: int, max_rounds: int = MAX_OUTLINE_CONFIRMATION_ROUNDS
) -> bool:
    """Return True if the outline-confirmation round bound has been reached."""
    return rounds_used >= max_rounds


def advance_after_review_turn(
    state: OutlineConfirmationState, action: OutlineReviewAction
) -> OutlineConfirmationState:
    """Advance the outline-confirmation loop given the classified action of the user's latest turn."""
    if state.stage is not OutlineConfirmationStage.REVIEWING:
        raise ValueError(
            f"advance_after_review_turn called outside REVIEWING stage: {state.stage}"
        )
    if action is OutlineReviewAction.CONFIRM:
        return OutlineConfirmationState(
            stage=OutlineConfirmationStage.CONFIRMED, rounds_used=state.rounds_used
        )
    if action is OutlineReviewAction.QUESTION:
        return state
    rounds_used = state.rounds_used + 1
    next_stage = (
        OutlineConfirmationStage.BOUND_REACHED
        if has_reached_outline_confirmation_bound(rounds_used)
        else OutlineConfirmationStage.REVIEWING
    )
    return OutlineConfirmationState(stage=next_stage, rounds_used=rounds_used)
