"""Bounded-loop state/round-counting for every interactive loop in this
codebase (CLAUDE.md guardrail #8: "the clarify gate and outline
confirmation are both bounded (~2 rounds)... any new interactive loop
needs the same treatment") — plus first-pass stated-goal classification
and structural reject detection, which are Clarify-Gate-specific.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per LLM Call Discipline, `classify_stated_goal` runs
on raw user input before it reaches any clarify-gate LLM call — never
after any content-processing step that could corrupt the pattern it needs
to catch.

Two independent bounded loops live here, each with its own state type,
sharing no fields (they are unrelated flows that happen to share the same
"~2 rounds, then a graceful exit" mechanical shape):

1. **Clarify Gate** (PRD §7.2, `ClarifyGateState`/`ClarifyGateStage`):
   narrowing rounds (bounded to ~2), then propose-best-guess ->
   explain-role -> accept-own-words -> zero-signal-exit. Also owns
   `classify_stated_goal`, the deterministic, lexical/plausibility
   first-pass classification of a raw stated goal into real /
   vague-but-genuine / nonsense — a vocabulary- and pattern-based
   judgment call, never an LLM call and never a market-existence check: a
   niche-but-real role must never be gate-rejected just because this
   module hasn't seen it before (see `classify_stated_goal`'s docstring
   for the mechanism).
2. **Outline Confirmation** (PRD §7.5, `OutlineConfirmationState`/
   `OutlineConfirmationStage`): a single reviewing stage the user can
   loop on by raising concerns or requesting additions (each consuming
   one of 2 bounded rounds) or asking free, unbounded questions (never
   consuming a round), until they explicitly confirm or the round bound
   is reached.

Everything downstream of the first-pass decisions above stays Agent 1's
job (Architecture §3): the *content* of each narrowing question,
proposal, explanation, outline "why" presentation, and question/concern
response, and — critically — interpreting a user's free-text reply into
one of this module's already-defined action/outcome values (accepted/
rejected for the clarify gate; `OutlineReviewAction` for outline
confirmation). That interpretation is genuinely open-ended natural
language, unlike the bounded classification/action vocabulary itself,
which is why interpretation stays an LLM reasoning task rather than being
pulled into this module. The values this module's `advance_*` functions
take as input are assumed already decided by that reasoning step.
"""

import re
from dataclasses import dataclass
from enum import Enum

MAX_NARROWING_ROUNDS = 2
MAX_OUTLINE_CONFIRMATION_ROUNDS = 2


class ClarifyGateStage(Enum):
    """A stage in the PRD §7.2 bounded-loop sequence."""

    NARROWING = "narrowing"
    PROPOSE_BEST_GUESS = "propose_best_guess"
    EXPLAIN_ROLE = "explain_role"
    ACCEPT_OWN_WORDS = "accept_own_words"
    RESOLVED = "resolved"
    EXITED = "exited"


class GoalClassification(Enum):
    """The first-pass, deterministic classification of a raw stated goal
    (PRD §7.2's Gate behavior list) — see `classify_stated_goal`.
    """

    REAL = "real"
    VAGUE = "vague"
    NONSENSE = "nonsense"


@dataclass(frozen=True)
class ClarifyGateState:
    """Immutable snapshot of the clarify gate's loop state."""

    stage: ClarifyGateStage
    narrowing_rounds_used: int = 0


class OutlineConfirmationStage(Enum):
    """A stage in the PRD §7.5 Outline Confirmation bounded loop."""

    REVIEWING = "reviewing"
    CONFIRMED = "confirmed"
    BOUND_REACHED = "bound_reached"


class OutlineReviewAction(Enum):
    """How Agent 1 classified a user's turn during outline confirmation
    (PRD §7.5) — see `advance_after_review_turn`.

    `QUESTION` is free and never advances the round bound. `CONCERN` and
    `ADDITION_REQUEST` both consume one of the 2 bounded rounds.
    `CONFIRM` ends the review immediately, regardless of rounds used so
    far. Classification itself is Agent 1's reasoning job (interpreting
    open-ended free text) — this module only tracks what happens to the
    loop once that classification is already decided.
    """

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
    """Detect blank/whitespace-only raw input — the narrowest possible
    reject condition, with no content to classify at all.

    Used as `classify_stated_goal`'s first check; kept as its own function
    since it is also meaningful on its own wherever only a blank/non-blank
    distinction is needed, not a full classification.
    """
    return not raw_input.strip()


_VOWELS = frozenset("aeiou")

# Real short tech/professional acronyms that would otherwise fail the
# consonant-run check below purely for lacking a vowel — a phonetic
# heuristic alone can't distinguish "UX" from keyboard mash without an
# allowlist. Judgment call; extend as real usage surfaces gaps.
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

# A run of this many consecutive consonants is treated as implausible for
# genuine English (rare real exceptions like "strengths" exist, but this
# is a narrow, low-stakes lexical filter, not a dictionary) — the
# mechanism that catches "asdkjfh"-style mash even though it contains a
# vowel and so cannot be caught by a bare vowel-presence check alone.
_MAX_PLAUSIBLE_CONSONANT_RUN = 4

# Single-word inputs that signal genuine (if vague) tech/career interest —
# accepted into the narrowing loop rather than rejected as an unrelated
# non-goal word (PRD §7.2's "AI", "coding" examples). Includes bare
# occupation nouns (a lone "Engineer" or "Analyst" is a vague-but-genuine
# goal, not nonsense) and a few of the most commonly stated single-word
# skills. Coarse, vocabulary-based judgment call — same status as
# data/tavily_parser.py's TECH_SKILL_VOCABULARY, flagged for tuning as
# real usage is observed, not presented as exhaustive.
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

# Additional keywords (beyond the single-word set above) that mark a
# longer, multi-word phrase as genuine vague tech interest — e.g.
# "something with computers", "fixing things on my laptop".
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

# Vocabulary that marks an otherwise title-shaped phrase as a joke/
# fabricated title rather than a real one (PRD §7.2's "Dragon Whisperer",
# "Chief Vibes Officer") — lexical-implausibility judgment only, never a
# market-existence check. Necessarily a small, curated, non-exhaustive
# denylist: a fabricated title this list doesn't recognize fails open to
# REAL rather than wrongly rejecting an unfamiliar-but-real title, which
# is the safer failure direction per PRD §7.2's explicit instruction to
# never gate-reject a niche/obscure real role.
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
    """Crude lexical plausibility check: does `token` read as an actual
    word rather than keyboard mash?

    A token with no vowel anywhere is mash unless it's a known vowelless
    acronym (`_VOWELLESS_REAL_WORDS`). A token *with* a vowel can still be
    mash — "asdkjfh" contains one 'a' — so this also rejects any run of
    `_MAX_PLAUSIBLE_CONSONANT_RUN`-or-more consecutive consonants, which a
    bare vowel-presence check alone cannot catch.
    """
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
    """First-pass, deterministic classification of a raw stated career
    goal (PRD §7.2's Gate behavior list) into `REAL` / `VAGUE` / `NONSENSE`.

    Purely lexical/plausibility judgment — no LLM call, no market-existence
    check. In particular this must never be used to decide whether a role
    actually exists in the job market: that determination is deferred to
    live grounding downstream (`data/cross_validation.py`), per PRD §7.2's
    explicit instruction that real-but-obscure or niche job titles must
    never be gate-rejected here however unfamiliar they are.

    Classification mechanism, in order:
    1. Blank/whitespace-only, or no alphabetic character at all (pure
       digits/symbols/emoji, e.g. "1234!!!", emoji-only) -> `NONSENSE`.
    2. No token in the input reads as an actual word at all (all tokens
       are keyboard mash per `_looks_like_a_word`, e.g. "asdkjfh") ->
       `NONSENSE`.
    3. A single-word input: `VAGUE` if the word is in the tech/career
       vocabulary (`_VAGUE_TECH_SINGLE_WORDS`, e.g. "AI", "coding", a bare
       occupation noun like "Engineer"); otherwise `NONSENSE` — a single
       real word with no tech/career connection at all (PRD §7.2's
       "banana", "purple" examples).
    4. A multi-word input ending in a recognized occupation noun
       (`_ROLE_NOUN_SUFFIXES`, e.g. "...Engineer", "...Analyst") is
       title-shaped: `NONSENSE` if it also contains a fabricated-title
       marker (`_FABRICATED_TITLE_MARKERS`, e.g. "Chief Vibes Officer"),
       else `REAL` — this is what lets a title this module has never seen
       before (e.g. "Site Reliability Engineer") through as real rather
       than nonsense or vague, per PRD §7.2's niche-title instruction.
    5. Not title-shaped, but contains a fabricated-title marker anyway
       (e.g. "Dragon Whisperer", which doesn't end in an occupation noun
       at all) -> `NONSENSE`.
    6. Not title-shaped, no fabricated marker, but contains a recognized
       tech/career keyword anywhere (`_VAGUE_TECH_PHRASE_KEYWORDS`, e.g.
       "something with computers", "data stuff") -> `VAGUE`.
    7. None of the above: a well-formed multi-word phrase with no title
       structure and no lexical connection to the product's domain at
       all. Not covered by any of PRD §7.2's worked examples; treated as
       `NONSENSE` rather than silently starting a narrowing loop on a
       statement with no domain connection whatsoever. Flagged as a
       genuine, revisable judgment call for this uncovered case, not a
       settled reading.
    """
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


def start_outline_confirmation() -> OutlineConfirmationState:
    """Return the initial outline-confirmation state: the reviewing
    stage, zero rounds used (PRD §7.5).
    """
    return OutlineConfirmationState(
        stage=OutlineConfirmationStage.REVIEWING, rounds_used=0
    )


def has_reached_outline_confirmation_bound(
    rounds_used: int, max_rounds: int = MAX_OUTLINE_CONFIRMATION_ROUNDS
) -> bool:
    """Determine whether the outline-confirmation round bound (exactly
    2, same as the clarify gate — confirmed directly, not inferred from
    PRD §7.5, which left the exact number unspecified) has been reached.
    """
    return rounds_used >= max_rounds


def advance_after_review_turn(
    state: OutlineConfirmationState, action: OutlineReviewAction
) -> OutlineConfirmationState:
    """Advance the outline-confirmation loop given Agent 1's
    classification of the user's latest review-turn message (PRD §7.5).

    - `CONFIRM` -> `CONFIRMED` immediately, regardless of rounds used so
      far (the user is explicitly done).
    - `QUESTION` -> the state is returned unchanged (same stage, same
      round count) — questions are free and unbounded, confirmed
      directly rather than inferred from PRD §7.5, which left this
      unspecified.
    - `CONCERN` / `ADDITION_REQUEST` -> both consume one of the 2 bounded
      rounds; moves to `BOUND_REACHED` once the bound is reached, or
      stays in `REVIEWING` for another round.

    Raises ValueError if called outside the REVIEWING stage.
    """
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
