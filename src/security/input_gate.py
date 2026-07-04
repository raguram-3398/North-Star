"""Clarify-gate bound/loop state and reject detection.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per LLM Call Discipline, this runs on raw user input
before it reaches any clarify-gate LLM call — never after any
content-processing step that could corrupt the pattern it needs to catch.
"""


def detect_reject(raw_input: str) -> bool:
    """Determine whether raw user input is clearly nonsense and should be
    rejected outright, per PRD §7.2 Clarify Gate bounded-loop resolution.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def has_reached_round_bound(current_round: int, max_rounds: int = 2) -> bool:
    """Determine whether the clarify-gate narrowing-question bound
    (~2 rounds, PRD §7.2) has been reached.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
