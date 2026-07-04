"""Research & Outline Agent — reasoning/generation only.

Owns (Architecture_North_Star.md §3): the *content* of clarify-gate
narrowing questions and best-guess role proposals/explanations,
cross-validation normalization judgment (anchored to roles_cache), and
initial full-outline hierarchy creation (sequencing sourced skills into
dependency order).

Calls as tools (deterministic, not owned — never reimplemented inline):
security/input_gate.py, security/output_guard.py, data/roles_cache.py,
outline/significant_event.py, outline/hierarchy.py, patches/patch_manager.py.

Tools: Himalayas MCP, Tavily search, Postgres (via gated write paths only
— never a raw insert).
"""

from typing import Any


def generate_clarify_gate_response(
    conversation_so_far: list[dict[str, str]], current_round: int
) -> dict[str, Any]:
    """Generate the next clarify-gate turn: a narrowing question, a
    best-guess role proposal, or an explanation of a rejected proposal,
    per PRD §7.2. Round-counting and bound enforcement are delegated to
    security/input_gate.py, not decided here.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def cross_validate_market_data(
    himalayas_result: dict[str, Any], tavily_result: dict[str, Any]
) -> dict[str, Any]:
    """Judge whether Himalayas and Tavily results agree, normalizing
    against roles_cache as a grounding anchor, per PRD §7.3. Confidence
    tier assignment itself is delegated to security/output_guard.py.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def create_initial_outline(
    resolved_role: str, grounded_skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Sequence grounded skill data into a dependency hierarchy (basics ->
    full role requirements), per PRD §7.4.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
