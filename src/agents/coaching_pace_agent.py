"""Coaching & Pace Agent — reasoning/generation only.

Owns (Architecture_North_Star.md §3): day-by-day content generation
(summary, theory framing, hands-on exercise design, reflection prompts)
and goal-completion closing-note composition.

Calls as tools (deterministic, not owned — never reimplemented inline):
the Verification Question Generator Skill, pace/calculator.py,
data/progress_log.py.

Tools: Verification Skill, Postgres (progress log, outline status — via
gated write paths), roles_cache (read-only, for closing note + enrichment
source).
"""


def generate_day_content(
    user_id: str, topic_id: str, day_number: int, is_conceptual_only: bool
) -> dict[str, str]:
    """Generate one day's content. Hands-on-eligible days follow the
    7-step structure (summary, theory, hands-on, review, reflection,
    verification, preview); conceptual-only days omit steps 3-4, per PRD
    §7.6.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def generate_closing_note(user_id: str) -> str:
    """Compose the goal-completion closing note, reusing roles_cache
    infrastructure for current hiring signal and in-demand skills, per PRD
    §7.11. Never makes a seniority, grading, or leveling claim.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
