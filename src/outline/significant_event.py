"""Significant-event detection — bucket/confidence-crossing diff.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per Architecture_North_Star.md §9: for each skill,
compare bucket membership (absent / emerging_skills / core_skills) and
confidence tier between the old and new roles_cache snapshot. Any upward
crossing generates a patch-note candidate for every user with a completed
topic matching that skill. Downward crossings are diffed but discarded —
never deleted from history, just not acted upon.
"""

from typing import Any


def detect_significant_events(
    old_snapshot: dict[str, Any], new_snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    """Diff two roles_cache snapshots for a role and return the skills that
    crossed a bucket or confidence boundary upward.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
