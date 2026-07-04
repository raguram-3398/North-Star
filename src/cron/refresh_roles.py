"""Shared roles_cache refresh function.

Called identically by both the GitHub Actions scheduled workflow and the
Streamlit startup staleness check (Architecture_North_Star.md §3) — the
seed run and the recurring cron job are the same code path. Deliberately
not agentic: the trigger is wall-clock time, not judgment. Re-runs Agent
1's Research/Grounding pipeline for the seed role list and writes results
to roles_cache. CLAUDE.md guardrail #9: never touch this function's core
logic without checking both call sites still work.
"""


def refresh_roles_cache(seed_roles: list[str]) -> None:
    """Re-run the Research/Grounding pipeline for each role in
    `seed_roles` and write the results into roles_cache.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
