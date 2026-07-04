"""roles_cache I/O.

Per Architecture_North_Star.md §5/§7.3: structured, cron-refreshed market
data (core_skills / emerging_skills per role, each with a confidence
tier), used as fallback data and a normalization anchor — never a shortcut
that skips live research for a new user.
"""

from typing import Any


def get_role(role_name: str) -> dict[str, Any] | None:
    """Read a single role's cached market data, or None if no entry
    exists.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def upsert_role(
    role_name: str,
    core_skills: list[dict[str, Any]],
    emerging_skills: list[dict[str, Any]],
) -> None:
    """Write or refresh a role's cached market data.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def is_stale(role_name: str, max_age_days: int = 30) -> bool:
    """Determine whether a cached role's last_updated is past the 30-day
    floor, per Architecture_North_Star.md §3's startup staleness check.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
