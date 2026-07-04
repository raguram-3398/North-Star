"""Logging utilities, including tool-call audit logging.

Per Architecture_North_Star.md §6 and CLAUDE.md's Cost & Usage Tracking:
every Gemini, Tavily, and Himalayas call is logged with cost/usage and a
request_id — traceable, not aggregate-only. Cost is recorded only on
success (never a silently wrong $0.00 for a failed call). Daily spend
accumulates in a module-level tracker with a one-time threshold alert.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def log_tool_call(
    service: str, request_id: str, cost: float, usage: dict[str, int]
) -> None:
    """Record a single external tool call's cost/usage, keyed by request_id.

    Per CLAUDE.md: only called on a successful response, using actual
    token/usage counts from the API response, never estimates.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def get_daily_spend() -> float:
    """Return the accumulated cost tracked so far in this process's daily
    spend tracker.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
