"""Shared pytest fixtures for the whole test suite."""

import pytest

import agents.research_outline_agent as roa


@pytest.fixture(autouse=True)
def _disable_gemini_call_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real pacing guard between Gemini calls
    (`agents.research_outline_agent.GEMINI_MIN_CALL_INTERVAL_SECONDS`/
    `_pace_gemini_call`) exists to space out real API calls during a live
    pipeline run — it has no reason to add real wall-clock delay to the
    test suite, where dozens of tests call a fake Gemini client in quick
    succession within the same process. The guard's "last call" timestamp
    is module-level state, so without this it would persist across tests
    and trigger delays for entirely unrelated tests sharing one pytest
    process. Disabled globally here rather than per test file.
    """
    monkeypatch.setattr(roa, "GEMINI_MIN_CALL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(roa, "_last_gemini_call_started_at", None)
