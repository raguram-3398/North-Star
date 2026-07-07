"""Shared pytest fixtures for the whole test suite."""

import pytest

import utils.adk_runtime as adk_runtime


@pytest.fixture(autouse=True)
def _disable_gemini_call_pacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the real Gemini call-pacing guard globally so tests never incur real wall-clock delay."""
    monkeypatch.setattr(adk_runtime, "GEMINI_MIN_CALL_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(adk_runtime, "_last_gemini_call_started_at", None)
