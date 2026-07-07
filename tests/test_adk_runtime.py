"""Tests for utils/adk_runtime.py: the shared ADK LLM-call infrastructure covering pacing, timeout, retry, and the Runner integration layer."""

import asyncio
import json

import pytest
from google.adk.agents import LlmAgent

import utils.adk_runtime as adk_runtime
from utils.exceptions import GeminiCallError


class _FakeRunLlmAgent:
    """Stand-in for run_llm_agent that returns queued responses or exceptions in order, one per call."""

    def __init__(
        self,
        responses: list[str] | None = None,
        raise_exc: Exception | None = None,
        raise_exc_sequence: list[Exception | None] | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._responses = list(responses or [])
        self._raise_exc = raise_exc
        self._raise_exc_sequence = (
            list(raise_exc_sequence) if raise_exc_sequence is not None else None
        )
        self._sleep_seconds = sleep_seconds
        self.calls: list[dict] = []

    async def __call__(
        self, agent: LlmAgent, user_content: str, *, timeout: float
    ) -> str:
        self.calls.append(
            {"agent": agent, "user_content": user_content, "timeout": timeout}
        )
        if self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_exc_sequence is not None:
            if not self._raise_exc_sequence:
                raise AssertionError(
                    "_FakeRunLlmAgent ran out of queued raise_exc_sequence "
                    "entries — the test queued fewer outcomes than the code "
                    "under test actually calls run_llm_agent"
                )
            exc = self._raise_exc_sequence.pop(0)
            if exc is not None:
                raise exc
        elif self._raise_exc is not None:
            raise self._raise_exc
        if not self._responses:
            raise AssertionError(
                "_FakeRunLlmAgent ran out of queued responses — the test "
                "queued fewer canned responses than the code under test "
                "actually calls run_llm_agent"
            )
        return self._responses.pop(0)


def _patch_adk_runtime(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str] | None = None,
    raise_exc: Exception | None = None,
    raise_exc_sequence: list[Exception | None] | None = None,
    sleep_seconds: float = 0.0,
) -> _FakeRunLlmAgent:
    fake = _FakeRunLlmAgent(
        responses=responses,
        raise_exc=raise_exc,
        raise_exc_sequence=raise_exc_sequence,
        sleep_seconds=sleep_seconds,
    )
    monkeypatch.setattr(adk_runtime, "run_llm_agent", fake)
    return fake


async def test_call_agent_json_raises_on_malformed_json_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adk_runtime(
        monkeypatch,
        responses=["not valid json at all"]
        * (adk_runtime.GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1),
    )
    with pytest.raises(GeminiCallError):
        await adk_runtime.call_agent_json(
            LlmAgent(name="fake"), "some prompt", {"accepted"}
        )


async def test_call_agent_json_raises_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adk_runtime(
        monkeypatch,
        responses=[json.dumps({"unexpected": "shape"})]
        * (adk_runtime.GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1),
    )
    with pytest.raises(GeminiCallError):
        await adk_runtime.call_agent_json(
            LlmAgent(name="fake"), "some prompt", {"accepted"}
        )


async def test_call_agent_json_retries_on_malformed_json_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed JSON must be retried with the concrete parse error fed back into the next attempt's prompt, not a generic retry."""
    good_response = json.dumps({"accepted": True})
    fake = _patch_adk_runtime(
        monkeypatch,
        responses=['{"accepted": true', good_response],
    )

    result = await adk_runtime.call_agent_json(
        LlmAgent(name="fake"), "some proposal prompt", {"accepted"}
    )

    assert result == {"accepted": True}
    assert len(fake.calls) == 2
    retry_prompt = fake.calls[1]["user_content"]
    assert "not valid JSON" in retry_prompt
    assert "some proposal prompt" in retry_prompt


async def test_call_agent_json_never_overwrites_original_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _patch_adk_runtime(
        monkeypatch,
        responses=["still not json", json.dumps({"accepted": True})],
    )
    await adk_runtime.call_agent_json(
        LlmAgent(name="fake"), "original prompt text", {"accepted"}
    )
    assert fake.calls[0]["user_content"] == "original prompt text"
    assert "original prompt text" in fake.calls[1]["user_content"]


async def test_call_agent_text_returns_stripped_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_adk_runtime(monkeypatch, responses=["  padded response  ".strip()])
    result = await adk_runtime.call_agent_text(LlmAgent(name="fake"), "some prompt")
    assert result == "padded response"


def test_build_retry_config_matches_gemini_retry_max_attempts() -> None:
    """max_attempts must include the initial attempt, so it equals GEMINI_RETRY_MAX_ATTEMPTS + 1."""
    config = adk_runtime.build_retry_config()
    assert config.max_attempts == adk_runtime.GEMINI_RETRY_MAX_ATTEMPTS + 1
    assert config.initial_delay == adk_runtime.GEMINI_RETRY_BASE_DELAY_SECONDS
    assert config.max_delay == adk_runtime.GEMINI_RETRY_MAX_DELAY_SECONDS


def test_json_response_config_requests_json_mime_type() -> None:
    config = adk_runtime.json_response_config()
    assert config.response_mime_type == "application/json"


class _FakePart:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, text: str | None) -> None:
        self.parts = [_FakePart(text)] if text is not None else []


class _FakeEvent:
    def __init__(
        self,
        *,
        text: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        final: bool = True,
    ) -> None:
        self.error_code = error_code
        self.error_message = error_message
        self._final = final
        self.content = _FakeContent(text) if text is not None else None

    def is_final_response(self) -> bool:
        return self._final


class _FakeRunner:
    def __init__(
        self,
        events: list[_FakeEvent] | None = None,
        sleep_seconds: float = 0.0,
        **_kwargs: object,
    ) -> None:
        self._events = events or []
        self._sleep_seconds = sleep_seconds

    async def run_async(self, *, user_id: str, session_id: str, new_message: object):
        if self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        for event in self._events:
            yield event


async def test_run_llm_agent_returns_final_response_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adk_runtime,
        "Runner",
        lambda **kwargs: _FakeRunner(events=[_FakeEvent(text="hello world")]),
    )
    result = await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "hi", timeout=5)
    assert result == "hello world"


async def test_run_llm_agent_raises_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forces the timeout path for run_llm_agent rather than only exercising the happy path."""
    monkeypatch.setattr(
        adk_runtime, "Runner", lambda **kwargs: _FakeRunner(sleep_seconds=0.2)
    )
    with pytest.raises(GeminiCallError):
        await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "hi", timeout=0.01)


async def test_run_llm_agent_raises_on_adk_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_event = _FakeEvent(error_code="500", error_message="boom", final=False)
    monkeypatch.setattr(
        adk_runtime, "Runner", lambda **kwargs: _FakeRunner(events=[error_event])
    )
    with pytest.raises(GeminiCallError, match="boom"):
        await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "hi", timeout=5)


async def test_run_llm_agent_raises_on_empty_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adk_runtime,
        "Runner",
        lambda **kwargs: _FakeRunner(events=[_FakeEvent(text=None)]),
    )
    with pytest.raises(GeminiCallError):
        await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "hi", timeout=5)


async def test_run_llm_agent_wraps_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception from the runner must be wrapped, never left as a bare unlabeled exception."""

    class _ExplodingRunner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_async(self, **_kwargs: object):
            raise ValueError("boom")
            yield  # pragma: no cover

    monkeypatch.setattr(adk_runtime, "Runner", _ExplodingRunner)
    with pytest.raises(GeminiCallError, match="boom"):
        await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "hi", timeout=5)


async def test_run_llm_agent_enforces_pacing_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies the pacing lock itself, re-enabled here since conftest.py disables it globally for other tests."""
    monkeypatch.setattr(adk_runtime, "GEMINI_MIN_CALL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(adk_runtime, "_last_gemini_call_started_at", None)
    monkeypatch.setattr(
        adk_runtime,
        "Runner",
        lambda **kwargs: _FakeRunner(events=[_FakeEvent(text="ok")]),
    )
    start = asyncio.get_event_loop().time()
    await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "one", timeout=5)
    await adk_runtime.run_llm_agent(LlmAgent(name="fake"), "two", timeout=5)
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed >= 0.05
