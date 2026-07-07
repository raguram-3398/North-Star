"""Tests for utils/gemini_client.py — the shared Gemini-call
infrastructure (pacing, timeout, transient-error retry/backoff, JSON
parsing/error-fed retry) extracted from `agents/research_outline_agent.py`
once `agents/coaching_pace_agent.py` and the Verification Question
Generator Skill both needed the identical behavior.

These tests exercise the shared infrastructure directly (`gc._call_gemini_
text`/`gc._call_gemini_json`/etc.), rather than through any one business
caller — `tests/test_research_outline_agent.py` still owns the tests for
`create_initial_outline`'s/the clarify gate's own wiring onto this
infrastructure (e.g. "always passes timeout=HEAVY_GENERATION_TIMEOUT_
SECONDS"), and imports `_patch_gemini`/`_FakeGeminiClient` from this file
rather than redefining them, the same way `tests/test_coaching_pace_agent.py`
and `tests/test_verification_skill.py` do.

Patches target `utils.gemini_client.<name>` (where each name is *used* —
`_generate_content_with_retry`'s own internal calls to `_get_gemini_client`/
`_pace_gemini_call` resolve in *this* module's namespace now that the
infrastructure lives here), not `agents.research_outline_agent.<name>` —
CLAUDE.md's flagged wrong-patch-target anti-pattern. Before this
extraction, patching `agents.research_outline_agent._get_gemini_client`
was correct, since that's where the code used to live; a patch there today
would silently have no effect.
"""

import asyncio
import json

import pytest

import utils.gemini_client as gc
from utils.exceptions import GeminiCallError


class _FakeGeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeGeminiModels:
    """Returns each of `responses` in order, one per call. Raises
    `AssertionError` (a test-authoring bug, not a code-under-test bug) if
    a test calls the fake client more times than it queued responses for.

    `raise_exc` (always raised, every call) and `raise_exc_sequence` (one
    entry consumed per call — `None` means "fall through to the next
    queued response instead of raising") are mutually exclusive; the
    latter is what the retry/backoff tests use to simulate "fails once,
    then succeeds."
    """

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

    async def generate_content(self, *, model: str, contents: str, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._sleep_seconds:
            await asyncio.sleep(self._sleep_seconds)
        if self._raise_exc_sequence is not None:
            if not self._raise_exc_sequence:
                raise AssertionError(
                    "_FakeGeminiModels ran out of queued raise_exc_sequence "
                    "entries — the test queued fewer outcomes than the code "
                    "under test actually calls Gemini"
                )
            exc = self._raise_exc_sequence.pop(0)
            if exc is not None:
                raise exc
        elif self._raise_exc is not None:
            raise self._raise_exc
        if not self._responses:
            raise AssertionError(
                "_FakeGeminiModels ran out of queued responses — the test "
                "queued fewer canned responses than the code under test "
                "actually calls Gemini"
            )
        return _FakeGeminiResponse(self._responses.pop(0))


class _FakeGeminiAio:
    def __init__(self, models: _FakeGeminiModels) -> None:
        self.models = models


class _FakeGeminiClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        raise_exc: Exception | None = None,
        raise_exc_sequence: list[Exception | None] | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self.aio = _FakeGeminiAio(
            _FakeGeminiModels(responses, raise_exc, raise_exc_sequence, sleep_seconds)
        )


def _patch_gemini(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str] | None = None,
    raise_exc: Exception | None = None,
    raise_exc_sequence: list[Exception | None] | None = None,
    sleep_seconds: float = 0.0,
) -> _FakeGeminiClient:
    client = _FakeGeminiClient(
        responses=responses,
        raise_exc=raise_exc,
        raise_exc_sequence=raise_exc_sequence,
        sleep_seconds=sleep_seconds,
    )
    monkeypatch.setattr(gc, "_get_gemini_client", lambda: client)
    return client


class _FakeGeminiAPIError(Exception):
    """A minimal stand-in for `google.genai.errors.APIError`'s shape
    (`.code`/`.status`/`.message`, the real attributes `_is_retryable_
    gemini_error`/`_format_gemini_error` read) — the real `APIError`
    requires a genuine HTTP response object to construct, which these
    tests have no need for.
    """

    def __init__(self, code: int, status: str, message: str) -> None:
        self.code = code
        self.status = status
        self.message = message
        super().__init__(f"{code} {status}. {message}")


def _fast_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the retry backoff delays to near-zero so retry tests run
    fast — a separate concern from tests/conftest.py's pacing-guard
    fixture (`GEMINI_MIN_CALL_INTERVAL_SECONDS`), which only covers the
    gap *before* each call, not the backoff *between* retries of the
    same call.
    """
    monkeypatch.setattr(gc, "GEMINI_RETRY_BASE_DELAY_SECONDS", 0.001)
    monkeypatch.setattr(gc, "GEMINI_RETRY_MAX_DELAY_SECONDS", 0.005)


async def test_gemini_json_helper_raises_gemini_call_error_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every attempt (1 + GEMINI_JSON_RETRY_MAX_ATTEMPTS) comes back
    malformed here, so the retry loop must still exhaust and raise —
    queues one bad response per attempt the retry loop will actually
    make."""
    _patch_gemini(
        monkeypatch,
        responses=["not valid json at all"] * (gc.GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1),
    )

    with pytest.raises(GeminiCallError):
        await gc._call_gemini_json("some prompt", {"accepted"})


async def test_gemini_json_helper_raises_gemini_call_error_on_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(
        monkeypatch,
        responses=[json.dumps({"unexpected": "shape"})]
        * (gc.GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1),
    )

    with pytest.raises(GeminiCallError):
        await gc._call_gemini_json("some prompt", {"accepted"})


async def test_gemini_json_helper_retries_on_malformed_json_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the real, live 'Outline Confirmation failed:
    Gemini response was not valid JSON: ...' incident (and the same class
    of failure reported earlier on Outline Creation): a large one-shot
    structured-generation call occasionally comes back as a genuine 200 OK
    response with syntactically broken JSON (observed live: a single
    missing '}' after the first entry in a ~70-item list) — not a 429/503,
    so `_generate_content_with_retry`'s transport-level retry never
    engages. `_call_gemini_json` must retry this itself and recover,
    feeding the concrete parse error back into the next attempt's prompt
    (CLAUDE.md's error-fed retry discipline) rather than a generic
    "try again".
    """
    good_response = json.dumps({"accepted": True})
    client = _patch_gemini(
        monkeypatch,
        responses=[
            '{"accepted": true',  # malformed: missing closing brace
            good_response,
        ],
    )

    result = await gc._call_gemini_json("some proposal prompt", {"accepted"})

    assert result == {"accepted": True}
    assert len(client.aio.models.calls) == 2
    retry_prompt = client.aio.models.calls[1]["contents"]
    assert "not valid JSON" in retry_prompt
    assert "some proposal prompt" in retry_prompt


async def test_gemini_timeout_raises_gemini_call_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit timeout-path test (CLAUDE.md testing expectations), mirroring
    tests/test_research_outline_agent.py's Himalayas/Tavily timeout tests."""
    monkeypatch.setattr(gc, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_gemini(monkeypatch, responses=["irrelevant"], sleep_seconds=0.2)

    with pytest.raises(GeminiCallError):
        await gc._call_gemini_text("some prompt")


async def test_explicit_timeout_overrides_the_short_turn_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the real 'Outline Creation failed: Gemini call
    failed: TimeoutError()' incident: `EXTERNAL_CALL_TIMEOUT_SECONDS` (10s)
    is too short for a one-shot structured-generation call's real latency
    (a live probe of `create_initial_outline` measured 13.5s — see
    HEAVY_GENERATION_TIMEOUT_SECONDS's own comment), and a bare
    `asyncio.TimeoutError` is never retried. This shrinks the short-turn
    default to below the fake call's sleep time, then proves an explicit
    `timeout=` argument — not the default — is what determines whether
    the call survives.
    """
    monkeypatch.setattr(gc, "EXTERNAL_CALL_TIMEOUT_SECONDS", 0.01)
    _patch_gemini(monkeypatch, responses=["ok"], sleep_seconds=0.05)

    with pytest.raises(GeminiCallError):
        await gc._call_gemini_text("some prompt")

    result = await gc._call_gemini_text("some prompt", timeout=0.5)
    assert result == "ok"


def test_reset_gemini_client_for_new_event_loop_forces_fresh_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the real, live-reproduced 'Outline Confirmation
    failed: Gemini call failed: Event loop is closed' incident: the
    memoized Gemini client caches an async transport bound to whichever
    event loop first used it, but `main.py._run_async` runs every
    Gemini-backed call on its own fresh `asyncio.run(...)` loop, so a
    reused client's transport is bound to an already-closed loop by the
    second call. Proves `reset_gemini_client_for_new_event_loop` discards
    the memoized client so the next `_get_gemini_client()` call constructs
    a genuinely new instance, rather than returning the same one across
    loops.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(gc, "_gemini_client", None)

    first = gc._get_gemini_client()
    assert gc._get_gemini_client() is first  # memoized within one loop

    gc.reset_gemini_client_for_new_event_loop()

    second = gc._get_gemini_client()
    assert second is not first


def test_compute_gemini_retry_loop_timeout_matches_original_short_turn_ceiling() -> (
    None
):
    """`_compute_gemini_retry_loop_timeout(EXTERNAL_CALL_TIMEOUT_SECONDS)`
    must reproduce this codebase's original fixed 120s ceiling exactly —
    a formula regression here would silently change short-turn behavior
    that was already working."""
    assert gc._compute_gemini_retry_loop_timeout(10) == 120.0


def test_compute_gemini_retry_loop_timeout_scales_for_heavier_calls() -> None:
    """A larger per-attempt timeout (HEAVY_GENERATION_TIMEOUT_SECONDS) must
    produce a proportionally larger outer ceiling — a fixed 120s ceiling
    sized only for the 10s short-turn case would cut off a still-healthy
    heavy-generation retry sequence under a 429/503 burst, recreating the
    exact incident this mechanism exists to prevent."""
    heavy_ceiling = gc._compute_gemini_retry_loop_timeout(
        gc.HEAVY_GENERATION_TIMEOUT_SECONDS
    )
    assert heavy_ceiling > (gc.GEMINI_RETRY_MAX_ATTEMPTS + 1) * (
        gc.HEAVY_GENERATION_TIMEOUT_SECONDS
    )
    assert heavy_ceiling > gc._compute_gemini_retry_loop_timeout(10)


async def test_gemini_retries_transient_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient 429 (RESOURCE_EXHAUSTED) on the first attempt is
    retried, and the second attempt's success is returned normally — the
    caller never sees the transient failure at all.
    """
    _fast_retry_backoff(monkeypatch)
    client = _patch_gemini(
        monkeypatch,
        responses=["Backend, frontend, or something else?"],
        raise_exc_sequence=[
            _FakeGeminiAPIError(429, "RESOURCE_EXHAUSTED", "quota exceeded"),
            None,
        ],
    )

    result = await gc._call_gemini_text("some prompt")

    assert result == "Backend, frontend, or something else?"
    assert len(client.aio.models.calls) == 2


async def test_gemini_retries_transient_503_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as the 429 case above, for the other retryable status (503
    UNAVAILABLE) — both are named explicitly in
    `GEMINI_RETRYABLE_STATUS_CODES`, not inferred from one shared branch.
    """
    _fast_retry_backoff(monkeypatch)
    client = _patch_gemini(
        monkeypatch,
        responses=["ok"],
        raise_exc_sequence=[
            _FakeGeminiAPIError(503, "UNAVAILABLE", "model overloaded"),
            None,
        ],
    )

    result = await gc._call_gemini_text("some prompt")

    assert result == "ok"
    assert len(client.aio.models.calls) == 2


async def test_gemini_does_not_retry_non_transient_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-transient error (400 bad request) must never be retried —
    retrying it would just waste a call on an outcome that can't change.
    Exactly one call is made, and the real error surfaces.
    """
    _fast_retry_backoff(monkeypatch)
    client = _patch_gemini(
        monkeypatch,
        raise_exc_sequence=[
            _FakeGeminiAPIError(400, "INVALID_ARGUMENT", "bad request shape")
        ],
    )

    with pytest.raises(GeminiCallError) as exc_info:
        await gc._call_gemini_text("some prompt")

    assert len(client.aio.models.calls) == 1
    assert "400" in str(exc_info.value)
    assert "bad request shape" in str(exc_info.value)


async def test_gemini_retry_is_capped_and_still_surfaces_real_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient error that never clears exhausts the retry budget
    (`GEMINI_RETRY_MAX_ATTEMPTS` retries beyond the first attempt) rather
    than retrying forever, and the final failure still carries the real
    429 message, not a generic "gave up" placeholder.
    """
    _fast_retry_backoff(monkeypatch)
    always_429 = _FakeGeminiAPIError(429, "RESOURCE_EXHAUSTED", "quota exceeded")
    client = _patch_gemini(
        monkeypatch,
        raise_exc_sequence=[always_429] * (gc.GEMINI_RETRY_MAX_ATTEMPTS + 1),
    )

    with pytest.raises(GeminiCallError) as exc_info:
        await gc._call_gemini_text("some prompt")

    assert len(client.aio.models.calls) == gc.GEMINI_RETRY_MAX_ATTEMPTS + 1
    assert "429" in str(exc_info.value)
    assert "quota exceeded" in str(exc_info.value)


async def test_gemini_call_error_message_is_never_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real bug this task fixes: some exceptions (a bare
    `asyncio.TimeoutError`/`TimeoutError`, raised directly rather than via
    a real timeout) stringify to an EMPTY string, which previously
    produced exactly "Gemini call failed: " with nothing after the colon.
    The fix must surface something diagnosable even then.
    """
    _fast_retry_backoff(monkeypatch)
    _patch_gemini(monkeypatch, raise_exc_sequence=[TimeoutError()])

    with pytest.raises(GeminiCallError) as exc_info:
        await gc._call_gemini_text("some prompt")

    message = str(exc_info.value)
    assert message != "Gemini call failed: "
    assert message.strip() != "Gemini call failed:"
    assert "TimeoutError" in message


async def test_gemini_call_error_message_surfaces_api_error_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-retryable API error's real `code`/`status`/`message` (not
    just a bare `str(exc)`) end up in the raised `GeminiCallError`, so a
    future 429/503/400 is diagnosable from the message alone.
    """
    _patch_gemini(
        monkeypatch,
        raise_exc_sequence=[
            _FakeGeminiAPIError(403, "PERMISSION_DENIED", "API key invalid")
        ],
    )

    with pytest.raises(GeminiCallError) as exc_info:
        await gc._call_gemini_text("some prompt")

    message = str(exc_info.value)
    assert "403" in message
    assert "PERMISSION_DENIED" in message
    assert "API key invalid" in message
