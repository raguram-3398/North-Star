"""Shared Gemini-call infrastructure — the one place that owns pacing, the
explicit timeout (CLAUDE.md guardrail #14), transient-error retry/backoff,
and JSON-response parsing/error-fed retry for every Gemini-backed call in
this codebase.

Extracted from `agents/research_outline_agent.py`, where this logic was
originally built and tested (that module's own git history/docstrings
carry the full incident history — the "Event loop is closed" bug,
the malformed-JSON retry incident, the TimeoutError-swallowing bug — this
module's docstrings summarize the mechanism, not re-litigate the
incidents). Promoted here once a second and third real consumer
(`agents/coaching_pace_agent.py`'s day-content/closing-note generation,
and the Verification Question Generator Skill's question
generation/grading) needed the same call/timeout/retry/error-handling
behavior — CLAUDE.md guardrail #10 ("never write agent reasoning that
duplicates logic already living in a shared module") applies here at the
infrastructure grain, not just the domain-logic grain: `agents/
research_outline_agent.py`, `agents/coaching_pace_agent.py`, and
`.agent/skills/verification_question_generator/generator.py` all import
this module as legitimate peers now, none of them reaching into another
agent's private (underscore-prefixed) namespace across a package boundary.

Every module-level global here (`_gemini_client`, `_last_gemini_call_
started_at`) is genuinely process-wide state, not owned by whichever
caller happens to import it first — see `reset_gemini_client_for_new_event_
loop`'s own docstring for why a fresh client per Streamlit rerun is the
correct grain of "one client per module" (CLAUDE.md coding conventions),
not a regression of it.
"""

import asyncio
import json
import random
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from utils.exceptions import GeminiCallError

# CLAUDE.md guardrail #14: explicit timeout on every external call.
# Reuses the 10s convention already established in db/connection.py and
# tests/spike_grounding_connectivity.py. Sized for short conversational
# turns (a handful of sentences) — never used for a one-shot call that
# generates a full multi-field/multi-item structured payload; see
# HEAVY_GENERATION_TIMEOUT_SECONDS for those.
EXTERNAL_CALL_TIMEOUT_SECONDS = 10

# Real, measured root cause of a live "Outline Creation failed: Gemini
# call failed: TimeoutError()" bug: a live probe of create_initial_outline
# (10 grounded skills -> 21 sequenced topics, gemini-2.5-flash) took 13.5s
# end to end — already past EXTERNAL_CALL_TIMEOUT_SECONDS, and a bare
# asyncio.TimeoutError has no `.code` attribute so `_is_retryable_gemini_
# error` never retries it — meaning every real Outline Creation call was
# guaranteed to fail on the very first attempt, not just an occasional
# slow one. Used for every one-shot call whose output is a full
# multi-field or multi-item structured generation (outline hierarchy
# sequencing, per-topic explanations, day-content generation, gap-study
# content, the closing note, verification-question batches) rather than a
# short conversational turn. 45s gives ~3x headroom above the measured
# 13.5s for a larger real skill/topic list.
HEAVY_GENERATION_TIMEOUT_SECONDS = 45

# Retry/backoff for transient Gemini errors only — 429 (RESOURCE_EXHAUSTED,
# rate limit) and 503 (UNAVAILABLE, transient overload), Google's own
# documented pattern for these two specific response codes: exponential
# backoff with jitter, bounded by a small retry count. Deliberately never
# applied to any other error (a 400 bad request, a 401/403 auth failure,
# a genuine timeout) — those aren't transient, so retrying them would
# only waste calls/time, never change the outcome.
# Real, live-reproduced incidents on both `create_initial_outline` (the
# outline-hierarchy call) and `_generate_topic_explanations` (~70 topics
# in one shot): a genuine 200 OK response whose JSON body is syntactically
# broken (once, a single missing `}` after the first list entry) —
# not a 429/503, so GEMINI_RETRY_MAX_ATTEMPTS/`_generate_content_with_retry`
# never engages; a bad structured-output generation roll, not a transient
# network/rate-limit failure. `_call_gemini_json` retries this itself,
# separately, per CLAUDE.md's "error-fed retry" LLM Call Discipline: each
# retry's prompt includes the concrete parse/validation error from the
# previous attempt, not a generic "try again".
GEMINI_JSON_RETRY_MAX_ATTEMPTS = 2  # retries beyond the first attempt
GEMINI_RETRYABLE_STATUS_CODES = frozenset({429, 503})
GEMINI_RETRY_MAX_ATTEMPTS = 4  # retries beyond the first attempt
GEMINI_RETRY_BASE_DELAY_SECONDS = 1.0
# Per-attempt cap. Nominal (jitter-free) delays across 4 retries starting
# at 1s double each time (1, 2, 4, 8s) — this cap is headroom, not the
# expected value, sized to keep worst-case total retry time within the
# ~30-60s ceiling this was scoped against.
GEMINI_RETRY_MAX_DELAY_SECONDS = 30.0

# Buffer added on top of the exact worst-case retry-sequence duration
# (see _compute_gemini_retry_loop_timeout) to get the outer ceiling on the
# ENTIRE retry sequence inside `_generate_content_with_retry` (every
# attempt's own per-attempt timeout wait *plus* every backoff sleep
# between them) — distinct from, and always larger than, that per-attempt
# timeout. Original incident this fixes: a live Outline Creation call
# raised a bare TimeoutError during a burst of transient 503s, before the
# retry loop had a chance to finish, because nothing bounded the whole
# sequence. A fixed buffer (rather than a proportional multiplier) is used
# so the ceiling doesn't scale unboundedly once a much larger per-attempt
# timeout (HEAVY_GENERATION_TIMEOUT_SECONDS) is passed in — 55s matches
# this margin's original sizing (a 65s worst case at the 10s short-turn
# timeout was rounded up to a 120s ceiling).
GEMINI_RETRY_LOOP_TIMEOUT_MARGIN_SECONDS = 55.0


def _compute_gemini_retry_loop_timeout(per_attempt_timeout: float) -> float:
    """Worst-case duration of `_run_attempts`' whole retry sequence at a
    given per-attempt timeout, plus `GEMINI_RETRY_LOOP_TIMEOUT_MARGIN_SECONDS`
    headroom: every attempt (1 + `GEMINI_RETRY_MAX_ATTEMPTS`) using its
    full per-attempt timeout, plus every nominal (jitter-free) backoff
    wait between them. Computed per call rather than a single fixed
    constant because different callers pass different per-attempt
    timeouts (`EXTERNAL_CALL_TIMEOUT_SECONDS` for a short conversational
    turn, `HEAVY_GENERATION_TIMEOUT_SECONDS` for a one-shot structured
    generation) — a ceiling sized only for the short-turn case would cut
    off a still-healthy heavy-generation retry sequence under a 429/503
    burst, recreating the exact bug this mechanism exists to prevent.
    """
    total_attempts = GEMINI_RETRY_MAX_ATTEMPTS + 1
    nominal_backoff_total: float = sum(
        min(GEMINI_RETRY_MAX_DELAY_SECONDS, GEMINI_RETRY_BASE_DELAY_SECONDS * (2**a))
        for a in range(GEMINI_RETRY_MAX_ATTEMPTS)
    )
    return (
        total_attempts * per_attempt_timeout
        + nominal_backoff_total
        + GEMINI_RETRY_LOOP_TIMEOUT_MARGIN_SECONDS
    )


# Simple pacing guard between Gemini calls (not a queue): a pipeline run
# routinely fires several Gemini-backed steps back to back (Clarify Gate
# -> Grounding -> Outline Creation), and Gemini's free tier enforces its
# rate limit per-project across every call this process makes, not
# per-key (confirmed via Google's own docs) — so a minimum gap between
# calls is real pacing, not a workaround for a key-specific limit.
# Module-level state, not per-caller, since the limit itself is
# process-wide. tests/conftest.py disables this globally for the test
# suite (module-level state would otherwise persist and trigger delays
# across unrelated tests sharing one pytest process).
GEMINI_MIN_CALL_INTERVAL_SECONDS = 1.0
_last_gemini_call_started_at: float | None = None
# Serializes the guard's check-wait-update sequence. Without this, two
# concurrent callers can both read the same stale
# `_last_gemini_call_started_at`, both compute a wait relative to it, and
# both finish their waits and fire their real calls around the same time
# — the guard would hold "by convention" but not actually prevent two
# real Gemini calls from firing close together, since `await
# asyncio.sleep(...)` yields the event loop between the check and the
# update.
_gemini_pacing_lock = asyncio.Lock()

# Judgment call: "flash" tier chosen for short, low-latency conversational
# turns (a handful of sentences at most) rather than long grounded
# generation. Used as the default `model` for every frequent, short
# Gemini-backed turn across every consumer of this module. A caller doing
# a one-time, more substantive generation passes its own stronger-tier
# constant explicitly (e.g. `research_outline_agent.
# OUTLINE_HIERARCHY_GEMINI_MODEL`) rather than relying on this default.
SHORT_TURN_GEMINI_MODEL = "gemini-2.5-flash"

_gemini_client: genai.Client | None = None


def _get_gemini_client() -> genai.Client:
    """Return the module-level Gemini client, creating it on first use
    (one client per module — CLAUDE.md coding conventions; same
    lazy-but-memoized pattern as `research_outline_agent._get_tavily_client`).

    Raises RuntimeError if GEMINI_API_KEY is not set.
    """
    global _gemini_client
    if _gemini_client is None:
        import os

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def reset_gemini_client_for_new_event_loop() -> None:
    """Discard the memoized Gemini client so the next `_get_gemini_client()`
    call constructs a fresh one on the caller's own new event loop.

    Real, live-reproduced root cause of "Outline Confirmation failed:
    Gemini call failed: Event loop is closed": `genai.Client` lazily
    creates and caches an async httpx transport bound to whichever event
    loop is running the first time `client.aio` is used. `main.py` runs
    every Gemini-backed call via a fresh `asyncio.run(...)` per Streamlit
    script rerun (`main._run_async`'s own docstring: a new loop per call is
    correct here, not a workaround) — so without this reset, the
    module-level client singleton (CLAUDE.md's "one client per module"
    convention) keeps a transport bound to a loop that's already closed by
    the time the *next* rerun's Gemini call reaches it. Reproduced with two
    back-to-back `asyncio.run(_call_gemini_text(...))` calls sharing the
    memoized client: the first succeeds, the second raises exactly
    `RuntimeError: Event loop is closed` deep in httpcore's connection
    teardown. `main.py._run_async` calls this after every `asyncio.run(...)`
    completes, so the *next* call constructs a client (and transport) fresh,
    inside its own new loop — cheap (holds only the API key until first
    use), so paying construction cost once per rerun is not a regression of
    the "one client per module" convention, just its correct grain here.
    """
    global _gemini_client
    _gemini_client = None


def _format_gemini_error(exc: BaseException) -> str:
    """Build a diagnosable message from a Gemini SDK exception — never
    just the bare exception with nothing after it (the "swallowed error"
    bug this function exists to fix). Prefers the google-genai SDK's own
    structured `code`/`status`/`message` fields, present on
    `google.genai.errors.APIError` and its `ClientError`/`ServerError`
    subclasses (e.g. `429 RESOURCE_EXHAUSTED: <quota message>` for a rate
    limit) — `str(exc)` already includes these for that SDK's own error
    types, but this is defensive against any exception type where it
    doesn't. Falls back to `str(exc)` if that's non-empty, and to
    `repr(exc)` as a last resort: a bare `asyncio.TimeoutError()` (raised
    by `asyncio.wait_for` on a real timeout) stringifies to an EMPTY
    string, which previously produced exactly "Gemini call failed: " with
    nothing after the colon.
    """
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    message = getattr(exc, "message", None)
    if code is not None or status is not None or message is not None:
        return f"{code} {status}: {message}".strip()
    text = str(exc)
    if text:
        return text
    return repr(exc)


def _is_retryable_gemini_error(exc: BaseException) -> bool:
    """True only for a transient 429/503 (`GEMINI_RETRYABLE_STATUS_CODES`)
    — identified via the google-genai SDK's own `.code` attribute, the
    real HTTP status code on `ClientError`/`ServerError`
    (`google.genai.errors.APIError` subclasses). Any exception without a
    matching `.code` (a malformed-request 400, an auth failure, a bare
    `asyncio.TimeoutError`, a connection error) is never retryable here.
    """
    return getattr(exc, "code", None) in GEMINI_RETRYABLE_STATUS_CODES


async def _pace_gemini_call() -> None:
    """Enforce `GEMINI_MIN_CALL_INTERVAL_SECONDS` between the start of any
    two Gemini calls this process makes — see that constant's own comment
    for why this is a simple pacing guard, not a queue, and why it's
    module-level rather than per-caller.

    Serialized via `_gemini_pacing_lock`: the check (read
    `_last_gemini_call_started_at`), the wait, and the update all happen
    while holding the lock, so a second concurrent caller can't read the
    same stale timestamp while the first is still sleeping — without the
    lock, `await asyncio.sleep(...)` yields the event loop between the
    check and the update, letting two callers both compute a wait from
    the same stale baseline and fire their real calls close together
    anyway (the guard would hold "by convention" but not in fact).
    """
    global _last_gemini_call_started_at
    async with _gemini_pacing_lock:
        now = time.monotonic()
        if _last_gemini_call_started_at is not None:
            wait = GEMINI_MIN_CALL_INTERVAL_SECONDS - (
                now - _last_gemini_call_started_at
            )
            if wait > 0:
                await asyncio.sleep(wait)
        _last_gemini_call_started_at = time.monotonic()


async def _generate_content_with_retry(
    model: str,
    contents: str,
    config: genai_types.GenerateContentConfig | None = None,
    timeout: float | None = None,
) -> Any:
    """The one shared low-level Gemini call every reasoning step in this
    codebase eventually routes through — directly here, or via
    `_call_gemini_text`/`_call_gemini_json` below, which every
    Gemini-backed function in this codebase calls (`agents/
    research_outline_agent.py`, `agents/coaching_pace_agent.py`,
    `.agent/skills/verification_question_generator/generator.py`). This
    was previously duplicated between `_call_gemini_text` and
    `_call_gemini_json` (each called `client.aio.models.generate_content`
    directly); now there is genuinely one place, not two, that owns
    pacing, the explicit timeout (CLAUDE.md guardrail #14), and retry.

    `timeout` is the per-attempt budget — defaults to
    `EXTERNAL_CALL_TIMEOUT_SECONDS` for a short conversational turn;
    callers doing a one-shot structured generation pass
    `HEAVY_GENERATION_TIMEOUT_SECONDS` explicitly (see that constant's own
    comment for the real incident this parameter fixes).

    Retries only a transient 429/503 (`_is_retryable_gemini_error`), up to
    `GEMINI_RETRY_MAX_ATTEMPTS` additional attempts, with exponential
    backoff plus equal-jitter (half the computed delay, plus a random
    amount up to the other half) — Google's own documented pattern for
    these two specific transient errors. Any other failure (a bad
    request, an auth error, a genuine timeout, an exhausted retry budget)
    raises `GeminiCallError` immediately, with a real, diagnosable
    message (`_format_gemini_error`) rather than a bare
    "Gemini call failed:" with nothing after the colon.

    The whole retry sequence (every attempt plus every backoff wait) is
    itself wrapped in a ceiling computed by
    `_compute_gemini_retry_loop_timeout(timeout)` — see that function's
    own comment for why: without an outer ceiling here, only each
    *individual* attempt was ever bounded, so nothing prevented some
    other, tighter external deadline from cutting the whole sequence off
    with a bare, unhelpful `TimeoutError` before it could finish.
    """
    # `timeout` defaults to `None`, resolved here rather than in the
    # signature (`timeout: float = EXTERNAL_CALL_TIMEOUT_SECONDS`) —
    # a default bound in the signature is evaluated once at function-
    # definition time, so a test's `monkeypatch.setattr(gc,
    # "EXTERNAL_CALL_TIMEOUT_SECONDS", ...)` would silently have no effect
    # on every caller that relies on the default. Resolving the module
    # global here, at call time, is what makes that monkeypatch work.
    resolved_timeout = EXTERNAL_CALL_TIMEOUT_SECONDS if timeout is None else timeout

    async def _run_attempts() -> Any:
        client = _get_gemini_client()
        last_error: Exception = GeminiCallError(
            "Gemini call failed: no attempt was made"
        )
        for attempt in range(GEMINI_RETRY_MAX_ATTEMPTS + 1):
            await _pace_gemini_call()
            try:
                return await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    ),
                    timeout=resolved_timeout,
                )
            except Exception as exc:  # noqa: BLE001 — see this function's
                # own docstring: every Gemini-side failure becomes a
                # GeminiCallError with a real message; only a transient
                # 429/503 is retried, everything else raises immediately.
                last_error = exc
                if attempt < GEMINI_RETRY_MAX_ATTEMPTS and _is_retryable_gemini_error(
                    exc
                ):
                    delay = min(
                        GEMINI_RETRY_MAX_DELAY_SECONDS,
                        GEMINI_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                    )
                    await asyncio.sleep(delay / 2 + random.uniform(0, delay / 2))
                    continue
                raise GeminiCallError(
                    f"Gemini call failed: {_format_gemini_error(exc)}"
                ) from exc
        # Unreachable: GEMINI_RETRY_MAX_ATTEMPTS + 1 >= 1, so the loop
        # above always either returns or raises on its final iteration.
        # Kept as an explicit fallback (not `assert False`) so a future
        # change to the loop bounds fails loudly with the real last
        # error, not a silent `None`/missing-return bug.
        raise GeminiCallError(
            f"Gemini call failed: {_format_gemini_error(last_error)}"
        ) from last_error

    retry_loop_timeout = _compute_gemini_retry_loop_timeout(resolved_timeout)
    try:
        return await asyncio.wait_for(_run_attempts(), timeout=retry_loop_timeout)
    except TimeoutError as exc:  # asyncio.TimeoutError is this since 3.11
        raise GeminiCallError(
            "Gemini call failed: the full retry sequence exceeded its "
            f"computed outer ceiling ({retry_loop_timeout:.0f}s, based on a "
            f"{resolved_timeout}s per-attempt timeout) across all attempts "
            "and backoff waits"
        ) from exc


async def _call_gemini_text(
    prompt: str,
    model: str = SHORT_TURN_GEMINI_MODEL,
    timeout: float | None = None,
) -> str:
    """Call Gemini with a plain-text prompt and return the response text,
    stripped. Raises `GeminiCallError` if the call fails, times out, or
    returns no text at all — see `_generate_content_with_retry` for the
    retry/backoff/pacing this now goes through.

    Shared across every Gemini-backed reasoning step in this codebase (not
    caller-specific despite the default `model`) — callers pass an
    explicit `model` (e.g. `research_outline_agent.
    OUTLINE_HIERARCHY_GEMINI_MODEL`) and, for a one-shot structured
    generation rather than a short conversational turn, an explicit
    `timeout=HEAVY_GENERATION_TIMEOUT_SECONDS`. Defaults to `None`
    (resolved to `EXTERNAL_CALL_TIMEOUT_SECONDS` inside
    `_generate_content_with_retry`), not the constant itself, for the same
    monkeypatch-at-call-time reason documented there.
    """
    response = await _generate_content_with_retry(
        model=model, contents=prompt, timeout=timeout
    )

    text = response.text
    if not text or not text.strip():
        raise GeminiCallError("Gemini returned an empty response")
    return str(text).strip()


def _parse_gemini_json_object(raw_text: str, required_keys: set[str]) -> dict[str, Any]:
    """Parse and validate one Gemini JSON response. Raises `GeminiCallError`
    with a specific, diagnosable message on an empty response, invalid
    JSON syntax, a non-object JSON value, or a missing required key —
    exactly the message `_call_gemini_json`'s retry loop feeds back into
    its next attempt's prompt (CLAUDE.md's error-fed retry discipline).
    """
    if not raw_text or not raw_text.strip():
        raise GeminiCallError("Gemini returned an empty response")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise GeminiCallError(
            f"Gemini response was not valid JSON: {raw_text!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise GeminiCallError(f"Gemini JSON response was not an object: {raw_text!r}")
    missing = required_keys - parsed.keys()
    if missing:
        raise GeminiCallError(
            f"Gemini JSON response is missing required keys {missing}: {raw_text!r}"
        )
    return parsed


async def _call_gemini_json(
    prompt: str,
    required_keys: set[str],
    model: str = SHORT_TURN_GEMINI_MODEL,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call Gemini requesting a JSON object response and return it parsed.

    Raises `GeminiCallError` if the call fails/times out, or every attempt's
    response is not valid JSON, is not a JSON object, or is missing any of
    `required_keys` — never returns a partially-valid dict for the caller
    to guess at. See `_generate_content_with_retry` for the transport-level
    retry/backoff/pacing (429/503 only) every attempt below also goes
    through.

    Retries up to `GEMINI_JSON_RETRY_MAX_ATTEMPTS` additional times if the
    response comes back but fails to parse/validate (`_parse_gemini_json_
    object`) — a real, live-reproduced failure mode on a large one-shot
    structured generation (see that constant's own comment), distinct from
    the 429/503 transport retry above it. `prompt` is captured once as
    `original_prompt` before the loop, never overwritten by a retry's
    augmented prompt, so a later retry never contaminates itself with a
    previous attempt's correction as if it were the original input.

    Shared across every Gemini-backed reasoning step in this codebase (not
    caller-specific despite the default `model`) — callers pass an
    explicit `model` (e.g. `research_outline_agent.
    OUTLINE_HIERARCHY_GEMINI_MODEL`) and, for a one-shot structured
    generation rather than a short conversational turn, an explicit
    `timeout=HEAVY_GENERATION_TIMEOUT_SECONDS`. Defaults to `None`
    (resolved to `EXTERNAL_CALL_TIMEOUT_SECONDS` inside
    `_generate_content_with_retry`), not the constant itself, for the same
    monkeypatch-at-call-time reason documented there.
    """
    original_prompt = prompt
    last_error: GeminiCallError = GeminiCallError(
        "Gemini JSON call failed: no attempt was made"
    )
    for attempt in range(GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1):
        call_prompt = (
            original_prompt
            if attempt == 0
            else (
                f"{original_prompt}\n\nYour previous response failed with this "
                f"error: {last_error}\nReturn ONLY a single, complete, valid "
                "JSON object this time — no other text, no truncation."
            )
        )
        response = await _generate_content_with_retry(
            model=model,
            contents=call_prompt,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
            timeout=timeout,
        )
        try:
            return _parse_gemini_json_object(response.text, required_keys)
        except GeminiCallError as exc:
            last_error = exc
            if attempt == GEMINI_JSON_RETRY_MAX_ATTEMPTS:
                raise
    # Unreachable: GEMINI_JSON_RETRY_MAX_ATTEMPTS + 1 >= 1, so the loop
    # above always either returns or raises on its final iteration. Kept
    # as an explicit fallback, mirroring `_generate_content_with_retry`'s
    # own, so a future change to the loop bounds fails loudly with the
    # real last error rather than silently returning `None`.
    raise last_error
