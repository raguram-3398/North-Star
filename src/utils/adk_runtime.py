"""Shared ADK LlmAgent/Runner call infrastructure: timeouts, pacing, and error-fed JSON-parse retry."""

import asyncio
import time
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow._retry_config import RetryConfig
from google.genai import types as genai_types

from utils.exceptions import GeminiCallError

EXTERNAL_CALL_TIMEOUT_SECONDS = 10
HEAVY_GENERATION_TIMEOUT_SECONDS = 45

GEMINI_JSON_RETRY_MAX_ATTEMPTS = 2

GEMINI_RETRY_MAX_ATTEMPTS = 4
GEMINI_RETRY_BASE_DELAY_SECONDS = 1.0
GEMINI_RETRY_MAX_DELAY_SECONDS = 30.0

GEMINI_MIN_CALL_INTERVAL_SECONDS = 1.0
_last_gemini_call_started_at: float | None = None
_gemini_pacing_lock = asyncio.Lock()

SHORT_TURN_GEMINI_MODEL = "gemini-2.5-flash"

_APP_NAME = "north_star"
_session_service = InMemorySessionService()


def build_retry_config() -> RetryConfig:
    """Build the shared transport-level 429/503 retry config attached to every LlmAgent."""
    return RetryConfig(
        max_attempts=GEMINI_RETRY_MAX_ATTEMPTS + 1,
        initial_delay=GEMINI_RETRY_BASE_DELAY_SECONDS,
        max_delay=GEMINI_RETRY_MAX_DELAY_SECONDS,
        backoff_factor=2.0,
    )


def json_response_config() -> genai_types.GenerateContentConfig:
    """Build the shared JSON-mime generate_content_config for every JSON-returning LlmAgent."""
    return genai_types.GenerateContentConfig(response_mime_type="application/json")


async def _pace_gemini_call() -> None:
    """Enforce the minimum interval between the start of any two Gemini calls this process makes."""
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


def _extract_final_text(event: Any) -> str | None:
    """Extract plain text from a final-response Event's content parts, or None if it carried none."""
    content = event.content
    if content is None or not content.parts:
        return None
    text = "".join(part.text or "" for part in content.parts if part.text)
    return text or None


async def run_llm_agent(agent: LlmAgent, user_content: str, *, timeout: float) -> str:
    """Run agent once with user_content as a fresh single-turn session and return its final response text."""
    await _pace_gemini_call()
    user_id = "north-star-user"
    session_id = str(uuid.uuid4())
    message = genai_types.Content(
        role="user", parts=[genai_types.Part(text=user_content)]
    )

    async def _drain() -> str:
        session = await _session_service.create_session(
            app_name=_APP_NAME, user_id=user_id, session_id=session_id
        )
        runner = Runner(
            agent=agent, app_name=_APP_NAME, session_service=_session_service
        )
        final_text: str | None = None
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=message
        ):
            if event.error_code is not None or event.error_message is not None:
                raise GeminiCallError(
                    f"Gemini call failed: {event.error_code} {event.error_message}"
                )
            if event.is_final_response():
                text = _extract_final_text(event)
                if text is not None:
                    final_text = text
        if not final_text or not final_text.strip():
            raise GeminiCallError("Gemini returned an empty response")
        return final_text.strip()

    try:
        return await asyncio.wait_for(_drain(), timeout=timeout)
    except GeminiCallError:
        raise
    except TimeoutError as exc:
        raise GeminiCallError(f"Gemini call failed: {exc!r}") from exc
    except Exception as exc:  # noqa: BLE001
        raise GeminiCallError(f"Gemini call failed: {exc!r}") from exc


async def call_agent_text(
    agent: LlmAgent, prompt: str, *, timeout: float | None = None
) -> str:
    """Call agent with a plain-text prompt and return the response text, defaulting the timeout."""
    resolved_timeout = EXTERNAL_CALL_TIMEOUT_SECONDS if timeout is None else timeout
    return await run_llm_agent(agent, prompt, timeout=resolved_timeout)


def _parse_gemini_json_object(raw_text: str, required_keys: set[str]) -> dict[str, Any]:
    """Parse raw_text as a JSON object and check it carries every key in required_keys."""
    import json

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


async def call_agent_json(
    agent: LlmAgent,
    prompt: str,
    required_keys: set[str],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Call agent for a JSON object, retrying with the validation error fed back on parse/schema failure."""
    resolved_timeout = EXTERNAL_CALL_TIMEOUT_SECONDS if timeout is None else timeout
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
        raw_text = await run_llm_agent(agent, call_prompt, timeout=resolved_timeout)
        try:
            return _parse_gemini_json_object(raw_text, required_keys)
        except GeminiCallError as exc:
            last_error = exc
            if attempt == GEMINI_JSON_RETRY_MAX_ATTEMPTS:
                raise
    raise last_error
