"""Typed exceptions raised across Project North Star instead of error strings or None-as-error."""


class ConfidenceValidationError(Exception):
    """Raised when a candidate outline item, patch-note, or grounding result fails the confidence gate."""


class HimalayasParseError(Exception):
    """Raised when a Himalayas MCP response doesn't match the expected search_jobs format."""


class TavilyParseError(Exception):
    """Raised when a Tavily search result is missing a required structural field."""


class GroundingSourceCallError(Exception):
    """Raised when a live grounding source call (Himalayas or Tavily) fails or times out."""


class GeminiCallError(Exception):
    """Raised when a Gemini-backed reasoning call fails, times out, or returns unparseable output."""
