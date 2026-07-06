"""Typed exceptions for Project North Star.

Per CLAUDE.md's LLM Call Discipline: raise exceptions, never return error
strings or None-as-error, so callers can reason about what they received.
"""


class ConfidenceValidationError(Exception):
    """Raised by security/output_guard.py when a candidate outline item,
    patch-note, or grounding result is missing a required source_url or
    confidence value and cannot pass the structural gate before a DB write.
    """


class HimalayasParseError(Exception):
    """Raised by data/himalayas_parser.py when a Himalayas MCP response's
    raw text doesn't match the known, consistently-bulleted search_jobs
    format at all (no recognizable header, or a listing block missing a
    required field) — signals that the source format has changed or the
    input isn't a search_jobs response, as distinct from a listing merely
    missing an optional field (which is not an error).
    """


class TavilyParseError(Exception):
    """Raised by data/tavily_parser.py when a Tavily search result dict is
    missing a required structural field (`url` or `title`) — Tavily's
    response is structured JSON, not fragile prose like Himalayas's, so a
    missing `url`/`title` signals a genuinely malformed/unexpected API
    response, not merely a result with no extractable skills (which is
    not an error — see data/tavily_parser.py's module docstring).
    """


class GroundingSourceCallError(Exception):
    """Raised by agents/research_outline_agent.py when a live grounding
    source call itself (Himalayas MCP or Tavily) fails — connection
    error, non-success API response, or the call exceeding its explicit
    timeout (CLAUDE.md guardrail #14). Distinct, intentionally, from that
    source legitimately returning zero relevant results (not an error —
    see data/himalayas_relevance.py): a call failure and an empty/
    irrelevant result both end up treated as "no live signal from this
    source" for confidence-ladder purposes, but only a call failure
    raises this exception internally before being caught and downgraded,
    so the two cases stay distinguishable in logs/tests rather than
    silently collapsing into one code path by accident.
    """


class GeminiCallError(Exception):
    """Raised by any Gemini-backed reasoning step in
    agents/research_outline_agent.py (the clarify gate's turn functions
    and initial outline-hierarchy sequencing) when a Gemini call itself
    fails (connection error, non-success response, or exceeding its
    explicit timeout) or when the response cannot be parsed into the
    structured shape the caller requires — malformed JSON, a required
    field missing from an otherwise-parsed object, or (for outline
    sequencing specifically) a topic referencing a skill that was never
    in the grounded input, or a grounded skill never covered by any
    topic. Distinct from a genuinely ambiguous or negative
    *interpretation* of the model's output (e.g. the model honestly
    reports it could not resolve a role) — that is expected, handled
    conversational content, not an error.
    """
