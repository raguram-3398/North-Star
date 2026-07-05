"""Typed exceptions for Project North Star.

Per CLAUDE.md's LLM Call Discipline: raise exceptions, never return error
strings or None-as-error, so callers can reason about what they received.
"""


class GroundingError(Exception):
    """Raised when a piece of content cannot be traced to a valid, live or
    cached source (source_url/source_type/confidence unresolved).
    """


class VerificationTimeoutError(Exception):
    """Raised when an external call made during verification-question
    generation or grading does not complete within its explicit timeout.
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
