"""Deterministic parser for Himalayas MCP's `search_jobs` text-blob
responses (Architecture_North_Star.md §1/§6/§12).

The connectivity spike (tests/spike_grounding_connectivity.py) revealed
that Himalayas MCP does not return structured JSON fields per job
listing — every tool response has the shape
`{"content": [{"type": "text", "text": "<markdown/emoji prose>"}],
"isError": false}`, where the prose is a "Found N jobs matching '<kw>'"
header followed by a series of `🚀`-prefixed listing blocks, each with a
title, company, an optional "Key Skills" line, and an "Apply on
Himalayas" URL.

Per PRD §7.0 ("Shift Intelligence Left"): extracting values out of this
known, consistently-bulleted format is pattern extraction on a fixed
shape, not semantic judgment — so it lives here as a plain, deterministic
module (no I/O, no LLM calls), not as agent reasoning. research_outline_
agent.py's cross-validation logic (not built yet) will call this module
on the raw text it already received from a Himalayas tool call, the same
way it calls security/output_guard.py or outline/hierarchy.py.

Only the two guardrail-#1-relevant fields (`skills`, `source_url`) plus
`title`/`company` are modeled — every one of the 78 real listings
gathered across 4 seed roles (tests/fixtures/himalayas_search_jobs_*.txt)
had all four. Employment type, location, and salary lines were observed
to vary in format (missing emoji prefix for some employment types,
"Competitive salary" vs. a numeric range) and aren't reliably parseable
the same way across listings, so they are deliberately not extracted
here — out of scope per the task that authored this module, revisit if a
future caller needs them.
"""

import re
from dataclasses import dataclass

from utils.exceptions import HimalayasParseError

_HEADER_PATTERN = re.compile(r"Found (\d+) jobs? matching")
_TITLE_PATTERN = re.compile(r"^🚀 \*\*(.+?)\*\*")
_COMPANY_PATTERN = re.compile(r"^🏢\s*(.+?)\s*✅?\s*$", re.MULTILINE)
_SKILLS_PATTERN = re.compile(r"🛠️\s*\*\*Key Skills:\*\*\s*(.+)")
_SOURCE_URL_PATTERN = re.compile(r"🔗\s*\*\*Apply on Himalayas:\*\*\s*(\S+)")
_MORE_SUFFIX_PATTERN = re.compile(r"\+\d+\s*more$")


@dataclass(frozen=True)
class ParsedJobListing:
    """One parsed job listing from a Himalayas `search_jobs` response.

    `skills` is `[]`, never fabricated, when a listing's "Key Skills"
    line is absent — not observed in any of the 78 real listings sampled,
    but the parser must not crash or invent data if a future listing
    omits it (CLAUDE.md guardrail #6's "never silently fabricate"
    applies here just as much as it does to output_guard.py). Likewise
    `source_url` is `None` under the same discipline, though every
    listing observed had one.
    """

    title: str
    company: str
    skills: list[str]
    source_url: str | None


def parse_search_jobs_response(raw_text: str) -> list[ParsedJobListing]:
    """Parse a Himalayas MCP `search_jobs` response's raw text
    (`response["content"][0]["text"]`) into a list of `ParsedJobListing`,
    one per listing block.

    Returns an empty list for a genuine zero-match response (a "Found 0
    jobs matching" header with no listing blocks) — this is a valid
    result, not an error; a caller needs to be able to tell "the search
    legitimately found nothing" from "the parser broke."

    Raises `HimalayasParseError` if `raw_text` doesn't contain a
    recognizable "Found N jobs matching" header at all (the input isn't
    a `search_jobs` response), or if the header reports a non-zero total
    but zero listing blocks were found (the known text format has
    changed or is otherwise broken, not a legitimate empty result).
    """
    header_match = _HEADER_PATTERN.search(raw_text)
    if header_match is None:
        raise HimalayasParseError(
            "raw_text does not contain a recognizable Himalayas "
            "'Found N jobs matching' search_jobs header"
        )
    reported_total = int(header_match.group(1))

    blocks = re.split(r"(?=🚀 )", raw_text)[1:]
    if reported_total > 0 and not blocks:
        raise HimalayasParseError(
            f"header reports {reported_total} jobs but no listing blocks "
            "were found — Himalayas's search_jobs text format may have changed"
        )

    return [_parse_listing_block(block, index) for index, block in enumerate(blocks)]


def _parse_listing_block(block: str, index: int) -> ParsedJobListing:
    """Parse one `🚀`-delimited listing block.

    `title` and `company` are treated as required — every one of the 78
    real listings sampled has both — so a block missing either raises
    `HimalayasParseError` rather than silently fabricating a placeholder
    value. `skills` and `source_url` are genuinely optional per the
    observed data and default to `[]` / `None`.
    """
    title_match = _TITLE_PATTERN.match(block)
    if title_match is None:
        raise HimalayasParseError(
            f"listing block {index} has no extractable title — expected a "
            "leading '🚀 **title**' line"
        )
    title = title_match.group(1).strip()

    company_match = _COMPANY_PATTERN.search(block)
    if company_match is None:
        raise HimalayasParseError(
            f"listing block {index} ({title!r}) has no extractable company line"
        )
    company = company_match.group(1).strip()

    skills_match = _SKILLS_PATTERN.search(block)
    skills = _parse_skills(skills_match.group(1)) if skills_match else []

    source_url_match = _SOURCE_URL_PATTERN.search(block)
    source_url = source_url_match.group(1) if source_url_match else None

    return ParsedJobListing(
        title=title, company=company, skills=skills, source_url=source_url
    )


def _parse_skills(skills_text: str) -> list[str]:
    """Split a "Key Skills:" line's value on its `•` bullet separator and
    strip the trailing "+N more" suffix Himalayas appends to the last
    skill when the full list is longer than what's displayed inline.
    """
    parts = [part.strip() for part in skills_text.split("•")]
    if parts:
        parts[-1] = _MORE_SUFFIX_PATTERN.sub("", parts[-1]).strip()
    return [part for part in parts if part]
