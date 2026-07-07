"""Deterministic parser that extracts title, company, skills, and source URL from Himalayas MCP's search_jobs text-blob responses."""

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
    """One parsed job listing from a Himalayas search_jobs response, with empty/None fields rather than fabricated data when absent."""

    title: str
    company: str
    skills: list[str]
    source_url: str | None


def parse_search_jobs_response(raw_text: str) -> list[ParsedJobListing]:
    """Parse a Himalayas search_jobs response's raw text into a list of parsed job listings, one per listing block."""
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
    """Parse one listing block into its title, company, skills, and source URL, requiring title and company to be present."""
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
    """Split a Key Skills line's value into individual skill strings, stripping the trailing '+N more' suffix."""
    parts = [part.strip() for part in skills_text.split("•")]
    if parts:
        parts[-1] = _MORE_SUFFIX_PATTERN.sub("", parts[-1]).strip()
    return [part for part in parts if part]
