"""Tests for data/himalayas_parser.py: deterministic parsing of Himalayas MCP's search_jobs text-blob responses, using real captured fixtures."""

from pathlib import Path

import pytest

from data.himalayas_parser import ParsedJobListing, parse_search_jobs_response
from utils.exceptions import HimalayasParseError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def test_parses_expected_listing_count_across_multiple_real_roles() -> None:
    """Checks exact listing counts across three different real roles so a subtly-wrong block splitter would be caught."""
    cases = [
        ("himalayas_search_jobs_frontend_engineer.txt", 20),
        ("himalayas_search_jobs_data_analyst.txt", 19),
        ("himalayas_search_jobs_devops_engineer.txt", 19),
    ]
    for filename, expected_count in cases:
        listings = parse_search_jobs_response(_load_fixture(filename))
        assert len(listings) == expected_count, filename


def test_every_real_listing_has_title_company_and_source_url() -> None:
    """Across all real listings in the three fixtures, title, company, and source_url are always non-empty."""
    for filename in [
        "himalayas_search_jobs_frontend_engineer.txt",
        "himalayas_search_jobs_data_analyst.txt",
        "himalayas_search_jobs_devops_engineer.txt",
    ]:
        listings = parse_search_jobs_response(_load_fixture(filename))
        assert listings, filename
        for listing in listings:
            assert listing.title, filename
            assert listing.company, filename
            assert listing.source_url is not None, filename
            assert listing.source_url.startswith("https://himalayas.app/"), filename


def test_parses_multi_skill_listing_from_real_fixture() -> None:
    listings = parse_search_jobs_response(
        _load_fixture("himalayas_search_jobs_data_analyst.txt")
    )
    first = listings[0]
    assert first.title == "Data Analyst- German Language"
    assert first.company == "Cision"
    assert first.skills == [
        "Excel",
        "PowerPoint",
        "Data Analysis",
        "Data Management",
        "Coding",
        "Qualitative Research",
        "German",
        "English",
    ]


def test_strips_plus_n_more_suffix_from_final_skill() -> None:
    """Himalayas's own "+N more" display-truncation marker must never leak into a skill string."""
    listings = parse_search_jobs_response(
        _load_fixture("himalayas_search_jobs_devops_engineer.txt")
    )
    truncated = [listing for listing in listings if len(listing.skills) == 8]
    assert truncated, "expected at least one listing with a '+N more' suffix to strip"
    for listing in truncated:
        for skill in listing.skills:
            assert "more" not in skill.lower()
            assert "+" not in skill


def test_parses_real_single_skill_listing() -> None:
    """A real listing can have a Key Skills line with exactly one skill and no bullet separator."""
    listings = parse_search_jobs_response(
        _load_fixture("himalayas_search_jobs_frontend_engineer.txt")
    )
    single_skill_listings = [
        listing for listing in listings if listing.company == "Nagarro"
    ]
    assert len(single_skill_listings) == 1
    assert single_skill_listings[0].skills == ["Dynamic"]


def test_strips_verified_company_checkmark() -> None:
    """Himalayas's verified-company checkmark badge must not leak into the parsed company name."""
    block = (
        "🚀 **Senior Backend Software Engineer | BASE**\n"
        "🏢 Wellhub ✅ \n\n"
        "🕘 Full-time • Brazil\n"
        "💵 💰 Competitive salary\n"
        "🛠️ **Key Skills:** RESTful APIs • AWS • GCP\n\n"
        "🔗 **Apply on Himalayas:** https://himalayas.app/companies/wellhub/jobs/x\n"
    )
    raw_text = "Found 1 jobs matching 'x' (showing page 1)\n\n\n" + block
    listings = parse_search_jobs_response(raw_text)
    assert listings[0].company == "Wellhub"


def test_handles_missing_skills_line_without_crashing() -> None:
    """A listing block missing the Key Skills line must not crash or fabricate skills, just report an empty list."""
    block = (
        "🚀 **Data Analyst- German Language**\n"
        "🏢 Cision \n\n"
        "🕘 Full-time • Germany\n"
        "💵 💰 Competitive salary\n"
        "\n"
        "🔗 **Apply on Himalayas:** https://himalayas.app/companies/cision/jobs/x\n"
    )
    raw_text = "Found 1 jobs matching 'x' (showing page 1)\n\n\n" + block
    listings = parse_search_jobs_response(raw_text)
    assert len(listings) == 1
    assert listings[0].skills == []
    assert listings[0].title == "Data Analyst- German Language"
    assert listings[0].source_url is not None


def test_returns_empty_list_for_genuine_zero_result_header() -> None:
    """A "Found 0 jobs matching" header with no listing blocks must be treated as a valid empty result, not an error."""
    raw_text = "Found 0 jobs matching 'zzz_no_such_role' (showing page 1)\n"
    assert parse_search_jobs_response(raw_text) == []


def test_raises_when_no_recognizable_header_is_present() -> None:
    """Completely unrecognized input must raise rather than silently returning an empty list."""
    with pytest.raises(HimalayasParseError):
        parse_search_jobs_response("<html><body>502 Bad Gateway</body></html>")


def test_raises_when_header_claims_jobs_but_no_blocks_found() -> None:
    """A header reporting a non-zero total with zero parsable listing blocks must raise, not silently return an empty list."""
    raw_text = "Found 5 jobs matching 'backend engineer' (showing page 1)\n"
    with pytest.raises(HimalayasParseError):
        parse_search_jobs_response(raw_text)


def test_raises_when_listing_block_has_no_title() -> None:
    raw_text = (
        "Found 1 jobs matching 'x' (showing page 1)\n\n\n"
        "🚀 not-bold-title\n🏢 Some Company\n"
    )
    with pytest.raises(HimalayasParseError):
        parse_search_jobs_response(raw_text)


def test_raises_when_listing_block_has_no_company() -> None:
    raw_text = (
        "Found 1 jobs matching 'x' (showing page 1)\n\n\n"
        "🚀 **Some Title**\nno company line here\n"
    )
    with pytest.raises(HimalayasParseError):
        parse_search_jobs_response(raw_text)


def test_parsed_job_listing_is_frozen() -> None:
    listing = ParsedJobListing(
        title="X", company="Y", skills=["A"], source_url="https://x.test"
    )
    with pytest.raises(Exception):
        listing.title = "changed"  # type: ignore[misc]
