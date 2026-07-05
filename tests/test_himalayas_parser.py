"""Tests for data/himalayas_parser.py — deterministic parsing of
Himalayas MCP's search_jobs text-blob responses.

Real fixtures (tests/fixtures/himalayas_search_jobs_*.txt) were captured
live via the same McpToolset + StreamableHTTPConnectionParams path
confirmed in tests/spike_grounding_connectivity.py, for 4 of PRD §7.3's
seed roles: Backend Engineer, Frontend Engineer, Data Analyst, DevOps
Engineer. Per this project's multi-item testing discipline (see
context-transfer.md's interleaving-order and drift-detection lessons), a
single real sample isn't enough — these fixtures are used together to
make sure the parser generalizes rather than happening to work on one
lucky example.

A true zero-result search_jobs response was never observed live, despite
several attempts (nonsense keywords, extreme salary_min, obscure country
+ exclude_worldwide filters) — Himalayas's search appears to always
return *something* rather than truly filtering to zero. The zero-result
and missing-skills-line cases below are therefore constructed from the
real header/block format actually observed (not invented from scratch),
clearly labeled as such, since the parser must still handle them
correctly per the task that authored this module.
"""

from pathlib import Path

import pytest

from data.himalayas_parser import ParsedJobListing, parse_search_jobs_response
from utils.exceptions import HimalayasParseError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def test_parses_expected_listing_count_across_multiple_real_roles() -> None:
    """Exact listing counts, cross-checked against `grep -c '^🚀'` on the
    raw fixture files, across three different real roles — not just one
    — so a subtly-wrong block splitter (e.g. off-by-one, or swallowing
    the last block) would be caught rather than coincidentally passing.
    """
    cases = [
        ("himalayas_search_jobs_frontend_engineer.txt", 20),
        ("himalayas_search_jobs_data_analyst.txt", 19),
        ("himalayas_search_jobs_devops_engineer.txt", 19),
    ]
    for filename, expected_count in cases:
        listings = parse_search_jobs_response(_load_fixture(filename))
        assert len(listings) == expected_count, filename


def test_every_real_listing_has_title_company_and_source_url() -> None:
    """Across all 58 real listings in the three fixtures, title, company,
    and source_url are always non-empty — confirmed by direct inspection
    of the raw text (every listing has an "Apply on Himalayas" line).
    Skills are checked separately below since that's the one field that
    does legitimately vary.
    """
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
    """Most real listings show only the first ~8 skills inline, with the
    last one carrying a "+N more" suffix (e.g. "AWS +6 more") that is
    Himalayas's own display truncation marker, not part of the skill
    name — it must never leak into a skill string.
    """
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
    """A real listing (Nagarro's "Engineer, Frontend") has a Key Skills
    line with exactly one skill and no bullet separator at all —
    confirmed present in tests/fixtures/himalayas_search_jobs_frontend_engineer.txt.
    """
    listings = parse_search_jobs_response(
        _load_fixture("himalayas_search_jobs_frontend_engineer.txt")
    )
    single_skill_listings = [
        listing for listing in listings if listing.company == "Nagarro"
    ]
    assert len(single_skill_listings) == 1
    assert single_skill_listings[0].skills == ["Dynamic"]


def test_strips_verified_company_checkmark() -> None:
    """Some companies render as "🏢 Company ✅" (Himalayas's verified-
    company badge) — the checkmark must not leak into `company`.
    Constructed from a real block observed in the connectivity spike
    (Wellhub's listing), reduced to only the fields this test needs.
    """
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
    """No real listing sampled (58 across 3 fixtures) omits the "Key
    Skills" line, so this is constructed by removing that one line from
    a real block (Cision's Data Analyst listing) — the parser must not
    crash or fabricate skills, just report an empty list for this field.
    """
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
    """A true zero-match response was never observed live (see module
    docstring), but the real header format ("Found N jobs matching") is
    known, so a "Found 0 jobs matching" header with no listing blocks is
    constructed to confirm this is treated as a valid empty result, not
    an error.
    """
    raw_text = "Found 0 jobs matching 'zzz_no_such_role' (showing page 1)\n"
    assert parse_search_jobs_response(raw_text) == []


def test_raises_when_no_recognizable_header_is_present() -> None:
    """Completely unrecognized input (not a search_jobs response at all,
    e.g. an HTML error page or a different tool's output) must raise
    rather than silently returning an empty list indistinguishable from
    a genuine zero-result response.
    """
    with pytest.raises(HimalayasParseError):
        parse_search_jobs_response("<html><body>502 Bad Gateway</body></html>")


def test_raises_when_header_claims_jobs_but_no_blocks_found() -> None:
    """A header reporting a non-zero total with zero parsable listing
    blocks indicates the known text format has changed, not a legitimate
    empty result — must raise, not silently return [].
    """
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
