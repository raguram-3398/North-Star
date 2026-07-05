"""Tests for data/tavily_parser.py — the coarse, vocabulary-based skill
extractor for Tavily search results.

Real fixtures (tests/fixtures/tavily_search_*.json) were captured live
via TavilyClient.search for the same 4 seed roles as the Himalayas
fixtures (Backend Engineer, Frontend Engineer, Data Analyst, DevOps
Engineer), so cross-validation tests can later use both sources for the
same role. Per this project's multi-example testing discipline, these
tests check several different real results, not just one.
"""

import json
from pathlib import Path

import pytest

from data.tavily_parser import (
    TECH_SKILL_VOCABULARY,
    ParsedSearchResult,
    extract_skills_from_content,
    parse_tavily_response,
    parse_tavily_result,
)
from utils.exceptions import TavilyParseError

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text())


def _result_by_title_prefix(response: dict, title_prefix: str) -> dict:
    for result in response["results"]:
        if result["title"].startswith(title_prefix):
            return result
    raise AssertionError(f"no result titled {title_prefix!r} in fixture")


# --- extract_skills_from_content -----------------------------------------


def test_extract_skills_from_content_finds_multiple_real_terms() -> None:
    assert extract_skills_from_content(
        "you need strong programming skills in languages like Java, Python, or Node.js"
    ) == ["Java", "Node.js", "Python"]


def test_extract_skills_from_content_empty_or_missing_is_empty_list() -> None:
    assert extract_skills_from_content("") == []


def test_extract_skills_from_content_no_vocabulary_hit_is_empty_list() -> None:
    assert (
        extract_skills_from_content(
            "Degree in Computer Science, Engineering, or a related field."
        )
        == []
    )


def test_extract_skills_from_content_is_case_insensitive() -> None:
    assert "Python" in extract_skills_from_content("i know python and JAVA well")
    assert "Java" in extract_skills_from_content("i know python and JAVA well")


def test_extract_skills_from_content_uses_word_boundaries() -> None:
    """ "R" is in the vocabulary but must not match inside an unrelated
    word like "Regression" or "correlation" — a substring-only match
    would false-positive constantly on a single-letter term like this.
    """
    assert "R" not in extract_skills_from_content("correlation and regression analysis")
    assert "R" in extract_skills_from_content("statistical modeling in R for analysts")


# --- parse_tavily_result / parse_tavily_response, real fixtures ----------


def test_parse_result_with_clearly_extractable_skills() -> None:
    """A real, mid-scoring Data Analyst result that explicitly names
    several tools — the clearly-extractable case.
    """
    response = _load_fixture("tavily_search_data_analyst.json")
    raw_result = _result_by_title_prefix(response, "What Skills Are Needed")

    parsed = parse_tavily_result(raw_result)

    assert isinstance(parsed, ParsedSearchResult)
    assert set(parsed.skills) == {
        "Data Visualization",
        "Excel",
        "Power BI",
        "Python",
        "SQL",
        "Tableau",
    }
    assert parsed.source_url == raw_result["url"]
    assert parsed.score == pytest.approx(raw_result["score"])


def test_parse_result_with_no_extractable_skills() -> None:
    """A real, highest-scoring Backend Engineer result whose content is
    generic ("Degree in Computer Science...") and names no specific
    technology at all — confirms score alone doesn't guarantee anything
    is extractable, and confirms the module doesn't fabricate skills to
    compensate.
    """
    response = _load_fixture("tavily_search_backend_engineer.json")
    raw_result = _result_by_title_prefix(response, "Back-end Engineer Job Description")

    parsed = parse_tavily_result(raw_result)

    assert parsed.skills == []
    assert parsed.score == pytest.approx(raw_result["score"])


def test_high_score_does_not_guarantee_extractable_skills() -> None:
    """The single highest-scoring result across all 4 gathered fixtures
    (Indeed's Data Analyst page) is a sitemap-style list of unrelated job
    titles (Zookeeper, Welder, Teacher, ...) — real, off-topic, and the
    top-scored result for its query. Documents the module docstring's
    real-data finding directly: score does not reliably predict
    extractability, so it must not be used as an implicit filter inside
    this module.
    """
    response = _load_fixture("tavily_search_data_analyst.json")
    raw_result = _result_by_title_prefix(response, "Data Analyst Job Description")

    parsed = parse_tavily_result(raw_result)

    assert parsed.score == pytest.approx(raw_result["score"])
    assert parsed.skills == []
    # confirm this really is the highest-scoring result in the fixture,
    # not an arbitrary one — that's the whole point of this test.
    assert raw_result["score"] == max(r["score"] for r in response["results"])


def test_parse_tavily_response_across_multiple_real_roles() -> None:
    """Cross-checked against three different real, live-captured roles —
    not just one.
    """
    cases = [
        ("tavily_search_frontend_engineer.json", 10),
        ("tavily_search_data_analyst.json", 9),
        ("tavily_search_devops_engineer.json", 9),
    ]
    for filename, expected_count in cases:
        response = _load_fixture(filename)
        parsed = parse_tavily_response(response)
        assert len(parsed) == expected_count, filename
        for result in parsed:
            assert result.title
            assert result.source_url.startswith("http")
            assert 0.0 <= result.score <= 1.0
            assert isinstance(result.skills, list)


def test_parse_tavily_response_at_least_one_result_per_role_has_skills() -> None:
    """A sanity check that the extractor isn't silently returning empty
    skills for everything — at least one real result per role should
    have matched something, given the vocabulary is grounded in real
    skills for these exact roles.
    """
    for filename in [
        "tavily_search_frontend_engineer.json",
        "tavily_search_data_analyst.json",
        "tavily_search_devops_engineer.json",
    ]:
        parsed = parse_tavily_response(_load_fixture(filename))
        assert any(result.skills for result in parsed), filename


def test_parse_tavily_response_empty_results_is_empty_list() -> None:
    assert parse_tavily_response({"results": []}) == []


def test_parse_tavily_response_missing_results_key_is_empty_list() -> None:
    assert parse_tavily_response({}) == []


# --- malformed input (structural errors, not "nothing extractable") ------


def test_parse_tavily_result_missing_url_raises() -> None:
    with pytest.raises(TavilyParseError):
        parse_tavily_result({"title": "x", "content": "Python", "score": 0.5})


def test_parse_tavily_result_empty_url_raises() -> None:
    with pytest.raises(TavilyParseError):
        parse_tavily_result({"url": "  ", "title": "x", "content": "Python"})


def test_parse_tavily_result_missing_title_raises() -> None:
    with pytest.raises(TavilyParseError):
        parse_tavily_result({"url": "https://x.test", "content": "Python"})


def test_parse_tavily_result_missing_content_is_not_an_error() -> None:
    parsed = parse_tavily_result({"url": "https://x.test", "title": "x"})
    assert parsed.skills == []
    assert parsed.score == 0.0


def test_parse_tavily_result_missing_score_defaults_to_zero() -> None:
    parsed = parse_tavily_result(
        {"url": "https://x.test", "title": "x", "content": "y"}
    )
    assert parsed.score == 0.0


def test_parsed_search_result_is_frozen() -> None:
    result = ParsedSearchResult(
        title="x", skills=["Python"], source_url="https://x.test", score=0.5
    )
    with pytest.raises(Exception):
        result.title = "changed"  # type: ignore[misc]


def test_vocabulary_has_no_duplicate_terms() -> None:
    assert len(TECH_SKILL_VOCABULARY) == len(set(TECH_SKILL_VOCABULARY))
