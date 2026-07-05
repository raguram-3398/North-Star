"""Deterministic skill extractor for Tavily search results (PRD §7.3) —
parallel in purpose to `data/himalayas_parser.py`, but for a
fundamentally different and structurally weaker source shape.

Tavily's response is already structured at the top level
(`{"query", "answer", "results": [...], "response_time", "request_id"}`,
each result `{"url", "title", "content", "score", "raw_content"}`) — no
text-blob splitting is needed the way `himalayas_parser.py` needs to
split Himalayas's single prose blob into listing blocks. `url` is used
directly as `source_url`; no URL-parsing is needed either.

**The skill extraction itself is NOT comparable in reliability to
`himalayas_parser.py`'s parsing.** Himalayas's "Key Skills:" line is a
consistent, bullet-delimited structural element — extracting from it is
pattern matching on a known format. Tavily's `content` field is
unstructured free-text prose (a short snippet from a career-advice
article, job-description template, forum post, or even a YouTube video
transcript — see `tests/fixtures/tavily_search_*.json`, gathered live for
the same 4 seed roles as the Himalayas fixtures) with no reliable
delimiter at all. This module's approach — scanning `content` for
case-insensitive, word-bounded matches against a fixed vocabulary of
known skill/technology terms — is a coarse keyword-spotting heuristic,
not real extraction. It will miss skills phrased in ways the vocabulary
doesn't cover, and it can be fooled by a term appearing in an unrelated
context. Treat its output as directional signal, not a trustworthy
skill list, and do not mistake it for the same quality of data
`himalayas_parser.py` produces.

**Real-data finding (flagged for `data/cross_validation.py`'s eventual
use of this module, not acted on here — see the task that introduced
this module):** Tavily's own `score` field does *not* reliably predict
whether a result's `content` contains any extractable skill terms at
all. The single highest-scoring result across all 4 gathered fixtures
(Indeed's Data Analyst page, score 0.87) is a sitemap-style list of
unrelated job titles (Zookeeper, Welder, Teacher, ...) with zero
extractable skills; a lower-scoring result on the same query (0.78)
explicitly names "SQL, Excel, and Python." `score` is therefore passed
through on `ParsedSearchResult` untouched — this module does not filter
or reweight by it — so that decision is left to whatever consumes it
later (PRD §7.0: extraction is pattern matching, filtering-by-relevance
is a judgment call that belongs closer to where confidence tiers get
decided).
"""

import re
from dataclasses import dataclass

from utils.exceptions import TavilyParseError

# Evidence-based vocabulary: the union of skills data/himalayas_parser.py
# actually extracted from tests/fixtures/himalayas_search_jobs_*.txt for
# the same 4 seed roles (Frontend Engineer, Data Analyst, DevOps
# Engineer — Backend Engineer's original spike output was inspected too),
# canonicalized to drop near-duplicate casing/naming variants (e.g.
# "ReactJS"/"React", "NodeJS"/"Node.js", "CSS3"/"CSS"), plus two terms
# ("Django", "Kafka") confirmed present in the real Tavily content itself
# but absent from the Himalayas set. Grounded in real data for these
# roles, not exhaustive — will need extending for roles not yet sampled
# (e.g. AI/ML Engineer). Short/generic-looking entries ("R", "D3", "Mode")
# carry a real, acknowledged false-positive risk despite the word-boundary
# match — flagged here rather than silently trusted.
TECH_SKILL_VOCABULARY: tuple[str, ...] = (
    "Amazon EKS",
    "Angular",
    "Ansible",
    "AWS",
    "Azure",
    "BASH",
    "BigQuery",
    "Bitbucket",
    "Bitbucket Pipelines",
    "Cassandra",
    "CI/CD",
    "CloudFormation",
    "Cloudwatch",
    "CSS",
    "D3",
    "Data Analysis",
    "Data Modeling",
    "Data Science",
    "Data Structures",
    "Data Visualization",
    "Datadog",
    "DBT",
    "DevOps",
    "Django",
    "Docker",
    "ECS",
    "ES6",
    "ESLint",
    "ETL",
    "Excel",
    "Flux",
    "GCP",
    "Git",
    "GitHub Actions",
    "GitLab CI",
    "GoCD",
    "Golang",
    "Grafana",
    "GraphQL",
    "Headless CMS",
    "Helm",
    "HTML",
    "Java",
    "JavaScript",
    "Jenkins",
    "jQuery",
    "Kafka",
    "Kubernetes",
    "Leaflet",
    "LESS",
    "Linux",
    "Looker",
    "Machine Learning",
    "Metabase",
    "Microsoft Fabric",
    "MobX",
    "Mode",
    "MySQL",
    "NestJS",
    "New Relic",
    "Next.js",
    "Node.js",
    "OpenShift",
    "Pandas",
    "PHP",
    "Pivot Tables",
    "Playwright",
    "Power BI",
    "PowerPoint",
    "Prometheus",
    "Pulumi",
    "Pyspark",
    "Python",
    "R",
    "React",
    "Redux",
    "REST APIs",
    "Ruby",
    "SASS",
    "scikit-learn",
    "Snowflake",
    "Snowpipe",
    "SQL",
    "Statistics",
    "Tableau",
    "Tailwind",
    "Terraform",
    "Terragrunt",
    "TypeScript",
    "Vite",
    "Vitest",
    "Vue",
    "Vuex",
    "Webpack",
)

_COMPILED_VOCABULARY: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (term, re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE))
    for term in TECH_SKILL_VOCABULARY
)


@dataclass(frozen=True)
class ParsedSearchResult:
    """One parsed Tavily search result. `skills` is `[]`, never
    fabricated, when no vocabulary term is found in `content` — the same
    "don't crash, don't invent" discipline `data/himalayas_parser.py`
    follows for a listing missing its "Key Skills" line. `score` is
    Tavily's own value, passed through unmodified (see module docstring).
    """

    title: str
    skills: list[str]
    source_url: str
    score: float


def extract_skills_from_content(content: str) -> list[str]:
    """Scan `content` for case-insensitive, word-bounded matches against
    `TECH_SKILL_VOCABULARY`, returning matched canonical terms in
    vocabulary order (not the order they appear in the text), deduplicated
    by construction (each vocabulary term appears at most once).

    Returns `[]` for empty/missing content or content with no vocabulary
    hits — this is the expected, common case (see module docstring's
    real-data finding), not an error.
    """
    if not content:
        return []
    return [term for term, pattern in _COMPILED_VOCABULARY if pattern.search(content)]


def parse_tavily_result(result: dict[str, object]) -> ParsedSearchResult:
    """Parse a single Tavily result dict
    (`{"url", "title", "content", "score", "raw_content"}`) into a
    `ParsedSearchResult`.

    Raises `TavilyParseError` if `url` or `title` is missing or empty —
    Tavily's response is structured JSON, so a missing required field
    indicates a malformed/unexpected API response, not a legitimate
    "nothing extractable" case. `content` being missing/empty is *not*
    an error (see `extract_skills_from_content`); neither is `score`
    being absent, which defaults to `0.0`.
    """
    url = result.get("url")
    if not isinstance(url, str) or not url.strip():
        raise TavilyParseError("Tavily result is missing a non-empty 'url'")

    title = result.get("title")
    if not isinstance(title, str) or not title.strip():
        raise TavilyParseError("Tavily result is missing a non-empty 'title'")

    content = result.get("content")
    skills = extract_skills_from_content(content if isinstance(content, str) else "")

    score_raw = result.get("score")
    score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0

    return ParsedSearchResult(title=title, skills=skills, source_url=url, score=score)


def parse_tavily_response(response: dict[str, object]) -> list[ParsedSearchResult]:
    """Parse a full Tavily search response into a list of
    `ParsedSearchResult`, one per entry in `response["results"]`.

    Returns `[]` if `results` is missing or empty — unlike Himalayas,
    Tavily's structured shape makes "no results" unambiguous, with no
    equivalent of `himalayas_parser.py`'s header-vs-blocks consistency
    check needed.
    """
    results = response.get("results")
    if not isinstance(results, list):
        return []
    return [parse_tavily_result(result) for result in results]
