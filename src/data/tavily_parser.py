"""Deterministic, coarse keyword-spotting extractor that scans Tavily search results for known skill/technology terms."""

import re
from dataclasses import dataclass

from utils.exceptions import TavilyParseError

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
    """One parsed Tavily search result, with an empty skills list when no vocabulary term is found in its content."""

    title: str
    skills: list[str]
    source_url: str
    score: float


def extract_skills_from_content(content: str) -> list[str]:
    """Scan content for case-insensitive, word-bounded matches against the technical skill vocabulary and return the matched terms."""
    if not content:
        return []
    return [term for term, pattern in _COMPILED_VOCABULARY if pattern.search(content)]


def parse_tavily_result(result: dict[str, object]) -> ParsedSearchResult:
    """Parse a single Tavily result dict into a ParsedSearchResult, requiring a non-empty url and title."""
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
    """Parse a full Tavily search response into a list of parsed search results, or an empty list if none are present."""
    results = response.get("results")
    if not isinstance(results, list):
        return []
    return [parse_tavily_result(result) for result in results]
