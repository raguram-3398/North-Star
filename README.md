---
title: North Star
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# North Star

Grounded, verified, market-aware career learning coach — Kaggle Agents for Good capstone submission.

## Problem

People breaking into tech with no bootcamp, mentor, or career coach have no way to know what to study, in what order, from which sources, or whether they've actually understood it. Generic AI chat answers are static, ungrounded, and don't adapt.

North Star is designed specifically for that underserved case: no mentor, no industry vocabulary, no one to sanity-check a plan or verify learning. It builds a market-grounded curriculum for a real tech role, teaches it day by day, and verifies understanding against the same sources it taught from — adapting the plan as the market and the user's own pace actually change.

## Why agents, not a single prompt

The value here depends on acting on live external state and remembering the user over time: a live, cited market claim; a plan that changes because of measured pace or a real market shift; a completion signal actually checked against source material. None of this is possible from a single chat completion — it requires an orchestrated pipeline that reads and writes persistent state between calls.

## Core flow

1. **Intake** — collects background, current job, years of experience, prior self-study specifics, stated goal, and available time.
2. **Clarify Gate** — resolves the stated goal into a concrete, real role. A real role proceeds immediately; nonsense is rejected and asked to clarify; a vague-but-genuine goal gets a narrowing question, bounded to ~2 rounds; if still unclear, a best-guess role is proposed, then explained, then the user's own words are accepted and grounding-checked. Any market signal found proceeds at low confidence; zero signal exits plainly, with no outline built.
3. **Research & Market Grounding** — Himalayas MCP and web search run in parallel and are cross-validated. Agreement yields high confidence; a single source or minor disagreement yields medium; a niche or no-anchor role falls back to a single source plus a sanity check, at low confidence. A cron-refreshed roles cache (at least every 30 days, or immediately on a significant market event) serves as a fallback and normalization anchor — never a shortcut that skips live research for a new user. Full confidence ladder: high → medium → low → cached-low → general-knowledge-only floor → reject. Every item persisted carries a source URL, source type, and confidence tier — nothing ungrounded is ever written.
4. **Outline Creation** — a grounded dependency hierarchy (basics through full role requirements), not a flat list. Additive only: content is never removed. Updates trigger on the 30-day floor, a significant market event (a skill crossing a bucket or confidence boundary upward — deterministic, not a judgment call), or 30+ days of dormancy. Two update types: a new addition (a new slot, positioned by hierarchy) or an augmentation (an existing topic refreshed in place).
5. **Outline Confirmation** — a one-time, pre-Day-1 checkpoint. The user sees the outline with grounded "why" reasoning per topic, can push back in a bounded loop (same propose-then-proceed pattern as the Clarify Gate), and no further outline edits are accepted once Day 1 begins.
6. **Day-by-Day Coaching** — each day is generated fresh, that day, nothing pre-chunked. Structure: summary → theory (grounded links) → hands-on (ramping in per topic-group) → review → reflection → verification → tomorrow's preview. A test-out option lets a user verify first and skip or gap-target study based on the result. Content is dynamically sized to the user's available time.
7. **Verification** — 5 fresh, source-anchored questions per topic, generated with grading criteria from the same source. Strict pass/fail, with a retry cap of exactly 3 attempts — each retry a fresh question, never a repeat. Failing all 3 teaches the answer inline for half credit. A topic completes only once all 5 slots pass, at full or half credit.
8. **Pace** — reflects understanding, not throughput. topic_score comes from question credit, not raw completion count; timing_ratio only pulls the combined signal when it's a genuine outlier against the user's own baseline. Weeks 1–2 are calibration only, and only a sustained trend (not a single day) triggers adaptation: falling behind extends pacing without cutting anything; getting ahead triggers enrichment.
9. **Patch-Notes** — market events that affect an already-completed topic never reopen its original status. High-confidence patches are prioritized into near-term delivery; low-confidence patches ask the user to learn now or defer. Deferred patches park permanently and resurface at goal completion or on request. Delivery is ordered by hierarchy position, never by detection time.
10. **Enrichment** — triggered by sustained-ahead pace, using the same insertion mechanism as outline updates, tagged as extra credit. It gets the full day-content treatment but is isolated from pace consequences — struggling on enrichment never counts against the user.
11. **Goal Completion** — "goal reached" means core scope only, full stop; enrichment is bonus, never required. The closing note reuses the same market-data infrastructure for next-step guidance and job tips. The system never makes seniority or grading claims — explicitly rejected as unsupported by what verification actually measures.

## Architecture

Two agents hold reasoning and generation only:

- **Research & Outline Agent** — Clarify Gate turns, market grounding, outline hierarchy sequencing, outline confirmation review.
- **Coaching & Pace Agent** — day content generation, verification question delivery, pace/patch/enrichment decisions.

Deterministic logic (confidence-ladder validation, pace math, drift thresholds, significant-event diffing, patch confidence-branching) lives in plain, independently-tested modules the agents call — never inline agent reasoning. State passes between agents and stages by database reference (`user_id`, `topic_id`), never as raw output dumped into a shared prompt context. Gates are structural: a candidate outline item, patch-note, or grounding result cannot reach persistence without first passing its validation gate — enforced by the write path itself, not by an instruction the agent is merely told to follow.

## Stack

- **Python 3.12** + **Google ADK** (`google-adk`, `LlmAgent`/`Runner`) for agent orchestration
- **Streamlit** for the UI
- **Neon Postgres** (via SQLAlchemy + `psycopg`) for persistence
- **Himalayas MCP server** for structured market/job data, consumed via ADK's MCP toolset
- **Tavily** for web search
- **Gemini** as the underlying LLM

## Course concepts demonstrated

- **Agent/Multi-agent system (ADK)** — both agents are built on real `google.adk.agents.LlmAgent` + `google.adk.runners.Runner` calls, not a bare model client.
- **MCP Server consumption** — the Himalayas MCP server is consumed via ADK's MCP toolset for structured market data.
- **Security features** — a structural confidence-ladder gate before any database write, bounded interactive loops with defined graceful exits, and raw-input reject detection that runs before any LLM call touches it.
- **Deployability** — a working `Dockerfile`, `requirements.txt`, and CI workflow (`ruff`, `black`, `mypy`, `pytest` on every push/PR).
- **Agent Skills** — a source-anchored verification question generator/grader (`src/skills/verification_question_generator.py`), independently reusable from either agent.
- **Evaluation-Driven Development** — JSON eval cases (input, expected output shape, rubric) written for the Verification Question Generator before finalizing its implementation.

## Setup

```bash
git clone <this-repo>
cd north-star
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp .env.example .env  # fill in real values
python -m src.db.create_schema  # one-time schema creation against your Neon instance
streamlit run streamlit_app.py
```

Required environment variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `NEON_CONNECTION_STRING` | Postgres connection string (Neon) |
| `TAVILY_API_KEY` | Tavily web search API key |
| `GEMINI_API_KEY` | Gemini / Google AI API key (ADK) |

### Running tests

```bash
pytest
pytest --cov=src tests/   # with coverage
ruff check .
black --check .
mypy src/
```

## Test coverage

Measured via `pytest --cov=src tests/` against the current codebase:

```
466 passed
TOTAL coverage: 94% (2293 statements, ~145 missed)
```

| Module | Coverage |
|---|---|
| `agents/coaching_pace_agent.py` | 94% |
| `agents/research_outline_agent.py` | 93% |
| `cron/refresh_roles.py` | 88% |
| `data/` (11 modules) | 50–100%, most at 100% |
| `db/connection.py`, `db/create_schema.py` | 100%, 83% |
| `main.py` | 87–88% |
| `models/schemas.py` | 100% |
| `outline/`, `pace/`, `patches/`, `security/` | 96–100% |
| `skills/verification_question_generator.py` | 99% |
| `utils/adk_runtime.py` | 98% |

`ruff check .`, `black --check .`, and `mypy src/` all pass clean.

## What I did not build, and why

At least three explicit, recorded scope cuts for this submission — a production version would do the following instead:

1. **No cost/usage or tool-call audit logging.** Every Gemini/Tavily/Himalayas call proceeds without a per-call cost/token log entry. A production version would record real API usage counts (never estimates) per call, plus a one-time daily-spend threshold alert.
2. **No database migration tool.** Schema creation is a one-time, idempotent `python -m src.db.create_schema` script (`Base.metadata.create_all()`), not Alembic — and `user_id`/`topic_id`/`origin_topic_id` columns are plain UUID columns with no foreign-key constraints yet. A production version would introduce Alembic-managed migrations and real FK constraints once the schema stabilizes.
3. **No mid-journey goal or role changes.** Once an outline is confirmed, a user cannot switch roles or goals without starting over. A production version would support re-grounding an in-progress plan against a new goal without discarding completed progress.
4. **No seniority, grading, or leveling claims.** The system never tells a user they are "junior" or "senior" at anything — this is rejected by design as unsupported by what verification actually measures (source-anchored pass/fail on specific questions, not a holistic skill assessment).
5. **No classroom or cohort features.** This is a single-individual skill-development tool, not a social or comparative platform, by design.
