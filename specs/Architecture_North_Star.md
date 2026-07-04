# Project North Star — Technical Architecture Doc

Companion to PRD_North_Star.md. This document defines *how* the system is built: stack, data model, component boundaries, and explicit out-of-scope decisions — so implementation choices are made once, here, not improvised turn-by-turn during the build.

---

## 1. Stack

| Layer | Choice | Notes |
|---|---|---|
| Language / agent framework | Python + Google ADK | Per course, per existing setup |
| UI | **Streamlit** | Native multi-step session-state model fits the intake→outline-confirm→day-by-day flow better than a single input/output demo tool; deploys natively on HF Spaces |
| Persistence | **Neon (Postgres)** | Free-tier, external — avoids HF Spaces' ephemeral local disk entirely |
| DB access | SQLAlchemy (or SQLModel) | Standard, avoids hand-rolled SQL string management |
| Market data (structured) | **Himalayas MCP server** | Consumed via ADK's MCP tool integration; free, no-auth |
| Market data (search) | **Tavily API** | Agent-native structured search results; free tier; single API key, no CSE setup |
| LLM | Gemini (via ADK) | Per course |
| Deployment | HF Spaces (Streamlit SDK) | Public link satisfies project-link requirement with no login |
| Secrets | HF Space secrets | Neon connection string, Tavily key, Gemini key — never committed to repo |

---

## 2. Orchestration Principles (course-aligned)

Per course guidance on DAG orchestration and "Shift Intelligence Left": agents hold reasoning and generation only. Deterministic computation is pulled into plain, testable modules that agents call as tools — never left as an instruction the agent is expected to remember and apply correctly at runtime. Concretely, the following are **not** agent logic, regardless of how they were described at the requirements stage — they are plain functions:

- Confidence-ladder tier assignment and source/schema validation (`security/output_guard.py`)
- Clarify-gate bound-counting and loop-termination (`security/input_gate.py`)
- Significant-event detection — bucket/confidence-crossing diff (`outline/significant_event.py`)
- Pace calculation — topic_score, timing_ratio, the 80/20 blend, sustained-drift threshold check (`pace/calculator.py`)
- Patch-note confidence branching (prioritize vs. ask-user, a threshold lookup on an already-computed value) (`patches/patch_manager.py`)
- Outline insertion/positioning into an *already-known* dependency structure (`outline/hierarchy.py`) — distinct from initial full-hierarchy *creation*, which does require reasoning and stays in Agent 1

**State passing is by database reference, not shared prompt context.** Agent 1 persists its output (resolved role, grounded outline rows, confidence tiers) to Postgres. Agent 2 receives only `user_id` / `topic_id` references and reads what it needs directly from the database. Raw agent output is never passed as accumulated text into the other agent's context window — this follows the course's explicit "Decouple State... pass only URIs or pointers" guidance and keeps each agent's context scoped and small.

**Gates are structural, not advisory ("Reviewer & Gate" node pattern).** The database write functions for outline items, patch-notes, and grounding results accept only a pre-validated object type (post-`output_guard`), not a raw agent-generated dict. An agent cannot persist an ungrounded item because the write path itself rejects anything that hasn't passed the gate — this is a software constraint, not a prompt instruction the agent could forget.

## 3. Component Boundaries (Agent Assignment)

Two real agents, consistent with the "don't inflate agent count for rubric-checkbox reasons" principle established during requirements:

### Agent 1 — Research & Outline Agent
**Owns (reasoning/generation only):** clarify-gate conversation (asking narrowing questions, proposing/explaining role interpretations — the *content* of what to ask, not the round-counting), cross-validation normalization judgment (anchored to `roles_cache`), initial full-outline hierarchy creation (sequencing sourced skills into dependency order).
**Calls as tools (deterministic, not owned):** `security/input_gate.py` (bound state, reject detection), `security/output_guard.py` (confidence/source validation before any write), `data/roles_cache.py` (I/O), `outline/significant_event.py` (diff logic), `outline/hierarchy.py` (insertion into existing structure), `patches/patch_manager.py` (confidence branching).
**Tools:** Himalayas MCP, Tavily search, Postgres (via the gated write paths above only — never a raw insert).

### Agent 2 — Coaching & Pace Agent
**Owns (reasoning/generation only):** day-by-day content generation (summary, theory framing, hands-on exercise design, reflection prompts), goal-completion closing-note composition.
**Calls as tools (deterministic, not owned):** the Verification Question Generator Skill (§4), `pace/calculator.py` (topic_score, timing_ratio, drift check), `data/progress_log.py` (I/O).
**Tools:** Verification Skill, Postgres (progress log, outline status — via gated write paths), `roles_cache` (read-only, for closing note + enrichment source).

### Cron job (not an agent)
**Owns:** the deterministic scheduled trigger (minimum every 30 days) that invokes Agent 1's Research pipeline against the seed role list to refresh `roles.json`. Deliberately not agentic — the trigger is wall-clock time, not judgment.

**Implementation — two layers, one shared function:**
- **Refresh function** (single, reusable): re-runs Agent 1's Research/Grounding pipeline for the seed role list, writes results to `roles_cache`. Called by both triggers below — no duplicated logic.
- **Primary trigger — GitHub Actions scheduled workflow** (`schedule:` cron, e.g. every 30 days): calls the refresh function directly (either via a script with its own DB/API credentials, or by hitting an app endpoint). Genuinely wall-clock triggered, independent of app traffic — this is the real "deployability"/"scheduled job" demonstration for the video.
- **Resilience layer — startup/session staleness check**: on Streamlit app startup (or first session of the day), check `roles_cache.last_updated` per cached role; if past the 30-day floor, call the same refresh function inline before continuing. Ensures the system stays honest even if the GitHub Action hasn't fired yet relative to a live demo session — not a replacement for the scheduled trigger, a safety net alongside it.

---

## 4. Agent Skill

**Verification Question Generator** — packaged as a real Skill artifact (SKILL.md + implementation), not inline agent logic.

- **Input:** topic source material (text/URL), number of questions needed (5, or 1 for a targeted retry)
- **Output:** structured question objects — `{question, grading_criteria, source_url}` — schema-validated
- **Trigger description (for SKILL.md):** "Generate source-anchored comprehension questions with grading criteria from study material. Use when a topic needs verification questions (initial 5-question set, a fresh retry question, or a test-out check). Do not use for market-data grounding or general Q&A unrelated to a specific study topic."
- **Reused by:** Agent 2, for every topic's initial verification, every retry (fresh question each time), and test-out checks.
- **Why this one:** narrowest, most repeated, cleanest single-responsibility unit in the whole pipeline — textbook "one skill, one job."

---

## 5. Data Model

```sql
-- Users
users (
  id UUID PRIMARY KEY,
  background TEXT,
  current_job TEXT,
  years_experience INT,
  prior_self_study TEXT,       -- free text, specifics not yes/no
  available_time_per_week INT, -- hours
  resolved_role TEXT,
  role_confidence TEXT,        -- high | medium | low | cached-low | general-knowledge-only
  pacing_profile TEXT,         -- initial expectation: high | medium | low (background-derived)
  created_at TIMESTAMP
)

-- roles.json equivalent — cached market data, cron-refreshed
roles_cache (
  role_name TEXT PRIMARY KEY,
  core_skills JSONB,           -- [{skill, source_url, confidence}]
  emerging_skills JSONB,       -- [{skill, source_url, confidence}]
  last_updated TIMESTAMP
)

-- Outline: dependency hierarchy per user
outline_topics (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  topic_name TEXT,
  hierarchy_position INT,       -- ordering within the dependency graph
  topic_group TEXT,             -- e.g. "Python", "SQL" — used for hands-on ramping logic
  position_in_group INT,        -- day-within-group, drives conceptual-only vs hands-on
  source_url TEXT,
  source_type TEXT,
  confidence TEXT,
  is_enrichment BOOLEAN DEFAULT FALSE,
  status TEXT,                  -- not_started | in_progress | completed | completed_test_out
  completed_at TIMESTAMP
)

-- Progress log: canonical record of everything that feeds pace
progress_log (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  topic_id UUID REFERENCES outline_topics(id),
  day_number INT,
  step TEXT,                    -- summary | theory | hands_on | review | reflection | verification | preview
  reflection_text TEXT,
  created_at TIMESTAMP
)

-- Verification attempts (feeds topic_score)
verification_attempts (
  id UUID PRIMARY KEY,
  topic_id UUID REFERENCES outline_topics(id),
  question_number INT,          -- 1-5
  attempt_number INT,
  question_text TEXT,
  grading_criteria TEXT,
  user_answer TEXT,
  passed BOOLEAN,
  credit FLOAT,                 -- 1.0 full, 0.5 half (taught-answer de-escalation)
  is_test_out BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP
)

-- Patch-notes
patch_notes (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  origin_topic_id UUID REFERENCES outline_topics(id),
  new_content TEXT,
  source_url TEXT,
  confidence TEXT,
  status TEXT,                  -- pending | delivered | deferred
  created_at TIMESTAMP,
  resolved_at TIMESTAMP
)

-- Pace snapshots (rolling window inputs)
pace_snapshots (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  topic_id UUID REFERENCES outline_topics(id),
  topic_score FLOAT,
  timing_ratio FLOAT,
  days_taken INT,
  days_expected INT,
  computed_at TIMESTAMP
)
```

---

## 6. MCP & Tool Security Governance (course-aligned)

Per course guidance on consuming (not building) MCP servers, and on MCP spoofing/contextual authorization risk:

- **Himalayas is a public, unverified MCP server** — acceptable for this prototype per course guidance ("public community servers are fine for weekend prototypes"), but explicitly *not* treated as if it were a vetted first-party integration. No credentials are passed to it beyond what its public tools require (none, per its no-auth design).
- **Read-only scoping**: all Himalayas and Tavily calls are read-only by construction — North Star never writes to either external service, only reads market data. No write-capable tool is given to either agent.
- **HITL before tool calls where the course requires it**: the outline confirmation checkpoint (§7.5, PRD) already functions as a human-in-the-loop gate before the *outcome* of Research/Outline tool calls is committed to the user's active plan — the user sees and can contest what the tools produced before it's acted on.
- **No hardcoded credentials, anywhere** — Himalayas needs none; Tavily and Gemini keys, and the Neon connection string, are environment variables / HF Space secrets only, never in code, prompts, or logs.
- **Audit logging**: tool calls (Himalayas queries, Tavily queries, Gemini calls) are logged via `observability`-equivalent logging in `utils/logger.py` for basic auditability, even without a full observability stack.
- **Scope, not breadth**: MCP/search tools are only ever invoked with the specific role/query needed for the current user's request — no broad, unscoped queries.

## 7. Agent Skill Evaluation (scoped to hackathon timeline)

Full production skill-graduation tiers (Read-Only → Draft → Action-Allowed, adversarial red-teaming, pass^k sustained-success testing) are out of scope for a 2.5-day build. The realistic, still-course-aligned subset:

- **EDD (Evaluation-Driven Development)**: 3 JSON eval cases (input, expected output shape, rubric) written for the Verification Question Generator *before* finalizing its implementation — per course's inversion-path guidance.
- **Positive + negative trigger cases**: at minimum 2 positive (should generate questions) and 1 negative (should not fire — e.g., empty/invalid source material) case.
- **Golden dataset**: a small set (5–10, not 20+) of representative (source material → expected question shape) pairs, versioned in `evaluation/golden_dataset.json`.
- Full adversarial/red-team and canary-rollout testing are explicitly noted as future work, not attempted here.

## 8. Confidence Ladder — canonical values

Used consistently across `roles_cache`, `outline_topics`, `patch_notes`:
`high` → `medium` → `low` → `cached-low` → `general-knowledge-only` → (reject, no record created)

## 9. Significant Event Detection — implementation note

Deterministic diff on `roles_cache` between refreshes: for each skill, compare bucket membership (`core_skills` / `emerging_skills` / absent) and confidence tier between the old and new snapshot. Any *upward* crossing generates a patch-note candidate for every user with a completed topic matching that skill. Downward crossings are diffed but discarded (no action) — never deleted from history, just not acted upon.

## 10. Explicit Out-of-Scope (implementation)

- No custom-built MCP server — Himalayas is consumed only
- No authentication/login system — public tool, no user accounts beyond a session/browser-local identifier
- No multi-user concurrency hardening beyond basic correctness (no load testing, no connection pooling tuning)
- No mobile-specific UI work — Streamlit default responsive behavior only
- No admin dashboard for `roles_cache` — cron script output is sufficient for v1
- Full skill-graduation tier system, adversarial red-teaming, and canary rollout (see §7) — noted as future work
- `outline/hierarchy.py` only supports must-follow (prerequisite) positioning constraints, not must-precede — see code comment in `insert_new_topic` for detail. Flagged during implementation, not from original design.

## 11. Non-negotiable Guardrails (carry into CLAUDE.md)

- Never accept or store an outline item / patch-note without a `source_url` and `confidence` value
- Never delete or reduce outline content under any pace condition
- Never repeat an identical verification question on retry — always regenerate fresh
- Never let enrichment topic outcomes write into `pace_snapshots`
- Never expose Neon/Tavily/Gemini credentials in code, logs, or the repo
- Never allow an agent to write to Postgres without passing through its corresponding gate function first
- Never pass one agent's raw output into the other agent's prompt context — pass a database reference instead
- Never call an external API (Gemini, Tavily, Himalayas, Neon) without an explicit timeout
- Never record a cost/usage log entry without a `request_id`, and never on a failed call
- Never delete a prior version from `PROMPT_REGISTRY` — versions are append-only, frozen once their baseline test locks them in

## 12. Repository Structure

```
north-star/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .gitignore
├── .claudeignore          # secrets/env exclusions for Claude Code's own context
├── .env / .env.example
├── .dockerignore
├── Dockerfile
├── requirements.txt
├── streamlit_app.py
│
├── .agent/
│   └── skills/
│       └── verification_question_generator/
│           ├── SKILL.md
│           └── generator.py
│
├── .github/workflows/
│   ├── ci.yml
│   └── refresh_roles.yml
│
├── specs/
│   ├── PRD_North_Star.md
│   ├── Architecture_North_Star.md
│   ├── scenarios/
│   │   └── high_risk_flows.feature      # Gherkin: clarify gate, confidence ladder, outline confirmation
│   └── architecture.png
│
├── src/
│   ├── main.py
│   ├── agents/
│   │   ├── research_outline_agent.py   # reasoning/generation only — see §3
│   │   └── coaching_pace_agent.py      # reasoning/generation only — see §3
│   ├── security/
│   │   ├── input_gate.py               # clarify-gate bound/loop state, reject detection
│   │   └── output_guard.py             # confidence-ladder enforcement, source validation — the structural gate
│   ├── pace/
│   │   └── calculator.py               # topic_score, timing_ratio, 80/20 blend, sustained-drift check
│   ├── outline/
│   │   ├── hierarchy.py                # insertion/positioning into an existing hierarchy
│   │   └── significant_event.py        # bucket/confidence-crossing diff
│   ├── patches/
│   │   └── patch_manager.py            # confidence branching, delivery ordering
│   ├── cron/
│   │   └── refresh_roles.py            # shared refresh function — called by GitHub Action and startup check
│   ├── data/
│   │   ├── roles_cache.py              # roles_cache I/O
│   │   └── progress_log.py             # progress_log I/O
│   ├── models/
│   │   └── schemas.py                  # SQLAlchemy models, mirrors §4 exactly
│   ├── db/
│   │   └── connection.py
│   └── utils/
│       ├── logger.py                   # includes basic tool-call audit logging, per §5
│       └── exceptions.py
│
├── evaluation/
│   ├── golden_dataset.json             # 5-10 (input, expected output) pairs for the Verification Skill
│   └── eval_cases.json                 # EDD-style: 2 positive + 1 negative trigger case
│
└── tests/
    ├── test_input_gate.py
    ├── test_output_guard.py
    ├── test_research_grounding.py
    ├── test_hierarchy.py
    ├── test_significant_event.py
    ├── test_verification_skill.py
    ├── test_pace_calculator.py
    └── test_patch_manager.py
```

**Skill location note:** the Verification Question Generator lives in `.agent/skills/`, not `src/skills/` — this is required for the skill to be recognized by the Antigravity workspace manager, per course convention. `src/` contains only agent reasoning code and plain deterministic modules; the Skill artifact is deliberately kept separate since it's a portable, standalone capability, not application source.

**Spec location note:** `specs/` (not `docs/`) holds the version-controlled behavioral/technical design, per the course's Spec-Driven Development convention — this is the source of truth an agent indexes to build and verify code, distinct from ad-hoc chat-window instructions. A small Gherkin (`Given/When/Then`) scenario file covers the highest-ambiguity-risk flows (§13) to remove guesswork exactly where it would be costliest.

This structure keeps agent files scoped to reasoning/generation only (per §2/§3), gives the deterministic logic (pace, gates, event-detection, patches) dedicated, independently testable homes, and includes the scoped-down evaluation artifacts from §7 — without importing RAG-specific infrastructure (retrievers, rerankers, vector stores) that this system, which grounds via live API calls and structured JSON rather than a document corpus, does not use.

## 13. Behavior Scenarios (Gherkin, high-ambiguity-risk flows only)

Per course guidance that BDD/Gherkin scenarios remove the ambiguity that causes agents to guess — applied selectively to the flows where a wrong guess is costliest, not exhaustively to the whole system:

```gherkin
Feature: Clarify Gate bounded resolution

  Scenario: Vague-but-genuine input resolves within the round bound
    Given a user enters a vague goal like "I want to make apps"
    When the gate asks a narrowing question
    And the user answers with more specificity
    Then the gate either accepts a resolved role or asks one more narrowing question
    And the total narrowing rounds never exceed 2

  Scenario: User rejects the proposed interpretation twice
    Given the bounded rounds are exhausted
    And the system has proposed a best-guess role and the user rejected it
    And the system explained the role clearly and the user rejected it again
    When the grounding check runs on the user's own words
    Then if any market signal is found, the system proceeds at low confidence
    And if zero market signal is found, the system exits and builds no outline

Feature: Confidence Ladder enforcement

  Scenario: Both sources agree
    Given Himalayas and Tavily both return the same skill for a role
    When the Research Agent cross-validates the result
    Then the outline item is tagged confidence "high"

  Scenario: No source returns usable data
    Given Himalayas and Tavily both fail or return nothing for a role
    And roles_cache has no entry for the role
    When the Research Agent attempts to ground the outline
    Then no outline item is created
    And the system reports the general-knowledge-only floor explicitly to the user

Feature: Outline content is never removed

  Scenario: User sustains a "behind" pace
    Given a user's rolling-window pace is sustained below the drift threshold
    When the Coaching Agent triggers a pacing adjustment
    Then the outline's topic list is unchanged
    And only the day-by-day delivery schedule is extended

Feature: Verification retry cap is exactly 3 attempts

  Scenario: User fails a question twice, passes on the third attempt
    Given a user answers a verification question incorrectly on attempt 1
    And a fresh regenerated question incorrectly on attempt 2
    When the user answers a fresh regenerated question correctly on attempt 3
    Then the question is marked passed at full credit
    And exactly 3 question-generation calls have occurred for that question slot, not 4
    And the first attempt counted as attempt 1, not as a call made before the retry loop began

  Scenario: User fails all 3 attempts
    Given a user answers incorrectly on attempts 1, 2, and 3 for the same question slot
    When the retry cap is reached
    Then the system teaches the answer inline, citing the source material
    And the question is marked passed at half credit, not left unresolved
    And no fourth question-generation attempt occurs
```

**`specs/scenarios/high_risk_flows.feature` is the canonical, version-controlled copy** — the block above mirrors it for readability in this document, but the `.feature` file is what implementation and any future scenario-runner should reference.