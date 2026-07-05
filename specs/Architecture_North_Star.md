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
| Market data (Himalayas) | **Himalayas MCP server** | Consumed via ADK's MCP tool integration; free, no-auth. Despite the name, its tool responses are **not** structured JSON — see resolved detail below |
| Market data (search) | **Tavily API** | Agent-native structured search results; free tier; single API key, no CSE setup. The response envelope is genuinely structured (`{url, title, content, score}` per result) — but each result's `content` is unstructured prose, needing its own coarse extractor — see resolved detail below |
| LLM | Gemini (via ADK) | Per course |
| Deployment | HF Spaces (Streamlit SDK) | Public link satisfies project-link requirement with no login |
| Secrets | HF Space secrets | Neon connection string, Tavily key, Gemini key — never committed to repo |

**Resolved implementation details (flagged during `src/db/connection.py` / `src/data/roles_cache.py` implementation, not pinned down above at the time of writing):**
- **DB access library**: plain SQLAlchemy 2.0 (declarative `Mapped`/`mapped_column` style), not SQLModel — the table above left this open ("SQLAlchemy (or SQLModel)").
- **Driver**: `psycopg` v3 (`postgresql+psycopg://` SQLAlchemy dialect), not the legacy `psycopg2` a bare `postgresql://` connection string resolves to by default — `pyproject.toml` installs `psycopg[binary]>=3.2.0`, not `psycopg2`. `db/connection.py` rewrites a bare `postgresql://` prefix automatically so `.env`'s existing `NEON_CONNECTION_STRING` format doesn't need to change.
- **Neon timeouts** (CLAUDE.md guardrail #14 requires an explicit timeout on every external call, but names no duration): `connect_timeout=10` seconds (TCP handshake) and `statement_timeout=10` seconds (per-query, set via a Postgres connection option) — both generous-but-bounded defaults for a free-tier instance, to be tuned once real latency is measured (ship-day README requirement).
- **Engine lifecycle**: the Engine is a lazy-but-memoized module-level singleton in `db/connection.py` (created on first use, cached, never recreated) rather than instantiated at raw import time — this avoids making `NEON_CONNECTION_STRING` a hard import-time requirement (e.g. in CI/test contexts with no DB configured) while still satisfying "one client per module."
- **`mcp` (the official Python MCP SDK) is a required explicit dependency, not something `google-adk` pulls in on its own.** `google.adk.tools.mcp_tool.mcp_toolset` imports `mcp` directly (`ClientSession`, `streamablehttp_client`, etc.), but `google-adk`'s own package metadata does not declare it — importing ADK's MCP toolset fails with `ModuleNotFoundError: No module named 'mcp'` until `mcp` is installed and added to `pyproject.toml` separately. Discovered during the Himalayas MCP connectivity spike (`tests/spike_grounding_connectivity.py`); resolved by adding `mcp>=1.28.1` to `pyproject.toml`.
- **Himalayas MCP's tool responses are not structured JSON, contrary to this table's original "structured" description.** Every tool call (`search_jobs`, `get_salary_data`, etc.) returns `{"content": [{"type": "text", "text": "<markdown/emoji-formatted prose>"}], "isError": bool}` — a human/LLM-readable text blob, not named fields. `search_jobs`'s text is a series of `🚀`-prefixed listing blocks, each with a title, company, an optional "🛠️ Key Skills:" line (bullet-separated, `+N more` truncation suffix on the last item), and an "Apply on Himalayas" URL. `src/data/himalayas_parser.py` is the deterministic pattern-extraction module that turns this text into structured `ParsedJobListing` objects (`title`, `company`, `skills`, `source_url`) for Agent 1's cross-validation to consume — see that module's docstring and `tests/fixtures/himalayas_search_jobs_*.txt` (real captured samples across 4 seed roles) for the full shape. `get_salary_data`'s text has a different, aggregate-statistics shape with no skills and no per-entry URL (only one shared URL for the whole response) — parsing that is out of scope for `himalayas_parser.py`, which targets `search_jobs` only.
- **`src/data/tavily_parser.py` extracts skills from Tavily's `content` field** — structurally the inverse problem from Himalayas: Tavily's response envelope is already per-result JSON (`url`/`title`/`content`/`score`, no blob-splitting needed, `url` used directly as `source_url`), but `content` itself is unstructured free-text prose (career-advice articles, job-description templates, forum posts, even a YouTube transcript — see `tests/fixtures/tavily_search_*.json`, real captures for the same 4 seed roles as the Himalayas fixtures) with no equivalent of Himalayas's consistent "Key Skills:" line. `parse_tavily_result`/`parse_tavily_response` extract skills via case-insensitive, word-bounded matching against a fixed, evidence-based vocabulary (`TECH_SKILL_VOCABULARY`, ~94 terms derived from skills `himalayas_parser.py` actually extracted for these same 4 roles, plus 2 terms observed directly in the Tavily content). This is explicitly a coarser, less trustworthy mechanism than `himalayas_parser.py`'s structural parsing — flagged in the module's own docstring, not presented as equally reliable. **Real-data finding:** Tavily's own `score` does not reliably predict whether a result's `content` has anything extractable at all — the single highest-scoring result across all 4 fixtures (Indeed's Data Analyst page, score 0.87) is a sitemap-style list of unrelated job titles with zero extractable skills, while several lower-scoring results on the same query name concrete tools explicitly. `score` is passed through on `ParsedSearchResult` untouched (no filtering inside this module) — what to do with that finding (e.g. whether/how `data/cross_validation.py` should weight or threshold on it) is left to a follow-up task, not decided here.

---

## 2. Orchestration Principles (course-aligned)

Per course guidance on DAG orchestration and "Shift Intelligence Left": agents hold reasoning and generation only. Deterministic computation is pulled into plain, testable modules that agents call as tools — never left as an instruction the agent is expected to remember and apply correctly at runtime. Concretely, the following are **not** agent logic, regardless of how they were described at the requirements stage — they are plain functions:

- Confidence-ladder tier assignment and source/schema validation (`security/output_guard.py`)
- Clarify-gate bound-counting, loop-termination, and first-pass real/vague/nonsense stated-goal classification (`security/input_gate.py`)
- Significant-event detection — bucket/confidence-crossing diff (`outline/significant_event.py`)
- Pace calculation — topic_score, timing_ratio, the 80/20 blend, sustained-drift threshold check (`pace/calculator.py`)
- Patch-note confidence branching (prioritize vs. ask-user, a threshold lookup on an already-computed value) (`patches/patch_manager.py`) — implemented as `branch_by_confidence` returning the literal `"prioritize"` / `"needs_user_decision"` (only `high` confidence prioritizes; see PRD §7.9 for the resolved cutoff)
- Outline insertion/positioning into an *already-known* dependency structure (`outline/hierarchy.py`) — distinct from initial full-hierarchy *creation*, which does require reasoning and stays in Agent 1

**State passing is by database reference, not shared prompt context.** Agent 1 persists its output (resolved role, grounded outline rows, confidence tiers) to Postgres. Agent 2 receives only `user_id` / `topic_id` references and reads what it needs directly from the database. Raw agent output is never passed as accumulated text into the other agent's context window — this follows the course's explicit "Decouple State... pass only URIs or pointers" guidance and keeps each agent's context scoped and small.

**Gates are structural, not advisory ("Reviewer & Gate" node pattern).** The database write functions for outline items, patch-notes, and grounding results accept only a pre-validated object type (post-`output_guard`), not a raw agent-generated dict. An agent cannot persist an ungrounded item because the write path itself rejects anything that hasn't passed the gate — this is a software constraint, not a prompt instruction the agent could forget.

## 3. Component Boundaries (Agent Assignment)

Two real agents, consistent with the "don't inflate agent count for rubric-checkbox reasons" principle established during requirements:

### Agent 1 — Research & Outline Agent
**Owns (reasoning/generation only):** clarify-gate conversation (asking narrowing questions, proposing/explaining role interpretations — the *content* of what to ask, not the round-counting), cross-validation normalization judgment (anchored to `roles_cache`), initial full-outline hierarchy creation (sequencing sourced skills into dependency order).
**Calls as tools (deterministic, not owned):** `security/input_gate.py` (bound state, first-pass real/vague/nonsense stated-goal classification, reject detection), `security/output_guard.py` (confidence/source validation before any write), `data/roles_cache.py` (I/O), `data/himalayas_parser.py` (search_jobs text-blob parsing), `data/himalayas_relevance.py` (relevance heuristic, since Himalayas cannot independently signal zero results), `data/cross_validation.py` (PRD §7.3 tier-decision rules), `data/grounding_fallback.py` (cached-fallback and general-knowledge-only floor rungs, invoked only once live Himalayas/Tavily grounding has already failed for the current request), `outline/significant_event.py` (diff logic), `outline/hierarchy.py` (insertion into existing structure), `patches/patch_manager.py` (confidence branching).
**Tools:** Himalayas MCP, Tavily search, Gemini (via `google-genai`, clarify-gate conversational content only), Postgres (via the gated write paths above only — never a raw insert).

**Resolved (`src/security/input_gate.py`'s `classify_stated_goal`, `src/agents/research_outline_agent.py`'s `begin_clarify_gate`/`advance_clarify_gate`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **First-pass classification mechanism (`classify_stated_goal`):** purely lexical/vocabulary-based, never a market-existence check. A multi-word phrase ending in a recognized occupation noun (`_ROLE_NOUN_SUFFIXES` — Engineer, Analyst, Manager, etc.) classifies as `REAL` unless it also contains a small, curated fabricated-title marker (`_FABRICATED_TITLE_MARKERS` — vibes, dragon, whisperer, wizard, etc.), which is what lets a title the module has never seen before (e.g. "Site Reliability Engineer") through as real rather than gate-rejected, per PRD §7.2's niche-title instruction. A single word classifies as `VAGUE` only if it's in a coarse tech/career vocabulary (`_VAGUE_TECH_SINGLE_WORDS`); otherwise a real single word with no tech connection (PRD §7.2's "banana", "purple") is `NONSENSE`. Keyboard-mash detection combines a vowel-presence check with a max-consecutive-consonant-run threshold (`_MAX_PLAUSIBLE_CONSONANT_RUN = 4`), since a bare vowel-presence check alone cannot catch a mash string that happens to contain one vowel (e.g. "asdkjfh"). All vocabulary lists are coarse and non-exhaustive — same status as `data/tavily_parser.py`'s `TECH_SKILL_VOCABULARY`, flagged for tuning as real usage is observed. The fabricated-title denylist in particular fails open to `REAL` for an unrecognized fabricated title, which is the deliberately safer failure direction per PRD §7.2's "never gate-reject a niche real title" instruction.
- **Interface redesign, not implemented as originally stubbed:** the pre-existing `generate_clarify_gate_response(conversation_so_far, current_round)` scaffold (a bare round-number int) predated `security/input_gate.py`'s actual `ClarifyGateStage` state machine and couldn't represent it — a round number alone can't distinguish NARROWING from PROPOSE_BEST_GUESS from EXPLAIN_ROLE. Replaced with `begin_clarify_gate` (the first turn) and `advance_clarify_gate` (every subsequent turn, dispatching on `ClarifyGateState.stage`), threading a new `ClarifyGateContext` (the original stated goal, captured once and never overwritten, plus the most recently proposed role) alongside `security/input_gate.py`'s state.
- **Two fixed, non-LLM messages:** the nonsense re-prompt and the zero-market-signal exit message are both plain constants, not Gemini-generated — both outcomes are already fully decided by deterministic logic (`classify_stated_goal`, `ground_role`'s confidence tier) before the message is chosen, so an LLM paraphrase would add cost/latency without adding information.
- **Model choice:** `gemini-2.5-flash` for all clarify-gate conversational turns — short, low-latency generation, not grounded content; a stronger tier may be warranted for outline/hierarchy sequencing later, a separate decision.
- **`PROMPT_REGISTRY`:** five versioned prompts (narrowing question, narrowing-answer evaluation, best-guess proposal, role explanation, acceptance evaluation) — the first real use of CLAUDE.md's LLM Call Discipline registry in this codebase; `security/input_gate.py`'s classification is deliberately *not* prompt-based at all (no registry entry needed), per its own no-LLM mandate.
- **Known gap, carried forward, not resolved by this task:** `utils/logger.py` still doesn't exist (Architecture's own prior-flagged gap for `ground_role`'s Himalayas/Tavily calls) — these new Gemini calls likewise produce zero cost/usage log entries for now. Worth flagging loudly before any real/demo run, same as before.

### Agent 2 — Coaching & Pace Agent
**Owns (reasoning/generation only):** day-by-day content generation (summary, theory framing, hands-on exercise design, reflection prompts), goal-completion closing-note composition.
**Calls as tools (deterministic, not owned):** the Verification Question Generator Skill (§4), `pace/calculator.py` (topic_score, timing_ratio, drift check), `data/progress_log.py` (I/O).
**Tools:** Verification Skill, Postgres (progress log, outline status — via gated write paths), `roles_cache` (read-only, for closing note + enrichment source).

### Cron job (not an agent)
**Owns:** the deterministic scheduled trigger (minimum every 30 days) that invokes Agent 1's Research pipeline against the seed role list to refresh `roles.json`. Deliberately not agentic — the trigger is wall-clock time, not judgment.

**Implementation — two layers, one shared function:**
- **Refresh function** (single, reusable): re-runs Agent 1's Research/Grounding pipeline for the seed role list, writes results to `roles_cache`. Called by both triggers below — no duplicated logic.
- **Primary trigger — GitHub Actions scheduled workflow** (`schedule:` cron, e.g. every 30 days): calls the refresh function directly (either via a script with its own DB/API credentials, or by hitting an app endpoint). Genuinely wall-clock triggered, independent of app traffic — this is the real "deployability"/"scheduled job" demonstration for the video.
- **Resilience layer — startup/session staleness check**: on Streamlit app startup (or first session of the day), check `roles_cache.last_updated` per cached role; if past the 30-day floor, call the same refresh function inline before continuing. Ensures the system stays honest even if the GitHub Action hasn't fired yet relative to a live demo session — not a replacement for the scheduled trigger, a safety net alongside it. **Implementation note:** the staleness comparison itself (`is_stale`) is a pure function living in `data/roles_cache.py` rather than a separate pure module — it operates on a value only that module reads (`last_updated`) with no other caller, so a separate module would be pure indirection; it still takes an explicit `reference_time` rather than calling the wall clock internally, keeping it deterministic and testable like the project's other pure modules.

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
-- Write boundary note: per guardrail #12, each core_skills/emerging_skills
-- entry must be a validated grounding result (security/output_guard.py's
-- ValidatedGroundedContent) before data/roles_cache.py's upsert_role will
-- accept it — this table is an enforced structural-gate boundary, not
-- just outline items and patch-notes.
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

**Resolved (`src/data/grounding_fallback.py`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **`general-knowledge-only` is structurally distinct from `ValidatedGroundedContent`, never coerced into it.** This rung means "no real source of any kind" by definition (PRD §7.3) — `security/output_guard.py`'s `ValidatedGroundedContent` requires a non-empty, structurally valid `source_url`, and fabricating one to force this rung through that type would violate guardrail #1 and the "never silently fabricate a source" rule. `grounding_fallback.py` instead returns a separate `GeneralKnowledgeFloorResult` (role_name, confidence, a human-readable label) with no `source_url` field to omit or fake. Per `specs/scenarios/high_risk_flows.feature`'s "No source returns usable data" scenario, this result is for honest user-facing reporting only and must never be written into `outline_topics`/`patch_notes`/`roles_cache`.
- **A stale `roles_cache` entry (past the 30-day floor) still counts as usable cached-fallback data**, not as equivalent to "no entry." Staleness only triggers the existing cron/startup refresh cycle; it does not disqualify already-grounded data from being served as today's fallback once live sources have failed. `is_stale` is surfaced as metadata on `CachedFallbackResult` so a caller can label the result honestly (e.g. "based on data from 47 days ago"), per PRD §7.3's "labeled with `last_updated`" phrasing. Flagged as a genuine judgment call, revisable if a stricter reading (stale should escalate straight to the floor) is preferred.
- **`roles_cache` skill entries carry no per-skill `source_type` on disk** (§5's JSONB shape is `{skill, source_url, confidence}` only), so one cannot be read back honestly. Cached-fallback results are stamped with a constant `source_type` of `"roles_cache-cached"`, naming the provenance layer truthfully (a previously-validated result now being re-served from cache) rather than guessing at the original external source's type.

**Known limitation (see PRD §7.2/§7.3 for the full finding):** the `reject` rung's "zero signal" trigger cannot rely on Himalayas independently reporting zero results. Live testing during `src/data/himalayas_parser.py`'s development (nonsense keyword, extreme `salary_min`, obscure `country` + `exclude_worldwide` combination) never produced a genuine empty `search_jobs` response — it fell back to broad/unrelated matches instead. Cross-validation's zero-signal determination therefore rests on Tavily + `roles_cache`, with any Himalayas results needing a relevance judgment rather than an empty-check. **Resolution mechanism below (`src/data/himalayas_relevance.py`, `src/data/cross_validation.py`); the underlying limitation itself still stands, only the inference mechanism around it is new.**

**Resolved (`src/data/himalayas_relevance.py`, `src/data/cross_validation.py`, `src/agents/research_outline_agent.py`'s `ground_role`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Relevance heuristic (`src/data/himalayas_relevance.py`):** infers "no usable Himalayas signal" from a non-empty `search_jobs` response by checking title-token overlap against the searched role. `compute_title_relevance` scores each listing (fraction of the role's tokens found in the title, simple lowercase word-set overlap, no stemming/synonyms); a listing counts as relevant at or above `PER_LISTING_TOKEN_OVERLAP_THRESHOLD` (0.5). The fraction of relevant listings required to trust the whole batch is banded by result count — `MIN_COUNT_THRESHOLD` (5) and below require `MIN_COUNT_RELEVANCE_FRACTION` (0.6); `MAX_COUNT_THRESHOLD` (25) and above require only `MAX_COUNT_RELEVANCE_FRACTION` (0.2); linearly scaled in between — mirroring `pace/calculator.py`'s `TIMING_OUTLIER_THRESHOLD`/`TIMING_SATURATION_DEVIATION` banding pattern. All five constants are unvalidated calibration judgment calls, flagged for tuning once real cross-validation runs are observed, exactly like the pace calculator's constants.
- **Cross-validation tier decision (`src/data/cross_validation.py`):** a pure function applying PRD §7.3's rules — both sources agree with the `roles_cache` anchor → `high`; single source (Himalayas or, now, Tavily — see below) → `medium`; both sources have signal but Himalayas's skills don't overlap the anchor → `medium` with an explicit `has_conflict=True` flag (a genuine conflict, PRD's "flagged" outcome); no anchor at all (Himalayas has signal) → `low` (PRD's niche/no-anchor rule — the LLM sanity-check pass that rule also describes remains explicitly deferred, not implemented); no usable signal from *either* source → `reject`. Anchor "agreement" requires at least `ANCHOR_OVERLAP_MINIMUM` (1) shared skill between Himalayas's extracted skills and the roles_cache anchor.
- **Tavily-only signal, resolved:** what was previously a hard scope limit (`himalayas_has_signal` as an unconditional precondition for any tier above `reject`) is now resolved. A Tavily-only batch (Himalayas has no usable signal) reaches `medium` confidence if it clears `TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD` (3 *distinct* skills — not total mentions, not count of skill-bearing results — found across the whole batch via `src/data/tavily_parser.py`'s extraction; unvalidated calibration judgment call, same status as the other constants on this page). Tavily's own `score` field is used **only** to select which already-skill-bearing result becomes the citation `source_url` (the highest-scoring one among those with ≥1 extracted skill) — never to decide trust, per `data/tavily_parser.py`'s real-data finding that `score` does not predict extractability. All distinct skills found across the whole trust-qualifying batch are written, attributed to that one citation URL (`TavilyCitation`) — not just the citation result's own skills; see `data/cross_validation.py`'s `TavilyCitation` docstring for why per-result attribution isn't attempted.
- **Known limitation, explicitly named (not resolved by this task):** `data/tavily_parser.py`'s `TECH_SKILL_VOCABULARY` was derived from skills `data/himalayas_parser.py` already extracted for the same 4 seed roles (Backend Engineer, Frontend Engineer, Data Analyst, DevOps Engineer), plus 2 manually-added terms — the Tavily trust check above can therefore only recognize skills Himalayas already surfaced for a role, not independently discover new ones. For any role outside those 4 (including the rest of PRD §7.3's seed list, e.g. AI/ML Engineer), Tavily's trust check has no special vocabulary coverage and will likely under-count real signal. **Future improvement path:** a real, independently-sourced skill vocabulary (not derived from Himalayas), or a smarter extraction method (e.g. an LLM-assisted pass, consistent with PRD §7.0's "reserve LLM judgment for genuinely ambiguous cases") — not attempted here.
- **Explicit attribution model for Tavily-sourced skills (previously only implicit in the bullet above):** a Tavily-sourced skill's `source_url` represents the trust-qualifying *batch* it was found in, not a per-skill provenance guarantee. Concretely: an individual skill's cited `source_url` may be a result that did *not itself* mention that specific skill — it mentions some skill(s) in the batch and won the citation ranking (highest `score` among skill-bearing results), while the skill actually being attributed to it may have been extracted from a *different* result in the same batch. The URL is always real and the skill was always genuinely found somewhere in that same trust-qualifying batch, for that role search — this is not fabrication — but it is a meaningfully different attribution model from Himalayas's, and is a new pattern in this codebase: `ground_role`'s Himalayas path (`himalayas_skill_map`, `src/agents/research_outline_agent.py`) attributes each skill to the specific listing it was actually extracted from — genuine per-skill provenance — with no precedent for the batch-level citation Tavily uses.
- **Fallback-only-on-failure ordering, now enforced:** `ground_role` (`src/agents/research_outline_agent.py`) only calls `data/grounding_fallback.py`'s `get_cached_fallback`/`get_general_knowledge_floor` when `data/cross_validation.py` resolves to `reject` — previously this ordering was documented as an assumption (see the earlier `grounding_fallback.py` task) but nothing in the codebase actually called live sources first; this is the first caller that does.
- **New exception, `utils.exceptions.GroundingSourceCallError`:** raised for a genuine Himalayas/Tavily call failure (connection error, non-success response, malformed response body, or exceeding `EXTERNAL_CALL_TIMEOUT_SECONDS` = 10s) — deliberately distinct from that source legitimately returning no relevant results, which is not an error. Both are collapsed to "no signal from this source" for tier-decision purposes, but `ground_role`'s internal `_safe_fetch_*` helpers track which happened (`"signal"` / `"no_signal"` / `"call_failed"`) so the distinction is intentional, not lost.
- **Skills written carry a source-specific `source_type`:** every `ValidatedGroundedContent` `ground_role` produces carries the confidence tier `data/cross_validation.py` decided, and either `source_type="job_listing"` (Himalayas-sourced, deduplicated by casefolded skill name across all relevant listings) or `source_type="web_search"` (Tavily-sourced, the new citation-based path above) — never both in the same result, since the two paths are mutually exclusive (Tavily-sourced skills only appear when Himalayas had no signal at all).

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

- Never accept or store an outline item / patch-note / `roles_cache` skill entry without a `source_url` and `confidence` value
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
│   │   ├── research_outline_agent.py   # reasoning/generation only, plus ground_role (cross-validation orchestrator) and begin_clarify_gate/advance_clarify_gate (clarify-gate conversational content) — see §3/§8
│   │   └── coaching_pace_agent.py      # reasoning/generation only — see §3
│   ├── security/
│   │   ├── input_gate.py               # clarify-gate bound/loop state, first-pass real/vague/nonsense classification, reject detection — see §3
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
│   │   ├── progress_log.py             # progress_log I/O
│   │   ├── grounding_fallback.py       # cached-fallback + general-knowledge-only floor rungs
│   │   ├── himalayas_parser.py         # search_jobs text-blob -> ParsedJobListing(title, company, skills, source_url) — see §1
│   │   ├── himalayas_relevance.py      # relevance heuristic inferring "no Himalayas signal" — see §8
│   │   ├── cross_validation.py         # PRD §7.3 tier-decision rules (pure) — see §8
│   │   └── tavily_parser.py            # content-field skill extractor, vocabulary-based — see §1
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