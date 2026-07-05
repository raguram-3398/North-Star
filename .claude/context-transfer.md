# Context Transfer — Project North Star

Written at the end of a session that built the full grounding/cross-
validation pipeline: `grounding_fallback.py`, the Himalayas + Tavily
connectivity spike, both source parsers, the relevance/trust heuristics,
and `cross_validation.py` + `research_outline_agent.py`'s real
`ground_role` implementation. Session history and judgment-call
rationale only — nothing here duplicates CLAUDE.md's standing policy.

## What was accomplished this session

1. **`src/data/grounding_fallback.py`** — the cached-fallback and
   general-knowledge-only floor rungs of the confidence ladder, plus
   `tests/test_grounding_fallback.py`.
2. **Connectivity spike** (`tests/spike_grounding_connectivity.py`) —
   first live Himalayas MCP + Tavily calls in this project. Discovered
   the `mcp` PyPI package gap (see below) and that Himalayas's tool
   responses are prose, not JSON.
3. **`src/data/himalayas_parser.py`** — deterministic `search_jobs`
   text-blob parser, built against real fixtures for 4 seed roles
   (`tests/fixtures/himalayas_search_jobs_*.txt`), gathered *before*
   writing any parsing logic.
4. **A missed-then-recovered spec-reconciliation pass**: the Himalayas
   zero-result finding was initially only captured as an implementation
   footnote (Architecture §1's "structured" correction), not as the
   behavioral limitation it actually is. Caught in a follow-up task,
   fixed across PRD §7.2/§7.3 and Architecture §8, and a new standing
   rule was added to CLAUDE.md's Git workflow section as a result (see
   Key decisions below).
5. **`src/data/himalayas_relevance.py`** — the title-token-overlap
   heuristic that infers "no usable Himalayas signal" despite a
   non-empty response (Himalayas structurally cannot return a true
   empty result — see traps below).
6. **`src/data/cross_validation.py`** + **`research_outline_agent.py`'s
   `ground_role`** — the real cross-validation orchestrator Architecture
   §3 assigns to Agent 1: calls Himalayas + Tavily in parallel, applies
   PRD §7.3's tier rules, enforces (for the first time anywhere in this
   codebase) that `grounding_fallback.py` is only invoked *after* live
   grounding fails.
7. **`src/data/tavily_parser.py`** — coarse, vocabulary-based skill
   extractor for Tavily's unstructured `content` field, built against
   real fixtures for the same 4 roles (`tests/fixtures/tavily_search_*.json`).
8. **`cross_validation.py` rewired** so a strong Tavily-only signal can
   reach `medium` confidence (previously a hard, explicitly-flagged
   scope limit) — distinct-skill-count trust threshold, `score` used
   only for citation selection.
9. **A documentation-precision fix**: the Tavily citation-attribution
   model (one URL representing a whole batch, not per-skill provenance)
   was only implicitly describable from the mechanism description: made
   explicit in Architecture §8 after a direct question caught the gap.

Every module above passed ruff/black/mypy clean and has dedicated tests;
nothing in this session has been committed — the user reviews and
commits personally (see Working pattern, unchanged from before this
session).

## Key decisions and why

**`grounding_fallback.py`**
- `GeneralKnowledgeFloorResult` deliberately bypasses
  `ValidatedGroundedContent` entirely (no `source_url` field to omit or
  fake) — confirmed against `specs/scenarios/high_risk_flows.feature`'s
  "No source returns usable data" scenario, which explicitly says no
  outline item is created at this rung, only honest reporting.
- A *stale* cached entry still counts as usable fallback data (`is_stale`
  is metadata for labeling, not a gate) — flagged as a genuine, revisable
  judgment call, not a settled reading.
- `CACHED_SOURCE_TYPE = "roles_cache-cached"` constant, since
  `roles_cache`'s JSONB shape never persists a per-skill `source_type` to
  read back honestly.

**Connectivity spike / `mcp` dependency**
- `google.adk.tools.mcp_tool.mcp_toolset` imports the `mcp` PyPI package
  directly, but `google-adk` does not declare it as a dependency —
  discovered as a real `ModuleNotFoundError`, not a hypothetical. Fixed
  with explicit user approval: `mcp>=1.28.1` added to `pyproject.toml`.
- Himalayas MCP's tool responses are **not structured JSON** despite
  Architecture's original "structured" description — every call returns
  `{"content": [{"type": "text", "text": "<prose>"}], "isError": bool}`.
  This is why `himalayas_parser.py` exists at all.

**Himalayas zero-result limitation (the one that got missed, then fixed)**
- Live testing (nonsense keyword, extreme `salary_min`, obscure
  `country` + `exclude_worldwide`) never produced a genuine empty
  `search_jobs` response — Himalayas falls back to broad/unrelated
  matching instead of strictly filtering to zero, even for garbage
  queries. This is a **behavioral limitation on what the confidence
  ladder can guarantee**, not just an implementation detail — it took a
  dedicated follow-up task to get it properly into PRD §7.2/§7.3 and
  Architecture §8 (see the new CLAUDE.md rule below).

**`himalayas_relevance.py`**
- Title-token overlap (`compute_title_relevance`), banded by result
  count (`MIN_COUNT_THRESHOLD=5`/`MAX_COUNT_THRESHOLD=25`,
  `MIN_COUNT_RELEVANCE_FRACTION=0.6`/`MAX_COUNT_RELEVANCE_FRACTION=0.2`,
  linear in between) — deliberately mirrors `pace/calculator.py`'s
  `TIMING_OUTLIER_THRESHOLD`/`TIMING_SATURATION_DEVIATION` banding
  pattern. All constants unvalidated, flagged for tuning, same status as
  the pace calculator's.

**`cross_validation.py` (original design, Task 2b)**
- The tier-decision logic is a separate, pure module rather than inline
  agent reasoning — consistent with `outline/significant_event.py` and
  `patches/patch_manager.py`'s existing precedent, and with PRD §7.3's
  own framing ("not open-ended LLM judgment").
- Original design made `himalayas_has_signal` a hard precondition for
  anything above `reject` — an explicitly flagged scope limit (no
  Tavily-side skill extractor existed yet), not a permanent decision.

**`tavily_parser.py` (Task 3a)**
- Vocabulary-based keyword-spotting (`TECH_SKILL_VOCABULARY`, ~94 terms),
  not real extraction — the module's own docstring says so plainly.
  Vocabulary was derived from skills `himalayas_parser.py` already
  extracted for the same 4 seed roles, plus 2 manually-added terms
  observed directly in Tavily content (`Django`, `Kafka`).
- **Real finding, load-bearing for Task 3b's design:** Tavily's own
  `score` does not predict extractability — the single highest-scoring
  result across all 4 fixtures (Indeed's Data Analyst sitemap page) had
  zero extractable skills, while several lower-scored results on the
  same query named concrete tools. `score` is passed through untouched,
  never filtered on, inside this module.

**`cross_validation.py` rewire (Task 3b)**
- `TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD = 3`, counting **distinct**
  skills (not total mentions, not count of skill-bearing results) —
  chosen because mentions let one repeated buzzword inflate trust
  without breadth, and result-count ignores how much each result
  contributes. Real batches (4 fixtures) had 4-11 distinct skills; 3 sits
  comfortably below that while still requiring more than an incidental
  one-token match.
- Citation selection: highest-`score` result **among skill-bearing
  results only** — a skill-less result must never win regardless of
  score (this is the concrete failure mode the Task 3a finding warns
  against, and has a dedicated regression test).
- All distinct skills in the trust-qualifying batch get attributed to
  **one shared citation URL** — a genuinely new attribution pattern in
  this codebase. Himalayas's path (`himalayas_skill_map` in
  `research_outline_agent.py`) attributes each skill to the specific
  listing it was actually found in; Tavily's does not. This was only
  implicitly describable until a direct follow-up question caught the
  gap — now stated explicitly in Architecture §8.

## Traps / failed approaches — don't repeat

- **Misleadingly-named fixture files must be deleted immediately, not
  left around "for later."** A fixture saved as
  `himalayas_search_jobs_zero_results.txt` actually contained 94 results
  (a filter combination that didn't do what it looked like it should) —
  deleted and re-gathered under an honest name
  (`..._nonsense_keyword_fallback.txt`) rather than leaving a
  self-contradicting artifact in the repo.
- **Stop iterating via throwaway inline Python heredocs once the shape
  is understood.** Repeatedly re-running near-duplicate exploratory
  scripts to poke at a live API drew explicit user pushback ("why are
  you repeating the same code — build the real thing once"). After 1-2
  exploratory calls, commit to writing the actual deliverable file and
  iterate on *that* directly.
- **A branch-ordering bug shipped in `cross_validation.py`'s first
  draft**: the `tavily_has_signal` branch returned unconditionally
  without ever checking Himalayas's anchor-overlap, making the HIGH tier
  unreachable. Caught by a test that held every precondition fixed and
  varied *only* the one decision-relevant input (anchor overlap) —
  a test that checks each branch in isolation without this "same
  preconditions, one variable changed" structure would not have caught
  it. Apply this pattern to any future branching pure function.
- **Dataclass definition order matters when one references another as a
  type.** `CrossValidationDecision.tavily_citation: TavilyCitation | None`
  needed `TavilyCitation` defined first in the file — simpler to just
  order definitions by dependency than to reach for a string
  forward-reference.
- **A real Himalayas zero-result response could not be obtained despite
  several genuine attempts** (nonsense keyword, extreme `salary_min`,
  obscure `country` + `exclude_worldwide`). Don't keep trying more filter
  combinations expecting a different outcome — document the limitation
  honestly (as a real, tested finding) and use a clearly-labeled
  constructed test case for the genuinely-unobservable scenario instead.
- **A "structural/stack correction" bullet does not automatically cover
  a "behavioral limitation" finding** — these are different bars. This
  session's own miss (documenting the Himalayas shape correction but not
  the zero-result behavioral consequence) is *why* CLAUDE.md's Git
  workflow section now has an explicit rule requiring a broader
  re-scan before every commit, not just "did I note the constants I
  chose."
- **Describing a mechanism is not the same as stating its consequence
  explicitly.** "All skills in the batch attribute to one citation URL"
  describes what happens; it doesn't say "a specific skill's cited URL
  may not be where that skill was found." When a reviewer asks "does the
  doc say X in plain terms," check for the literal sentence, not just a
  derivable inference — pyproject/spec precision bar is high here.
- **Pip installs (even `--dry-run`) need explicit user approval in this
  environment** — don't attempt them unprompted, even for a read-only
  version check.

## Working pattern (continuity note, not policy — matches prior session)

- Implement → test → ruff/black/mypy clean → flag every constant/
  judgment call explicitly in the chat response → **do not commit** →
  wait for explicit review. Not one commit has been made by the
  assistant this session either.
- **Real sample data gathered *before* writing any parsing/extraction
  logic**, for both Himalayas and Tavily — 3-4 real fixture files across
  different roles each, explicitly checked for edge cases (missing
  fields, single-item results, off-topic noise) rather than coding
  against one lucky example. This discipline held up twice this session
  and caught real behavior neither parser's author would have guessed
  from a single sample (Himalayas's zero-result fallback; Tavily's
  score/extractability mismatch).
- Every new deterministic module's judgment-call constants get an
  inline comment explaining the choice (mirroring
  `pace/calculator.py`'s established style) *and* get re-flagged in the
  chat report — constants are never silently finalized.
- Spec reconciliation happens in the same task as the code that
  necessitates it, and now explicitly includes a re-scan for
  behavioral/limitation findings, not just implementation details (see
  the new CLAUDE.md rule).

## Open items

- `src/agents/research_outline_agent.py`: `ground_role` is now real;
  `generate_clarify_gate_response` and `create_initial_outline` are
  still `NotImplementedError` stubs. `create_initial_outline` will need
  to consume `ground_role`'s output shape
  (`LiveGroundingResult`/`CachedFallbackResult`/`GeneralKnowledgeFloorResult`)
  — that consumption hasn't been designed yet.
- `src/agents/coaching_pace_agent.py`, `src/cron/refresh_roles.py`,
  `src/data/progress_log.py`, `src/utils/logger.py` — still fully
  stubbed, unchanged from before this session.
- **`utils/logger.py`'s absence means `ground_role`'s live Himalayas/
  Tavily calls currently produce zero cost/usage log entries**,
  contrary to CLAUDE.md's Cost & Usage Tracking section — worth flagging
  loudly before any real/demo run, not just at logger-build time.
- `security/output_guard.py`'s `assign_confidence_tier` — still
  unimplemented (flagged since before this session).
- `.agent/skills/verification_question_generator/generator.py` — still
  doesn't exist, only `SKILL.md`.
- `evaluation/golden_dataset.json`, `evaluation/eval_cases.json` — still
  empty placeholders.
- No Streamlit UI, no `.github/workflows/*.yml` — untouched.
- `roles_cache.py`/`db/connection.py` — still never exercised against a
  real Neon instance; all tests remain mocked.
- No `pytest --cov=src tests/` run yet; `README.md` not started.
- PRD §7.3's niche/no-anchor lightweight LLM sanity-check pass — still
  explicitly deferred, not implemented anywhere in `cross_validation.py`.
- `get_salary_data` integration — still out of scope/unbuilt; only
  `search_jobs` is parsed on the Himalayas side.
- **`TECH_SKILL_VOCABULARY` (`tavily_parser.py`) has zero coverage for
  any role outside the 4 already-fixtured seed roles** (e.g. AI/ML
  Engineer) — explicitly named in Architecture §8 as a future-
  improvement item (independent vocabulary or an LLM-assisted pass), not
  attempted.
- The new Tavily-only medium-confidence path (`cross_validation.py`'s
  Task 3b rewire) has only ever been exercised against mocked data in
  tests — never a live Tavily call. Worth a live smoke-test before
  relying on it in a demo.
- Multi-role/batch orchestration is unaddressed: `ground_role` grounds
  one role at a time. How PRD §7.3's seed-role bootstrap loop
  (5-8 roles, written into `roles_cache` via `upsert_role`) actually
  calls this repeatedly hasn't been wired up.
