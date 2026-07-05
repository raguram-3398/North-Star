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

---

# Context Transfer — Session 2: Clarify Gate → Outline → Verification Skill → Coaching/Pace Agent

Written at the end of a separate, later session (interleaved with — not a
continuation of — the grounding-pipeline session documented above; that
work's files/tests are untouched here). Covers: git-history splitting,
the Clarify Gate, Initial Outline Creation, an end-to-end pipeline
integration test, Outline Confirmation, the Verification Question
Generator Skill, and `coaching_pace_agent.py`. Session narrative and
judgment-call rationale only — nothing here duplicates CLAUDE.md.

## What was accomplished this session

1. **Git-history splitting** — an existing, unsplit working tree was
   broken into 8 logical commits.
2. **Clarify Gate** (`security/input_gate.py`'s `classify_stated_goal`;
   `research_outline_agent.py`'s `begin_clarify_gate`/
   `advance_clarify_gate`) — bounded (~2 round) goal-clarification loop.
3. **Initial Outline Creation** (`create_initial_outline`) — consumes
   `ground_role`'s output shape into a sequenced, hierarchy-positioned
   outline.
4. **End-to-end pipeline integration test**
   (`tests/test_pipeline_integration.py`) — exercises clarify gate →
   outline creation as one real flow, not just unit-isolated pieces.
5. **Outline Confirmation** (`OutlineConfirmationState`/
   `OutlineReviewAction`, `begin_outline_confirmation`/
   `handle_review_turn`/`regenerate_outline_with_addition`) — the second
   bounded (~2 round) interactive loop, plus regeneration-not-insertion
   semantics for user-requested additions.
6. **Verification Question Generator Skill**
   (`.agent/skills/verification_question_generator/generator.py` +
   `SKILL.md` + EDD eval files) — stateless per-call question generation
   and answer grading, packaged as a real Antigravity Skill artifact.
7. **`coaching_pace_agent.py`** — day-by-day content generation (7-step
   hands-on-eligible structure + conceptual-only variant), verification
   retry-cap orchestration, and pace-signal computation/persistence.
   Required fleshing out four previously-stub pieces first:
   `models/schemas.py` (`OutlineTopic`/`ProgressLog`/
   `VerificationAttempt`/`PaceSnapshot` fully mapped), a rewritten
   `data/progress_log.py` (the stub had no `session` param at all),
   and two new modules, `data/verification_log.py` and
   `data/pace_snapshots.py`, plus a read/status-update-only
   `data/outline_topics.py`.
8. **`tests/test_coaching_pace_agent.py`** — 24 tests, all passing,
   covering both day-structure variants, the exact-3 retry cap (with the
   specific anti-pattern check CLAUDE.md names), half-credit teach-and-
   de-escalate, success-stops-retrying at every attempt number, the
   all-5-slots-resolved completion gate, and pace-calculator wiring.

Every module above passed ruff/black/mypy clean; full suite finished at
313 passed / 1 pre-existing unrelated failure (`test_research_grounding.py`'s
placeholder). Nothing in this session has been committed — the user
reviews and commits personally, and has said so explicitly more than
once (e.g. "do no stage ill handle git").

## Key decisions and why

**Turn-based state machine, not a blocking loop (recurring pattern across 3 features)**
- Clarify Gate, Outline Confirmation, and Verification retry-cap
  orchestration all use the same shape: an immutable, frozen dataclass
  carrying `stage`/`attempt_number`/etc., advanced one call at a time by
  a function that takes the current state and one new input, rather than
  an internal loop that blocks for multiple turns. Chosen because a real
  Streamlit request/response cycle cannot synchronously wait on several
  separate human inputs inside one function call.
- For verification specifically, this is *how* CLAUDE.md's named
  anti-pattern ("first attempt must live inside the same counter as
  retries 2/3") is satisfied without an actual `for` loop: every attempt
  number (1, 2, or 3) calls the identical `submit_verification_answer`
  function — no special-cased first-attempt code path exists to
  accidentally exclude attempt 1 from the count.

**Hands-on ramping formula (a real ambiguity, resolved and flagged, not guessed silently)**
- `(position_in_group - 1) / (group_size - 1)`, linear across a
  topic-group, 0.0 on day 1 to 1.0 on the last day. `group_size == 1` is
  special-cased to return 1.0 rather than 0.0 — otherwise a single-day
  topic-group would never get any hands-on practice at all. Written into
  PRD §7.6 and Architecture §3 as a "Resolved" block, per the task's
  explicit instruction to propose and name a concrete rule rather than
  leave the ramp undecided.

**Weekly-hours-to-daily-minutes conversion**
- `STUDY_DAYS_PER_WEEK = 5` (Monday-through-Friday assumption) — PRD
  never states a study cadence; flagged as revisable, not asserted as
  correct.

**Taught-answer message is deterministic, not a second LLM call**
- Built directly from the Verification Skill's own `grading_criteria` +
  `source_url` (`_build_taught_answer_message`). Reasoning: the grading
  criteria already **is** the rubric; asking Gemini to restate it as
  "teaching" prose risks it subtly contradicting its own rubric for no
  benefit. Consistent with this codebase's broader pattern of preferring
  structural correctness over prompt-only correctness wherever possible.

**`verification_attempts` (not caller-tracked state) is the source of truth for topic completion**
- `complete_topic_verification` reads each question slot's *final*
  attempt back from the database rather than trusting a value threaded
  through the calling code. This also turned out to be where "all 5
  slots resolved (not just attempted)" is actually enforced — a slot
  that failed but hasn't hit the retry cap yet is genuinely still in
  progress, and `_get_final_credits_per_question` raises `ValueError` in
  that case rather than treating "attempted" as "done."

**Dynamic-sizing/spillover mechanism: one generic string pair, not a typed union**
- `carried_over_content` (input) / `remaining_content` (output) on
  `generate_day_content`, deliberately not specific to *why* something
  spilled. Reasoned through explicitly (per the task's own ask) that a
  future patch-note's content could plug into the same
  `carried_over_content` parameter without reworking the mechanism —
  this task only wires it for regular-content overflow, but the seam is
  designed to be reusable, not something that will need revisiting.

**Prompt duplication over dynamic templating for hands-on vs. conceptual-only**
- Two full, separately-frozen `PROMPT_REGISTRY` strings
  (`day_content_generation_hands_on_v1` /
  `..._conceptual_v1`) rather than one template with conditional
  sections — keeps each version's baseline regression test asserting on
  a single literal string (per CLAUDE.md's LLM Call Discipline), with no
  risk of a conditional-assembly bug silently changing a frozen version.

## Traps / failed approaches — don't repeat

- **A private-name import resolves in its *defining* module's namespace,
  not the caller's — this determines where a test must patch it.**
  `_call_gemini_json`'s internal `_get_gemini_client()` lookup always
  resolves inside `research_outline_agent.py` regardless of which module
  calls `_call_gemini_json`, so `test_research_outline_agent.py`'s
  existing `_patch_gemini` helper could be reused *unchanged* for
  `coaching_pace_agent.py`'s day-content tests. But `_get_tavily_client`
  is separately imported *into* `coaching_pace_agent.py`'s own
  namespace, and `_fetch_theory_material_links` (defined in that module)
  resolves the bare name there — reusing `research_outline_agent.py`'s
  `_patch_tavily` would have silently patched the wrong module and the
  fake would never have been called. Always trace where a name is
  looked up from, not just where the underlying function is defined,
  before deciding what a test should patch.
- **An IDE "unused parameter" diagnostic caught a real design mismatch,
  not just a lint nit.** `begin_verification_question` was first written
  taking a `session` param it never used — worth checking, when a
  diagnostic like this fires, whether the function's docstring/contract
  actually promised that parameter would do something (it didn't: this
  function doesn't write to the DB, `submit_verification_answer` does).
- **An "unused import" diagnostic caught a genuine missing wire-up, not
  dead code.** After building `data/outline_topics.py`'s
  `mark_topic_completed` specifically to be called from
  `complete_topic_verification`, the import was added but the actual
  call was initially forgotten — the task's own requirement ("topic
  requires all 5 slots... to complete") wasn't fully satisfied until the
  diagnostic forced a second look and the call was added.
- **Resist writing a read function "for symmetry" before its consumer
  exists.** A `get_recent_pace_signals` read function was written into
  `data/pace_snapshots.py`, then deleted before being used — it would
  have guessed at the rolling-window shape `detect_sustained_drift`'s
  eventual caller needs (a decision explicitly deferred to a later
  task), which is the same "don't guess at an undecided next-task shape"
  principle already applied earlier (core/emerging skill split gap,
  addition-grounding gap). A one-line comment explaining the omission
  replaced it instead of a speculative function.
- **A "Resolved" spec block needs to name the actual chosen constant,
  not just gesture at "a formula was picked."** Both PRD §7.6 and §7.7
  now carry inline `Resolved (`src/agents/coaching_pace_agent.py`; ...)`
  blocks matching the exact style already established for the Clarify
  Gate, Outline Creation, and Outline Confirmation sections earlier in
  the same document — consistency here matters because future spec
  reconciliation passes pattern-match on that heading style.

## Open items

- **Test-out (verification-first)**, **patch-note delivery/surfacing**
  into the new spillover mechanism, **enrichment triggering/generation**,
  and **goal-completion closing-note content** are real, named PRD items
  deliberately not built in the `coaching_pace_agent.py` task —
  `generate_closing_note` is still an untouched `NotImplementedError`
  stub.
- **Acting on the pace signal is unbuilt**: `complete_topic_verification`
  computes and persists `pace_snapshots`, but nothing calls
  `pace/calculator.py`'s `detect_sustained_drift` or responds to
  "behind"/"ahead" yet.
- **No `outline_topics` insert path exists anywhere** — `data/
  outline_topics.py` only reads and status-updates rows;
  `create_initial_outline`/`regenerate_outline_with_addition`'s output
  is never persisted into real `outline_topics` rows. Every function in
  `coaching_pace_agent.py` assumes the row already exists. Flagged in
  PRD §11 item 8 / Architecture §10.
- **Third consumer of `research_outline_agent.py`'s private Gemini/
  Tavily helpers** (`_call_gemini_json`, `_get_tavily_client`) —
  `coaching_pace_agent.py` joins the Verification Skill as a module
  reaching across a package boundary into underscore-prefixed names.
  Extraction to a shared `src/utils/` module keeps getting more
  justified and keeps not happening (would mean touching already-tested,
  already-committed Agent code as a side effect of a narrower task).
  Flagged in PRD §11 item 7 / Architecture §10.
- **The `.agent/skills/` sys.path bootstrap is now duplicated** —
  `coaching_pace_agent.py` repeats the exact `sys.path.insert` hack
  `tests/test_verification_skill.py` already uses, since the Skill's
  required location (outside `src/`, for Antigravity workspace-manager
  recognition) isn't on the normal editable-install import path. Flagged
  in PRD §11 item 9 / Architecture §10 — worth revisiting if more Skills
  get added.
- **No `pytest --cov=src tests/` run yet**; `README.md` not started
  (shared with the grounding-pipeline session's open items above).
- Nothing from this session has been staged or committed — standing
  workflow throughout was implement → test → ruff/black/mypy clean →
  reconcile specs → report back → stop, per explicit user instruction
  each time.

---

# Context Transfer — Session 3: outline_topics Insert Gap + Test-Out (Verification-First)

Written at the end of a session covering two discrete tasks: closing the
`outline_topics` insert gap flagged at the end of Session 2, and building
test-out (verification-first) for `coaching_pace_agent.py` — including a
real mid-session correction made after direct user pushback on the
initial test-out design. Session narrative and judgment-call rationale
only — nothing here duplicates CLAUDE.md.

## What was accomplished this session

1. **`data/outline_topics.py`'s `insert_outline_topics`** — the
   previously-flagged persistence gap (Session 2's last open item):
   persists `create_initial_outline`/`regenerate_outline_with_addition`'s
   output into real rows. Replaces the user's entire prior row set
   (delete-then-insert, never a partial upsert), refuses to touch a row
   that has progressed past `not_started`, and structurally rejects a raw
   dict via a same-module `@runtime_checkable` Protocol
   (`SequencedOutlineTopic`). 5 new tests in `tests/test_outline_topics.py`.
2. **Test-out (verification-first) in `coaching_pace_agent.py`** —
   `submit_verification_answer` and `complete_topic_verification` both
   gained an additive `is_test_out: bool = False` parameter;
   `data/outline_topics.py`'s `mark_topic_completed` gained a `status`
   parameter (validated against `{completed, completed_test_out}`, since
   Architecture's schema lists them as distinct values, not synonyms);
   new `TestOutResult`/`complete_topic_test_out` orchestrate a topic's
   test-out completion.
3. **A real design correction, made mid-session after direct user
   pushback, not self-caught:** the first implementation of
   `complete_topic_test_out` built `generate_gap_study_content` (a new
   content-generation path) and wired it into the partial-pass branch,
   scoped to the questions that only resolved at `HALF_CREDIT`. The user
   pushed back with a specific question — had the interaction with
   `submit_verification_answer`'s existing inline teach-in
   (`_build_taught_answer_message`) actually been considered? It had not.
   On inspection: a `HALF_CREDIT` slot is *only* reachable by failing all
   3 attempts and receiving the teach-in, built from the identical
   `grading_criteria` `generate_gap_study_content`'s prompt also used —
   so the wiring was a genuine double-remediation bug (re-teaching the
   same rubric a second time, in different words, in the same session),
   not a richer second pass. Corrected: `complete_topic_test_out` no
   longer calls `generate_gap_study_content` at all; a partial pass now
   does exactly what a full pass does (mark `completed_test_out`) and
   generates nothing further. The now-unused `_get_failed_questions_for_topic`
   helper was deleted; `generate_gap_study_content` itself was kept
   (function is sound, just wrong to call here), unwired, flagged as a
   possible building block for a future non-test-out remediation feature.
   Tests and both specs' "Resolved" blocks were rewritten to state this
   finding directly (mechanism *and* why *and* what actually happens
   instead), per explicit instruction — not left as an implicit "gap
   content generation was descoped" gloss.
4. Spec reconciliation for both tasks in PRD/Architecture, including the
   corrected-not-just-descoped test-out narrative.

## Key decisions and why

**`insert_outline_topics`'s structural type-check, without an `agents/` import**
- `SequencedOutlineTopic`, a `@runtime_checkable` `typing.Protocol` defined
  in `data/outline_topics.py` itself, structurally matching
  `agents/research_outline_agent.py`'s `InitialOutlineTopic` — chosen over
  importing that dataclass directly because `agents/research_outline_agent.py`
  already calls `data/outline_topics.py` as a tool; the reverse import
  would invert that dependency direction and risks a real circular import
  the moment a caller inside that agent module wires this function in.
  Declared with read-only `@property` members (not plain attribute
  annotations) specifically so a frozen dataclass satisfies it under
  static type checking, not just at runtime — a plain `name: str`
  Protocol attribute is implicitly read-write and a frozen dataclass
  fails that check even though `isinstance` at runtime is fine.
- A raw dict is rejected by the same mechanism (`isinstance` against the
  Protocol fails — a dict has no `.topic_name` attribute) — satisfies
  CLAUDE.md guardrail #12 without the import.

**Regeneration-replaces-prior-unstarted-rows (a genuine, flagged judgment call)**
- Neither PRD nor Architecture specify DB-level persistence semantics for
  outline confirmation (only the conversational/regeneration behavior).
  Resolved as: delete every existing row for the user, insert the new set,
  in one transaction — safe because Outline Confirmation is provably
  pre-Day-1, so no row can have progressed past `not_started` while this
  is still possible. Not merely assumed: raises `ValueError` if it ever
  finds an already-progressed row, rather than trusting that invariant
  (CLAUDE.md guardrail #2 — never delete/overwrite started or completed
  content). This same mechanism also answers "what if this is called
  twice for the same user" (second call overwrites the first) — flagged
  as revisable if double-submission ever needs distinguishing from
  genuine regeneration.

**IDs generated explicitly (`uuid.uuid4()`) inside `insert_outline_topics`**
- Not left to `OutlineTopic.id`'s mapped-column `default=uuid.uuid4`,
  because this module's tests use a mocked `Session` (matching
  `data/roles_cache.py`'s established no-SQLite-substitute convention —
  CLAUDE.md's Stack section forbids SQLite in place of Neon), which
  cannot execute SQLAlchemy's own flush-time default-generation machinery.

**Test-out's "full pass" / "partial pass" reuse the existing credit scale**
- A slot counts as a full pass if it resolved at `FULL_CREDIT` (passed
  within the retry cap, any attempt); a partial pass is any slot at
  `HALF_CREDIT`. Chosen to mirror §7.7's own completion rule ("all 5
  resolved, full or half credit, to complete") rather than invent a
  second, competing definition of "passed" specific to test-out. This
  reuse was *correct*; what was missing (see Traps below) was checking it
  against the teach-in interaction before wiring a consumer to it.

**`mark_topic_completed` gained a `status` parameter**
- Checked directly against Architecture's schema (as the task explicitly
  instructed) rather than assuming: `completed_test_out` is listed as a
  distinct value from `completed`, not a synonym, so a test-out
  completion must write that specific status. `complete_topic_verification`
  always passes `status=` explicitly via a ternary on its own `is_test_out`
  parameter — never relies on `mark_topic_completed`'s own default in
  practice, even for the regular (non-test-out) path.

## Traps / failed approaches — don't repeat

- **Mapping a new requirement onto a readily-available existing concept
  is a good instinct, but it is not a substitute for checking that
  concept's interactions with every other mechanism already keyed off
  it.** "Partial pass = any `HALF_CREDIT` slot" is a correct, minimal
  definition on its own — the miss was not checking that `HALF_CREDIT`
  is *only* ever reached via a path (`submit_verification_answer`'s
  retry-cap teach-in) that already performs the exact remediation the
  new partial-pass branch was about to perform a second time. When a
  judgment call reuses an existing scale/enum/concept, explicitly trace
  every other place that concept already drives behavior before wiring
  a new consumer to it — don't stop at "this satisfies the PRD's literal
  wording."
- **This was caught by direct user pushback, not self-review.** The
  original report described the redundancy risk only as a hypothetical
  question to ask, correctly declining to guess — but the deeper lesson
  is that the interaction should have been checked during design, before
  writing the wiring, not surfaced only when asked to defend it.
- **Removing a wired mechanism leaves a dead, always-`None` field if the
  surrounding dataclass isn't revisited.** `TestOutResult.gap_study_content`
  would have been permanently `None` after `complete_topic_test_out`
  stopped calling `generate_gap_study_content` — deleted along with the
  wiring rather than left as vestigial, per "no half-finished
  implementations" applied to fields, not just functions.
- **Removing code without re-scanning every prose sentence that described
  it leaves specs silently wrong.** After un-wiring
  `generate_gap_study_content`, an Architecture "Resolved" bullet ("gap-
  content generation runs before the completion write, not after") was
  initially left in place — accurate for the code that no longer existed,
  describing an ordering guarantee for a step that had been deleted. Had
  to be caught and rewritten in a second pass. When code that a spec bullet
  describes changes or is removed, re-read that entire bullet for
  continued accuracy, not just the bullet that named the removed function.

## Open items

- **`generate_gap_study_content` exists but is called from nowhere** —
  kept as a possible building block for a future, non-test-out remediation
  flow (e.g. a dedicated "review what you missed" feature), not wired
  into anything.
- **Neither `insert_outline_topics` nor `complete_topic_test_out` is wired
  into a real caller.** No orchestration layer yet decides when to call
  `ground_role` → `create_initial_outline` → `insert_outline_topics`, or
  when to route a topic through test-out vs. regular day-by-day coaching
  — both are UI/orchestration decisions with no home yet (no `main.py`
  wiring, no Streamlit layer).
- Everything already open at the end of Session 2 remains open and
  untouched by this session: patch-note delivery/surfacing, enrichment
  triggering/generation, goal-completion closing-note content
  (`generate_closing_note` still `NotImplementedError`), acting on the
  pace signal (`detect_sustained_drift` never called),
  `agents/research_outline_agent.py`'s private Gemini/Tavily helpers now
  have a fourth-plus consumer with no extraction attempted, the
  `.agent/skills/` `sys.path` bootstrap duplication, no `pytest --cov`
  run yet, `README.md` not started, `roles_cache`/`db/connection.py`
  never exercised against a real Neon instance.
- Nothing from this session has been staged or committed — same standing
  workflow (implement → test → ruff/black/mypy clean → reconcile specs →
  report → stop).
