# Context Transfer — Project North Star

Consolidated continuity notes. Restructured (not appended to) each session
down from raw journaling — that history is superseded by git history and by
the "Resolved"/"Known limitation" blocks already carried in
`specs/PRD_North_Star.md` and `specs/Architecture_North_Star.md`, which is
where a landed decision's full rationale belongs. This file's job is
standing continuity — what's open, what to avoid repeating, and a compact
account of what just happened — not a chronological diary. Nothing here
duplicates CLAUDE.md's standing policy.

**Current state:** the full pipeline is built and wired end-to-end in
`src/main.py`'s Streamlit app (Intake → Clarify Gate → Research & Market
Grounding → Outline Creation → Outline Confirmation, including a real
addition-request grounding path → Day-by-Day Coaching → Verification →
pace-drift/patch-note/enrichment side effects → Goal Completion).
`db/create_schema.py` has run against the real Neon instance; all 7 tables
exist. 446 tests pass, ruff/black/mypy clean. Nothing has ever been
committed by the assistant — the user reviews and commits personally.

## What was accomplished this session

Four real, user-reported issues from a live run were root-caused and
fixed, plus one reported error investigated and confirmed *not* a bug.
Full mechanism/rationale for each lives in the specs' own "Resolved"
blocks (Architecture §3 — Gemini-client/event-loop, JSON-retry,
addition-grounding, Day-by-Day/citation blocks; PRD §11 item 6) — not
repeated here.

1. **Live-reproduced and fixed "Gemini call failed: Event loop is
   closed"** on the second Gemini-backed call of any session. Root cause:
   `genai.Client`'s cached async transport binds to whichever event loop
   is running the first time it's used, but `main.py` gives every call
   its own fresh `asyncio.run()` per Streamlit rerun. Fixed with
   `reset_gemini_client_for_new_event_loop()`, called from `_run_async`'s
   `finally` block.
2. **Live-reproduced and fixed "Gemini response was not valid JSON"** on
   large one-shot generations (outline hierarchy, ~70-item topic
   explanations) — a genuine 200 OK response occasionally comes back
   syntactically broken. `_call_gemini_json` now retries up to
   `GEMINI_JSON_RETRY_MAX_ATTEMPTS = 2` more times, feeding the concrete
   parse error back into the retry prompt (CLAUDE.md's error-fed retry
   discipline) — separate from the existing 429/503 transport retry.
3. **Closed the addition-grounding gap (PRD §11 item 6 / Architecture
   §10), previously a named, deliberate scope boundary.** New
   `ground_addition_request` (extract a clean skill name via Gemini, then
   a live Tavily-only search, never a fabricated `source_url`) wired into
   `main.py`'s Outline Confirmation via a new `OutlineConfirmationTurn
   .action` field. Verified against the real Gemini + Tavily APIs, not
   just mocks. The "not wired up yet" UI caption is removed.
4. **Day-by-Day Coaching UX, requested from the live app:** the 6
   sections now reveal one at a time via a "Next" button (new
   `day_coaching_step_index` session key) instead of all at once; the
   final button is relabeled "Start Quiz" (copy-only — stage/function
   names unchanged).
5. **Fixed a real citation-link bug, reported from the live app:**
   `_render_citations` rendered only the bare domain as plain text
   (discarding the real path), so whatever became clickable was an
   accident of the renderer's own autolink heuristics and always pointed
   at that site's homepage — reported live as a YouTube citation going to
   a broken YouTube page instead of the actual video. Fixed with an
   explicit `[label](url)` markdown link to the real, full URL.
6. **Investigated a reported "503 UNAVAILABLE" Gemini error and confirmed
   it is *not* a bug** — the retry/backoff logic (5 total attempts,
   exponential backoff) already ran to completion and correctly surfaced
   the real error once Gemini's service stayed overloaded across the
   whole window. No fallback tier exists for Gemini itself (unlike
   Himalayas/Tavily's confidence ladder) since Gemini *is* the reasoning
   engine, not a swappable data source. No code change made; confirmed
   with the user before closing.

## Key decisions made and why

- **The Gemini client singleton is reset after *every* `_run_async` call
  (success or failure), not left memoized across Streamlit reruns.** A
  fresh event loop per rerun is correct (each rerun is a genuinely fresh
  script execution), but that means the module-level "one client per
  module" convention (CLAUDE.md) has to apply at the grain of "one client
  per event loop," not "one client for the process's whole lifetime" —
  `genai.Client` is cheap to reconstruct (holds only the API key until
  first real use), so this isn't a regression of that convention, just
  its correct grain here.
- **`ground_addition_request` is deliberately Tavily-only, never
  `ground_role`'s full Himalayas+Tavily+cross-validation pipeline.**
  Himalayas is job-listing search keyed by a whole role, with no per-skill
  lookup at all — there's no such thing as "Himalayas signal for one ad
  hoc skill." Cited at `ConfidenceTier.MEDIUM`, reusing this codebase's
  existing "Tavily confirmed it, nothing cross-validated it against a
  role anchor" meaning rather than inventing a new tier.
- **`_fetch_tavily_results` now takes a raw `query: str` instead of
  building a role-specific query internally**, so both role-grounding and
  single-skill-grounding share the one low-level Tavily-call/timeout/
  error-handling primitive instead of duplicating it.
- **`OutlineConfirmationTurn` gained an `action` field so `main.py` can
  detect an `ADDITION_REQUEST` turn and follow up with grounding+
  regeneration itself**, rather than `handle_review_turn` doing the
  (potentially slow, potentially failing) live Tavily call inline —
  keeps classification fast and unconditional.
- **`insert_outline_topics`'s parameter widened from `list[
  SequencedOutlineTopic]` to `Sequence[SequencedOutlineTopic]`** — a real,
  previously-dormant type-correctness gap (`list` is invariant; every
  prior call site's `topics` variable was typed `Any` via `_run_async`'s
  `Any` return, which mypy never checks). Surfaced only once a new call
  site explicitly annotated its return type. `Sequence` is mypy's own
  suggested fix, not a cast/ignore workaround.
- **A separate, dedicated `PROMPT_REGISTRY` entry
  (`outline_addition_skill_name_extraction_v1`) for skill-name
  extraction, not folded into the existing review-turn classifier** — one
  prompt, one responsibility, per CLAUDE.md's LLM Call Discipline.

## Traps / failed approaches — don't repeat

- **A module-level async API client (`genai.Client`) silently binds its
  transport to whichever event loop is running the first time it's used.**
  Reusing that client from a *second*, different `asyncio.run()` call
  (a fresh loop each time — the correct pattern for Streamlit reruns)
  raises `RuntimeError: Event loop is closed`, reproduced by two
  back-to-back `asyncio.run(...)` calls sharing one memoized client. Any
  module-level async client in this codebase needs the same
  reset-after-each-loop treatment if it's ever called from more than one
  `asyncio.run()`.
- **An `async def` pytest test cannot call a sync wrapper that itself
  calls `asyncio.run(...)`** — `pytest-asyncio` (`asyncio_mode = "auto"`)
  already runs an `async def` test inside its own event loop, and nesting
  `asyncio.run()` inside a running loop fails. `_run_async`-calling tests
  must be plain `def`, not `async def`.
- **`list` invariance can mask a real type mismatch for a long time if
  every call site's variable happens to be `Any`-typed** (e.g., from a
  function like `_run_async` that returns `Any`). The mismatch (`list[
  Concrete]` passed where `list[Protocol]` is declared) only surfaces
  once some caller explicitly annotates the variable's real type — it is
  not a new bug at that point, just a newly-visible old one. Fix at the
  boundary with `Sequence` (covariant), not by suppressing the check.
- **Adding a retry loop to a previously-single-attempt function breaks
  every existing test that queued exactly one failing response expecting
  immediate failure** — those tests must queue `N + 1` failing responses
  (one per attempt the retry loop will actually make) to still exercise
  true exhaustion, or queue a failing-then-succeeding pair to exercise
  recovery.
- **`python-dotenv`'s `load_dotenv()` with no explicit path searches
  upward from the *calling file's own location*, not the process's cwd.**
  A throwaway diagnostic script living outside the repo silently fails to
  find the project's `.env` unless the path is passed explicitly.
- **`MagicMock`'s default behavior on an unmocked new DB read looks like
  a benign no-op, but isn't** — adding a new, unconditional DB read inside
  an already-tested function makes every pre-existing test using a bare
  `MagicMock()` session exercise real arithmetic against a `MagicMock`
  attribute instead of a value. Whenever a new DB read is added inside an
  already-tested function, every existing test through that path needs an
  explicit, well-formed mock for the new dependency.
- **A wrong-patch-target bug can hide behind the `MagicMock`-default trap
  above.** A test can patch a function where it's *defined* instead of
  where it's *used*, "pass" anyway because the real code path was never
  exercised, and look like a genuine regression guard when it verifies
  nothing. Always trace where a bare name is looked up from before
  deciding what a test should patch.
- **Gemini's free tier caps `gemini-2.5-flash` at roughly 20 requests per
  day.** Prefer targeted, isolated live probes (one function, hand-built
  inputs) over full UI walkthroughs when confirming a specific fix.
- **A script that is the literal `streamlit run` (or `AppTest.from_file`)
  target re-executes its entire top-level code — including every `class`
  statement — from scratch on every rerun.** A class instance (e.g. an
  `Enum` member) stored in `st.session_state` from a prior rerun becomes a
  non-equal object once the next rerun redefines the class. `main.py`
  works around this by storing plain `.value` strings, never the `Enum`
  member, in `session_state`.
- **No browser/Playwright/chromium binary exists in this environment** —
  `streamlit.testing.v1.AppTest` (real script execution, inspectable
  rendered element tree) plus a real `streamlit run` server-boot HTTP
  smoke test is the working substitute for visual QA.

## What remains open

**Real, tested, plan-named features with no UI/orchestration wiring**
(each needs a genuine product/UX decision, not just plumbing):
- Test-out (verify-first) — `complete_topic_test_out`/`TestOutResult`
  exist; `main.py` hardcodes `is_test_out=False` on every completion.
- Patch-note "ask_user" (low-confidence learn-now-or-defer) —
  `resolve_patch_decision`/`PatchDecisionState` exist; `maybe_deliver_patch`
  returns `None` on this branch and stops, no UI resolves it.
- Outline augmentation — `augment_existing_topic` exists and is tested;
  nothing decides that a given significant event should refresh an existing
  topic in place rather than always inserting a new one.
- Startup staleness check — `check_and_refresh_stale_roles`/
  `get_stale_or_missing_roles` exist and are tested; explicitly never wired
  into `main.py` (would mean a session start potentially blocking on
  several live grounding calls).
- Deferred patch-notes "on request" — resurfacing at goal completion is
  built; a mid-journey, user-initiated "show me what's deferred" has no UI.

**Everything else still open, unchanged from before this session:**
- `days_expected` has no real baseline formula — `main.py` uses a flat
  `DAYS_EXPECTED_PER_TOPIC = 1`, not yet consuming
  `available_time_per_week`/topic-group size/`pace_extension_days`.
- `streamlit_app.py` (root-level thin wrapper) unbuilt — see the Streamlit
  rerun trap above for why this matters architecturally, not just tidiness.
- No Alembic/FK constraints; no `Dockerfile`/`requirements.txt`/`ci.yml`/
  `README.md`.
- The shared Gemini-call helpers (`_call_gemini_json`/`GeminiCallError`)
  still live in `agents/research_outline_agent.py`, reached into across a
  package boundary by three other consumers (`coaching_pace_agent.py`,
  the Verification Skill, and now implicitly `main.py`'s addition-grounding
  path), not yet extracted to `src/utils/`.
