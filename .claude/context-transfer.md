# Context Transfer — Project North Star

Consolidated continuity notes, rewritten during a refactor/cleanup pass after
six build sessions' worth of session-by-session journaling accumulated here
(~1400 lines). That raw history is not reproduced below — it is fully
superseded by (a) git history, (b) the "Resolved"/"Known limitation" blocks
already carried in `specs/PRD_North_Star.md` and
`specs/Architecture_North_Star.md`, which is where session-by-session
decisions belong once a task lands, and (c) this file's own job, restated:
**current state, open gaps, and lessons still worth not re-learning** — not a
chronological diary. Nothing here duplicates CLAUDE.md's standing policy.

## Current state (as of this cleanup pass)

The full pipeline is built and wired end-to-end in `src/main.py`'s Streamlit
app: Intake → Clarify Gate → Research & Market Grounding → Outline Creation →
Outline Confirmation → Day-by-Day Coaching → Verification → pace-drift /
patch-note / enrichment side effects → Goal Completion. `db/create_schema.py`
has been run against the real Neon instance; all 7 tables exist. 434 tests
pass (`pytest tests/`), ruff/black/mypy clean.

## Known, deliberate gaps versus the approved plan

These are real, tested, backend-complete features with no UI/orchestration
wiring — each was checked during this cleanup pass and confirmed to require
a genuine product/UX decision (not just plumbing), so none were wired in
unilaterally:

- **Test-out (verify-first).** `agents/coaching_pace_agent.py`'s
  `complete_topic_test_out`/`TestOutResult` are real and tested, but
  `main.py` never offers the choice — every completion hardcodes
  `is_test_out=False`.
- **Patch-note "ask_user" (low-confidence learn-now-or-defer).**
  `patches/patch_manager.py`'s `resolve_patch_decision`/`PatchDecisionState`
  are real and tested, but nothing in `main.py` ever constructs or resolves
  one — `maybe_deliver_patch` returns `None` on this branch and stops.
- **Outline augmentation.** `outline/hierarchy.py`'s `augment_existing_topic`
  (PRD §7.4's second update type, alongside "new addition") has zero real
  callers — nothing anywhere decides that a given significant event should
  refresh an existing topic in place rather than insert a new one.
- **Startup staleness check.** `cron/refresh_roles.py`'s
  `check_and_refresh_stale_roles`/`get_stale_or_missing_roles` are real and
  tested but were explicitly scoped as "not wired into any Streamlit app,"
  and still aren't — wiring it means deciding whether a session start should
  block on up to several live grounding calls, a real UX tradeoff.
- **Deferred patch-notes "on request."** They resurface at goal completion
  (built); a mid-journey, user-initiated "show me what's deferred" has no UI.

Also still open, unchanged from before this pass: `days_expected` has no real
baseline formula (`main.py` uses a flat `DAYS_EXPECTED_PER_TOPIC = 1`, not yet
consuming `available_time_per_week`/topic-group size/`pace_extension_days`);
`streamlit_app.py` (root-level thin wrapper) unbuilt; no Alembic/FK
constraints; no `Dockerfile`/`requirements.txt`/`ci.yml`/`README.md`; the
addition-request → grounding gap (Outline Confirmation can't fold a raw
user-requested addition into a *grounded* skill — no single-skill grounding
function exists); the shared Gemini-call helpers
(`_call_gemini_json`/`GeminiCallError`) still live in
`agents/research_outline_agent.py` and are reached into across a package
boundary by three other consumers, not yet extracted to `src/utils/`; no
browser/Playwright in this environment, so no pixel-level visual QA has ever
been done (`streamlit.testing.v1.AppTest` plus a real server-boot smoke test
is the working substitute).

## This cleanup pass (dead-code audit + bug fix + doc reconciliation)

- **Root-caused and fixed the "Outline Creation failed: Gemini call failed:
  TimeoutError()" crash.** `EXTERNAL_CALL_TIMEOUT_SECONDS` (10s, sized for a
  short conversational turn) was also the per-attempt timeout for
  `create_initial_outline`'s call — a one-shot structured generation over
  potentially dozens of skills. A live probe measured 13.5s end to end for a
  10-skill/21-topic outline; since a bare `asyncio.TimeoutError` isn't
  retried, the call failed on its first attempt, every time. Fixed with a
  new `HEAVY_GENERATION_TIMEOUT_SECONDS = 45`, passed explicitly by every
  one-shot structured-generation call site; the outer retry-loop ceiling is
  now computed per call from whichever per-attempt timeout is in play
  (`_compute_gemini_retry_loop_timeout`), not a single fixed constant sized
  only for the short-turn case. Live-reproduced, live-confirmed fixed (both
  via a direct `create_initial_outline` call, not just unit tests), and
  covered by new regression tests. See Architecture §3's dedicated
  "Resolved" block for the full mechanism.
- **Found and closed a real, previously-undetected PRD gap: "weeks 1-2 are
  calibration only" was never actually implemented.** `detect_sustained_
  drift`'s `DRIFT_WINDOW_SIZE` gating is a *count* of snapshots, not
  calendar time — a user completing 3 topics in 2 days would already trigger
  real drift/enrichment/pacing-extension actions. `complete_topic_
  verification` now also gates on `users.created_at` vs. a new
  `COLD_START_CALIBRATION_DAYS = 14`. See PRD §7.8 / Architecture §3's
  updated "Resolved" blocks.
- **Audited every function in `src/` and `.agent/skills/` for real callers**
  (word-boundary usage counts + manual verification, not just grep noise)
  and removed genuine dead code: `security/output_guard.py`'s
  `assign_confidence_tier` (an unimplemented stub, superseded by
  `data/cross_validation.py`'s real `decide_confidence_tier`);
  `utils/exceptions.py`'s `GroundingError`/`VerificationTimeoutError` (never
  raised or caught anywhere); `data/progress_log.py`'s
  `get_progress_for_topic` (no consumer, no test); `patches/patch_manager.py`'s
  `mark_patch_delivered`/`mark_patch_deferred` (superseded by
  `data/patch_notes.py`'s `update_patch_note_status`, a direct DB-status
  write); `agents/coaching_pace_agent.py`'s `generate_gap_study_content` +
  its `_format_failed_questions_for_prompt` helper (built, then deliberately
  left unwired in an earlier session as a "possible future building block"
  that never materialized and isn't named in the approved plan); and
  `tests/test_research_grounding.py` (an always-failing placeholder fully
  superseded by `tests/test_research_outline_agent.py`'s real coverage).
  Distinguished throughout from the "known, deliberate gaps" list above —
  those are real, tested, plan-named features simply missing UI wiring, and
  were kept.
- **Reconciled `specs/PRD_North_Star.md` and `specs/Architecture_North_Star.md`**
  against all of the above, including a debt carried since a prior session
  (its own incident-driven Gemini retry-loop-timeout/pacing-lock follow-up
  fix was never spec-reconciled at the time — closed here alongside this
  pass's own timeout fix).
- Nothing in this pass has been committed — the user reviews and commits
  personally, per every prior session's standing workflow.

## Lessons still worth not re-learning

- **A default argument value bound in a function signature
  (`def f(x: float = SOME_MODULE_CONSTANT)`) is evaluated once at
  function-*definition* (import) time.** A test's
  `monkeypatch.setattr(module, "SOME_MODULE_CONSTANT", ...)` then silently
  has no effect on any caller relying on that default — the fix is a `None`
  sentinel default, resolved to the live module constant inside the
  function body. Bit this exact pass while adding `timeout` parameters to
  `_generate_content_with_retry`/`_call_gemini_text`/`_call_gemini_json`;
  caught by the existing timeout-path tests suddenly failing, not by
  inspection.
- **`python-dotenv`'s `load_dotenv()` with no explicit path searches upward
  from the *calling file's own location*, not the process's cwd.** A
  throwaway diagnostic script living outside the repo (e.g. a scratchpad
  probe) will silently fail to find the project's `.env` unless the path is
  passed explicitly — surfaced as a confusing `RuntimeError: GEMINI_API_KEY
  environment variable is not set` even though the key was genuinely set in
  `.env`, at first misread as a real app bug before being traced to the
  probe script itself.
- **`MagicMock`'s default behavior on an unmocked new DB read looks like a
  benign no-op, but isn't.** Adding a new, unconditional DB read inside an
  already-tested function (`get_user`, in this pass's cold-start fix) makes
  every pre-existing test that uses a bare `MagicMock()` session start
  exercising that real function's arithmetic against a `MagicMock`
  attribute (e.g. `.created_at`), not a value — surfaces as a `TypeError`
  comparing `MagicMock` to `int`, not a silent pass. Whenever a new DB read
  is added inside an already-tested function, every existing test through
  that path needs an explicit, well-formed mock for the new dependency, not
  just the pre-existing ones.
- **Gemini's free tier caps `gemini-2.5-flash` at roughly 20 requests per
  day, not just per-minute** — confirmed via real 429s in more than one past
  session. A single live end-to-end pipeline walkthrough can exhaust it.
  Budget live/unmocked verification calls deliberately; prefer targeted,
  isolated live probes (one function, hand-built inputs) over full
  UI walkthroughs when confirming a specific fix, and fall back to mocked
  `AppTest`/unit-test runs once the day's quota is likely gone.
- **A script that is the literal `streamlit run` (or `AppTest.from_file`)
  target re-executes its entire top-level code — including every `class`
  statement — from scratch on every rerun.** A class instance (e.g. an
  `Enum` member) defined in that script and stored in `st.session_state`
  from a prior rerun becomes a non-equal object once the next rerun
  redefines the class — a real, reproduced `KeyError`. `main.py` works
  around this by storing plain `.value` strings, never the `Enum` member,
  everywhere in `session_state`. Building the root-level `streamlit_app.py`
  this repo's structure already names, as a thin `import`-and-call wrapper
  around `src/main.py`'s `main()`, would close this architecturally instead
  (only ever normally imported, never re-executed) — still unbuilt.
- **A wrong-patch-target bug can hide behind the `MagicMock`-default trap
  above, compounding.** A test can patch a function where it's *defined*
  instead of where it's *used* (a direct import gives the importing module
  its own name binding), "pass" anyway because the real code path was never
  exercised, and look like a genuine regression guard when it verifies
  nothing. Always trace where a bare name is looked up from — the defining
  module, or an importing module's own namespace — before deciding what a
  test should patch.
- **No pixel-level visual QA has ever been possible in this environment** —
  no browser/Playwright/chromium binary. `streamlit.testing.v1.AppTest`
  (real script execution, inspectable rendered element tree) plus a real
  `streamlit run` server-boot HTTP smoke test is the working substitute.
