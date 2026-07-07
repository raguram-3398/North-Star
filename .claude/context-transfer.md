# Context Transfer — Project North Star

Consolidated continuity notes. Restructured (not appended to) each session
down from raw journaling — that history is superseded by git history and by
the "Resolved"/"Superseded" blocks already carried in
`specs/PRD_North_Star.md` and `specs/Architecture_North_Star.md`, which is
where a landed decision's full rationale belongs. This file's job is
standing continuity — what's open, what to avoid repeating, and a compact
account of what just happened — not a chronological diary. Nothing here
duplicates CLAUDE.md's standing policy.

**Current state:** the full pipeline is built and wired end-to-end in
`src/main.py`'s Streamlit app (Intake → Clarify Gate → Research & Market
Grounding → Outline Creation → Outline Confirmation → Day-by-Day Coaching
— including a test-out entry point and a real `days_expected` baseline —
→ Verification — including a non-blocking patch ask_user banner → Goal
Completion). `streamlit_app.py` is the real `streamlit run` target now,
not `src/main.py` directly. `Dockerfile`/`requirements.txt`/
`.github/workflows/ci.yml` all exist and are verified working. 465 tests
pass, ruff/black/mypy clean. Nothing has ever been committed by the
assistant — the user reviews and commits personally.

## What was accomplished this session

The user handed over the prior session's full "what remains open" list
(5 real-but-unwired features, plus days_expected/streamlit_app.py/
Dockerfile-CI gaps) and said "fix all the open items." All 9 were closed
in one long session. Alembic/FK-constraints and README.md were explicitly
descoped by the user partway through (dropped from scope entirely, not
deferred) — see "What remains open" below.

1. **Extracted shared Gemini-call infrastructure to `src/utils/gemini_client.py`**
   (pacing, timeout/retry constants, `_call_gemini_text`/`_call_gemini_json`,
   `reset_gemini_client_for_new_event_loop`) out of
   `agents/research_outline_agent.py` — closing a long-flagged architectural
   seam (`.agent/skills/verification_question_generator/generator.py` and
   `agents/coaching_pace_agent.py` were both reaching into another module's
   private, underscore-prefixed namespace). All three now import it as
   peers. Required a matching test split: `tests/test_gemini_client.py` is
   the new home for the shared-infra tests (fixtures rewritten to patch
   `utils.gemini_client._get_gemini_client`, not
   `agents.research_outline_agent._get_gemini_client` — patching the old
   location now silently does nothing, since that's not where
   `_generate_content_with_retry` resolves the name from anymore).
2. **Wired test-out (verify-first) into `main.py`.** A one-time choice
   screen (`_render_test_out_choice`) appears exactly when a topic's day
   content hasn't been generated yet. Choosing "Test out" calls
   `fetch_theory_material_links` (promoted from private to public — a
   second real consumer) directly, never `generate_day_content` — the
   whole point of PRD §7.6's "verification first, before study content is
   generated" is that the Gemini-generated content is never produced, not
   just never shown. No usable theory material found -> test-out refused
   for that topic (an honest error), never a question set with no real
   grounding.
3. **Wired the startup staleness check into `main.py`.** `_maybe_check_
   stale_roles`, called once per browser session (a `startup_staleness_
   checked` flag) right after `_init_session_state`, checks `SEED_ROLES`
   (not `resolved_role` — nothing's resolved yet at this point in the
   pipeline). Failure degrades to a dismissable `st.warning`, never a
   crash — a resilience layer must never become a hard dependency.
4. **Resolved the augment-vs-addition decision rule** (Architecture §9's
   long-standing "neither PRD §7.4 nor this document resolves" gap), per
   the user's explicit choice: augment if the significant event's skill
   already exists as an outline topic. In practice this is *always* true
   for `maybe_deliver_patch`'s one real caller (a patch-note's origin
   topic already exists and matches by construction), so "insert_now" now
   calls a new `data/outline_topics.py`'s `augment_outline_topic` instead
   of `insert_new_outline_topic` — no more `" (Update)"` sibling topics.
5. **Built the patch ask_user UI**, per the user's choice of a
   non-blocking dismissible banner. `maybe_deliver_patch` now returns a
   new `PendingPatchDecision` (patch content + origin topic id) instead of
   silently returning `None` on the "ask_user" branch. `main.py` renders
   it as a banner alongside — never blocking — "Continue to next topic".
   A new `resolve_pending_patch_decision` (mirrors "insert_now"'s
   augmentation exactly on "learn now"; no-op on "defer") is the
   resolution counterpart.
6. **Replaced the flat `DAYS_EXPECTED_PER_TOPIC = 1` placeholder** with
   `agents/coaching_pace_agent.py`'s `calculate_days_expected` — a new
   `ESTIMATED_MINUTES_PER_TOPIC = 120` constant divided by
   `convert_weekly_hours_to_daily_minutes`'s daily budget (the same
   conversion `generate_day_content` already used), rounded up, floored
   at 1. Per the user's chosen approach. Does **not** factor in
   topic-group size or accumulated `pace_extension_days` — flagged as
   still-open, not silently solved.
7. **Built `streamlit_app.py`** — a thin `from main import main` wrapper,
   closing the Enum/session_state re-execution bug class structurally
   (Architecture §3's "Known limitation"). Verified live: boots and
   serves HTTP 200.
8. **Built `Dockerfile`/`requirements.txt`/`.dockerignore`.** Real
   verification caught two genuine bugs before they shipped: (a) a plain
   (non-editable) `pip install .` never installs `main.py` at all, since
   `setuptools`' package-discovery only picks up real packages
   (directories), not a lone top-level module sitting next to them; (b) a
   non-editable install copies source into site-packages, which breaks
   `agents/coaching_pace_agent.py`'s `_SKILLS_DIR` path (computed relative
   to its own `__file__`) — it would resolve against site-packages, not
   `/app/.agent/skills`. Fixed by using an editable install
   (`pip install --no-deps -e .`) in the image, which keeps every module's
   `__file__` pointing at the real `/app/src/...` location. Docker itself
   isn't available in this environment — verified instead via a fully
   equivalent fresh-venv simulation (same COPY layout, same install
   commands, then a real `streamlit run streamlit_app.py` boot to HTTP 200).
9. **Built `.github/workflows/ci.yml`** (ruff/black/mypy/pytest on
   push+PR) — needs no repo secrets, since every test mocks its external
   calls. Verified by running the exact same steps in a separate fresh
   venv via `pip install -e .[dev]`.
10. **Reconciled both specs against every change above** — see their own
    "Resolved"/"Superseded" blocks (PRD §7.4/§7.6/§7.9/§11; Architecture
    §3/§5/§9/§10) for full rationale on each. Also updated CLAUDE.md's
    repo-structure comments (removed stale "NOT YET BUILT" markers for
    the 4 files now built).

## Key decisions made and why

- **The augment-vs-addition rule and the patch ask_user UI shape were
  both genuine product decisions the user made explicitly** (via
  clarifying questions), not inferred: augment-if-skill-exists
  (regardless of completion status), and a non-blocking banner rather
  than a blocking prompt or auto-defer. Both are now the spec's own
  "Resolved" language, not implementation-detail asides.
- **`fetch_theory_material_links` was promoted from private
  (`_fetch_theory_material_links`) to public** the moment `main.py`
  became its second real caller (test-out) — the same "promote once a
  second real consumer needs it" reasoning already used for the Gemini
  client extraction. Renaming required updating the one existing test
  reference and one docstring line; no behavior change.
- **`calculate_days_expected` intentionally does not consume
  `pace_extension_days`** — the user's chosen formula was scoped to
  `available_time_per_week` only. Recorded explicitly in both specs as
  still-open, not quietly folded into "resolved."
- **The Dockerfile's editable-install requirement is load-bearing, not a
  style preference** — see "Traps" below. Any future change to how this
  image installs the package must re-verify `main`/`agents.*` both still
  import and that `.agent/skills/` still resolves correctly.

## Traps / failed approaches — don't repeat

- **A non-editable `pip install .` silently fails to install a lone
  top-level module (`main.py`) that sits next to real packages in
  `package-dir = {"": "src"}`.** `setuptools`'s `packages.find` only
  discovers directories with an `__init__.py`; a stray `.py` file at the
  same level needs an explicit `py-modules` entry, or (the fix used here)
  an editable install, which sidesteps discovery entirely by pointing
  back at the source tree. Caught by actually trying `import main` after
  a real `pip install --no-deps .` in a fresh venv — would have shipped
  broken otherwise.
- **A non-editable install breaks any `Path(__file__).resolve().parent...`
  based path computation**, since the module's `__file__` now points into
  site-packages, not the original source tree.
  `agents/coaching_pace_agent.py`'s `_SKILLS_DIR` is exactly this pattern.
  Editable installs keep `__file__` pointing at the real source location
  — verify this explicitly (`python -c "import main; print(main.__file__)"`)
  whenever a Dockerfile/packaging change touches how the local package
  gets installed.
- **`AppTest`-based tests that don't seed `startup_staleness_checked =
  True` will silently exercise a real, unmocked `check_and_refresh_stale_
  roles` call against the bare `MagicMock` session every other test in
  `test_main.py` already uses** — `MagicMock`'s default truthy comparison
  behavior (`__gt__`/`__rsub__` on `datetime` arithmetic) makes every
  `SEED_ROLES` entry look "stale," which would attempt real `ground_role`
  network calls once per test if `GEMINI_API_KEY`/`TAVILY_API_KEY` happen
  to be set in the ambient environment. Fixed by making `_make_at()`
  itself pre-seed `startup_staleness_checked = True` for every test by
  default; the handful of tests that actually exercise this wiring
  explicitly re-enable it. A `_make_at()`-style shared test helper is
  exactly the right place to bake in a new default like this — adding it
  per-test would have been 30+ near-identical edits and easy to miss on
  the next new test.
- **Deleting a large, multi-hundred-line code block via the `Edit` tool's
  exact-string matching is fragile against em-dashes/paraphrasing —
  precise `python -c` line-index slicing (with `assert`ed boundary lines
  first) is more reliable for a "delete lines N..M" refactor** than trying
  to reproduce a large block of prose verbatim in `old_string`. Watch the
  1-indexed-line-number vs. 0-indexed-list-index conversion carefully;
  both directions of off-by-one were hit at least once while extracting
  the Gemini-client module.
- **Patches on a shared fixture (`_patch_gemini`) must move to the new
  defining module the moment the code they patch moves** — patching
  `agents.research_outline_agent._get_gemini_client` after that function's
  real definition (and its caller, `_generate_content_with_retry`) moves
  to `utils.gemini_client` silently stops working (no error, the patch
  just has no effect, because the real call site now resolves the name
  in the new module's globals). This is CLAUDE.md's "wrong patch target"
  anti-pattern showing up as a consequence of a refactor, not just an
  authoring mistake.

## What remains open

**Everything from the prior session's list is now closed.** Newly
surfaced or still-open items, from this session's own work:

- `calculate_days_expected` doesn't factor in topic-group size or
  accumulated `users.pace_extension_days` — a fixed 120-minutes-per-topic
  constant regardless of subject matter or how much pacing extension has
  already accrued. Flagged in both specs as still-open, not solved.
- Deferred patch-notes' "on-demand if the user explicitly asks"
  resurfacing (PRD §7.9) still has no UI/trigger — only goal-completion
  resurfacing is built. Explicitly skipped this session (user's choice),
  a genuine product/UX decision (where in the pipeline a mid-journey
  "show me what's deferred" action would live) still needs making.
- No Alembic, no FK constraints on `outline_topics`/`patch_notes`/etc.'s
  UUID columns — explicitly dropped from this session's scope by the
  user (not deferred as "still owed," genuinely out of scope now).
- `README.md` — explicitly dropped from this session's scope by the user
  (not deferred as "still owed" either). `pyproject.toml` still declares
  `readme = "README.md"` referencing a file that doesn't exist; confirmed
  this doesn't break `pip install` (tested directly), so it's cosmetic,
  not blocking.
- The shared Gemini-call helpers are now properly extracted, but
  `_get_tavily_client` (Tavily, not Gemini, infrastructure) is still
  reached into by `agents/coaching_pace_agent.py` directly from
  `agents/research_outline_agent.py` — a smaller, separate seam, not
  addressed by this session's extraction (out of scope; only the Gemini
  infra was named as an open item).
- No production cost/usage tracking (`utils/logger.py`) — still an
  explicit, recorded scope cut from a much earlier session, unaffected by
  anything done here.
