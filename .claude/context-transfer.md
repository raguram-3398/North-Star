# Context Transfer — Project North Star

Written at the end of a session that scaffolded `src/` and implemented six
deterministic modules end-to-end. Everything below is session history and
judgment-call rationale — nothing here duplicates what's already stated as
policy in `CLAUDE.md`; read that first for the standing rules.

## What was accomplished this session

1. Read all four specs in full and summarized the two-agent architecture,
   confidence ladder, and verification retry mechanism back to the user for
   confirmation before writing any code.
2. Built `specs/architecture.png` (Graphviz `dot`), iterated twice on layout
   to reduce edge crossings, added an explicit HITL-checkpoint node for
   outline confirmation.
3. Scaffolded the full `src/`, `evaluation/`, `tests/` tree per CLAUDE.md's
   repo structure — every file as a docstring + typed stub raising
   `NotImplementedError`. Added `__init__.py` to every `src/` package dir and
   `tests/` in a follow-up pass (originally omitted; fixed after the editable
   install (`pip install -e '.[dev]'`) showed cross-module imports failing).
4. Implemented, tested, and got sign-off on, in order:
   `security/output_guard.py`, `security/input_gate.py`,
   `outline/significant_event.py`, `pace/calculator.py`,
   `outline/hierarchy.py`, `patches/patch_manager.py`,
   `data/roles_cache.py` + `db/connection.py` (+ fleshed out `RolesCache` in
   `models/schemas.py`).
5. Added a new CLAUDE.md standing rule mid-session: reconcile
   `PRD_North_Star.md`/`Architecture_North_Star.md` against every commit's
   actual implementation decisions, not just at the end.
6. Current state: **100 passed**, 2 pre-existing failures
   (`test_research_grounding.py`, `test_verification_skill.py`) — both are
   placeholders for modules not yet started (see Open Items). Ruff, black,
   and mypy all clean across `src/` and `tests/`.
7. All commits made by the user personally, never by the assistant — matches
   the working pattern established this session (see "Working pattern" below).

## Key decisions and why

**Confidence ladder & output guard**
- `ConfidenceTier` is a `StrEnum` (not `str, Enum` — ruff on py312 flags the
  latter and suggests `StrEnum` directly).
- `ValidatedGroundedContent` (frozen dataclass: `source_url`, `source_type`,
  `confidence`, `extra: dict`) is *the* "post-output_guard object" guardrail
  #12 requires. `extra` carries whatever the candidate dict had beyond the
  three validated fields (e.g. a skill's `skill` name).
- `validate_output_object` checks `source_url` is a structurally valid
  absolute URL via `urlparse` (catches `"https://"` with no host, not just
  blank/missing) and explicitly rejects the `reject` tier itself — reject
  means "no record ever created," so it can't produce a writable object.

**Clarify gate**
- `ClarifyGateStage` state machine: `NARROWING → PROPOSE_BEST_GUESS →
  EXPLAIN_ROLE → ACCEPT_OWN_WORDS → RESOLVED/EXITED`.
- `detect_reject` is scoped **narrowly** to blank/whitespace input only.
  Semantic "is this nonsense" classification stays Agent 1's reasoning job —
  input_gate.py only tracks structural loop state, never content judgment.

**Significant event detection**
- Bucket-rank comparison happens **before** any confidence comparison, and
  short-circuits on any bucket-rank change. This matters for one specific
  edge case: a bucket *decrease* paired with a confidence *increase*
  (e.g. `core_skills`/`low` → `emerging_skills`/`high`) must still resolve
  to "not significant" — a naive confidence-only check would get this
  wrong. There's a dedicated test for exactly this combination.

**Pace calculator**
- Judgment-call constants (all flagged, all still open to revision):
  `TIMING_OUTLIER_THRESHOLD=0.5`, `TIMING_SATURATION_DEVIATION=1.0`,
  `MAX_TIMING_INFLUENCE=0.2` (this one is PRD-specified, not a guess),
  `DRIFT_WINDOW_SIZE=3`, `SUSTAINED_BEHIND_THRESHOLD=0.7`,
  `SUSTAINED_AHEAD_THRESHOLD=0.95`.
- `detect_sustained_drift` is **mean-based** over the trailing window, not
  "every entry must cross the threshold." This was a direct revision after
  initial feedback: an all-entries check means one good day inside an
  otherwise-bad window fully resets the streak, which the user felt was too
  strict for "sustained." Mean-based lets a single outlier day still count
  if the average still crosses.

**Outline hierarchy**
- `prerequisite_topic_ids` is a plain function parameter (`frozenset[str]`),
  not a schema field — `outline_topics` has no dependency-graph column, only
  a flat `hierarchy_position` int. Only "must-follow" (prerequisite)
  constraints are modeled; "must-precede" is an explicitly documented known
  limitation (code comment in `insert_new_topic`, Architecture §10, PRD
  Future Improvements #4) — flag this if enrichment positioning (PRD §7.9)
  is ever found to need inserting something *before* an existing topic.

**Patch manager**
- Only `ConfidenceTier.HIGH` auto-prioritizes; `medium` and below all route
  to `"needs_user_decision"`. PRD only named `high`/`low` explicitly —
  `medium` was a genuine gap, resolved conservatively (medium isn't fully
  cross-validated per PRD §7.3, so it reads as "uncertain").
- `order_pending_items` sorts a **mixed** list of patches and regular topics
  purely by `hierarchy_position`. Needed a test with 2+ items of *each* kind
  specifically — a single-pair test can't distinguish true interleaving from
  a bug that sorts each kind separately then concatenates by kind (both
  would pass a 1-and-1 test if the pair happens to already be in the right
  relative order).

**roles_cache / db connection**
- `roles_cache.py` functions take `session: Session` as an explicit
  parameter (dependency injection) rather than calling
  `db.connection.get_session()` internally. This is *the* reason the test
  suite works with a plain `MagicMock()` and touches no real DB.
- Chose a **mocked session over a test database**: CLAUDE.md forbids
  substituting SQLite for Neon, and `upsert_role` uses Postgres-specific
  `INSERT ... ON CONFLICT`, which SQLite can't even execute.
- `db/connection.py`'s Engine is a **lazy-but-memoized singleton** — created
  on first call, not at raw import time. Avoids making
  `NEON_CONNECTION_STRING` a hard import-time requirement in CI/test
  contexts with no DB configured, while still keeping exactly one Engine
  per process.
- Driver fix: `.env.example`'s bare `postgresql://` scheme resolves to
  legacy `psycopg2` by default, but `pyproject.toml` installs `psycopg` v3.
  `db/connection.py` rewrites the prefix to `postgresql+psycopg://` in
  code rather than requiring `.env` to change.
- Timeouts: `connect_timeout=10s`, `statement_timeout=10s` — both flagged
  judgment calls, no duration specified anywhere in the specs.
- `upsert_role`'s `core_skills`/`emerging_skills` went through two review
  rounds: first accepted raw `list[dict]`, then per guardrail #12 (grounding
  results need a post-output_guard object) changed to require
  `list[ValidatedGroundedContent]`, with an internal `_to_skill_entry`
  serializer producing the `{skill, source_url, confidence}` JSONB shape
  (skill name pulled from `.extra["skill"]`). Final polish: that function's
  type/field checks raise `ConfidenceValidationError` via an explicit
  `isinstance()` check, not an incidental `AttributeError` from a dict
  lacking `.extra` — every guard in this codebase should fail intentionally
  and by name, not by accident of duck typing.
- `models/schemas.py`: only `RolesCache` is fully mapped (real `Base` +
  `Mapped`/`mapped_column`). The other 6 model classes are still
  docstring-only stubs — intentional, to be fleshed out as their own I/O
  modules get built, not an oversight.

## Traps / failed approaches — don't repeat

- **`uv pip install graphviz` does not get you the `dot` binary.** The PyPI
  `graphviz` package is only a Python wrapper around a system Graphviz
  install. Needs brew/conda/manual download; got explicit user approval for
  a scoped `brew install graphviz` after clarifying this.
- **Graphviz `rank=same` + cluster membership silently breaks clusters.**
  First diagram attempt mixed `{rank=same; ...}` blocks with nodes that were
  also inside `subgraph cluster_x` blocks — dot printed
  `"X was already in a rankset, deleted from cluster"` and quietly emptied
  the clusters. Fix: don't force rank hints across cluster-member nodes; let
  the natural layered layout handle it, or split into sibling clusters.
- **Reduce Graphviz edge crossings by splitting shared clusters per-consumer**,
  not by fighting the layout engine — e.g. splitting one big "Deterministic
  Modules" cluster into per-agent sub-clusters (positioned next to the agent
  that owns them) fixed most of the crossing problem in one pass.
- **Naive vs. aware `datetime` mismatches.** `TIMESTAMP` (not `TIMESTAMPTZ`)
  columns need naive UTC consistently. Established convention throughout:
  `datetime.now(UTC).replace(tzinfo=None)`, documented in every docstring
  that touches a timestamp.
- **`postgresql.insert(...).on_conflict_do_update(set_={...})` auto-suffixes
  the update-side bound params** (`param_1`, `param_2`, ...) — they do *not*
  reuse the insert-side column-named params. Don't write tests asserting on
  those anonymous names; assert on `compiled.params` (column-named, insert
  side) and just check `"ON CONFLICT"` is present in the compiled SQL string
  for the update side.
- **`create_engine()` is lazy about network I/O but not about driver
  import** — it imports the DBAPI module immediately. A bare `postgresql://`
  URL with only `psycopg` (v3) installed (not `psycopg2`) fails at
  `create_engine()` call time, not at actual connection time. Caught this
  before it became a runtime surprise.
- **Frozen-dataclass mutation tests and deliberately-wrong-type tests both
  need explicit `# type: ignore` comments** (`[misc]` for the frozen-field
  assignment, `[arg-type]` for the wrong-type argument) — mypy correctly
  flags what the test intentionally verifies raises at runtime.
- **A test with only one item of each kind/category can pass under a subtly
  wrong implementation.** The interleaving-order bug (sort-within-groups vs.
  true interleave) and the earlier all-entries-vs-mean sustained-drift logic
  both needed 2+ items per category to actually distinguish correct from
  incorrect. General lesson: before trusting a test, ask "could a plausible
  bug still pass this?"

## Working pattern established this session (for continuity, not policy)

- Every module: implement → write full test coverage → compile/ruff/black/
  mypy/pytest all clean → flag every constant/ambiguity explicitly in the
  chat response → **do not commit** → wait for explicit review.
- The user reviews, sometimes requests a specific fix or a specific missing
  test case, then commits personally — the assistant has not run `git
  commit` once this session.
- When a resolved ambiguity/constant/limitation surfaces, it gets reconciled
  into `PRD_North_Star.md` and/or `Architecture_North_Star.md` in the same
  pass, before reporting back — not left for later cleanup.

## Open items

- `src/agents/research_outline_agent.py` — still a stub. Real ADK/Gemini
  agent reasoning (clarify-gate content, cross-validation judgment, initial
  outline creation) hasn't been started.
- `src/agents/coaching_pace_agent.py` — still a stub (day content
  generation, closing note).
- `src/cron/refresh_roles.py` — still a stub.
- `src/data/progress_log.py` — still a stub (parallel to `roles_cache.py`
  but untouched).
- `src/utils/logger.py` — still a stub (tool-call audit logging, cost/usage
  tracking with `request_id`).
- `security/output_guard.py`'s `assign_confidence_tier` — deliberately left
  unimplemented; only `validate_output_object` and the enum were in scope
  when that module was built.
- `src/main.py` — still a stub.
- `.agent/skills/verification_question_generator/generator.py` — **does not
  exist at all**, only `SKILL.md` (predates this session). The actual Skill
  implementation hasn't been started.
- `evaluation/golden_dataset.json` and `evaluation/eval_cases.json` — still
  empty `[]` placeholders from initial scaffolding.
- No Streamlit UI, no `.github/workflows/*.yml` — untouched.
- Asymmetry not yet resolved: `get_role` returns plain dicts and can't fully
  reconstruct a `ValidatedGroundedContent` on read-back anyway, since
  `source_type` isn't part of the persisted `{skill, source_url,
  confidence}` JSONB shape. Flagged, not fixed — revisit if a caller ever
  needs the read path to produce validated objects too.
- `roles_cache.py`/`db/connection.py` have never been exercised against a
  real Neon instance — all tests are mocked/monkeypatched. First real
  integration check is still pending.
- No `pytest --cov=src tests/` run yet; README.md not started — both are
  explicit ship-day requirements per CLAUDE.md, deliberately deferred until
  there's real code to measure.
