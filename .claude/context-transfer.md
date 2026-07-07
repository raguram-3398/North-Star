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
`src/main.py`'s Streamlit app (Landing → Intake → Clarify Gate → Research &
Market Grounding → Outline Creation → Outline Confirmation → Day-by-Day
Coaching → Verification → Goal Completion), auto-advancing with no button
through Clarify Gate → Research & Market Grounding → Outline Creation.
Every LLM-backed reasoning call in both agents (`agents/
research_outline_agent.py`, `agents/coaching_pace_agent.py`) and the
Verification Question Generator (`src/skills/
verification_question_generator.py`, a plain `src/` module — no longer a
separately-packaged `.agent/skills/` Agent Skill artifact, see below) now
goes through real `google.adk.agents.LlmAgent` + `google.adk.runners.Runner`
calls via `src/utils/adk_runtime.py`, which replaced `utils/gemini_client.py`
(deleted). 466 tests pass, ruff/black/mypy clean, 94% coverage
(`pytest --cov=src`). `streamlit_app.py` is the real `streamlit run` target,
verified booting to HTTP 200. Nothing has ever been committed by the
assistant — the user reviews and commits personally. **`README.md` still
does not exist** — see "What remains open."

## What was accomplished this session

Two connected pieces of work, both completed:

1. **Executed the full ADK refactor** that a prior session had planned and
   gotten approval-in-substance for but not yet run (the plan lived at
   `/Users/ram/.claude/plans/eager-giggling-babbage.md`). Built
   `src/utils/adk_runtime.py` (real `LlmAgent`/`Runner`/`RetryConfig`
   infrastructure), converted all ~14 LLM-calling functions across both
   agents and the Verification Skill to module-level `LlmAgent` instances,
   added two documentation-only composite agents
   (`RESEARCH_OUTLINE_AGENT`, `COACHING_PACE_AGENT`), migrated the entire
   test suite onto a new `_patch_adk_runtime`/`_FakeRunLlmAgent` fixture
   (`tests/test_adk_runtime.py`, replacing the deleted
   `tests/test_gemini_client.py`), and reconciled
   `specs/Architecture_North_Star.md`/`specs/PRD_North_Star.md`/
   `CLAUDE.md` in the same pass. Full detail (design deviations, the real
   bug found, etc.) lives in Architecture §3's "Resolved (the ADK-refactor
   task...)" block, not repeated here.
2. **A follow-up polish pass, four requested changes:** (a) center-aligned
   the "North Star" title (the only place that literal string was
   rendered as UI, not just docstring/comment text); (b) removed the
   confidence-tier message from the two places it was user-facing — the
   Day-by-Day Coaching "Confidence Stamp" (`_render_stamp`, deleted
   outright once its one call site was removed — zero remaining callers)
   and the three "Confidence: ..." lines in Research & Market Grounding
   (kept the cached-fallback staleness caption, dropped the confidence
   value itself); (c) moved `generator.py` from `.agent/skills/
   verification_question_generator/` to `src/skills/
   verification_question_generator.py` (`git mv`, history preserved),
   deleted `.agent/` entirely (including `SKILL.md`), removed both
   `sys.path` bootstraps this required before (`agents/
   coaching_pace_agent.py` and `tests/test_verification_skill.py`); (d)
   re-checked the whole project against the hackathon rubric's actual
   text (read directly from `course/Hackathon Rules.docx`, not recalled
   from memory) given (a)-(c)'s changes.

## Key decisions made and why

- **JSON-returning `LlmAgent`s use `generate_content_config=
  GenerateContentConfig(response_mime_type="application/json")` +
  `call_agent_json`'s own parse/retry loop, NOT ADK's `output_schema`/
  `output_key`.** This was a deliberate deviation from the originally
  sketched plan, made to preserve the existing, already-tested
  `_parse_gemini_json_object` required-keys/error-fed-retry contract
  byte-for-byte rather than deriving a bespoke pydantic schema per task on
  a same-day deadline for no behavioral gain. Documented in
  `adk_runtime.py`'s own module docstring.
- **Every task `LlmAgent` explicitly sets `disallow_transfer_to_parent=True,
  disallow_transfer_to_peers=True`.** Not present in the original plan —
  added after live verification surfaced a real bug (see Traps below).
  Consistent with, and reinforcing, this refactor's own design principle
  that dispatch among task agents is deterministic Python, never ADK
  auto-routing — these agents were never supposed to be transferable, so
  this is a correctness fix, not a workaround.
- **The Verification Skill's relocation out of `.agent/skills/` was done
  as explicitly instructed, with the hackathon-rubric tradeoff surfaced
  rather than silently accepted.** CLAUDE.md's repo-structure section
  had flagged that location as "required for recognition by the
  Antigravity workspace manager, per course convention," and the
  hackathon rubric lists "Agent skills" as one of 6 evidenceable course
  concepts (need ≥3). Proceeded anyway (direct, unambiguous instruction,
  reversible via git) but flagged clearly in the same turn that this
  drops the strongest form of that specific rubric line's evidencing —
  not a blocker, since ADK/MCP Server/Security/Deployability already
  independently clear the ≥3 threshold.
- **Confidence-message removal scoped to exactly two spots, not the
  outline-confirmation presentation text.** `_format_outline_presentation`
  (`research_outline_agent.py`) also renders "(confidence: ...)" per topic,
  but that's LLM-adjacent generated markdown content, not a simple UI
  render line — left untouched since the user said "two places" and named
  a UI-level removal, not a prompt/content change.

## Traps / failed approaches — don't repeat

- **Mocking at the `adk_runtime.run_llm_agent` boundary (correct for unit
  tests) cannot catch bugs living inside real ADK/Gemini plumbing below
  that boundary.** Every phase's mocked tests passed (466 green,
  ruff/black/mypy clean) while a real, call-breaking bug sat undetected:
  any `LlmAgent` that is both JSON-mode (`response_mime_type=
  "application/json"`) AND a `sub_agents` member of a composite agent
  fails every real call with `400 INVALID_ARGUMENT: Function calling with
  a response mime type: 'application/json' is unsupported` — ADK
  auto-attaches a `transfer_to_agent` function tool to any agent with a
  `parent_agent` set, unless `disallow_transfer_to_parent`/
  `disallow_transfer_to_peers` are explicitly `True`, and Gemini rejects
  function-calling combined with JSON mime type. Only caught by
  deliberately driving `create_initial_outline`/`generate_day_content`/
  `generate_questions` against the real Gemini API after all phases were
  "complete" per the mocked suite — confirms this codebase's own
  established practice (a real `streamlit run` + live call before calling
  an ADK-integration task done) is load-bearing, not optional ceremony.
  Fixed by setting both `disallow_transfer_*` flags on every task agent;
  re-verified live afterward.
- **A structural file-location change (moving `.agent/skills/` code into
  `src/`) can silently break `Dockerfile`/CI even when every test and
  linter passes**, since neither exercises the Docker build. The
  `Dockerfile` had a `COPY .agent/ ./.agent/` line that would have failed
  the image build outright once `.agent/` was deleted — caught only
  because the hackathon-compliance re-check specifically walked
  deployability artifacts, not because any automated check flagged it.
  Fixed in the same turn. Lesson: any task that deletes/moves a
  top-level directory needs an explicit grep across `Dockerfile`/
  `.dockerignore`/`.github/workflows/*.yml`, not just `src`/`tests`.

## What remains open

- **`README.md` still does not exist — the single biggest concrete risk
  to today's hackathon score**, flagged again this session: Documentation
  is 20 of 70 implementation points, and the rules require a README for
  GitHub-based submissions (setup instructions, architecture,
  problem/solution). This was an explicit, deliberate scope cut by the
  user in an earlier session (not an oversight), so it was flagged, not
  written unprompted. `pyproject.toml` still declares `readme =
  "README.md"` referencing a file that doesn't exist; confirmed this
  doesn't break `pip install`. Test coverage (94%, `pytest --cov=src`) is
  ready to drop in whenever the README is written.
- **Agent Skills course-concept evidencing is now weaker, by explicit
  choice, not a bug to fix**: no `.agent/skills/SKILL.md`, no
  Antigravity-recognized Skill-artifact packaging. The underlying
  capability (source-anchored question generation/grading) is unchanged
  and still genuinely reusable — just no longer packaged in the course's
  specific discoverable form. Only revisit if the user decides they want
  that specific rubric line's strongest evidencing back.
- **Antigravity [Video] course concept can't be verified from code at
  all** — depends entirely on the user's own video demonstrating the
  browser-subagent UI-verification workflow CLAUDE.md's Build Environment
  section describes. Not something a future session can check or fix in
  the repo.
- Carried forward, unaffected by this session's work:
  - `calculate_days_expected` still doesn't factor in topic-group size or
    accumulated `users.pace_extension_days` — flagged in both specs as
    still-open, not solved.
  - Deferred patch-notes' "on-demand if the user explicitly asks"
    resurfacing (PRD §7.9) still has no UI/trigger — only goal-completion
    resurfacing is built.
  - No Alembic, no FK constraints on UUID columns — explicitly dropped
    from scope, not deferred.
  - No production cost/usage tracking (`utils/logger.py`) — an explicit,
    recorded scope cut from an earlier session.
  - No test exists yet for the `CachedFallbackResult` path through
    Research & Market Grounding's automatic advance (only
    `LiveGroundingResult` and the `GeneralKnowledgeFloorResult` dead end
    are covered) — not a known bug, just an untested branch.
