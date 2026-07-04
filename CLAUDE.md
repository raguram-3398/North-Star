# CLAUDE.md — Project North Star

This file is the standing constitution for all work in this repo. Read `specs/PRD_North_Star.md` and `specs/Architecture_North_Star.md` for full behavioral and technical spec — this file governs *how* to work, not *what* to build.

## Build environment

- **IDE:** Google Antigravity — the working environment for this build
- **Primary code generation:** Claude Code, invoked from within Antigravity, following this file as its constitution
- **Antigravity's own browser subagent** is used separately and specifically for UI verification (driving a sandboxed Chrome instance against the running Streamlit app) — this is the concrete, demonstrable use of Antigravity for the course's "Antigravity" concept in the video. It is not the code-generation driver; Claude Code is.
- Verify early (Day 1) that Claude Code runs correctly as a terminal-based agent inside Antigravity's shell before relying on this setup for the full build.

## Source of truth

- `specs/PRD_North_Star.md` — behavior, rules, user-facing logic. If code and this PRD disagree, the PRD wins unless we explicitly agreed to change it.
- `specs/Architecture_North_Star.md` — stack, schema, component boundaries, guardrails.
- `specs/scenarios/high_risk_flows.feature` — Gherkin (Given/When/Then) scenarios for the highest-ambiguity-risk flows (clarify gate, confidence ladder, outline content permanence). When implementing these specific flows, match the scenario exactly.
- If a requirement is ambiguous or missing from all of the above, **stop and ask** rather than inferring a design decision. Do not silently invent new behavior, thresholds, or schema fields.

## Stack (do not substitute without asking)

- Python + Google ADK
- Streamlit (UI) — not Gradio
- Neon Postgres (persistence) — not local SQLite, not HF Spaces local disk
- Himalayas MCP server (structured market data) — consumed only, never build a custom MCP server
- Tavily (search) — not a different search provider
- Secrets via environment variables / HF Space secrets — never hardcoded, never committed

## Repo structure

```
north-star/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .gitignore
├── .claudeignore
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
│   │   └── high_risk_flows.feature
│   └── architecture.png
│
├── src/
│   ├── main.py
│   ├── agents/
│   │   ├── research_outline_agent.py   # reasoning/generation only
│   │   └── coaching_pace_agent.py      # reasoning/generation only
│   ├── security/
│   │   ├── input_gate.py               # clarify-gate bound/loop state, reject detection
│   │   └── output_guard.py             # confidence-ladder enforcement — the structural gate before any DB write
│   ├── pace/
│   │   └── calculator.py               # topic_score, timing_ratio, 80/20 blend, sustained-drift check
│   ├── outline/
│   │   ├── hierarchy.py                # insertion into existing hierarchy
│   │   └── significant_event.py        # bucket/confidence-crossing diff
│   ├── patches/
│   │   └── patch_manager.py            # confidence branching, delivery ordering
│   ├── cron/
│   │   └── refresh_roles.py            # shared refresh function — GitHub Action + startup check both call this
│   ├── data/
│   │   ├── roles_cache.py              # roles_cache I/O
│   │   └── progress_log.py             # progress_log I/O
│   ├── models/
│   │   └── schemas.py
│   ├── db/
│   │   └── connection.py
│   └── utils/
│       ├── logger.py                   # includes tool-call audit logging
│       └── exceptions.py
│
├── evaluation/
│   ├── golden_dataset.json
│   └── eval_cases.json
│
└── tests/
```

**Skills live in `.agent/skills/`, not `src/`** — required for recognition by the Antigravity workspace manager, per course convention.

**Specs live in `specs/`, not `docs/`** — the version-controlled source of truth per Spec-Driven Development convention, indexed by the agent to build and verify code.

**Agents hold reasoning and generation only.** Deterministic logic (confidence-ladder validation, pace math, drift thresholds, significant-event diffing, patch confidence-branching) lives in plain, independently-tested modules that agents call — never inline agent reasoning, never an instruction the agent is merely told to follow. See `specs/Architecture_North_Star.md` §2 for the full rationale.

**`specs/architecture.png` must exist and be reviewed before any `src/` code is written.** The diagram is the build contract — every component in Architecture §3/§12 must be visible in it. If it doesn't exist yet, draw it first.

## Non-negotiable guardrails

Never do these, even if a task seems to call for it — ask first:

1. Never create or store an outline item, patch-note, or roadmap content without a `source_url` and `confidence` value populated.
2. Never delete or reduce outline content for any reason, including "user is behind" — pacing extends, content is never removed.
3. Never repeat an identical verification question on retry — always generate a fresh one from the same source.
4. Never let enrichment topic results write into `pace_snapshots` or influence velocity in any way.
5. Never let a patch-note reopen or alter the completion/verification status of its origin topic.
6. Never skip the confidence ladder — if live sources fail, fall through to cached data, then to the labeled general-knowledge-only floor. Never silently fabricate a source.
7. Never hardcode or log Neon/Tavily/Gemini credentials, in code or in commit history.
8. Never make an unbounded user-facing loop — the clarify gate and outline confirmation are both bounded (~2 rounds) with a defined graceful exit. Any new interactive loop needs the same treatment.
9. Never touch the `roles_cache` bootstrap/refresh function's core logic (used by both the seed run and the cron job) without checking both call sites still work.
10. Never write agent reasoning that duplicates logic already living in `security/`, `pace/`, `outline/`, or `patches/` — call those modules, don't reimplement their checks inline in a prompt.
11. Never pass one agent's full raw output into the other agent's prompt/context — pass a database reference (`user_id`, `topic_id`) and let the receiving agent read what it needs.
12. Never let a database write function accept an unvalidated object — writes for outline items, patch-notes, and grounding results require a post-`output_guard` object, not a raw dict.
13. Never let input validation run after any content-processing step that could corrupt what it's checking for — `security/input_gate.py` always runs on raw input first.
14. Never let an external API call (Gemini, Tavily, Himalayas, Neon) run without an explicit timeout.

## Coding conventions

- Type hints on **every** function argument and return value, including `-> None` on constructors (`__init__`) — a function without type hints is not finished
- Docstrings on every public function/class stating what it does and why (not just what)
- SQLAlchemy models mirror `Architecture_North_Star.md`'s schema exactly — if a field needs to change, update the architecture doc in the same commit
- **Timeout on every external call, no exceptions** — Himalayas, Tavily, Gemini, and Neon all get explicit timeouts (`asyncio.wait_for` for coroutines, `asyncio.timeout()` as a context manager for streaming). Every timeout is tested, not assumed.
- **Raise exceptions, never return error strings or `None`-as-error.** Mixed return types destroy the caller's ability to reason about what it received. Use specific, typed exceptions (see `utils/exceptions.py`) — e.g. `GroundingError`, `VerificationTimeoutError`, `ConfidenceValidationError` — never a bare `Exception` and never a string that looks like data.
- **Pure functions stay pure.** `pace/calculator.py`, `outline/significant_event.py`, `security/output_guard.py`, `security/input_gate.py`'s bound-check logic — all side-effect-free, independently testable, no `print()`, no DB calls, no external API calls inside them. If a function needs to do I/O, it is not one of these modules.
- No bare `except:` — catch specific exceptions, especially around external calls (Himalayas, Tavily, Gemini) where failure is expected and must degrade gracefully per the confidence ladder, not crash
- **One client per module, instantiated at module level — never inside a function or per-request.** Applies to the Gemini client, Tavily client, and any Himalayas MCP client. Re-instantiating a client per call is an anti-pattern to avoid from the start, not fix later.
- ruff + black clean before every commit — zero errors, zero formatting changes. Fix before writing the commit message, not after.

## LLM Call Discipline

- **Every prompt used for grounded or safety-critical generation is versioned in a module-level `PROMPT_REGISTRY`** — the Verification Question Generator's prompt, the outline-hierarchy sequencing prompt, and the cross-validation normalization prompt all qualify. Old versions are never deleted. A version is frozen the moment its baseline regression test locks it in.
- **The baseline regression test for a registered prompt asserts on the prompt string itself, not on the LLM's output.** Output is probabilistic; the prompt string is deterministic. The test exists to catch an unintentional change to a frozen version.
- **Error-fed retry, not generic retry.** When a schema-validation or grounding failure triggers a retry (e.g., `output_guard` rejects a missing `source_url`), the retry prompt includes the specific validation error (`str(ValidationError)` or equivalent), not a generic "try again." Specificity is the mechanism that makes the retry different from the first attempt.
- **Capture the original input before a retry loop overwrites it.** E.g., `original_question_source = question_source` before any retry logic runs — a retry must never contaminate itself with a previous attempt's correction as if it were the original input.
- **Pipeline order is non-negotiable where order affects correctness.** Input validation/reject-detection (`security/input_gate.py`) runs on raw user input *before* it reaches any clarify-gate LLM call — never after, since downstream processing could corrupt a pattern the gate needs to catch.

## Cost & Usage Tracking

- Every Gemini, Tavily, and Himalayas call is logged with cost/usage and a `request_id`, via `utils/logger.py` — traceable, not aggregate-only.
- Use actual token/usage counts from the API response, never estimates.
- Cost is recorded only on success, explicitly guarded (e.g. `if response is not None`) — a failed call produces no cost record, never a silently wrong `$0.00`.
- Daily spend accumulates in a module-level tracker with a one-time threshold alert (a boolean flag prevents repeated firing). Document the reset-on-restart limitation in code comments and the README rather than solving it — out of scope for this timeline.

## Anti-Patterns to Avoid From the Start

Caught and corrected in prior projects — avoid on first pass here, don't wait to self-correct later:

- **Client-per-request** — instantiating an API client inside a function instead of at module level. One client per module.
- **Wrong patch target in tests** — patch where a name is *used*, not where it is *defined* (e.g. `agents.research_outline_agent.tavily_client`, not `src.tavily_client`).
- **A retry loop overwriting the variable it needs later** — capture the original input before the loop runs (see LLM Call Discipline above).
- **Off-by-one on a hard-capped count.** The verification retry cap is exactly 3 (PRD §7.7) — the first attempt must live inside the same loop as retries 2 and 3, not run once outside the loop before it starts. Verify actual call count against the spec's number, don't assume the loop is right.
- **Truthiness check on a result/dataclass object.** `if not guard:` tests whether the object exists (almost always `True`), not whether validation passed. Always check the specific field: `if not guard.is_valid:` / `if not guard.is_safe:`.
- **A method reference used where a method call was intended** — a missing `()` silently passes the wrong thing through, especially dangerous around HTTP status checks and validation calls.

## Testing expectations

- Every function in `.agent/skills/` and `src/cron/` needs at least one test before being considered done
- Confidence ladder branches (high/medium/low/cached/floor/reject) each need at least one test case exercising that branch
- Every external-call timeout has an explicit test forcing the timeout path, not just the happy path
- Never mark a task complete with a failing or skipped test — fix or flag, don't hide

## Git workflow

- **Before every commit, reconcile `specs/PRD_North_Star.md` and `specs/Architecture_North_Star.md` against what was actually implemented in this commit's changes.** If an implementation decision (a chosen constant, a resolved ambiguity, a discovered limitation, a design choice not previously specified) isn't yet reflected in the specs, update the specs in the same commit — specs must never silently drift behind the code they're supposed to govern.
- Commit after each working, reviewed unit — not after each generated diff
- Never commit `.env` or any file containing a real secret
- **Semantic commits only** — every commit message starts with `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, or `chore:`, followed by a specific, concrete description. No vague messages (`update stuff`, `wip`, `changes`).
- ruff + black must pass clean before the commit message is written — not after committing

## What Goes in the README on Ship Day — Not Before

Real, measured numbers — never estimates or "approximately":
- Gemini/Tavily/Himalayas latency, pulled from actual logs
- Cost per call, pulled from actual cost-tracker log entries
- Test coverage, from `pytest --cov=src tests/`
- "What I did not build and why" — at least three explicit scope cuts (see PRD §6 Non-Goals, Architecture §10 Explicit Out-of-Scope), each naming what a production version would do instead
- No vague claims — if a number can't be measured yet, the section stays unwritten until it can

## What to do when uncertain

If a task touches one of the guardrails above, or requires a design decision not covered in the PRD/architecture doc, stop and surface the question rather than guessing. A wrong guess costs more time to unwind than a short pause to ask.