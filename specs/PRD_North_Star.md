# Project North Star — Product Requirements Document

**Hackathon:** Kaggle "AI Agents: Intensive Vibe Coding" Capstone
**Track:** Agents for Good
**Deadline:** July 6, 2026, 11:59 PM PT (build stops afternoon of July 6; evening reserved for video + writeup)

---

## 1. Problem Statement

Breaking into a tech career requires knowing *what* to learn, *in what order*, from *which sources*, and *whether you've actually understood it* — knowledge that is normally supplied by a bootcamp, a mentor, or a career coach. People without access to any of these — no industry vocabulary, no one to sanity-check a study plan, no one to verify they're actually learning correctly — are left to piece together a path from scattered, often outdated, unverified content.

Generic AI chat answers this poorly: a single prompt produces a plausible-sounding study list with no live market grounding, no memory of the learner's actual progress, and no way to verify understanding. It is static the moment it's generated.

## 2. Target User

Someone pursuing a tech role who lacks a bootcamp, mentor, or career coach — regardless of starting background (a high schooler, a career switcher from an unrelated field, a self-taught hobbyist). They may not know correct industry terminology for what they want. They have no one else to ask "is this plan actually right" or "did I really understand that." Resourced users may also use the system, but it is designed for and prioritizes the underserved case.

## 3. Value Proposition

North Star provides a study path that is:
- **Grounded** — every piece of content traces to a real, cited source (official docs, established free courses, current job market data)
- **Currently relevant** — built from live labor-market signal, not stale training data, and kept current as the market shifts
- **Verified, not honor-system** — progress is only "complete" when understanding is demonstrated, not self-reported
- **Adaptive** — pace and content respond to how the individual is actually doing, not a fixed calendar
- **Honest about its own confidence** — every claim is labeled by how well-supported it is; the system says plainly when it doesn't know

## 4. Why Agents (not a single prompt)

The core value requires acting on live external state and remembering the user over time — structurally impossible for a single chat completion:
- Live, cross-validated market data with real timestamps and sources
- A plan that changes because of a specific, measured change in the user (pace) or the world (market shift) — not a static document
- A completion signal that is actually checked against source material, not self-declared

## 5. Success Criteria (for this submission)

- A working end-to-end flow: intake → grounded outline → day-by-day coaching → verification → adaptive replanning → goal completion
- At least 3 course concepts demonstrably applied: Agent/multi-agent (ADK), MCP Server (Himalayas), Security features, Deployability — ideally Agent Skills and Antigravity as well
- A public GitHub repo with a clear README, and a public HF Spaces link requiring no login
- A 5-minute video and Kaggle Writeup articulating problem, agentic rationale, architecture, and demo
- The system never silently fails: every failure mode (bad input, no market data, external tool failure) degrades gracefully to a labeled, honest state rather than crashing or fabricating
- Ship-day documentation (README) reports only real, measured numbers (latency, cost, test coverage) — never estimates or "approximately," consistent with the system's own no-fabrication principle applied to its own engineering claims

## 6. Non-Goals (explicitly out of scope for this submission)

- No seniority/grading/leveling claims (junior vs. senior) — rejected as unsupported by what verification actually measures
- No mid-journey goal/role changes without losing progress (future improvement)
- No visible base-knowledge indicator to the user (internal signal only, future improvement)
- No distinct short-gap/pause handling separate from behind-pace-drift and 30-day dormancy (future improvement)
- No live deployment requirement per hackathon rules — deployed anyway to HF Spaces for demo strength, but not required for judging
- No classroom/cohort features — this is a single-individual skill-development tool, not a social or comparative platform

---

## 7. System Overview

### 7.0 Orchestration & State-Passing Principles

Per course guidance ("Write Software, Not Rules" / "Shift Intelligence Left" / DAG orchestration over prompt-chaining): subjective-sounding but actually-deterministic logic is implemented as plain, testable code the agents call — never as an instruction the agent is merely told to follow. This applies specifically to: confidence-ladder enforcement, significant-event detection, pace calculation, sustained-drift thresholds, and patch confidence-branching. None of these require LLM judgment; all are deterministic given already-computed inputs, and are built as such.

**State passes between agents by database reference, not by accumulating raw output in a shared prompt context.** Agent 1 writes its output (resolved role, grounded outline, confidence tiers) to Postgres; Agent 2 reads by reference (`user_id`, `topic_id`) and fetches only what it needs. Neither agent receives the other's full raw output dumped into its own context window — this avoids context bloat and keeps each agent's reasoning scoped to its own job.

**Gates are structural, not advisory.** Per the course's "Reviewer & Gate" node pattern: a candidate output (an outline item, a patch-note, a market-grounding result) cannot reach persistence without first passing through its corresponding gate (confidence/source validation, clarify-gate bound-check). This is enforced by the write path itself requiring a validated object, not by instructing the agent to "remember to check" — making the invalid action structurally impossible rather than merely discouraged.

### 7.1 Pipeline stages

1. **Intake** — background, current role, years of experience, prior self-study specifics, stated goal, available time
2. **Clarify Gate** — bounded-loop resolution of the stated goal into a concrete, real role
3. **Research & Market Grounding** — parallel live market data (Himalayas MCP + web search), cross-validated, confidence-scored
4. **Outline Creation** — grounded dependency hierarchy (basics → full role requirements)
5. **Outline Confirmation** — user reviews and can request changes, bounded loop, before Day 1 begins
6. **Day-by-Day Coaching** — just-in-time generation of each day's content, structured in 7 steps
7. **Verification** — 5 source-anchored questions per topic, strict grading, retry-with-de-escalation
8. **Pace Tracking** — understanding-weighted (not throughput-weighted) velocity, sustained-drift triggered adaptation
9. **Patch-Notes** — market-driven updates to already-completed topics, confidence-branched delivery
10. **Enrichment** — bonus content for users pulling ahead, isolated from pace consequences
11. **Goal Completion** — closing career-guidance note, reusing market-data infrastructure

### 7.2 Intake & Clarify Gate

**Inputs collected:** background, current job, years of experience, prior self-study (specific, not yes/no), goal (free text), available time.

**Gate behavior:**
- Clearly real role → accept, proceed to Research
- Clearly nonsense → reject, ask to clarify
- Vague-but-genuine ("I want to make apps") → one narrowing question at a time, bounded to ~2 rounds
- Still unclear after bound → system proposes a best-guess role interpretation
  - Accepted → proceed
  - Rejected → system clearly explains what that role actually involves, asks again
    - Accepted → proceed
    - Rejected again → accept the user's own words verbatim; run the grounding check anyway before committing
      - Any market signal found (even weak) → proceed at low confidence, framed as "starting here, refining as we go"
      - **Zero market signal found → exit.** State plainly that no current hiring activity exists for this; no outline is built.

Output: a **pacing profile** (background-derived initial pace expectation) and a **resolved role** — not an outline yet.

### 7.3 Research & Market Grounding

**Sources, in parallel:**
- Himalayas MCP server (structured, free, no-auth — job listings, salary data, market stats)
- Web search extraction (JD snippets from search results, not direct board scraping)

**Cross-validation:**
- Both sources agree → **high confidence**
- Single source only, or minor disagreement → **medium confidence**
- Genuine conflict → flagged
- Agreement/normalization judged against `roles.json` as a grounding anchor (not open-ended LLM judgment)

**Niche/no-anchor roles** (no `roles.json` entry): single-source only, lightweight LLM sanity-check pass using only already-present in-snippet company context (no extra fetch) → **confidence: low**. Never written back into `roles.json`.

**Full confidence ladder:** high → medium → low → cached-fallback (low, labeled with `last_updated`) → general-knowledge-only floor (explicitly labeled as such) → reject (zero signal, see Clarify Gate exit).

**Grounding rule (absolute):** every downstream content item carries `source_url`, `source_type`, `confidence`. Nothing enters the system ungrounded.

**`roles.json`:** structured with `core_skills` / `emerging_skills` per role, each carrying a confidence tier. Refreshed on a deterministic cron (minimum every 30 days, immediately on a significant event). Serves as fallback data and normalization anchor — never a shortcut that skips live research for a new user; every user gets a fresh live research pass.

**Bootstrap (initial creation):** there is no separate seeding process — the initial file is produced by manually running the same Research/Grounding pipeline once per role in a small seed list (5–8 common roles relevant to the target user, e.g. Backend Engineer, Frontend Engineer, Data Analyst, AI/ML Engineer, DevOps Engineer), writing each result into `roles.json` in its normal structure. The manual seed run and the recurring cron job are the same code path — the seed run is simply the first invocation, triggered by hand instead of by a timer. No hand-curated data is written.

**Significant event (deterministic rule):** a skill crosses a bucket or confidence boundary *upward* — absent → `emerging_skills`, `emerging_skills` → `core_skills`, or confidence tier strengthens. Downgrades produce no action (content is never removed).

### 7.4 Outline Creation

Output is a **dependency hierarchy** (prerequisite order: basics → full role requirements), not a flat list — and the hierarchy itself is grounded, derived from the same sourced skill data as Research, not a separate ungrounded reasoning step.

**Update policy:** additive/refreshing only — **content is never removed**.

**Triggers:** fixed floor of 30 days, OR immediately on a significant event, OR dormancy >30 days (subsumed into the same significant-event mechanism — no separate handling needed).

**Update types:**
- *New addition* — a genuinely new topic, inserted at its correct hierarchical position
- *Augmentation* — an existing topic's content refreshed in place

**Resolution by user position:**
- Not yet reached / current → picked up naturally when reached
- Already completed → becomes a **patch-note** (see 7.9), never reopens the original topic's status

### 7.5 Outline Confirmation

Before Day 1: outline shown to the user with grounded "why" reasoning per topic. User can raise concerns, request additions, ask questions. Bounded loop (matching the clarify gate's pattern) — if not resolved within the bound, proceeds with current outline, framed as "starting here, refine as we go." **One-time, pre-start window only** — no user-initiated outline editing once Day 1 begins (see Non-Goals / future improvements).

### 7.6 Day-by-Day Coaching

Each day's content is generated **fresh, that day** — nothing pre-chunked in advance. Inputs: current outline position, user's pace/time allotment, any pending high-confidence patches (ordered by hierarchy).

**Hands-on-eligible day structure (7 steps):**
1. Summary of today's topic
2. Theoretical study material — grounded links (YouTube, official docs/learning platforms, established third-party content)
3. Practical application (hands-on)
4. Review / refactor of the hands-on work
5. Reflection
6. Verification (5 questions)
7. Preview of tomorrow and how it connects

**Conceptual-only days** (early in any new topic-group — e.g., day 1 of a brand-new subject area): steps 3–4 omitted. Hands-on intensity ramps progressively within a topic-group as days progress, scaled to that group's size — this pattern repeats independently every time a new topic-group begins, anywhere in the hierarchy.

**Test-out exception:** for any topic, the user may trigger verification *first*, before study content is generated.
- Full pass → topic marked complete, no study content generated
- Partial pass → only the failed questions' underlying gaps are studied, not the whole topic

**Dynamic sizing:** content volume and structure size to the user's time allotment; anything that doesn't fit (including a pending patch-note) spills to the next day — same mechanism regardless of whether the overflow is regular content or a patch.

### 7.7 Verification

- 5 fresh, source-anchored questions per topic (question + grading criteria generated from the same source material as the topic itself)
- Strict pass/fail grading, no partial credit ambiguity — but retries generate a **new** question each time (never repeats the identical question)
- **Retry cap: 3 attempts per question.** After the cap, the system teaches the answer inline (points to source material) and the user passes at **half credit**
- Topic requires all 5 questions passed (full or half credit) to complete

### 7.8 Pace Tracking

**Core principle: pace reflects understanding, not throughput.**

Per-topic score: `topic_score = (sum of per-question credit, full=1 / half=0.5) / 5`

Timing ratio: `days_taken / days_expected`, benchmarked against the **user's own established baseline**, not a universal standard.

**Combined pace signal:** topic_score is dominant (~80% influence) under normal conditions; timing ratio exerts real pull (~20% ceiling) only when it's a genuine outlier relative to the user's own baseline — ordinary variation is ignored.

**Cold start:** weeks 1–2 are calibration only — no velocity judgment or triggering.

**Trigger condition:** sustained drift across a rolling window (consecutive check-ins) — a single day's performance never triggers anything.

**Behind (sustained):** pacing extends only — outline content is never reduced.
**Ahead (sustained):** triggers enrichment (see 7.10).

**Initial pace expectation:** background/experience sets the *starting* pace-expectation profile (used pre-calibration); actual hierarchy-position skipping only ever happens via test-out, never via background-based assumption.

### 7.9 Patch-Notes

Generated when a significant market event affects an already-completed topic. Never reopens or alters the original topic's completion/verification status — always a separate, small, independently-assessed unit.

**Confidence-based branching** (reuses the existing confidence signal, no new metric):
- High confidence → prioritized into near-term delivery
- Low/uncertain confidence → user is asked: learn now (confidence-labeled, folded in) or defer

**Deferred patches:** parked permanently — no expiry, no accumulation problem (append-only) — resurface at the goal-completion closing note, or on-demand if the user explicitly asks.

**Delivery position:** always at/near the user's current position, never retroactively reinserted at the original hierarchical slot.

**Ordering among multiple pending items on the same day:** governed by hierarchy/dependency order, not detection time or arrival order.

**Test-out completions are patch-eligible** — patches don't care how a topic was completed, only that it is.

### 7.10 Enrichment

Triggered by sustained-ahead pace. Uses the **same outline-update insertion mechanism** as market-driven updates (additive, hierarchy-positioned), tagged as **extra credit**.

- Gets the full day-content treatment (theory, hands-on ramping, reflection) — same structure as core topics
- **Isolated from pace/verification consequences** — struggling on enrichment never triggers anything, never feeds the pace formula
- Still requires its own pass/fail verification, purely for closing-note credit (not for pace)

### 7.11 Goal Completion

**"Goal reached" = original core scope complete, full stop.** Enrichment is extra credit, never a requirement to finish.

**Closing note** reuses `roles.json` infrastructure (current hiring signal, in-demand skills):
- Fast/enriched learner: completed enrichment topics listed as demonstrated strengths
- Slow/core-only learner: enrichment topics suggested as next steps, not framed as a deficiency
- Any deferred low-confidence patches surface here if never resolved earlier
- **No seniority, grading, or leveling claims** anywhere in this note

---

## 8. Data Model (high-level)

- **User profile** — background, experience, prior self-study, resolved role, pacing profile
- **`roles.json`** — cron-refreshed, common roles, `core_skills`/`emerging_skills` with confidence tiers
- **Outline** — dependency hierarchy per user, topics with source/confidence metadata, status (not started / in progress / completed / completed-via-test-out)
- **Progress log** — canonical store: verification results (per question, per attempt), hands-on/review outcomes, reflection entries, timing — feeds pace calculation
- **Patch-notes** — per user, linked to originating topic, confidence tier, status (pending / delivered / deferred)
- **Enrichment topics** — tagged distinctly from core, own completion status, excluded from pace queries

## 9. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Live external calls (Himalayas, search) fail mid-demo | Full fallback chain built and proven *before* live cross-validation is layered on |
| HF Spaces free-tier storage is ephemeral | Neon (external Postgres) used for all persistence, not Spaces local disk |
| Grounding produces a fabricated source under pressure | Schema validation + retry on missing/malformed `source_url`; reject rather than accept ungrounded output |
| Scope exceeds 2.5-day build window | Full behavior preserved, build *order* sequenced by risk, not features cut |
| Verification questions are low quality (LLM-generated) | Out of scope for v1 grading; acceptable risk, noted as a known limitation |

## 10. Course Concepts Demonstrated

- **Agent/Multi-agent (ADK):** Research/Grounding and Outline/Day-generation as coordinated components with distinct state and tool access
- **MCP Server:** Himalayas (consumed, not built) — matches course guidance to consume existing MCP servers when vibe coding
- **Security features:** no hardcoded credentials, env-based secrets, input validation at the clarify gate, HITL-style confirmation at outline review
- **Deployability:** HF Spaces + Neon deployment, documented reproducibly; a GitHub Actions scheduled workflow provides the real wall-clock cron trigger for `roles.json` refresh, backed by an in-app startup staleness check as a resilience layer
- **Agent Skills:** the verification-question generator (source material in → 5 fresh source-anchored questions + grading rubric out) is packaged as a real Agent Skill artifact (SKILL.md-anchored, with a sharp what/when/when-not description), not embedded as inline agent logic. Chosen because it is narrow, single-purpose, and invoked identically and repeatedly throughout the system (every topic, every retry, test-out) — the clearest one-skill-one-job fit in the pipeline, per course guidance. Demonstrated in code.
- **Antigravity:** used during build specifically for browser-based UI verification — Antigravity's sandboxed browser subagent drives the running Streamlit app to verify the day-by-day flow renders and behaves correctly. This is a genuine, narrow demonstration of the concept, distinct from Claude Code (the primary code-generation driver, invoked from within the Antigravity IDE). Shown in the video as a concrete verification step, not a claimed-but-unused tool.

---

## 11. Future Improvements (explicitly deferred, for README)

1. Support for changing goals/roles mid-journey without losing progress or pace history
2. Visible base-knowledge level indicator (low/high) to the user — v1 keeps this internal-only
3. Pause/leave-of-absence handling distinct from both "behind" pace-drift and 30-day dormancy