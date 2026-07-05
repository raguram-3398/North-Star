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
- No cost/usage tracking or tool-call audit logging (`utils/logger.py`) for this submission — a real scope cut, decided during the cron-refresh task, not an oversight. Every Gemini/Tavily/Himalayas call proceeds without a cost/usage log entry or audit trail. A production version would add per-call cost/token logging (real API usage counts, not estimates) and a one-time daily-spend threshold alert; see Architecture §10 for the technical framing.

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

**Known limitation (discovered live during `src/data/himalayas_parser.py`'s development, not anticipated in original design):** Himalayas MCP's `search_jobs` tool does not reliably signal "zero matches." Live testing — a nonsense keyword, an extreme `salary_min`, and an obscure `country` + `exclude_worldwide` combination — never produced a genuine empty result; the tool instead fell back to broad/unrelated matches. In practice, the "zero market signal found → exit" determination above cannot rest on Himalayas casting an independent "no" — it rests on Tavily returning nothing usable and `roles_cache` having no anchor, with any Himalayas results needing a relevance/quality judgment rather than a literal empty-check. See §7.3 and Architecture_North_Star.md §8 for the same finding. **Resolved:** `src/agents/research_outline_agent.py`'s `ground_role` now infers "no signal from Himalayas" via a title-relevance heuristic (`src/data/himalayas_relevance.py`) rather than an empty-result check — see §7.3's updated note and Architecture §8 for the mechanism. The limitation itself (Himalayas cannot independently signal zero) still stands; only the inference mechanism around it is new.

**Resolved (`src/security/input_gate.py`'s `classify_stated_goal`, `src/agents/research_outline_agent.py`'s `begin_clarify_gate`/`advance_clarify_gate`; implementation not specified above at the time of writing):** the Gate behavior list above is implemented as a deterministic, lexical/plausibility first-pass classification (never a market-existence check — a niche/obscure real title is never gate-rejected here, per this section's own instruction) that routes to real/vague/nonsense, followed by an LLM-driven conversational loop for everything past that first classification. See Architecture §3 for the full mechanism, constants, and judgment calls (vocabulary lists, the fabricated-title denylist, the model choice, the `PROMPT_REGISTRY`). The nonsense re-prompt and the zero-market-signal exit message are both fixed, non-LLM text — both outcomes are already fully decided deterministically before the message is chosen.

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

**Resolved (`src/data/himalayas_relevance.py`, `src/data/tavily_parser.py`, `src/data/cross_validation.py`, `src/agents/research_outline_agent.py`'s `ground_role`; previously flagged as an open design question during `src/data/himalayas_parser.py`, now implemented):** the cross-validation rules above assume each source can independently signal "no data." Live testing showed Himalayas's `search_jobs` does not do this reliably (see §7.2's limitation note) — it returns broad/unrelated matches rather than an empty result even for deliberately nonsensical or over-constrained queries. "No signal from Himalayas" is inferred from the *relevance* of returned listings (title-token overlap against the searched role, banded by result count — see Architecture §8 for the exact mechanism and constants) rather than read directly off an empty-result response.

**A Tavily-only result can now reach medium confidence** (previously a hard scope limit: a Tavily-only signal always fell through to the cached-fallback/general-knowledge-only rungs, regardless of how strong Tavily's signal actually was). Trust is decided by how many *distinct* skills `src/data/tavily_parser.py` extracts from Tavily's results across the whole batch, against a named threshold (`TAVILY_DISTINCT_SKILLS_TRUST_THRESHOLD`, Architecture §8) — Tavily's own relevance `score` is used only to choose which already-skill-bearing result becomes the cited source, never to decide whether to trust the batch (Tavily's `score` was shown not to predict extractability — see Architecture §1's resolved detail on `tavily_parser.py`). **Known limitation, carried forward, not resolved by this task:** the extraction vocabulary was derived from skills Himalayas already surfaced for the same 4 seed roles, so it does not independently discover skills for roles outside that set — see Architecture §8 for the named future-improvement path.

**Niche/no-anchor roles** (no `roles.json` entry): single-source only, lightweight LLM sanity-check pass using only already-present in-snippet company context (no extra fetch) → **confidence: low**. Never written back into `roles.json`.

**Full confidence ladder:** high → medium → low → cached-fallback (low, labeled with `last_updated`) → general-knowledge-only floor (explicitly labeled as such) → reject (zero signal, see Clarify Gate exit).

**Resolved (`src/data/grounding_fallback.py`; judgment calls made and flagged during implementation, not specified above at the time of writing):** see Architecture_North_Star.md §8 for the full reasoning on (1) why `general-knowledge-only` returns a structurally distinct, sourceless result rather than being forced through the same validated-object type every other rung uses, (2) why a stale cached entry still counts as usable fallback data instead of being treated as "no entry," and (3) the constant `source_type` stamp used for cached-fallback results, since `roles_cache` never persists a per-skill `source_type` to read back.

**Grounding rule (absolute):** every downstream content item carries `source_url`, `source_type`, `confidence`. Nothing enters the system ungrounded.

**`roles.json`:** structured with `core_skills` / `emerging_skills` per role, each carrying a confidence tier. Refreshed on a deterministic cron (minimum every 30 days, immediately on a significant event). Serves as fallback data and normalization anchor — never a shortcut that skips live research for a new user; every user gets a fresh live research pass.

**Bootstrap (initial creation):** there is no separate seeding process — the initial file is produced by manually running the same Research/Grounding pipeline once per role in a small seed list (5–8 common roles relevant to the target user, e.g. Backend Engineer, Frontend Engineer, Data Analyst, AI/ML Engineer, DevOps Engineer), writing each result into `roles.json` in its normal structure. The manual seed run and the recurring cron job are the same code path — the seed run is simply the first invocation, triggered by hand instead of by a timer. No hand-curated data is written.

**Significant event (deterministic rule):** a skill crosses a bucket or confidence boundary *upward* — absent → `emerging_skills`, `emerging_skills` → `core_skills`, or confidence tier strengthens. Downgrades produce no action (content is never removed).

### 7.4 Outline Creation

Output is a **dependency hierarchy** (prerequisite order: basics → full role requirements), not a flat list — and the hierarchy itself is grounded, derived from the same sourced skill data as Research, not a separate ungrounded reasoning step.

**Resolved (`src/agents/research_outline_agent.py`'s `create_initial_outline`; implementation not specified above at the time of writing):** initial creation requires genuine LLM domain-knowledge judgment (correct prerequisite order — e.g. HTML before CSS before JavaScript, Python fundamentals before Django — isn't derivable from the grounded skill list alone), so it's implemented as a single Gemini call that both orders topic groups relative to each other *and* orders topics within each group, per Architecture §3's mechanism. Sourcing is never at Gemini's discretion: it only names which grounded skill each topic came from, and `create_initial_outline` re-attaches that skill's actual `source_url`/`source_type`/`confidence` by exact-name lookup against the already-grounded input — Gemini's response never contains sourcing fields to invent, drop, or alter in the first place. Every grounded skill must be covered by at least one topic (a skill may fan out into several topics, e.g. "Python" into syntax/functions/OOP, but none may be silently dropped). See Architecture §3 for the model choice, prompt, and a known integration gap (`ground_role`'s live-grounding path doesn't yet produce the core/emerging skill split this function's input expects).

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

**Resolved (confirmed directly for `src/agents/research_outline_agent.py`'s `begin_outline_confirmation`/`handle_review_turn`/`regenerate_outline_with_addition` and `src/security/input_gate.py`'s `OutlineConfirmationState`; these three ambiguities were not derivable from this section or the Gherkin scenarios and were confirmed directly rather than inferred):**
- **Round bound is exactly 2** — the same number as the clarify gate's narrowing bound, not a new value chosen independently.
- **Questions are free and unbounded, asked alongside the bounded rounds.** Only raising a concern or requesting an addition consumes one of the 2 bounded rounds; a user can ask any number of clarifying questions about why a topic is included without ever advancing the bound.
- **An accepted addition regenerates the full outline from scratch via the same Gemini-sequencing path as initial creation (`create_initial_outline`), given the original grounded skills plus the new addition folded in — it does not use `outline/hierarchy.py`'s insertion logic.** That module is reserved for post-confirmation updates to an outline the user is already actively progressing through (market-driven significant events, patch-notes, §7.9). This pre-Day-1 window is the only time an outline is still being drafted rather than inserted into, so full regeneration is correct here.

See Architecture §3 for the mechanism, the fourth action value (`confirm`) this task's implementation needed beyond the three named above, and a flagged scope boundary: how a raw addition request becomes a *grounded* skill (with a real source_url) before it can be folded in is not addressed by this task.

**Resolved (`src/data/outline_topics.py`'s `insert_outline_topics`; the outline-topics-insert task — persisting a confirmed/regenerated outline was previously an unbuilt gap, see Architecture §10 and Non-Goals; this section covers behavior, Architecture §3/§5 covers the mechanism):**
- **Persisting an outline replaces the user's entire prior `outline_topics` row set — never a partial delta.** Since an accepted addition regenerates the *whole* hierarchy from scratch (the bullet above), the persistence layer mirrors that: every existing row for the user is deleted and the newly-sequenced set is inserted in the same transaction. A first-time outline (no prior rows) degenerates to a plain insert.
- **Safe only because this window is provably pre-Day-1:** no `outline_topics` row for a user can be anything but `not_started` while Outline Confirmation is still open (this section's own "no user-initiated outline editing once Day 1 begins"). `insert_outline_topics` does not merely assume this — it raises if it ever finds an existing row already `in_progress`/`completed`/`completed_test_out`, since silently replacing that would violate CLAUDE.md guardrail #2 ("never delete or reduce outline content"). This was a genuine judgment call (this section describes the conversational flow, not DB-write semantics) — revisable if a future, non-pre-Day-1 regeneration path is ever added.

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

**Resolved (`src/agents/coaching_pace_agent.py`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Hands-on ramping formula:** `compute_hands_on_intensity(position_in_group, group_size) = (position_in_group - 1) / (group_size - 1)`, linear across the topic-group — 0.0 (conceptual-only) on the group's first day, reaching 1.0 (full depth) on the group's last day. Driven entirely by `outline_topics.position_in_group` and the group's total size, never a fixed day-count constant, per this section's own requirement. Edge case: a single-topic group (`group_size == 1`) returns 1.0 (hands-on-eligible immediately) rather than 0.0 — otherwise a topic-group with only one day would never get any hands-on practice at all.
- **Time-budget conversion:** `users.available_time_per_week` (hours) is converted to a daily minute budget assuming a 5-day study cadence (`STUDY_DAYS_PER_WEEK = 5`, i.e. a Monday-through-Friday-shaped week) — not specified anywhere above; revisable if a different cadence (e.g. 7-day) is intended.
- **Dynamic-sizing/spillover mechanism:** implemented as a single generic `carried_over_content` (input) / `remaining_content` (output) string pair threaded through `generate_day_content` — content that doesn't fit becomes tomorrow's `carried_over_content`. Deliberately generic (not specific to *why* something spilled), so a future patch-note's content can plug into the same `carried_over_content` parameter without a rework of the mechanism itself — this task only wires it for regular-content overflow; patch-note overflow is not yet triggered into it (§7.9's delivery/surfacing logic is a later task).
- **Closing note still deferred; test-out is no longer deferred (see the "Resolved" block immediately below):** this task built day-content generation and the verification retry-cap/pace-signal wiring below, but not goal-completion closing-note content or (at the time this bullet was written) the test-out (verification-first) path — deliberately deferred rather than stubbed with a guessed shape (closing note still is; see Architecture §10).

**Resolved (`src/agents/coaching_pace_agent.py`'s `complete_topic_test_out`/`generate_gap_study_content`; the test-out task — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **"Full pass" / "partial pass" reuse the existing credit scale, not a new concept.** A question slot counts as a full pass if it ultimately resolved at `FULL_CREDIT` (passed within the retry cap, on any of the 3 attempts); a slot that only resolved via the retry-cap teach-and-de-escalate path (`HALF_CREDIT`) counts as a partial pass for that question. The topic-level "full pass" this section names is every one of the 5 slots individually full-passing. This was not derivable from this section as written and was confirmed directly rather than guessed — it mirrors §7.7's own completion rule ("all 5 questions passed, full or half credit, to complete") rather than inventing a second, competing definition of "passed" specific to test-out.
- **Test-out reuses the identical retry-cap machinery, not a second one:** `begin_verification_question`/`submit_verification_answer` (§7.7's turn-based state machine) are called unchanged for test-out — same exactly-3-attempt cap, same attempt-counting shape. The only change is an added `is_test_out` flag, threaded through to `verification_attempts.is_test_out` (Architecture §5's schema column — previously always written `False` regardless of caller, since nothing before this task ever passed `True`).
- **Completion reuses `complete_topic_verification` unchanged, extended with the same flag:** a full pass and a partial pass both complete the topic (§7.7's completion rule doesn't distinguish full/half credit, and test-out doesn't either) — `complete_topic_verification` gained an `is_test_out` parameter so it writes the schema's distinct `completed_test_out` status (not a synonym for `completed`) instead of adding a second completion function.
- **Partial pass generates NO separate gap-study content — this was implemented once, then corrected on review.** The first pass built `generate_gap_study_content` (a new, purpose-built content path, distinct from `generate_day_content`'s 7-step structure) and wired it into `complete_topic_test_out`'s partial-pass branch. On review this was caught as a real double-remediation bug, not a richer second pass: a partial pass's `HALF_CREDIT` slot(s) are, by construction, exactly the slot(s) that already fired §7.7's inline teach-in (`_build_taught_answer_message`) during the retry-cap attempt itself — there is no other way to reach `HALF_CREDIT`. `generate_gap_study_content`'s prompt built its text from the identical `grading_criteria` the teach-in already used, in the same session, moments earlier — calling it too would re-teach the same rubric a second time in different words, not add anything a user hasn't already just seen. **What actually happens on a partial pass:** the topic is marked `completed_test_out` exactly as on a full pass; nothing further is generated. `generate_gap_study_content` remains in the codebase, unwired, as a possible building block for a future, non-test-out remediation flow (e.g. a dedicated "review what you missed" feature) — not deleted, since the function itself is sound, only wrong to call from test-out.

### 7.7 Verification

- 5 fresh, source-anchored questions per topic (question + grading criteria generated from the same source material as the topic itself)
- Strict pass/fail grading, no partial credit ambiguity — but retries generate a **new** question each time (never repeats the identical question)
- **Retry cap: 3 attempts per question.** After the cap, the system teaches the answer inline (points to source material) and the user passes at **half credit**
- Topic requires all 5 questions passed (full or half credit) to complete

**Resolved (`.agent/skills/verification_question_generator/generator.py`; implementation not specified above at the time of writing):** question generation and answer grading are both this Skill's responsibility — it is stateless per call (no memory of prior attempts) and produces a strict boolean pass/fail with no ambiguous middle state. The retry cap (exactly 3), attempt-number tracking, and half-credit de-escalation above are orchestration-level concerns, not this Skill's — the Coaching & Pace Agent (a later task) calls this Skill fresh for every attempt and owns counting attempts and applying half credit after the cap. "Never repeats the identical question" is enforced structurally (an exact, case-insensitive match check against every prior question for that slot, which the caller must pass in) rather than left to a prompt instruction alone. See Architecture §4 for the full mechanism and judgment calls.

**Resolved (`src/agents/coaching_pace_agent.py`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Retry-cap orchestration shape:** a turn-based state machine (`VerificationSlotState`, advanced one attempt at a time via `begin_verification_question`/`submit_verification_answer`), not an internal blocking loop — a single function call can't synchronously wait for 3 separate human answers in Streamlit's request/response cycle. Every attempt (1st, 2nd, or 3rd) is graded and recorded through the exact same `submit_verification_answer` call, with no special-cased first-attempt code path, so the retry counter genuinely includes attempt 1 rather than treating it as a freebie before the loop starts.
- **Taught-answer message is deterministic, not LLM-generated:** built directly from the Skill's own `grading_criteria` and `source_url` (`_build_taught_answer_message`), rather than a fresh Gemini call — `grading_criteria` already is the rubric; regenerating prose from it risks restating or contradicting it, and this stays consistent with the system's structural (never prompt-only) sourcing-safety approach.
- **Credits are read from `verification_attempts`, not caller-tracked state:** `complete_topic_verification` reads each question slot's final attempt back from the database as the single source of truth, and this is also where "all 5 questions passed (full or half credit)" is actually enforced — it raises if any slot has failed but hasn't yet reached the retry cap (still genuinely in progress, not resolved).
- **Acting on the pace signal is deferred:** this task computes and persists the pace snapshot (§7.8) once a topic's 5 questions resolve, but does not call `detect_sustained_drift` or act on "behind"/"ahead" — that is a later task's scope.

### 7.8 Pace Tracking

**Core principle: pace reflects understanding, not throughput.**

Per-topic score: `topic_score = (sum of per-question credit, full=1 / half=0.5) / 5`

Timing ratio: `days_taken / days_expected`, benchmarked against the **user's own established baseline**, not a universal standard.

**Combined pace signal:** topic_score is dominant (~80% influence) under normal conditions; timing ratio exerts real pull (~20% ceiling) only when it's a genuine outlier relative to the user's own baseline — ordinary variation is ignored.

**Cold start:** weeks 1–2 are calibration only — no velocity judgment or triggering.

**Trigger condition:** sustained drift across a rolling window (consecutive check-ins) — a single day's performance never triggers anything.

**Behind (sustained):** pacing extends only — outline content is never reduced.
**Ahead (sustained):** triggers enrichment (see 7.10).

**Resolved calibration values (`src/pace/calculator.py`; judgment calls made and flagged during implementation, not specified above at the time of writing):**
- Outlier threshold: `timing_ratio` must deviate more than ±50% from 1.0 (on-baseline-pace) before timing exerts any pull at all on the combined signal (`TIMING_OUTLIER_THRESHOLD = 0.5`). Inside that band, timing is ignored entirely, not just down-weighted.
- Saturation point: timing's pull reaches its full ~20% ceiling once the deviation reaches ±100% (`TIMING_SATURATION_DEVIATION = 1.0`), scaling linearly between the outlier threshold and this point.
- Rolling window size: 3 consecutive check-ins (`DRIFT_WINDOW_SIZE = 3`).
- Behind threshold: the window's **mean** combined pace signal at or below 0.7 triggers "behind" (`SUSTAINED_BEHIND_THRESHOLD = 0.7`) — calibrated so it roughly corresponds to needing the half-credit teach-in fallback (§7.7) on 3 or more of the 5 questions in a topic.
- Ahead threshold: the window's mean at or above 0.95 triggers "ahead" (`SUSTAINED_AHEAD_THRESHOLD = 0.95`).
- The sustained-drift check is **mean-based** over the trailing window, not requiring every single entry to individually cross the threshold — a single unusually good or bad day within an otherwise-consistent window does not by itself reset the streak.

**Initial pace expectation:** background/experience sets the *starting* pace-expectation profile (used pre-calibration); actual hierarchy-position skipping only ever happens via test-out, never via background-based assumption.

### 7.9 Patch-Notes

Generated when a significant market event affects an already-completed topic. Never reopens or alters the original topic's completion/verification status — always a separate, small, independently-assessed unit.

**Confidence-based branching** (reuses the existing confidence signal, no new metric):
- High confidence → prioritized into near-term delivery
- Low/uncertain confidence → user is asked: learn now (confidence-labeled, folded in) or defer

**Resolved cutoff (`src/patches/patch_manager.py`; judgment call made and flagged during implementation, not specified above at the time of writing):** only `high` confidence auto-prioritizes. `medium`, `low`, `cached-low`, and `general-knowledge-only` all route to "needs a user decision." Reasoning: §7.3 defines `high` as both sources agreeing, and `medium` as a single source or minor disagreement — `medium` is therefore not fully cross-validated and reads as "uncertain" in this document's own terms, consistent with the system's "honest about its own confidence" value proposition (§3). This is an easy, low-risk constant to change if `medium` was intended to auto-prioritize too.

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
4. Support a "must-precede" positioning constraint in `outline/hierarchy.py` for cases where a new or enrichment topic must be inserted before an existing topic, not just after a prerequisite — currently unhandled, flagged as a limitation during implementation.
5. Deciding which grounded skills are "core" vs "emerging" for a *live* (non-cached) grounding result — `ground_role`'s `LiveGroundingResult` returns one flat skill list, unlike `roles_cache`/`data/grounding_fallback.py`'s cached path, which already carries the split. `create_initial_outline` currently receives the degenerate workaround (`core_skills=result.skills, emerging_skills=[]`), verified end-to-end (`tests/test_pipeline_integration.py`) not to block the pipeline, but the "(core)"/"(emerging)" hint given to Gemini during hierarchy sequencing is a placeholder, not an accurate signal, until this is resolved. Flagged during implementation (outline-creation task), not from original design.
6. Grounding a user's raw outline-confirmation addition request (e.g. "can you add GraphQL?") into a real, sourced skill before it can be folded into a regenerated outline — `handle_review_turn` classifies the request and consumes its round, but does not itself ground anything (no live Himalayas/Tavily lookup for the specific requested skill); `regenerate_outline_with_addition` requires an already-grounded object supplied by the caller. Flagged during implementation (outline-confirmation task), not from original design.
7. Extracting the shared Gemini call/timeout/error-handling infrastructure (`_call_gemini_json`, `GeminiCallError`) out of `agents/research_outline_agent.py` into a shared `src/utils/` module — `.agent/skills/verification_question_generator/generator.py` currently imports it directly from the Agent module (an underscore-prefixed, module-private-by-convention name, reached into across a real package boundary) rather than duplicating the logic, per that task's explicit reuse instruction. `agents/coaching_pace_agent.py` is now a *third* direct consumer of the same private helpers (`_call_gemini_json`, `_get_tavily_client`), strengthening the case for eventual extraction. Still not attempted, since it would mean refactoring already-tested, already-committed Agent code as a side effect of a narrower-scoped task. Flagged during implementation (verification-skill task; reaffirmed during coaching-pace-agent task), not from original design.
8. ~~Persisting `agents/research_outline_agent.py`'s `create_initial_outline`/`regenerate_outline_with_addition` output into real `outline_topics` rows~~ — **Resolved** (outline-topics-insert task): `data/outline_topics.py`'s `insert_outline_topics` now persists a freshly-created or regenerated outline, replacing the user's prior row set. See §7.5's "Resolved" block above and Architecture §3/§5/§10 for the mechanism and the still-open items this task did *not* touch (grounding a raw addition request into a real skill, item 6 above; wiring this function into the live pipeline's actual call sites).
9. The Verification Skill's mandated `.agent/skills/` location (outside `src/`, per CLAUDE.md, for Antigravity workspace-manager recognition) is not on the normal editable-install import path — every real importer, not just tests, needs an explicit `sys.path` bootstrap (see `agents/coaching_pace_agent.py`'s identical pattern to `tests/test_verification_skill.py`'s). A second data point (beyond the test file) that this required location creates real friction for legitimate importers; worth revisiting if more Skills are added. Flagged during implementation (coaching-pace-agent task), not from original design.
