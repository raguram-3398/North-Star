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

**Resolved (`src/agents/coaching_pace_agent.py`'s `complete_topic_test_out`; the test-out task — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **"Full pass" / "partial pass" reuse the existing credit scale, not a new concept.** A question slot counts as a full pass if it ultimately resolved at `FULL_CREDIT` (passed within the retry cap, on any of the 3 attempts); a slot that only resolved via the retry-cap teach-and-de-escalate path (`HALF_CREDIT`) counts as a partial pass for that question. The topic-level "full pass" this section names is every one of the 5 slots individually full-passing. This was not derivable from this section as written and was confirmed directly rather than guessed — it mirrors §7.7's own completion rule ("all 5 questions passed, full or half credit, to complete") rather than inventing a second, competing definition of "passed" specific to test-out.
- **Test-out reuses the identical retry-cap machinery, not a second one:** `begin_verification_question`/`submit_verification_answer` (§7.7's turn-based state machine) are called unchanged for test-out — same exactly-3-attempt cap, same attempt-counting shape. The only change is an added `is_test_out` flag, threaded through to `verification_attempts.is_test_out` (Architecture §5's schema column — previously always written `False` regardless of caller, since nothing before this task ever passed `True`).
- **Completion reuses `complete_topic_verification` unchanged, extended with the same flag:** a full pass and a partial pass both complete the topic (§7.7's completion rule doesn't distinguish full/half credit, and test-out doesn't either) — `complete_topic_verification` gained an `is_test_out` parameter so it writes the schema's distinct `completed_test_out` status (not a synonym for `completed`) instead of adding a second completion function.
- **Partial pass generates NO separate gap-study content.** A first implementation pass built a purpose-built gap-study content function and wired it into `complete_topic_test_out`'s partial-pass branch; review caught this as a real double-remediation bug, not a richer second pass: a partial pass's `HALF_CREDIT` slot(s) are, by construction, exactly the slot(s) that already fired §7.7's inline teach-in (`_build_taught_answer_message`) during the retry-cap attempt itself — there is no other way to reach `HALF_CREDIT`. Its prompt built text from the identical `grading_criteria` the teach-in already used, in the same session, moments earlier — calling it too would re-teach the same rubric a second time in different words, not add anything a user hasn't already just seen. It was left in place unwired at the time, as a possible building block for a future, non-test-out remediation flow — but no such feature was ever added or named in the approved plan, so a later refactor/cleanup pass deleted it outright. **What actually happens on a partial pass:** the topic is marked `completed_test_out` exactly as on a full pass; nothing further is generated.
- ~~No UI entry point exists for the user to actually trigger test-out~~ — **Resolved** (the test-out-UI-wiring task): `main.py`'s Day-by-Day Coaching stage shows a one-time choice screen (`_render_test_out_choice`) exactly when a topic's content hasn't been generated yet for the current day (this section's own "verification first, before study content is generated"). Choosing "Test out" calls `agents/coaching_pace_agent.py`'s `fetch_theory_material_links` (promoted from private to public — the same real teaching-material grounding `generate_day_content` uses) directly, never `generate_day_content` itself, so the Gemini-generated day content genuinely is never produced when the user tests out. If Tavily's search returns nothing at all, test-out is refused for that topic (an honest error, never a fabricated source) rather than proceeding with no real material to ground questions in.

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

**Resolved (`src/agents/coaching_pace_agent.py`'s `complete_topic_verification` drift wiring; the sustained-drift-wiring task — closes this section's own previously-flagged "detect_sustained_drift is not called" gap — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Wiring point:** immediately after a topic's `pace_snapshots` row is written (never for an enrichment topic — see §7.10's isolation rule), the user's full snapshot history is re-read, each entry's combined signal is recomputed via the existing blend function, and `detect_sustained_drift` is called — no drift-window or threshold logic is duplicated here.
- **Calendar-based cold-start gate, added in a later refactor/cleanup pass after this section's "weeks 1-2 are calibration only" was found to be genuinely unimplemented.** `detect_sustained_drift`'s own `DRIFT_WINDOW_SIZE` gating is a *count* of snapshots, not calendar time — a user completing 3 topics in their first 2 days would already trigger real drift/enrichment/pacing-extension actions well inside week 1, which a prior version of this bullet incorrectly treated as satisfying "weeks 1-2 are calibration only." `complete_topic_verification` now also checks `users.created_at`: still within `COLD_START_CALIBRATION_DAYS = 14` (a literal reading of "weeks 1-2," flagged for tuning like every other constant in this codebase) -> `detect_sustained_drift` is never called, `drift` is forced to `"on_track"`. The `pace_snapshots` write itself is never suppressed, only judgment/action on it.
- **"Ahead" -> enrichment (§7.10); "behind" -> pacing-extension mechanism, a new judgment call** — see §7.10's own "Resolved" block for enrichment, and Architecture §5's "Resolved" block for the pacing-extension mechanism (a new `users.pace_extension_days` column, incremented by a flagged, unvalidated constant each time behind fires). Consuming that accumulated value in a future `days_expected` calculation is not built by this task — only the accumulation mechanism itself.

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

**Resolved (`src/patches/patch_manager.py`'s `decide_patch_delivery`/`PatchDeliveryDecision`/`PatchDecisionState`/`resolve_patch_decision`; the delivery/surfacing-decision task — judgment call made and flagged during implementation, not specified above at the time of writing):** `decide_patch_delivery` surfaces at most one action per call — the earliest (hierarchy/dependency order, reusing `patch_manager.py`'s own existing `order_pending_items`) prioritized patch if any are pending, else the earliest needs-a-decision patch, else nothing; anything not chosen stays pending for a future call. `resolve_patch_decision` is the pure state transition for the "learn now or defer" choice itself (mirrors `security/input_gate.py`'s `OutlineConfirmationState` pattern) — it does not generate the conversational ask or call an LLM; that UI layer does not exist yet. See Architecture §9 for the full mechanism.

**Known limitation (see Architecture §3's Cron job "Known limitation" for the full technical finding):** the patch-note candidates generated when a significant market event is detected (`src/cron/refresh_roles.py`) carry a deterministic, mechanically-assembled `new_content` sentence (role, skill, and new confidence tier only) — not real narrative content. This section's own framing assigns patch-note content to Agent 1's reasoning, but the cron job that detects the event is explicitly not an agent and cannot call an LLM (the significant-event-wiring task's scope fence forbade it). **Not resolved by this task:** a production version routes each detected event through Agent 1 to generate the real explanation before the patch-note is persisted.

**Resolved (`src/agents/coaching_pace_agent.py`'s `maybe_deliver_patch`; the patch-delivery task — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Patch-note delivery is not pace-gated.** `maybe_deliver_patch` is called unconditionally on every non-enrichment topic completion, regardless of that call's own drift result — unlike enrichment (§7.10), which only fires on sustained-ahead drift, a market event has nothing to do with the completing user's own pace.
- ~~**"insert_now" reuses the identical insertion mechanism §7.10's enrichment already wired (`outline/hierarchy.py`'s `insert_new_topic`, via `data/outline_topics.py`'s `insert_new_outline_topic`) — not a second insertion path**~~ — **Superseded** (the augment-vs-addition-decision task, resolving Architecture §9's previously-flagged open question): "insert_now" now **augments** the patch's own origin topic in place (`data/outline_topics.py`'s `augment_outline_topic`, wrapping `outline/hierarchy.py`'s `augment_existing_topic`) instead of inserting a new topic. Every patch-note's origin topic is, by construction, a topic the user already completed with a skill name that exactly matches the significant event that created the patch (`refresh_roles_cache`'s `create_patch_notes_for_significant_events` only ever creates a patch-note via `get_completed_topics_matching_skill`) — so the update is always "this already-existing topic's own market data changed," never "a genuinely new topic is needed." CLAUDE.md guardrail #5 still holds: only `source_url`/`source_type`/`confidence` are refreshed, never `status`/`completed_at`.
- **Co-occurrence with enrichment, a genuine judgment call:** sustained-ahead drift and a pending high-confidence patch-note are independent conditions and can both fire in the same completion call — neither is suppressed in favor of the other. See Architecture §3's "Resolved" block for the exact resulting ordering (a low-stakes tie-break, not a spec-mandated rule).
- ~~**"ask_user" is not built or wired further by this task**~~ — **Resolved** (the patch ask_user-UI task): `maybe_deliver_patch` now returns a `PendingPatchDecision` (patch content + origin topic id, not just a bare `PatchDecisionState`) on this branch; `main.py`'s Verification stage renders it as a non-blocking, dismissible banner (`_render_patch_decision_banner`) alongside — never instead of — the normal "Continue to next topic"/"View closing note" actions, so ignoring it is always safe. Choosing "Learn now" or "Defer" calls `agents/coaching_pace_agent.py`'s new `resolve_pending_patch_decision`, which advances `resolve_patch_decision` and, on "learn now," augments the origin topic exactly the way an auto-prioritized patch already is (PRD's own "learn now (confidence-labeled, folded in)" language above) — never a second, competing mechanism.
- **`new_content` is inserted exactly as-is** — the already-flagged deterministic placeholder (see the "Known limitation" above); this task does not attempt real narrative content generation for it.

### 7.10 Enrichment

Triggered by sustained-ahead pace. Uses the **same outline-update insertion mechanism** as market-driven updates (additive, hierarchy-positioned), tagged as **extra credit**.

- Gets the full day-content treatment (theory, hands-on ramping, reflection) — same structure as core topics
- **Isolated from pace/verification consequences** — struggling on enrichment never triggers anything, never feeds the pace formula
- Still requires its own pass/fail verification, purely for closing-note credit (not for pace)

**Resolved (`src/agents/coaching_pace_agent.py`'s `maybe_trigger_enrichment`/`_select_enrichment_skill`, `data/outline_topics.py`'s `get_all_topics_for_user`/`has_pending_enrichment_topic`/`insert_new_outline_topic`; the sustained-drift-wiring task — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- **Candidate source: the resolved role's own `roles_cache` `emerging_skills` bucket**, not the user's own live grounding result — `emerging_skills` already carries the real core/emerging split `roles_cache` maintains; a live `LiveGroundingResult` (Architecture §3) only ever returns one flat skill list, per the already-flagged degenerate-split limitation (Future Improvements #5).
- **Selection rule, a genuine judgment call:** the first `emerging_skills` entry (in `roles_cache`'s own stored order) whose skill name doesn't already match an existing outline topic for the user (case-insensitive) — "first not-yet-used," not a weighted or ranked pick, since that list carries no other ordering signal (demand strength, recency) to prefer one entry over another.
- **One pending enrichment topic at a time, a judgment call not unambiguous from the schema alone:** a user with any `outline_topics` row that is `is_enrichment=True` and not yet resolved (`status` not in `{completed, completed_test_out}`) gets no second enrichment topic inserted on a later "ahead" trigger, regardless of which topic completion triggered it.
- **Insertion mechanism, actually wired for the first time by this task:** `outline/hierarchy.py`'s existing `insert_new_topic` — not a second insertion path. The new topic's sole prerequisite is the just-completed topic that triggered the check, so it lands immediately after the user's current position (matching this section's own "additive, hierarchy-positioned" framing), never at the very start of the whole hierarchy.
- **Isolation is structural, not advisory:** `complete_topic_verification`'s entire `pace_snapshots`-write-and-drift-detection block sits inside a single `if not is_enrichment:` guard — an enrichment topic's completion is architecturally incapable of writing a `pace_snapshots` row or influencing a future `detect_sustained_drift` call, regardless of credit earned, satisfying this section's "never feeds the pace formula" as an unconditional code-level guarantee rather than a caller convention.
- **Test-out + enrichment is treated as a real, possible combination** (both converge on the identical `complete_topic_verification` completion path) — built and tested, not assumed impossible.

### 7.11 Goal Completion

**"Goal reached" = original core scope complete, full stop.** Enrichment is extra credit, never a requirement to finish.

**Closing note** reuses `roles.json` infrastructure (current hiring signal, in-demand skills):
- Fast/enriched learner: completed enrichment topics listed as demonstrated strengths
- Slow/core-only learner: enrichment topics suggested as next steps, not framed as a deficiency
- Any deferred low-confidence patches surface here if never resolved earlier
- **No seniority, grading, or leveling claims** anywhere in this note

**Resolved (`src/agents/coaching_pace_agent.py`'s `is_goal_complete`/`generate_closing_note`/`ClosingNote`; the patch-delivery task — judgment calls made and flagged during implementation, not specified above at the time of writing):**
- ~~**`is_goal_complete` and `generate_closing_note` are separate, independently callable functions, not composed internally... Neither is wired into any trigger yet.**~~ — **Resolved** (`src/main.py`, the orchestration-wiring task): the Verification stage calls `is_goal_complete` after every non-enrichment topic completion; the Goal Completion stage calls `generate_closing_note` only once that returns true. The two functions themselves remain independently callable/testable — `main.py` is the composing caller, not a merge of the two into one function.
- **`is_goal_complete`: core topics only, enrichment entirely excluded from the check** — not merely allowed to be incomplete. A user with zero core topics returns `False` (not vacuously `True`): "goal reached" means the core plan that existed was completed, not that there was nothing to complete.
- **Fast/enriched vs. slow/core-only is decided by whether the user has *completed* at least one enrichment topic** (not merely has one in-progress or pending) — completed enrichment topics become `demonstrated_strengths`; their absence is what triggers `suggested_next_steps` (the role's `roles_cache` `emerging_skills`) instead.
- **LLM-vs-deterministic, decided explicitly as this task required:** unlike the market-event `new_content` placeholder (mechanical, because that caller structurally cannot call an LLM), this note is real, user-facing closing prose — so a genuine Gemini call composes it from deterministically-gathered facts, rather than a mechanical template. See Architecture §3's "Resolved" block for the prompt/model details.
- **"No seniority, grading, or leveling claims" is a hard content constraint, enforced structurally after generation** (a deterministic banned-term check against Gemini's actual output), not trusted to the prompt instruction alone — `generate_closing_note` raises rather than returning non-compliant text. No retry-with-feedback is attempted on a rejection in this task.
- **Deferred patch-notes are surfaced as raw structured data (`ClosingNote.deferred_patch_notes`), not folded into the Gemini prompt as content to narrate** — this is their one resurfacing point per this section's own rule, but a future caller/UI renders them directly rather than trusting the LLM to accurately restate patch-note facts it didn't ground itself.

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
6. ~~Grounding a user's raw outline-confirmation addition request (e.g. "can you add GraphQL?") into a real, sourced skill before it can be folded into a regenerated outline~~ — **Resolved** (the addition-grounding task, reported live from the running app as a real gap and closed the same session): `agents/research_outline_agent.py`'s `ground_addition_request` extracts a short, search-ready skill name from the user's raw message (a new, dedicated `PROMPT_REGISTRY` entry, `outline_addition_skill_name_extraction_v1`), then confirms it via a live Tavily-only search, citing the top result at `ConfidenceTier.MEDIUM` — never a fabricated `source_url` (CLAUDE.md guardrail #1); returns `None`, never an invented citation, if Tavily has nothing usable. `main.py`'s Outline Confirmation stage now calls this and, on success, `regenerate_outline_with_addition` + `insert_outline_topics` for real on an `ADDITION_REQUEST` turn (the previously-added `OutlineConfirmationTurn.action` field is how `main.py` learns a turn needs this follow-up); the previously-shown "not wired up yet" caption is removed. See Architecture §3's dedicated "Resolved" block after Agent 2 for the full mechanism.
7. ~~Extracting the shared Gemini call/timeout/error-handling infrastructure (`_call_gemini_json`, `GeminiCallError`) out of `agents/research_outline_agent.py` into a shared `src/utils/` module~~ — **Resolved** (the Gemini-client-extraction task): `src/utils/gemini_client.py` now owns this infrastructure (pacing, timeout/retry constants, `_call_gemini_text`/`_call_gemini_json`, `reset_gemini_client_for_new_event_loop`) in full — `agents/research_outline_agent.py`, `agents/coaching_pace_agent.py`, and `.agent/skills/verification_question_generator/generator.py` all import it as legitimate peers now, none reaching into another agent's private (underscore-prefixed) namespace across a package boundary. `_get_tavily_client` (Tavily, not Gemini, infrastructure) still lives in `research_outline_agent.py` and is still reached into by `coaching_pace_agent.py` — out of scope for this extraction, not yet promoted. **Further superseded** (the ADK-refactor task, see Architecture §3's dedicated "Resolved" block): `utils/gemini_client.py` is deleted; `utils/adk_runtime.py` — built on real `google.adk.agents.LlmAgent`/`google.adk.runners.Runner` calls, not a bare `google-genai` client — is the shared infrastructure today, closing this rubric line item's actual code-level gap (previously `google-adk` was only used for MCP tool consumption, never for agent reasoning).
8. ~~Persisting `agents/research_outline_agent.py`'s `create_initial_outline`/`regenerate_outline_with_addition` output into real `outline_topics` rows~~ — **Resolved** (outline-topics-insert task): `data/outline_topics.py`'s `insert_outline_topics` now persists a freshly-created or regenerated outline, replacing the user's prior row set. See §7.5's "Resolved" block above and Architecture §3/§5/§10 for the mechanism. ~~Wiring this function into the live pipeline's actual call sites~~ — **Resolved** (`src/main.py`, the orchestration-wiring task, plus the addition-grounding task for the addition-request case): `main.py`'s Outline Creation stage is the first real caller, and Outline Confirmation's addition-request path now also calls it (via `ground_addition_request` + `regenerate_outline_with_addition`) — see item 6 above.
9. The Verification Skill's mandated `.agent/skills/` location (outside `src/`, per CLAUDE.md, for Antigravity workspace-manager recognition) is not on the normal editable-install import path — every real importer, not just tests, needs an explicit `sys.path` bootstrap (see `agents/coaching_pace_agent.py`'s identical pattern to `tests/test_verification_skill.py`'s). A second data point (beyond the test file) that this required location creates real friction for legitimate importers; worth revisiting if more Skills are added. Flagged during implementation (coaching-pace-agent task), not from original design. — **Resolved** (the skill-relocation task, see Architecture §4's revised "Skill location note"): the Verification Skill moved to `src/skills/verification_question_generator.py`, a plain importable module; every `sys.path` bootstrap this bullet flagged is deleted. Explicit tradeoff: this drops the Antigravity-workspace-manager-recognized packaging form for the hackathon's "Agent Skills" course concept — not a blocker, since ADK/MCP/Security/Deployability already evidence 4 concepts independently.
10. ~~Actually consuming `users.pace_extension_days` in a real `days_expected` baseline calculation~~ — **Partially resolved** (the days_expected-formula task): `src/main.py`'s flat `DAYS_EXPECTED_PER_TOPIC = 1` placeholder is gone, replaced by `agents/coaching_pace_agent.py`'s `calculate_days_expected(available_time_per_week_hours)` — a real baseline derived from the user's own `available_time_per_week` (a new, flagged `ESTIMATED_MINUTES_PER_TOPIC = 120` constant divided by `convert_weekly_hours_to_daily_minutes`'s daily budget, rounded up). **Still not built:** this formula does not factor in topic-group size or the accumulated `users.pace_extension_days` — every topic still costs the same fixed 120 minutes regardless of subject matter or how much pacing extension has already accrued for that user.
11. ~~A real Streamlit/AppTest technical finding from the orchestration-wiring task: ... building the root-level `streamlit_app.py` this repo's structure already names ... would close this structurally instead~~ — **Resolved** (the streamlit_app.py task): the root-level `streamlit_app.py` now exists as a thin `from main import main` wrapper — `streamlit run streamlit_app.py` is the real entry point (verified live: boots and serves HTTP 200), not `streamlit run src/main.py`, so `main` is a normally cached, once-imported module rather than a re-executed script, closing this bug class structurally. `main.py`'s own `PipelineStage.X.value` string-storage workaround is left in place (harmless, and `main.py` can still be run directly during development), not removed.
12. **Deferred patch-notes' "on-demand if the user explicitly asks" resurfacing (§7.9) has no UI/trigger** — only goal-completion resurfacing (`ClosingNote.deferred_patch_notes`) is built. A genuine product/UX decision (where in the pipeline a mid-journey "show me what's deferred" action would live) that was deliberately left open during the patch ask_user-UI task, not attempted.
