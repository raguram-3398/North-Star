---
name: verification_question_generator
description: Generate source-anchored comprehension questions with grading criteria from study material. Use when a topic needs verification questions (initial 5-question set, a fresh retry question, or a test-out check). Do not use for market-data grounding or general Q&A unrelated to a specific study topic.
---

# Verification Question Generator

## Purpose

Produces strict, source-anchored comprehension questions and their grading criteria from a single piece of study material, and grades a user's answer against a question's grading criteria at strict pass/fail. This is the only mechanism in North Star that creates or grades verification content — it is invoked identically whether the call is the initial 5-question set for a topic, a fresh retry question after a failed attempt, or a test-out pre-check.

**Stateless per call.** This Skill has no memory of prior attempts — it does not count retries, track attempt numbers, or apply the half-credit de-escalation. Those are orchestration-level concerns (`coaching_pace_agent.py`, a later task), which calls this Skill fresh for every attempt and is responsible for passing whatever context (e.g. `previous_question_texts`) this Skill needs to do its job correctly.

## When to use

1. A topic is about to be studied for the first time and needs its initial 5-question verification set.
2. A user failed a specific question and needs a **fresh** question covering the same underlying concept — never the identical question repeated.
3. A user invokes "test out" on a topic and needs the verification check run *before* any study content is generated.

## When NOT to use

1. Market-data grounding, skill extraction from job descriptions, or any Research Agent task (that's `data/himalayas_parser.py`/`data/tavily_parser.py`/`agents/research_outline_agent.py`'s job, not this Skill's).
2. General question-answering unrelated to a specific, already-identified study topic (e.g. a user asking "what's a good laptop for programming?" — nothing here anchors to source material for a topic already in the outline).
3. Grading or generating content for anything other than the exact `{question_text, grading_criteria, source_url}` schema below — this Skill does not draft outline topics, day-by-day content, or patch-notes.

## Input — question generation

```json
{
  "topic_source_material": "text of the specific source this topic is grounded in",
  "source_url": "the same source_url already attached to this outline topic",
  "num_questions": 5,
  "previous_question_texts": []
}
```

- `num_questions` is `5` for an initial set or a test-out check, or `1` for a targeted retry.
- `previous_question_texts` lists every question already asked for this same slot — required for a retry, so the generator can guarantee the new question is not a repeat of any prior one. Empty for an initial set (nothing to avoid repeating yet).
- Both `topic_source_material` and `source_url` must be non-empty, and `num_questions` must be at least 1 — this Skill refuses to fire (raises) on empty/invalid input rather than fabricating questions from nothing.

## Output — question generation (schema-validated)

```json
{
  "questions": [
    {
      "question_text": "string",
      "grading_criteria": "string — the specific rubric a grader checks the answer against, not just the answer itself",
      "source_url": "string — always the input source_url, verbatim; never invented or altered by generation"
    }
  ]
}
```

- Exactly `num_questions` objects are always returned.
- Every `source_url` is the caller's input `source_url`, attached directly by this Skill's own code — never something the LLM produces or could alter (CLAUDE.md guardrail #1: never fabricate a source). A malformed or incomplete question object (missing `question_text`/`grading_criteria`, or an empty value) raises rather than being silently accepted.
- `grading_criteria` must be specific enough for strict pass/fail grading (PRD §7.7) — not just a restated answer, but what must be present for correctness.
- Every returned question is genuinely distinct from every entry in `previous_question_texts` — enforced structurally (an exact-match check on the generated text), not merely requested via the prompt.

## Input/Output — answer grading

```json
// input
{
  "question_text": "string",
  "grading_criteria": "string",
  "user_answer": "string"
}
// output
{
  "passed": true
}
```

Grading is strict pass/fail — no partial-credit ambiguity at this level. Half-credit is an orchestration-level concept (PRD §7.7), applied by the caller only after 3 failed attempts; this Skill never computes it.

## Trigger examples

**Positive (this Skill should fire):**
1. "The user just finished reading the material for the 'Git Branching' topic — generate the initial 5 verification questions."
2. "The user answered question 3 incorrectly on attempt 1 — generate one fresh retry question covering the same concept, making sure it's not the same question as before."
3. "The user wants to test out of the 'SQL Joins' topic before starting it — run the verification check now."

**Negative (this Skill should NOT fire):**
1. "What skills does a Backend Engineer need in 2026?" — market-data grounding, not verification of a specific already-studied topic (belongs to the Research Agent).
2. "Can you recommend a good book on system design?" — general Q&A unrelated to a specific study topic already in the user's outline.
3. "Write today's hands-on exercise for the 'Python Functions' topic." — day-by-day content generation (Agent 2's job), not comprehension verification.

## Retry cap and de-escalation (caller responsibility, not this Skill's)

The 3-attempt retry cap and half-credit taught-answer de-escalation (PRD §7.7) are orchestrated by the Coaching & Pace Agent, which calls this Skill fresh for each retry. This Skill has no memory of attempt count — it only guarantees non-repetition given `previous_question_texts`.

## Evaluation

See `/evaluation/golden_dataset.json` and `/evaluation/eval_cases.json` for EDD-style positive/negative trigger cases and representative (source → expected question shape) pairs, per Architecture §7. Written before finalizing this Skill's implementation, per the course's inversion-path guidance.
