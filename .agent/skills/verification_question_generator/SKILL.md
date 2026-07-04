---
name: verification_question_generator
description: Generate source-anchored comprehension questions with grading criteria from study material. Use when a topic needs verification questions — the initial 5-question set for a topic, a single fresh retry question (never repeats an identical question), or a test-out check performed before study content is generated. Do not use for market-data grounding, general Q&A unrelated to a specific study topic, or any content-generation task outside comprehension verification.
---

# Verification Question Generator

## Purpose

Produces strict, source-anchored comprehension questions and their grading criteria from a single piece of study material. This is the only mechanism in North Star that creates or grades verification content — it is invoked identically whether the call is the initial 5-question set for a topic, a fresh retry question after a failed attempt, or a test-out pre-check.

## When to use

- A topic is about to be studied for the first time and needs its initial 5-question verification set
- A user failed a specific question and needs a **fresh** question covering the same underlying concept (never the identical question — see Architecture §11 guardrails)
- A user invokes "test out" on a topic and needs the verification check run *before* any study content is generated

## When NOT to use

- Market-data grounding, skill extraction from job descriptions, or any Research Agent task
- General question-answering unrelated to a specific, already-identified study topic
- Anything requiring output other than the schema below

## Input

```json
{
  "topic_source_material": "text or URL of the specific source this topic is grounded in",
  "source_url": "the same source_url already attached to this outline topic",
  "num_questions": 5,
  "mode": "initial | retry | test_out"
}
```

For `retry`, also pass `previous_question_texts: [string]` so the generator can guarantee the new question is not a repeat of any prior one for this topic/question-slot.

## Output (schema-validated — reject and retry generation if malformed)

```json
{
  "questions": [
    {
      "question_text": "string",
      "grading_criteria": "string — the specific rubric a grader checks the answer against, not just the answer itself",
      "source_url": "string — must match or be a specific anchor within topic_source_material"
    }
  ]
}
```

- `num_questions` objects are always returned
- Every `source_url` must be non-empty and trace to the input source material — missing or fabricated sources fail validation in `security/output_guard.py` and must not reach the caller
- `grading_criteria` must be specific enough for strict pass/fail grading (per PRD §7.7) — not just a restated answer, but what must be present for correctness

## Grading contract (used by the caller, not performed by this skill)

This skill generates questions and rubrics; it does not itself grade user answers. Grading is strict pass/fail, no partial credit ambiguity, performed by the calling agent against the returned `grading_criteria`.

## Retry cap and de-escalation (caller responsibility, not this skill's)

The 3-attempt retry cap and half-credit taught-answer de-escalation (PRD §7.7) are orchestrated by the Coaching & Pace Agent, which calls this skill fresh for each retry. This skill has no memory of attempt count — it only guarantees non-repetition given `previous_question_texts`.

## Evaluation

See `/evaluation/golden_dataset.json` and `/evaluation/eval_cases.json` for EDD-style positive/negative trigger cases and representative (source → expected question shape) pairs, per Architecture §7.
