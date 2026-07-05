"""Verification Question Generator — an Agent Skill (Architecture §4),
packaged here rather than as inline agent logic, per CLAUDE.md's repo
structure ("Skills live in `.agent/skills/`, not `src/` — required for
recognition by the Antigravity workspace manager").

Generates source-anchored comprehension questions + grading criteria
from study material, and grades a user's answer against a question's
grading_criteria at strict pass/fail (PRD §7.7).

**Stateless per call.** This module has no memory of prior attempts —
retry-cap counting (exactly 3), attempt-number tracking, and half-credit
de-escalation are orchestration-level concerns (`coaching_pace_agent.py`,
a later task, not built here). That caller invokes this module fresh for
every attempt and is responsible for passing whatever context (e.g.
`previous_question_texts`) this module needs to do its job correctly —
statelessness does not make freshness automatic on its own.

Reuses `agents/research_outline_agent.py`'s existing Gemini call/timeout/
error-handling helper (`_call_gemini_json`, plus `GeminiCallError`)
directly rather than duplicating that logic, per this
task's explicit instruction. This does mean importing underscore-
prefixed (module-private-by-convention) names across a real package
boundary (`.agent/skills/` -> `src/agents/`) — a real architectural seam,
not an oversight: a full extraction of that Gemini-call infrastructure
into a shared `src/utils/` module would be the cleaner long-term fix, but
this task's scope is the Skill itself, not a refactor of already-tested,
already-committed Agent code. Flagged for a future promotion, same as
this project's other flagged-not-solved seams.

`PROMPT_REGISTRY` here is this module's own, separate from
`agents/research_outline_agent.py`'s — CLAUDE.md's LLM Call Discipline
says "a module-level PROMPT_REGISTRY", not one project-wide singleton,
and this Skill is architecturally independent (Architecture §4: "a real
Skill artifact... not inline agent logic").
"""

from dataclasses import dataclass
from typing import Any

from agents.research_outline_agent import _call_gemini_json
from utils.exceptions import GeminiCallError

# Judgment call: "flash" tier — question generation from a single piece
# of source material and single-answer grading are both short, bounded
# tasks, closer in shape to the clarify gate's per-turn calls than to
# outline-hierarchy sequencing's whole-curriculum reasoning. Kept as this
# Skill's own constant rather than importing
# `agents.research_outline_agent.SHORT_TURN_GEMINI_MODEL` — an
# independent choice for an independent artifact, even though the value
# happens to match today.
VERIFICATION_GEMINI_MODEL = "gemini-2.5-flash"

# CLAUDE.md's LLM Call Discipline: every prompt used for grounded or
# safety-critical generation is versioned here, never deleted once its
# baseline regression test (tests/test_verification_skill.py) locks it
# in — a version is frozen prose, not something later tasks may edit in
# place; a changed prompt gets a new "_v2" key instead.
PROMPT_REGISTRY: dict[str, str] = {
    "verification_question_generation_v1": (
        "Generate exactly {num_questions} comprehension question(s) that "
        "test understanding of the study material below. Each question "
        "must be answerable ONLY using the material given — do not test "
        "outside knowledge or facts not present in the material.\n\n"
        "Study material:\n{source_material}\n\n"
        "{previous_questions}"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"questions": [{{"question_text": "<question>", '
        '"grading_criteria": "<specific rubric describing exactly what '
        "must be present in a correct answer, not just a restated "
        'answer>"}}, ...]}}. Return exactly {num_questions} question(s).'
    ),
    "verification_answer_grading_v1": (
        "Grade a user's answer to a comprehension question strictly "
        "pass/fail, with no partial credit.\n\n"
        "Question: {question_text!r}\n"
        "Grading criteria: {grading_criteria!r}\n"
        "User's answer: {user_answer!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"passed": true or false}}. Pass only if the answer clearly '
        "satisfies the grading criteria; when genuinely ambiguous, fail "
        "rather than pass."
    ),
}


class SchemaValidationError(Exception):
    """Raised when a generated question object is missing a required
    field or has an empty/invalid value — never silently accepted. This
    Skill's output schema (`{question_text, grading_criteria,
    source_url}`) has no confidence-tier concept the way outline topics/
    grounding results do, so `security/output_guard.py`'s
    `validate_output_object` doesn't apply directly; this is a
    purpose-built validator for this Skill's own schema, still enforcing
    a non-empty `source_url` per CLAUDE.md guardrail #1.
    """


@dataclass(frozen=True)
class VerificationQuestion:
    """One verification question — this Skill's entire output schema
    (Architecture §4), one instance per requested question."""

    question_text: str
    grading_criteria: str
    source_url: str


def _validate_question_object(candidate: dict[str, Any]) -> VerificationQuestion:
    """Validate a single candidate question object against this Skill's
    schema. Raises `SchemaValidationError` if any required field is
    missing or empty — never returns a partially-valid object for the
    caller to guess at.
    """
    question_text = candidate.get("question_text")
    if not isinstance(question_text, str) or not question_text.strip():
        raise SchemaValidationError(
            f"candidate is missing a non-empty 'question_text': {candidate!r}"
        )
    grading_criteria = candidate.get("grading_criteria")
    if not isinstance(grading_criteria, str) or not grading_criteria.strip():
        raise SchemaValidationError(
            f"candidate is missing a non-empty 'grading_criteria': {candidate!r}"
        )
    source_url = candidate.get("source_url")
    if not isinstance(source_url, str) or not source_url.strip():
        raise SchemaValidationError(
            f"candidate is missing a non-empty 'source_url': {candidate!r}"
        )
    return VerificationQuestion(
        question_text=question_text,
        grading_criteria=grading_criteria,
        source_url=source_url,
    )


def _format_previous_questions(previous_question_texts: list[str] | None) -> str:
    if not previous_question_texts:
        return ""
    listed = "\n".join(f"- {q}" for q in previous_question_texts)
    return (
        "This is a retry. The following question(s) were already asked "
        "for this same slot — the new question must cover the same "
        "underlying concept but be genuinely different, never a "
        "reworded repeat:\n"
        f"{listed}\n\n"
    )


async def generate_questions(
    topic_source_material: str,
    source_url: str,
    num_questions: int,
    previous_question_texts: list[str] | None = None,
) -> list[VerificationQuestion]:
    """Generate `num_questions` source-anchored comprehension questions +
    grading criteria from `topic_source_material` (PRD §7.7).

    Every returned question's `source_url` is `source_url` verbatim,
    attached by this function directly — never trusted from the LLM's
    own output. Gemini's JSON response never even contains a
    `source_url` field to invent, drop, or alter (CLAUDE.md guardrail
    #1), the same structural sourcing-safety split
    `agents/research_outline_agent.py`'s `create_initial_outline` already
    uses, applied here to a different schema.

    Raises `ValueError` if `topic_source_material`/`source_url` is blank
    or `num_questions` is less than 1 — this Skill must refuse to fire on
    empty/invalid input (Architecture §7's negative trigger case) rather
    than fabricate questions from nothing.

    If `previous_question_texts` is given (the retry case), raises
    `GeminiCallError` if any returned question is an exact (case-folded)
    repeat of one already asked, or of another question in this same
    batch — enforced structurally, not merely requested via the prompt.
    True semantic-similarity detection (e.g. embeddings, beyond exact
    match) is explicitly out of scope for this task; see spec
    reconciliation.
    """
    if not topic_source_material or not topic_source_material.strip():
        raise ValueError("topic_source_material must be non-empty")
    if not source_url or not source_url.strip():
        raise ValueError("source_url must be non-empty")
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")

    prompt = PROMPT_REGISTRY["verification_question_generation_v1"].format(
        source_material=topic_source_material,
        num_questions=num_questions,
        previous_questions=_format_previous_questions(previous_question_texts),
    )
    parsed = await _call_gemini_json(
        prompt, required_keys={"questions"}, model=VERIFICATION_GEMINI_MODEL
    )
    raw_questions = parsed["questions"]
    if not isinstance(raw_questions, list) or len(raw_questions) != num_questions:
        raise GeminiCallError(
            f"expected exactly {num_questions} question(s), got: {raw_questions!r}"
        )

    previous_folded = {q.strip().casefold() for q in (previous_question_texts or [])}
    seen_in_batch: set[str] = set()
    questions: list[VerificationQuestion] = []
    for raw in raw_questions:
        if not isinstance(raw, dict):
            raise GeminiCallError(f"malformed question entry: {raw!r}")
        candidate = {**raw, "source_url": source_url}
        try:
            question = _validate_question_object(candidate)
        except SchemaValidationError as exc:
            raise GeminiCallError(str(exc)) from exc

        folded = question.question_text.strip().casefold()
        if folded in previous_folded:
            raise GeminiCallError(
                "generated question repeats a previous question verbatim: "
                f"{question.question_text!r}"
            )
        if folded in seen_in_batch:
            raise GeminiCallError(
                "generated question duplicates another question in this "
                f"same batch: {question.question_text!r}"
            )
        seen_in_batch.add(folded)
        questions.append(question)

    return questions


async def grade_answer(question: VerificationQuestion, user_answer: str) -> bool:
    """Grade `user_answer` against `question.grading_criteria` at strict
    pass/fail — no partial credit (PRD §7.7). Half-credit is an
    orchestration-level concept applied by the caller only after 3
    failed attempts; this function never computes it.

    Raises `GeminiCallError` if Gemini's response is malformed or
    `passed` is not a genuine boolean.
    """
    prompt = PROMPT_REGISTRY["verification_answer_grading_v1"].format(
        question_text=question.question_text,
        grading_criteria=question.grading_criteria,
        user_answer=user_answer,
    )
    parsed = await _call_gemini_json(
        prompt, required_keys={"passed"}, model=VERIFICATION_GEMINI_MODEL
    )
    passed = parsed["passed"]
    if not isinstance(passed, bool):
        raise GeminiCallError(
            f"Gemini grading response 'passed' was not a boolean: {parsed!r}"
        )
    return passed
