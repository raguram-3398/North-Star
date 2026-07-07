"""Standalone, stateless skill that generates source-anchored comprehension questions and grades answers against their criteria."""

from dataclasses import dataclass
from typing import Any

from google.adk.agents import LlmAgent

from utils.adk_runtime import build_retry_config, call_agent_json, json_response_config
from utils.exceptions import GeminiCallError

VERIFICATION_GEMINI_MODEL = "gemini-2.5-flash"

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


_question_generation_agent = LlmAgent(
    name="verification_question_generation_agent",
    model=VERIFICATION_GEMINI_MODEL,
    instruction=(
        "Generate source-anchored comprehension question(s) answerable "
        "only from the given study material, with a grading rubric per "
        "question."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

_answer_grading_agent = LlmAgent(
    name="verification_answer_grading_agent",
    model=VERIFICATION_GEMINI_MODEL,
    instruction=(
        "Grade a user's answer to a comprehension question strictly "
        "pass/fail against its grading criteria, no partial credit."
    ),
    generate_content_config=json_response_config(),
    retry_config=build_retry_config(),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)


class SchemaValidationError(Exception):
    """Raised when a generated question object is missing a required field or has an empty or invalid value."""


@dataclass(frozen=True)
class VerificationQuestion:
    """One verification question — this skill's entire output schema, one instance per requested question."""

    question_text: str
    grading_criteria: str
    source_url: str


def _validate_question_object(candidate: dict[str, Any]) -> VerificationQuestion:
    """Validate a single candidate question object against this skill's schema, raising an error if any required field is missing or empty."""
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
    """Generate a batch of source-anchored comprehension questions and grading criteria from study material, rejecting exact repeats."""
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
    parsed = await call_agent_json(
        _question_generation_agent, prompt, required_keys={"questions"}
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
    """Grade a user's answer against a question's grading criteria at strict pass/fail, with no partial credit."""
    prompt = PROMPT_REGISTRY["verification_answer_grading_v1"].format(
        question_text=question.question_text,
        grading_criteria=question.grading_criteria,
        user_answer=user_answer,
    )
    parsed = await call_agent_json(
        _answer_grading_agent, prompt, required_keys={"passed"}
    )
    passed = parsed["passed"]
    if not isinstance(passed, bool):
        raise GeminiCallError(
            f"Gemini grading response 'passed' was not a boolean: {parsed!r}"
        )
    return passed
