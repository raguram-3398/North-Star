"""Tests for the Verification Question Generator Skill
(.agent/skills/verification_question_generator/) — source material in,
fresh source-anchored questions + grading rubric out, plus strict
pass/fail grading of a user's answer.

`.agent/skills/` sits outside the `src/` package (required for Antigravity
workspace-manager recognition, per CLAUDE.md), so it isn't importable via
the normal editable-install path the way `src/` is — this file adds
`.agent/skills/` to `sys.path` explicitly before importing the Skill as
`verification_question_generator.generator` (an implicit namespace
package; no `__init__.py` needed, Python 3.3+).

Gemini is mocked by reusing tests/test_gemini_client.py's existing
`_patch_gemini`/`_FakeGeminiClient` fakes rather than duplicating them —
the Skill's `generate_questions`/`grade_answer` call
`utils.gemini_client._call_gemini_json` directly (imported by name), so
patching `utils.gemini_client._get_gemini_client` (the module where that
call actually resolves the client from) is the correct target, exactly as
`_patch_gemini` already does — not `generator._get_gemini_client`, which
doesn't exist as a name in this module's own namespace at all.
"""

import json
import sys
from pathlib import Path

import pytest

_SKILLS_DIR = Path(__file__).resolve().parent.parent / ".agent" / "skills"
if str(_SKILLS_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILLS_DIR))

from verification_question_generator import generator  # noqa: E402

from tests.test_gemini_client import _patch_gemini  # noqa: E402
from utils.exceptions import GeminiCallError  # noqa: E402

EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"

GIT_SOURCE_MATERIAL = (
    "Git is a distributed version control system that tracks changes in "
    "source code during software development. Unlike centralized version "
    "control systems, every developer's working copy of the code is also "
    "a repository that contains the full history of all changes."
)
GIT_SOURCE_URL = "https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control"


def _questions_response(count: int, prefix: str = "Question") -> str:
    """A synthetic, valid Gemini question-generation response with
    `count` distinct questions — used wherever the exact question text
    doesn't matter, only the shape/count/processing.
    """
    return json.dumps(
        {
            "questions": [
                {
                    "question_text": f"{prefix} {i}: what does concept {i} mean?",
                    "grading_criteria": f"Answer must explain concept {i} correctly.",
                }
                for i in range(1, count + 1)
            ]
        }
    )


# --- generate_questions: count per mode -----------------------------------


async def test_generate_questions_returns_five_for_initial_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[_questions_response(5)])

    questions = await generator.generate_questions(
        GIT_SOURCE_MATERIAL, GIT_SOURCE_URL, num_questions=5
    )

    assert len(questions) == 5
    for question in questions:
        assert isinstance(question, generator.VerificationQuestion)
        assert question.source_url == GIT_SOURCE_URL
        assert question.question_text
        assert question.grading_criteria


async def test_generate_questions_returns_one_for_targeted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[_questions_response(1)])

    questions = await generator.generate_questions(
        GIT_SOURCE_MATERIAL,
        GIT_SOURCE_URL,
        num_questions=1,
        previous_question_texts=["An older, already-asked question."],
    )

    assert len(questions) == 1
    assert questions[0].source_url == GIT_SOURCE_URL


async def test_generate_questions_raises_when_response_count_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gemini(monkeypatch, responses=[_questions_response(3)])

    with pytest.raises(GeminiCallError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL, GIT_SOURCE_URL, num_questions=5
        )


# --- freshness: never repeat a previous question, never dupe in-batch -----


async def test_regenerated_question_is_genuinely_different_from_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = "What makes Git 'distributed' rather than centralized?"
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question_text": "What is the staging area in Git?",
                            "grading_criteria": "Must mention reviewing changes "
                            "before commit.",
                        }
                    ]
                }
            )
        ],
    )

    questions = await generator.generate_questions(
        GIT_SOURCE_MATERIAL,
        GIT_SOURCE_URL,
        num_questions=1,
        previous_question_texts=[previous],
    )

    assert questions[0].question_text != previous


async def test_generate_questions_raises_on_verbatim_repeat_of_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = "What makes Git 'distributed' rather than centralized?"
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question_text": previous,
                            "grading_criteria": "Anything.",
                        }
                    ]
                }
            )
        ],
    )

    with pytest.raises(GeminiCallError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL,
            GIT_SOURCE_URL,
            num_questions=1,
            previous_question_texts=[previous],
        )


async def test_generate_questions_raises_on_repeat_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The freshness check is case-insensitive — a trivially re-cased
    repeat still counts as identical, not 'genuinely different'."""
    previous = "What makes Git 'distributed' rather than centralized?"
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question_text": previous.upper(),
                            "grading_criteria": "Anything.",
                        }
                    ]
                }
            )
        ],
    )

    with pytest.raises(GeminiCallError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL,
            GIT_SOURCE_URL,
            num_questions=1,
            previous_question_texts=[previous],
        )


async def test_generate_questions_raises_on_duplicate_within_same_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_text = "What is a commit in Git?"
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question_text": duplicate_text,
                            "grading_criteria": "Must define a commit.",
                        },
                        {
                            "question_text": duplicate_text,
                            "grading_criteria": "Must define a commit.",
                        },
                    ]
                }
            )
        ],
    )

    with pytest.raises(GeminiCallError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL, GIT_SOURCE_URL, num_questions=2
        )


# --- schema validation: malformed/incomplete questions must raise --------


def test_validate_question_object_rejects_missing_source_url() -> None:
    with pytest.raises(generator.SchemaValidationError):
        generator._validate_question_object(
            {"question_text": "A question?", "grading_criteria": "A rubric."}
        )


def test_validate_question_object_rejects_missing_question_text() -> None:
    with pytest.raises(generator.SchemaValidationError):
        generator._validate_question_object(
            {"grading_criteria": "A rubric.", "source_url": "https://example.com"}
        )


def test_validate_question_object_rejects_empty_grading_criteria() -> None:
    with pytest.raises(generator.SchemaValidationError):
        generator._validate_question_object(
            {
                "question_text": "A question?",
                "grading_criteria": "   ",
                "source_url": "https://example.com",
            }
        )


async def test_generate_questions_raises_on_malformed_entry_from_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed entry from Gemini (missing grading_criteria) must
    raise rather than silently reach the caller — the schema-validation
    failure is wrapped as GeminiCallError since it originates from the
    LLM's own response, not caller-supplied input."""
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps({"questions": [{"question_text": "A question with no rubric?"}]})
        ],
    )

    with pytest.raises(GeminiCallError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL, GIT_SOURCE_URL, num_questions=1
        )


# --- negative input: refuse to fire on empty/invalid input ---------------


async def test_generate_questions_raises_on_empty_source_material() -> None:
    with pytest.raises(ValueError):
        await generator.generate_questions("", GIT_SOURCE_URL, num_questions=5)


async def test_generate_questions_raises_on_empty_source_url() -> None:
    with pytest.raises(ValueError):
        await generator.generate_questions(GIT_SOURCE_MATERIAL, "", num_questions=5)


async def test_generate_questions_raises_on_non_positive_num_questions() -> None:
    with pytest.raises(ValueError):
        await generator.generate_questions(
            GIT_SOURCE_MATERIAL, GIT_SOURCE_URL, num_questions=0
        )


# --- grading: strict pass/fail, no ambiguous middle state -----------------


async def test_grade_answer_strict_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gemini(monkeypatch, responses=[json.dumps({"passed": True})])
    question = generator.VerificationQuestion(
        question_text="What is a commit?",
        grading_criteria="Must mention it's a snapshot of changes.",
        source_url=GIT_SOURCE_URL,
    )

    result = await generator.grade_answer(question, "A commit is a saved snapshot.")

    assert result is True
    assert isinstance(result, bool)


async def test_grade_answer_strict_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gemini(monkeypatch, responses=[json.dumps({"passed": False})])
    question = generator.VerificationQuestion(
        question_text="What is a commit?",
        grading_criteria="Must mention it's a snapshot of changes.",
        source_url=GIT_SOURCE_URL,
    )

    result = await generator.grade_answer(question, "I don't know.")

    assert result is False
    assert isinstance(result, bool)


async def test_grade_answer_raises_on_non_boolean_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ambiguous middle state: if Gemini's 'passed' isn't a genuine
    boolean (e.g. a string like 'maybe'), this must raise, not coerce it
    into some third state."""
    _patch_gemini(monkeypatch, responses=[json.dumps({"passed": "maybe"})])
    question = generator.VerificationQuestion(
        question_text="What is a commit?",
        grading_criteria="Must mention it's a snapshot of changes.",
        source_url=GIT_SOURCE_URL,
    )

    with pytest.raises(GeminiCallError):
        await generator.grade_answer(question, "Some answer.")


# --- PROMPT_REGISTRY: baseline regression asserts on the prompt string ----
# --- itself, not the LLM's output (CLAUDE.md's explicit instruction) -----


def test_verification_question_generation_prompt_v1_is_frozen() -> None:
    assert generator.PROMPT_REGISTRY["verification_question_generation_v1"] == (
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
    )


def test_verification_answer_grading_prompt_v1_is_frozen() -> None:
    assert generator.PROMPT_REGISTRY["verification_answer_grading_v1"] == (
        "Grade a user's answer to a comprehension question strictly "
        "pass/fail, with no partial credit.\n\n"
        "Question: {question_text!r}\n"
        "Grading criteria: {grading_criteria!r}\n"
        "User's answer: {user_answer!r}\n\n"
        "Respond with ONLY a JSON object matching this shape: "
        '{{"passed": true or false}}. Pass only if the answer clearly '
        "satisfies the grading criteria; when genuinely ambiguous, fail "
        "rather than pass."
    )


# --- EDD: evaluation/eval_cases.json exercised as real tests --------------


def _load_eval_cases() -> list[dict]:
    with open(EVALUATION_DIR / "eval_cases.json") as f:
        return json.load(f)


EVAL_CASES = _load_eval_cases()


def test_eval_cases_file_has_exactly_two_positive_and_one_negative() -> None:
    positive = [c for c in EVAL_CASES if c["type"] == "positive"]
    negative = [c for c in EVAL_CASES if c["type"] == "negative"]
    assert len(positive) == 2
    assert len(negative) == 1


async def test_eval_case_positive_initial_five_question_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        c for c in EVAL_CASES if c["case_id"] == "positive_initial_five_question_set"
    )
    inputs = case["input"]
    expected = case["expected"]
    _patch_gemini(monkeypatch, responses=[_questions_response(inputs["num_questions"])])

    questions = await generator.generate_questions(
        inputs["topic_source_material"],
        inputs["source_url"],
        inputs["num_questions"],
        inputs["previous_question_texts"],
    )

    assert len(questions) == expected["question_count"]
    assert (
        all(q.source_url for q in questions)
        == expected["every_question_has_non_empty_source_url"]
    )
    assert (
        all(q.source_url == inputs["source_url"] for q in questions)
        == expected["every_question_source_url_equals_input_source_url"]
    )


async def test_eval_case_positive_targeted_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = next(
        c
        for c in EVAL_CASES
        if c["case_id"] == "positive_targeted_retry_single_fresh_question"
    )
    inputs = case["input"]
    expected = case["expected"]
    _patch_gemini(
        monkeypatch,
        responses=[
            json.dumps(
                {
                    "questions": [
                        {
                            "question_text": "How does Git's staging area work?",
                            "grading_criteria": "Must mention reviewing/"
                            "formatting changes before commit.",
                        }
                    ]
                }
            )
        ],
    )

    questions = await generator.generate_questions(
        inputs["topic_source_material"],
        inputs["source_url"],
        inputs["num_questions"],
        inputs["previous_question_texts"],
    )

    assert len(questions) == expected["question_count"]
    differs = all(
        questions[0].question_text.strip().casefold() != prev.strip().casefold()
        for prev in inputs["previous_question_texts"]
    )
    assert differs == expected["question_text_differs_from_all_previous"]
    assert (
        bool(questions[0].source_url)
        == expected["every_question_has_non_empty_source_url"]
    )


async def test_eval_case_negative_empty_source_material_does_not_fire() -> None:
    case = next(
        c for c in EVAL_CASES if c["case_id"] == "negative_empty_source_material"
    )
    inputs = case["input"]
    expected = case["expected"]
    assert expected["should_fire"] is False

    with pytest.raises(ValueError):
        await generator.generate_questions(
            inputs["topic_source_material"],
            inputs["source_url"],
            inputs["num_questions"],
            inputs["previous_question_texts"],
        )


# --- Golden dataset: representative (source -> expected shape) pairs -----


def _load_golden_dataset() -> list[dict]:
    with open(EVALUATION_DIR / "golden_dataset.json") as f:
        return json.load(f)


GOLDEN_DATASET = _load_golden_dataset()


def test_golden_dataset_has_five_to_ten_entries() -> None:
    assert 5 <= len(GOLDEN_DATASET) <= 10


@pytest.mark.parametrize("entry", GOLDEN_DATASET, ids=[e["id"] for e in GOLDEN_DATASET])
async def test_golden_dataset_entry_matches_expected_shape(
    entry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    shape = entry["expected_question_shape"]
    _patch_gemini(
        monkeypatch,
        responses=[_questions_response(entry["num_questions"], prefix=entry["id"])],
    )

    questions = await generator.generate_questions(
        entry["topic_source_material"], entry["source_url"], entry["num_questions"]
    )

    assert len(questions) == shape["count"]
    for question in questions:
        assert question.question_text
        assert question.grading_criteria
        assert question.source_url
        if shape["source_url_equals_input"]:
            assert question.source_url == entry["source_url"]
