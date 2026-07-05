"""verification_attempts I/O.

Per Architecture_North_Star.md §5: the canonical, append-only record of
every verification attempt (one row per question, per attempt, up to 3
per question slot) — feeds `topic_score` via `pace/calculator.py`'s
`calculate_topic_score`. This module only reads and writes rows; the
retry-cap state machine, credit assignment, and freshness enforcement
belong to `agents/coaching_pace_agent.py` and
`.agent/skills/verification_question_generator/generator.py`
respectively — never reimplemented here.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern — testable with a mocked
Session, no real database required.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from models.schemas import VerificationAttempt


def write_verification_attempt(
    session: Session,
    topic_id: str,
    question_number: int,
    attempt_number: int,
    question_text: str,
    grading_criteria: str,
    user_answer: str,
    passed: bool,
    credit: float,
    is_test_out: bool = False,
) -> None:
    """Append one verification-attempt row. Every attempt is logged,
    whether it passed, failed-but-not-final, or failed at the retry cap
    — this is an append-only log, never an upsert (each attempt is a
    genuinely new row, never overwriting a prior attempt for the same
    question_number).

    `created_at` is stamped here as naive UTC, matching
    `data/roles_cache.py`'s established timestamp convention. Commits the
    transaction.
    """
    now = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    session.add(
        VerificationAttempt(
            topic_id=topic_id,
            question_number=question_number,
            attempt_number=attempt_number,
            question_text=question_text,
            grading_criteria=grading_criteria,
            user_answer=user_answer,
            passed=passed,
            credit=credit,
            is_test_out=is_test_out,
            created_at=now,
        )
    )
    session.commit()


def get_attempts_for_topic(session: Session, topic_id: str) -> list[dict[str, Any]]:
    """Read every recorded verification attempt for a topic, ordered by
    question_number then attempt_number — used to reconstruct a topic's
    per-question credit history (e.g. for `pace/calculator.py`'s
    `calculate_topic_score`, or an audit view).
    """
    rows = (
        session.query(VerificationAttempt)
        .filter(VerificationAttempt.topic_id == topic_id)
        .order_by(
            VerificationAttempt.question_number, VerificationAttempt.attempt_number
        )
        .all()
    )
    return [
        {
            "question_number": row.question_number,
            "attempt_number": row.attempt_number,
            "question_text": row.question_text,
            "grading_criteria": row.grading_criteria,
            "user_answer": row.user_answer,
            "passed": row.passed,
            "credit": row.credit,
            "is_test_out": row.is_test_out,
            "created_at": row.created_at,
        }
        for row in rows
    ]
