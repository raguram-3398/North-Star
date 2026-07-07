"""Append-only record of every verification attempt, feeding topic-score computation."""

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
    """Append one verification-attempt row to the log."""
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
    """Read every recorded verification attempt for a topic, ordered by question number then attempt number."""
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
