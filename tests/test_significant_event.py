"""Tests for outline/significant_event.py — bucket/confidence-crossing
diff between roles_cache snapshots.
"""

from outline.significant_event import SkillBucket, SkillSnapshot, is_significant_event
from security.output_guard import ConfidenceTier


def test_absent_to_emerging_is_significant() -> None:
    old = SkillSnapshot(bucket=SkillBucket.ABSENT)
    new = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.LOW
    )

    assert is_significant_event(old, new) is True


def test_emerging_to_core_is_significant() -> None:
    old = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.MEDIUM
    )
    new = SkillSnapshot(
        bucket=SkillBucket.CORE_SKILLS, confidence=ConfidenceTier.MEDIUM
    )

    assert is_significant_event(old, new) is True


def test_confidence_strengthens_within_same_bucket_is_significant() -> None:
    old = SkillSnapshot(
        bucket=SkillBucket.CORE_SKILLS, confidence=ConfidenceTier.MEDIUM
    )
    new = SkillSnapshot(bucket=SkillBucket.CORE_SKILLS, confidence=ConfidenceTier.HIGH)

    assert is_significant_event(old, new) is True


def test_core_to_emerging_is_discarded_not_significant() -> None:
    old = SkillSnapshot(bucket=SkillBucket.CORE_SKILLS, confidence=ConfidenceTier.HIGH)
    new = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.HIGH
    )

    assert is_significant_event(old, new) is False


def test_bucket_decrease_with_confidence_increase_is_not_significant() -> None:
    """Bucket rank must be checked first and short-circuit before any
    confidence comparison: a bucket downgrade is never significant, even
    when paired with a confidence *increase* — the one combination where
    a naive confidence-only comparison would get it wrong.
    """
    old = SkillSnapshot(bucket=SkillBucket.CORE_SKILLS, confidence=ConfidenceTier.LOW)
    new = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.HIGH
    )

    assert is_significant_event(old, new) is False


def test_emerging_to_absent_is_discarded_not_significant() -> None:
    old = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.LOW
    )
    new = SkillSnapshot(bucket=SkillBucket.ABSENT)

    assert is_significant_event(old, new) is False


def test_no_change_is_not_significant() -> None:
    old = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.MEDIUM
    )
    new = SkillSnapshot(
        bucket=SkillBucket.EMERGING_SKILLS, confidence=ConfidenceTier.MEDIUM
    )

    assert is_significant_event(old, new) is False
