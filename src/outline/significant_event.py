"""Detects whether a skill's bucket/confidence crossing between two snapshots counts as a significant event."""

from dataclasses import dataclass
from enum import Enum

from security.output_guard import ConfidenceTier


class SkillBucket(Enum):
    """A skill's membership bucket, in ascending order of market-demand strength."""

    ABSENT = "absent"
    EMERGING_SKILLS = "emerging_skills"
    CORE_SKILLS = "core_skills"


_BUCKET_RANK: dict[SkillBucket, int] = {
    SkillBucket.ABSENT: 0,
    SkillBucket.EMERGING_SKILLS: 1,
    SkillBucket.CORE_SKILLS: 2,
}

_CONFIDENCE_RANK: dict[ConfidenceTier, int] = {
    ConfidenceTier.GENERAL_KNOWLEDGE_ONLY: 0,
    ConfidenceTier.CACHED_LOW: 1,
    ConfidenceTier.LOW: 2,
    ConfidenceTier.MEDIUM: 3,
    ConfidenceTier.HIGH: 4,
}


@dataclass(frozen=True)
class SkillSnapshot:
    """A single skill's bucket membership and confidence tier at one point in time."""

    bucket: SkillBucket
    confidence: ConfidenceTier | None = None


def is_significant_event(old: SkillSnapshot, new: SkillSnapshot) -> bool:
    """Return True only if the skill crosses a bucket or confidence boundary upward from old to new."""
    old_rank = _BUCKET_RANK[old.bucket]
    new_rank = _BUCKET_RANK[new.bucket]

    if new_rank > old_rank:
        return True
    if new_rank < old_rank:
        return False

    if new.bucket is SkillBucket.ABSENT:
        return False

    assert old.confidence is not None
    assert new.confidence is not None
    return _CONFIDENCE_RANK[new.confidence] > _CONFIDENCE_RANK[old.confidence]
