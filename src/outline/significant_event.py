"""Significant-event detection — bucket/confidence-crossing diff.

Pure, deterministic, no LLM calls, no DB reads, no roles_cache I/O
(CLAUDE.md: pure functions stay pure) — takes two already-fetched skill
snapshots and returns whether the crossing between them is a significant
event. The caller (Agent 1 or src/cron/refresh_roles.py) is responsible
for fetching the old/new snapshots from roles_cache and passing them in.

Per Architecture_North_Star.md §9: for a single skill, compare bucket
membership (absent / emerging_skills / core_skills) and confidence tier
between the old and new roles_cache snapshot. Any *upward* crossing is a
significant event. Downward crossings are diffed but discarded — content
is never removed (PRD §7.4, CLAUDE.md guardrail #2), so a downgrade
triggers no action.
"""

from dataclasses import dataclass
from enum import Enum

from security.output_guard import ConfidenceTier


class SkillBucket(Enum):
    """A skill's membership bucket within a role's roles_cache entry, in
    ascending order of market-demand strength.
    """

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
    """A single skill's bucket membership and confidence tier at one
    point in time. `confidence` is None when `bucket` is ABSENT — an
    absent skill has no confidence tier to speak of.
    """

    bucket: SkillBucket
    confidence: ConfidenceTier | None = None


def is_significant_event(old: SkillSnapshot, new: SkillSnapshot) -> bool:
    """Determine whether the crossing from `old` to `new` is a
    significant event, per Architecture_North_Star.md §9's deterministic
    rule: significant if and only if the skill crosses a bucket or
    confidence boundary upward.

    - absent -> emerging_skills or core_skills: significant
    - emerging_skills -> core_skills: significant
    - confidence tier strengthens while the bucket stays the same:
      significant
    - any downward crossing (core_skills -> emerging_skills,
      emerging_skills/core_skills -> absent, confidence weakening): not
      significant — diffed but discarded, no action taken
    - no change at all: not significant
    """
    old_rank = _BUCKET_RANK[old.bucket]
    new_rank = _BUCKET_RANK[new.bucket]

    if new_rank > old_rank:
        return True
    if new_rank < old_rank:
        return False

    # Same bucket. absent -> absent has no confidence to compare.
    if new.bucket is SkillBucket.ABSENT:
        return False

    assert old.confidence is not None
    assert new.confidence is not None
    return _CONFIDENCE_RANK[new.confidence] > _CONFIDENCE_RANK[old.confidence]
