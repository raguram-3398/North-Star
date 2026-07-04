"""SQLAlchemy models mirroring Architecture_North_Star.md §5's data model
exactly. If a field needs to change, the architecture doc is updated in the
same commit (CLAUDE.md coding conventions).

Field-level definitions are deferred — this stage only scaffolds the table
entities the rest of the system references.
"""


class User:
    """`users` — user profile: background, current job, years of
    experience, prior self-study, resolved role, role confidence, pacing
    profile.
    """


class RolesCache:
    """`roles_cache` — cron-refreshed market data cache: core_skills /
    emerging_skills per role, each carrying a confidence tier, plus
    last_updated.
    """


class OutlineTopic:
    """`outline_topics` — dependency hierarchy per user: topic, hierarchy
    position, topic group / position-in-group (hands-on ramping), source
    metadata, enrichment flag, status.
    """


class ProgressLog:
    """`progress_log` — canonical record of everything that feeds pace:
    per-day, per-step entries (summary/theory/hands_on/review/reflection/
    verification/preview).
    """


class VerificationAttempt:
    """`verification_attempts` — per-question, per-attempt verification
    records that feed topic_score: question/grading criteria, answer,
    passed, credit (1.0 full / 0.5 half), test-out flag.
    """


class PatchNote:
    """`patch_notes` — market-driven updates to already-completed topics:
    origin topic, new content, source/confidence, status (pending /
    delivered / deferred).
    """


class PaceSnapshot:
    """`pace_snapshots` — rolling-window inputs to pace tracking:
    topic_score, timing_ratio, days_taken, days_expected per topic.
    """
