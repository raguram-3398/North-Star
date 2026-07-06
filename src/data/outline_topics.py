"""outline_topics I/O.

Per Architecture_North_Star.md §5. Reads/status-updates existing rows
(`get_topic`, `get_topics_in_group`, `mark_topic_completed` —
`agents/coaching_pace_agent.py`'s original consumers), plus
`insert_outline_topics`, which closes the previously-flagged integration
gap (PRD §11 / Architecture §10): persisting
`agents/research_outline_agent.py`'s `create_initial_outline`/
`regenerate_outline_with_addition` output into real rows. Every other
function in this module still assumes the row already exists by the time
it runs — only `insert_outline_topics` is a create path.

Sessions are passed in by the caller (dependency injection), matching
`data/roles_cache.py`'s established pattern.
"""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import func
from sqlalchemy.orm import Session

from models.schemas import OutlineTopic
from outline.hierarchy import insert_new_topic
from security.output_guard import ConfidenceTier

# Match Architecture §5's `status TEXT` column comment
# (`not_started | in_progress | completed | completed_test_out`) exactly.
# `NOT_STARTED_STATUS` duplicated here rather than imported from
# `agents/research_outline_agent.py` (which also defines it) deliberately:
# data/ modules do not import from agents/ (see `SequencedOutlineTopic`
# below) — agents call this module as a tool, never the other way around.
NOT_STARTED_STATUS = "not_started"
COMPLETED_STATUS = "completed"
COMPLETED_TEST_OUT_STATUS = "completed_test_out"
_VALID_COMPLETION_STATUSES = frozenset({COMPLETED_STATUS, COMPLETED_TEST_OUT_STATUS})


@runtime_checkable
class SequencedOutlineTopic(Protocol):
    """The structural shape `insert_outline_topics` requires — matches
    `agents/research_outline_agent.py`'s `InitialOutlineTopic` (the
    output of `create_initial_outline`/`regenerate_outline_with_addition`)
    field-for-field.

    Checked at runtime via `@runtime_checkable` rather than importing that
    agent-owned dataclass directly and using `isinstance` against it:
    `agents/research_outline_agent.py` is a *caller* of this module (see
    Architecture §3, "Calls as tools... data/outline_topics.py"), so this
    module importing back from `agents/` would invert that dependency
    direction and risks a real circular import the moment the agent's own
    caller wires `insert_outline_topics` in. A `runtime_checkable`
    Protocol gives the same practical guarantee CLAUDE.md guardrail #12
    asks for — a raw dict is structurally rejected (`isinstance` is False:
    a dict has no `.topic_name` attribute) — without that dependency.
    """

    # Declared as read-only `@property` members, not plain attribute
    # annotations: a plain `name: str` Protocol attribute is implicitly
    # read-write, which `InitialOutlineTopic` (a frozen, read-only
    # dataclass) structurally does not satisfy under static type checking
    # even though it satisfies the runtime `isinstance` check just fine.
    @property
    def topic_name(self) -> str: ...
    @property
    def hierarchy_position(self) -> int: ...
    @property
    def topic_group(self) -> str: ...
    @property
    def position_in_group(self) -> int: ...
    @property
    def source_url(self) -> str: ...
    @property
    def source_type(self) -> str: ...
    @property
    def confidence(self) -> ConfidenceTier: ...
    @property
    def is_enrichment(self) -> bool: ...
    @property
    def status(self) -> str: ...


def _to_dict(row: OutlineTopic) -> dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "topic_name": row.topic_name,
        "hierarchy_position": row.hierarchy_position,
        "topic_group": row.topic_group,
        "position_in_group": row.position_in_group,
        "source_url": row.source_url,
        "source_type": row.source_type,
        "confidence": row.confidence,
        "is_enrichment": row.is_enrichment,
        "status": row.status,
        "completed_at": row.completed_at,
    }


def get_topic(session: Session, topic_id: str) -> dict[str, Any] | None:
    """Read a single outline topic, or None if no entry exists."""
    row = session.get(OutlineTopic, topic_id)
    if row is None:
        return None
    return _to_dict(row)


def get_topics_in_group(
    session: Session, user_id: str, topic_group: str
) -> list[dict[str, Any]]:
    """Read every topic in `topic_group` for `user_id`, ordered by
    `position_in_group` — used to determine a topic-group's total size
    for hands-on ramping (Architecture §3's ramping rule needs
    `position_in_group` *and* the group's size, not a fixed day-count
    constant).
    """
    rows = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.user_id == user_id, OutlineTopic.topic_group == topic_group
        )
        .order_by(OutlineTopic.position_in_group)
        .all()
    )
    return [_to_dict(row) for row in rows]


def mark_topic_completed(
    session: Session, topic_id: str, status: str = COMPLETED_STATUS
) -> None:
    """Mark a topic completed and stamp `completed_at` — called once all
    5 verification question slots have resolved (PRD §7.7), whether via
    regular day-by-day coaching or test-out (PRD's day-by-day coaching
    section's "verification-first" exception).

    `status` defaults to `COMPLETED_STATUS` ("completed"); the test-out
    task added `COMPLETED_TEST_OUT_STATUS` ("completed_test_out") as an
    explicit alternative — Architecture §5's schema lists it as a
    distinct value from `completed`, not a synonym, so
    `agents/coaching_pace_agent.py`'s test-out completion path passes it
    explicitly rather than this function silently collapsing the two.

    Raises `ValueError` if `topic_id` does not exist, or if `status` is
    not one of the two recognized completion values.
    """
    if status not in _VALID_COMPLETION_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(_VALID_COMPLETION_STATUSES)}, "
            f"got {status!r}"
        )
    row = session.get(OutlineTopic, topic_id)
    if row is None:
        raise ValueError(f"outline topic {topic_id!r} not found")
    row.status = status
    row.completed_at = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    session.commit()


def insert_outline_topics(
    session: Session,
    user_id: str,
    topics: Sequence[SequencedOutlineTopic],
) -> list[dict[str, Any]]:
    """Persist a freshly-created or regenerated outline
    (`agents/research_outline_agent.py`'s `create_initial_outline`/
    `regenerate_outline_with_addition` output) as real `outline_topics`
    rows, and return the persisted rows (with generated `id`s).

    **Regeneration-replaces-prior-unstarted-rows:** `topics` is always the
    *entire* outline (Outline Confirmation's accepted-addition path
    regenerates the full hierarchy from scratch, per PRD §7.5 — it is
    never a partial delta). So this function deletes every existing
    `outline_topics` row for `user_id` and inserts the new set in the same
    transaction. A first-time call (no prior rows) degenerates to a plain
    insert.

    This is safe specifically because Outline Confirmation is a
    provably pre-Day-1 window (PRD §7.5: "no user-initiated outline
    editing once Day 1 begins") — no row for this user can have
    progressed past `not_started` while regeneration is still possible.
    This function does not merely assume that invariant: it raises
    `ValueError` if it ever finds an existing row already
    `in_progress`/`completed`/`completed_test_out`, rather than silently
    deleting it — replacing already-progressed content would violate
    CLAUDE.md guardrail #2 ("never delete or reduce outline content").

    Raises `ValueError` if `topics` is empty, or if an existing row for
    `user_id` has already progressed past `not_started`. Raises
    `TypeError` if any entry in `topics` is not a `SequencedOutlineTopic`
    (structurally checked — see that Protocol's docstring for why this is
    a runtime check rather than an `isinstance` against an imported
    agent-owned dataclass). Commits the transaction.
    """
    if not topics:
        raise ValueError("insert_outline_topics requires at least one topic")
    for topic in topics:
        if not isinstance(topic, SequencedOutlineTopic):
            raise TypeError(
                "insert_outline_topics requires already-sequenced topic "
                f"objects (e.g. InitialOutlineTopic), got {type(topic).__name__!r}"
            )

    existing_rows = (
        session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    )
    already_progressed = [
        row for row in existing_rows if row.status != NOT_STARTED_STATUS
    ]
    if already_progressed:
        raise ValueError(
            f"cannot persist a regenerated outline for user {user_id!r}: "
            f"{len(already_progressed)} existing row(s) have already "
            "progressed past 'not_started' — regenerating over "
            "started/completed content would violate CLAUDE.md guardrail #2"
        )
    for row in existing_rows:
        session.delete(row)

    # IDs are generated explicitly here (not left to OutlineTopic.id's
    # `default=uuid.uuid4` mapped-column default), so a persisted row's
    # id is available immediately on the returned dict without requiring
    # a real flush against a live engine — this module's tests use a
    # mocked Session (matching data/roles_cache.py's established
    # no-SQLite-substitute convention), which cannot execute SQLAlchemy's
    # own default-generation machinery.
    new_rows = [
        OutlineTopic(
            id=uuid.uuid4(),
            user_id=user_id,
            topic_name=topic.topic_name,
            hierarchy_position=topic.hierarchy_position,
            topic_group=topic.topic_group,
            position_in_group=topic.position_in_group,
            source_url=topic.source_url,
            source_type=topic.source_type,
            confidence=topic.confidence.value,
            is_enrichment=topic.is_enrichment,
            status=topic.status,
        )
        for topic in topics
    ]
    session.add_all(new_rows)
    persisted = [_to_dict(row) for row in new_rows]
    session.commit()
    return persisted


def get_completed_topics_matching_skill(
    session: Session, skill_name: str
) -> list[dict[str, Any]]:
    """Find every completed (`completed` or `completed_test_out`)
    `outline_topics` row, across all users, whose `topic_name` matches
    `skill_name` — used by `src/cron/refresh_roles.py` to find which
    users' already-completed topics are affected by a significant
    `roles_cache` crossing for that skill (Architecture §9's "generates a
    patch-note candidate for every user with a completed topic matching
    that skill").

    Matching is case-insensitive (`func.lower()` on the SQL side, `.lower()`
    on the Python side — not `.casefold()`, unlike this codebase's usual
    skill-matching convention elsewhere: this comparison must execute
    inside the DB query, and Postgres's `lower()` only approximates ASCII
    lowercasing, so pairing it with Python's more aggressive Unicode
    `.casefold()` on the other side of the comparison risks a silent
    mismatch for non-ASCII input. Skill names here are expected to be
    plain ASCII tech terms, so this is a narrow, flagged judgment call,
    not a correctness concern in practice) — outline topic names and
    roles_cache skill names are populated by two different pipelines
    (Gemini-sequenced topic hierarchy vs. Himalayas/Tavily-extracted skill
    strings) with no guaranteed identical casing convention.

    Not scoped to a single user — deliberately global, per Architecture
    §9's "every user" wording.
    """
    rows = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.status.in_(_VALID_COMPLETION_STATUSES),
            func.lower(OutlineTopic.topic_name) == skill_name.lower(),
        )
        .all()
    )
    return [_to_dict(row) for row in rows]


def get_all_topics_for_user(session: Session, user_id: str) -> list[dict[str, Any]]:
    """Read every `outline_topics` row for `user_id`, regardless of status
    or `topic_group` — used as the `existing_topics` input to
    `outline/hierarchy.py`'s `insert_new_topic` (not reimplemented here),
    which needs the user's whole existing hierarchy to renumber against,
    and by enrichment selection to check which skill names are already in
    use.
    """
    rows = session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    return [_to_dict(row) for row in rows]


def has_pending_enrichment_topic(session: Session, user_id: str) -> bool:
    """True if `user_id` already has an `is_enrichment=True` topic that
    has not yet resolved (`status` not in `{completed, completed_test_out}`).

    Used to prevent a second sustained-ahead trigger from inserting a
    second enrichment topic while one is still pending (PRD §7.10) —
    `agents/coaching_pace_agent.py`'s `maybe_trigger_enrichment` checks
    this before selecting a candidate skill.
    """
    row = (
        session.query(OutlineTopic)
        .filter(
            OutlineTopic.user_id == user_id,
            OutlineTopic.is_enrichment.is_(True),
            OutlineTopic.status.notin_(_VALID_COMPLETION_STATUSES),
        )
        .first()
    )
    return row is not None


def insert_new_outline_topic(
    session: Session,
    user_id: str,
    topic_name: str,
    topic_group: str,
    position_in_group: int,
    source_url: str,
    source_type: str,
    confidence: ConfidenceTier,
    is_enrichment: bool,
    prerequisite_topic_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Insert one new topic into `user_id`'s existing hierarchy via
    `outline/hierarchy.py`'s `insert_new_topic` (not reimplemented here) —
    the single insertion mechanism intended for any additive, hierarchy-
    positioned update (market-driven patch content, PRD §7.9; enrichment,
    PRD §7.10), not a second insertion path. Renumbers `hierarchy_position`
    on every existing row that shifts as a result, then persists the new
    row. Commits the transaction.

    `prerequisite_topic_ids` is passed straight through to
    `insert_new_topic` — e.g. the single just-completed topic that
    triggered an enrichment insertion, so the new topic lands immediately
    after the user's current position rather than at the very start of
    the whole hierarchy (`insert_new_topic`'s behavior with no
    prerequisites at all).

    Raises `ValueError` (via `insert_new_topic`) if any id in
    `prerequisite_topic_ids` is not found in `user_id`'s existing topics.
    """
    existing_rows = (
        session.query(OutlineTopic).filter(OutlineTopic.user_id == user_id).all()
    )
    existing_topics = [_to_dict(row) for row in existing_rows]

    new_topic_id = uuid.uuid4()
    new_topic_candidate = {
        "id": new_topic_id,
        "topic_name": topic_name,
        "topic_group": topic_group,
        "position_in_group": position_in_group,
        "source_url": source_url,
        "source_type": source_type,
        "confidence": confidence.value,
        "is_enrichment": is_enrichment,
        "status": NOT_STARTED_STATUS,
    }
    renumbered = insert_new_topic(
        existing_topics, new_topic_candidate, prerequisite_topic_ids
    )
    renumbered_by_id = {topic["id"]: topic for topic in renumbered}

    for existing_row in existing_rows:
        existing_row.hierarchy_position = renumbered_by_id[existing_row.id][
            "hierarchy_position"
        ]

    new_row = OutlineTopic(
        id=new_topic_id,
        user_id=user_id,
        topic_name=topic_name,
        hierarchy_position=renumbered_by_id[new_topic_id]["hierarchy_position"],
        topic_group=topic_group,
        position_in_group=position_in_group,
        source_url=source_url,
        source_type=source_type,
        confidence=confidence.value,
        is_enrichment=is_enrichment,
        status=NOT_STARTED_STATUS,
    )
    session.add(new_row)
    session.commit()
    return _to_dict(new_row)
