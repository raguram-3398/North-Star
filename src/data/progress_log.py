"""progress_log I/O.

Per Architecture_North_Star.md §5: the canonical store for verification
results, hands-on/review outcomes, reflection entries, and timing — feeds
pace calculation.
"""

from typing import Any


def log_progress_step(
    user_id: str,
    topic_id: str,
    day_number: int,
    step: str,
    reflection_text: str | None = None,
) -> None:
    """Record a single day/step entry (summary/theory/hands_on/review/
    reflection/verification/preview) to the progress log.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def get_progress_for_topic(topic_id: str) -> list[dict[str, Any]]:
    """Read all recorded progress-log entries for a given topic.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
