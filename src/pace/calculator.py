"""Pace calculation — topic_score, timing_ratio, the 80/20 blend, and the
sustained-drift threshold check.

Pure, deterministic, no LLM calls, no side effects (CLAUDE.md: pure
functions stay pure). Per PRD §7.8: pace reflects understanding, not
throughput.
"""


def calculate_topic_score(credits: list[float]) -> float:
    """Compute topic_score = (sum of per-question credit, full=1 /
    half=0.5) / 5, per PRD §7.8.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def calculate_timing_ratio(days_taken: int, days_expected: int) -> float:
    """Compute timing_ratio = days_taken / days_expected, benchmarked
    against the user's own established baseline, per PRD §7.8.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def is_cold_start(weeks_since_start: int) -> bool:
    """Determine whether the user is still in the weeks 1-2 calibration
    window, during which no velocity judgment or triggering occurs
    (PRD §7.8).

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def calculate_combined_pace_signal(
    topic_score: float, timing_ratio: float, baseline_timing_ratio: float
) -> float:
    """Blend topic_score (~80% influence) and timing_ratio (~20% ceiling,
    exerted only when it's a genuine outlier against the user's own
    baseline) into a single combined pace signal, per PRD §7.8.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError


def detect_sustained_drift(pace_signals: list[float]) -> str:
    """Determine whether a rolling window of combined pace signals shows
    sustained drift (behind/ahead/on-track) — a single day's performance
    never triggers anything, per PRD §7.8.

    Not yet implemented — scaffolding only.
    """
    raise NotImplementedError
