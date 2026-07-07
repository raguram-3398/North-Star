"""Pace calculation: topic_score, timing_ratio, the combined blend, and sustained-drift detection."""

from typing import Literal

TIMING_OUTLIER_THRESHOLD = 0.5

TIMING_SATURATION_DEVIATION = 1.0

MAX_TIMING_INFLUENCE = 0.2

DRIFT_WINDOW_SIZE = 3

SUSTAINED_BEHIND_THRESHOLD = 0.7

SUSTAINED_AHEAD_THRESHOLD = 0.95


def calculate_topic_score(credits: list[float]) -> float:
    """Compute topic_score as the average of exactly 5 per-question credits."""
    if len(credits) != 5:
        raise ValueError(f"expected exactly 5 per-question credits, got {len(credits)}")
    return sum(credits) / 5


def calculate_timing_ratio(days_taken: int, days_expected: int) -> float:
    """Compute timing_ratio as days_taken divided by days_expected."""
    if days_expected <= 0:
        raise ValueError(f"days_expected must be positive, got {days_expected}")
    return days_taken / days_expected


def calculate_combined_pace_signal(topic_score: float, timing_ratio: float) -> float:
    """Blend topic_score with timing_ratio, letting timing pull the signal only once it's a genuine outlier."""
    deviation = timing_ratio - 1.0
    magnitude = abs(deviation)

    if magnitude <= TIMING_OUTLIER_THRESHOLD:
        return topic_score

    excess = magnitude - TIMING_OUTLIER_THRESHOLD
    saturation_range = TIMING_SATURATION_DEVIATION - TIMING_OUTLIER_THRESHOLD
    influence = MAX_TIMING_INFLUENCE * min(1.0, excess / saturation_range)

    timing_score = max(0.0, min(1.0, 1.0 - deviation))

    return (1.0 - influence) * topic_score + influence * timing_score


def detect_sustained_drift(
    pace_signals: list[float],
) -> Literal["behind", "ahead", "on_track"]:
    """Classify the trailing window of pace signals as behind, ahead, or on_track by its mean."""
    if len(pace_signals) < DRIFT_WINDOW_SIZE:
        return "on_track"

    window = pace_signals[-DRIFT_WINDOW_SIZE:]
    window_mean = sum(window) / len(window)
    if window_mean <= SUSTAINED_BEHIND_THRESHOLD:
        return "behind"
    if window_mean >= SUSTAINED_AHEAD_THRESHOLD:
        return "ahead"
    return "on_track"
