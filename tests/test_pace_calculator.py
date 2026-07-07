"""Tests for pace/calculator.py: topic_score, timing_ratio, and pace-signal blending."""

import pytest

from pace.calculator import (
    MAX_TIMING_INFLUENCE,
    SUSTAINED_AHEAD_THRESHOLD,
    SUSTAINED_BEHIND_THRESHOLD,
    calculate_combined_pace_signal,
    calculate_timing_ratio,
    calculate_topic_score,
    detect_sustained_drift,
)


def test_topic_score_all_full_credit() -> None:
    assert calculate_topic_score([1.0, 1.0, 1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_topic_score_all_half_credit() -> None:
    assert calculate_topic_score([0.5, 0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_topic_score_mixed_credit() -> None:
    assert calculate_topic_score([1.0, 1.0, 1.0, 0.5, 0.5]) == pytest.approx(0.8)


def test_topic_score_rejects_wrong_length() -> None:
    with pytest.raises(ValueError):
        calculate_topic_score([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError):
        calculate_topic_score([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


def test_timing_ratio_on_baseline() -> None:
    assert calculate_timing_ratio(10, 10) == pytest.approx(1.0)


def test_timing_ratio_faster_than_expected() -> None:
    assert calculate_timing_ratio(5, 10) == pytest.approx(0.5)


def test_timing_ratio_slower_than_expected() -> None:
    assert calculate_timing_ratio(20, 10) == pytest.approx(2.0)


def test_timing_ratio_rejects_non_positive_days_expected() -> None:
    with pytest.raises(ValueError):
        calculate_timing_ratio(5, 0)

    with pytest.raises(ValueError):
        calculate_timing_ratio(5, -3)


def test_combined_signal_exactly_on_baseline_equals_topic_score() -> None:
    assert calculate_combined_pace_signal(
        topic_score=0.8, timing_ratio=1.0
    ) == pytest.approx(0.8)


def test_combined_signal_ordinary_variation_does_not_move_at_all() -> None:
    """Within the outlier threshold, timing is ignored entirely, not down-weighted."""
    assert calculate_combined_pace_signal(
        topic_score=0.9, timing_ratio=1.2
    ) == pytest.approx(0.9)
    assert calculate_combined_pace_signal(
        topic_score=0.9, timing_ratio=0.8
    ) == pytest.approx(0.9)


def test_combined_signal_severe_behind_outlier_saturates_at_ceiling() -> None:
    """At or beyond saturation, timing's pull maxes out at MAX_TIMING_INFLUENCE."""
    topic_score = 0.9
    result = calculate_combined_pace_signal(topic_score=topic_score, timing_ratio=2.0)

    assert result == pytest.approx((1.0 - MAX_TIMING_INFLUENCE) * topic_score)


def test_combined_signal_moderate_behind_outlier_shows_partial_real_pull() -> None:
    """Between threshold and saturation, timing pulls the signal down only partially."""
    topic_score = 0.9
    result = calculate_combined_pace_signal(topic_score=topic_score, timing_ratio=1.75)

    assert result < topic_score
    assert result > (1.0 - MAX_TIMING_INFLUENCE) * topic_score


def test_combined_signal_ahead_outlier_pulls_upward() -> None:
    """A genuine ahead-of-baseline outlier pulls the signal up, not down."""
    topic_score = 0.9
    result = calculate_combined_pace_signal(topic_score=topic_score, timing_ratio=0.4)

    assert result > topic_score


def test_sustained_drift_insufficient_data_is_on_track() -> None:
    assert detect_sustained_drift([]) == "on_track"
    assert detect_sustained_drift([0.5, 0.5]) == "on_track"


def test_sustained_drift_behind_when_average_crosses_threshold() -> None:
    behind_signal = SUSTAINED_BEHIND_THRESHOLD - 0.05
    assert detect_sustained_drift([behind_signal, behind_signal, behind_signal]) == (
        "behind"
    )


def test_sustained_drift_one_strong_day_among_weak_days_still_triggers_behind() -> None:
    """Average-based: one strong day among weak days can still trigger "behind"."""
    weak_signal = 0.55
    strong_signal = 0.9
    assert detect_sustained_drift([weak_signal, strong_signal, weak_signal]) == "behind"


def test_sustained_drift_evenly_mixed_window_near_boundary_does_not_trigger() -> None:
    """A window whose mean lands just above the threshold must not falsely trigger."""
    weak_signal = 0.6
    strong_signal = 0.96
    assert (
        detect_sustained_drift([weak_signal, strong_signal, weak_signal]) == "on_track"
    )


def test_sustained_drift_ahead_when_average_crosses_threshold() -> None:
    ahead_signal = SUSTAINED_AHEAD_THRESHOLD + 0.01
    assert detect_sustained_drift([ahead_signal, ahead_signal, ahead_signal]) == (
        "ahead"
    )


def test_sustained_drift_one_weak_day_among_strong_days_still_triggers_ahead() -> None:
    """Average-based: one weaker day among strong days can still trigger "ahead"."""
    very_strong_signal = 0.99
    weaker_signal = 0.92
    assert (
        detect_sustained_drift([very_strong_signal, very_strong_signal, weaker_signal])
        == "ahead"
    )


def test_sustained_drift_evenly_mixed_window_near_ahead_boundary_does_not_trigger() -> (
    None
):
    strong_signal = 0.95
    weaker_signal = 0.90
    assert (
        detect_sustained_drift([strong_signal, strong_signal, weaker_signal])
        == "on_track"
    )


def test_sustained_drift_exact_topic_score_boundary_triggers_behind() -> None:
    """A window sitting exactly on the behind threshold must trigger "behind"."""
    topic_score = calculate_topic_score([1.0, 1.0, 0.5, 0.5, 0.5])
    assert topic_score == pytest.approx(0.7)
    assert topic_score == pytest.approx(SUSTAINED_BEHIND_THRESHOLD)

    combined = calculate_combined_pace_signal(topic_score=topic_score, timing_ratio=1.0)
    assert combined == pytest.approx(0.7)

    assert detect_sustained_drift([combined, combined, combined]) == "behind"


def test_sustained_drift_only_examines_trailing_window() -> None:
    """A longer history is fine — only the trailing DRIFT_WINDOW_SIZE entries are examined."""
    ahead_signal = SUSTAINED_AHEAD_THRESHOLD + 0.01
    behind_signal = SUSTAINED_BEHIND_THRESHOLD - 0.05
    history = [behind_signal, behind_signal, ahead_signal, ahead_signal, ahead_signal]

    assert detect_sustained_drift(history) == "ahead"
