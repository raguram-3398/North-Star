"""Pace calculation — topic_score, timing_ratio, the 80/20 blend, and the
sustained-drift threshold check.

Pure, deterministic, no LLM calls, no DB calls, no date/calendar logic
(CLAUDE.md: pure functions stay pure). Per PRD §7.8: pace reflects
understanding, not throughput. Cold-start gating (no velocity judgment
during weeks 1-2) is the caller's responsibility (Agent 2) — this module
has no notion of calendar time and is never itself gated.
"""

from typing import Literal

# --- Judgment-call constants ------------------------------------------
# PRD §7.8 states the *principle* (topic_score dominant ~80%, timing_ratio
# a ~20% ceiling only on genuine outliers, sustained drift over a rolling
# window) but not exact numbers. The following are judgment calls flagged
# for review, not silently finalized.

# How far timing_ratio must deviate from 1.0 (exactly on the user's own
# baseline pace) before it counts as a genuine outlier rather than
# ordinary variation. 0.5 means a user must take less than 50% or more
# than 150% of their own expected days before timing exerts any pull at
# all on the combined signal.
TIMING_OUTLIER_THRESHOLD = 0.5

# The |timing_ratio - 1.0| deviation at which timing's pull reaches its
# full ceiling weight (MAX_TIMING_INFLUENCE below). Chosen as double the
# outlier threshold: taking >= 2x (or effectively 0x) the expected days is
# a large enough outlier to justify the full 20% pull. Deviations between
# the outlier threshold and this saturation point scale linearly.
TIMING_SATURATION_DEVIATION = 1.0

# The ~20% ceiling PRD §7.8 names explicitly for timing's maximum
# influence on the combined pace signal, once it is a genuine outlier.
MAX_TIMING_INFLUENCE = 0.2

# Number of consecutive rolling-window entries required before a trend
# counts as "sustained" rather than a single data point. PRD §7.8 requires
# a rolling window of "consecutive check-ins"; 3 was chosen as the
# smallest window that is unambiguously more than a single occurrence.
DRIFT_WINDOW_SIZE = 3

# Combined-pace-signal value at or below which a single check-in counts
# as "behind". topic_score alone lands at or below this when a user
# needed the taught-answer/half-credit fallback (PRD §7.7) on roughly 3 or
# more of the 5 questions in a topic (e.g. 2 full + 3 half = 0.7).
SUSTAINED_BEHIND_THRESHOLD = 0.7

# Combined-pace-signal value at or above which a single check-in counts
# as "ahead" — near-perfect first-attempt performance.
SUSTAINED_AHEAD_THRESHOLD = 0.95


def calculate_topic_score(credits: list[float]) -> float:
    """Compute topic_score = (sum of per-question credit, full=1.0 /
    half=0.5) / 5, per PRD §7.8.

    Raises ValueError if `credits` does not have exactly 5 entries — a
    topic is always exactly 5 questions (PRD §7.7).
    """
    if len(credits) != 5:
        raise ValueError(f"expected exactly 5 per-question credits, got {len(credits)}")
    return sum(credits) / 5


def calculate_timing_ratio(days_taken: int, days_expected: int) -> float:
    """Compute timing_ratio = days_taken / days_expected.

    `days_expected` is supplied by the caller as already derived from the
    user's own established baseline (PRD §7.8) — this function does not
    compute or know about that baseline itself.

    Raises ValueError if `days_expected` is not positive.
    """
    if days_expected <= 0:
        raise ValueError(f"days_expected must be positive, got {days_expected}")
    return days_taken / days_expected


def calculate_combined_pace_signal(topic_score: float, timing_ratio: float) -> float:
    """Blend topic_score (dominant) and timing_ratio (only when a genuine
    outlier) into a single combined pace signal, per PRD §7.8.

    topic_score is returned unchanged when timing_ratio is within ordinary
    variation of 1.0 (on-baseline-pace) — ordinary variation is ignored
    entirely, not just down-weighted. Once timing_ratio is a genuine
    outlier (TIMING_OUTLIER_THRESHOLD), timing exerts a proportional pull,
    capped at MAX_TIMING_INFLUENCE and saturating at
    TIMING_SATURATION_DEVIATION. Timing slower than baseline (behind)
    pulls the signal down; timing faster than baseline (ahead) pulls it
    up.
    """
    deviation = timing_ratio - 1.0
    magnitude = abs(deviation)

    if magnitude <= TIMING_OUTLIER_THRESHOLD:
        return topic_score

    excess = magnitude - TIMING_OUTLIER_THRESHOLD
    saturation_range = TIMING_SATURATION_DEVIATION - TIMING_OUTLIER_THRESHOLD
    influence = MAX_TIMING_INFLUENCE * min(1.0, excess / saturation_range)

    # 1.0 when on-or-ahead of baseline, falling toward 0.0 the further
    # behind baseline the user is.
    timing_score = max(0.0, min(1.0, 1.0 - deviation))

    return (1.0 - influence) * topic_score + influence * timing_score


def detect_sustained_drift(
    pace_signals: list[float],
) -> Literal["behind", "ahead", "on_track"]:
    """Determine whether the trailing DRIFT_WINDOW_SIZE entries of a
    rolling window of combined pace signals show sustained drift.

    Uses the *mean* of the trailing window against the behind/ahead
    thresholds, rather than requiring every single entry to individually
    cross — a single unusually good or bad day within an otherwise
    consistent window should not by itself reset the streak. Fewer
    signals than the window size never triggers anything (PRD §7.8):
    returns "on_track".
    """
    if len(pace_signals) < DRIFT_WINDOW_SIZE:
        return "on_track"

    window = pace_signals[-DRIFT_WINDOW_SIZE:]
    window_mean = sum(window) / len(window)
    if window_mean <= SUSTAINED_BEHIND_THRESHOLD:
        return "behind"
    if window_mean >= SUSTAINED_AHEAD_THRESHOLD:
        return "ahead"
    return "on_track"
