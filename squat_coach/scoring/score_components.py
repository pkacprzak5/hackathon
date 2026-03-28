"""Individual score component computations. All scores are 0-100."""
import numpy as np


def compute_depth_score(knee_angle_min: float, target_angle: float = 90.0) -> float:
    """Score how close the minimum knee angle was to target depth.

    Score 100 = at or below target angle. Decreases linearly above target.
    """
    if knee_angle_min <= target_angle:
        return 100.0
    gap = knee_angle_min - target_angle
    # 50 degrees short of target = 0 score
    return max(0.0, 100.0 - (gap / 50.0) * 100.0)


def compute_trunk_control_score(
    torso_variance: float,
    max_forward_lean: float,
    baseline_angle: float,
) -> float:
    """Score trunk stability through the rep.

    torso_variance: variance of frame-to-frame angle CHANGES (jitter).
        A smooth squat has near-zero jitter. Wobbly trunk has high jitter.
    max_forward_lean: peak torso angle during the rep.
        Up to ~45° forward lean is normal for a squat. Only penalize beyond that.
    """
    # Jitter penalty: variance of diffs. <2 = smooth, >10 = very wobbly
    var_score = max(0.0, 100.0 - (torso_variance / 10.0) * 100.0)

    # Lean penalty: only penalize excessive lean (>45° from baseline is a lot)
    lean_deviation = max(0.0, max_forward_lean - baseline_angle - 35.0)  # 35° lean is normal
    lean_score = max(0.0, 100.0 - (lean_deviation / 25.0) * 100.0)

    return 0.4 * var_score + 0.6 * lean_score


def compute_posture_stability_score(
    rounded_back_risk: float,
    head_drift: float,
    body_scale: float,
) -> float:
    """Score posture quality based on back rounding and head drift."""
    back_score = max(0.0, 100.0 - rounded_back_risk * 100.0)
    drift_normalized = head_drift / max(body_scale, 0.01)
    drift_score = max(0.0, 100.0 - (drift_normalized / 0.3) * 100.0)
    return 0.6 * back_score + 0.4 * drift_score


def compute_movement_consistency_score(
    descent_duration: float,
    ascent_duration: float,
    expected_descent: float = 1.0,
    expected_ascent: float = 1.0,
) -> float:
    """Score movement smoothness and tempo symmetry."""
    # Timing symmetry: how similar descent and ascent durations are
    if descent_duration + ascent_duration < 0.01:
        return 50.0
    ratio = min(descent_duration, ascent_duration) / max(descent_duration, ascent_duration)
    symmetry_score = ratio * 100.0

    # Tempo match: how close to expected durations
    desc_match = max(0.0, 1.0 - abs(descent_duration - expected_descent) / expected_descent)
    asc_match = max(0.0, 1.0 - abs(ascent_duration - expected_ascent) / expected_ascent)
    tempo_score = (desc_match + asc_match) / 2.0 * 100.0

    return 0.5 * symmetry_score + 0.5 * tempo_score
