"""Rounded back risk and posture proxy analysis.

IMPORTANT LIMITATIONS:
- This estimates rounded back RISK from surface landmarks only.
- It CANNOT detect actual vertebral flexion or spinal curvature.
- It measures visible postural indicators correlated with back rounding.
- Should NOT be used for medical assessment.
- Works best from true lateral (side) view; degrades at oblique angles.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE, LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.math_utils import angle_between_vectors, perpendicular_distance_to_line


@dataclass
class RoundedBackAssessment:
    """Result of rounded back risk estimation."""
    risk_score: float               # 0.0-1.0
    confidence: float               # 0.0-1.0
    trunk_curl_component: float     # 0.0-1.0
    head_drift_component: float     # 0.0-1.0
    spine_linearity_component: float  # 0.0-1.0
    rationale: str
    limitations: str = "Estimated from surface landmarks only. Not a medical assessment."


def compute_rounded_back_risk(
    world_landmarks: NDArray[np.float64],
    baseline_torso_angle: float,
    body_scale: float,
) -> RoundedBackAssessment:
    """Compute rounded back risk from side-view landmarks.

    Three components:
    1. Trunk curl proxy: trunk angle deviation from calibrated baseline.
    2. Head/neck drift: forward displacement of head relative to shoulders.
    3. Spine linearity: how much the mid-torso deviates from a straight
       shoulder-hip line (proxy for thoracic rounding).

    Args:
        world_landmarks: (33, 3) world landmarks.
        baseline_torso_angle: Calibrated upright trunk angle in degrees.
        body_scale: Torso length in meters (from calibration).
    """
    lm = world_landmarks
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0

    # Component 1: Trunk curl — deviation of current trunk angle from baseline
    trunk_vec = mid_hip - mid_sh
    vertical = np.array([0.0, 1.0, 0.0])
    current_angle = angle_between_vectors(trunk_vec, vertical)
    angle_deviation = max(0.0, current_angle - baseline_torso_angle)
    # Normalize: 30+ degrees deviation = risk 1.0
    trunk_curl = min(angle_deviation / 30.0, 1.0)

    # Component 2: Head drift — forward displacement of nose relative to shoulders
    nose = lm[NOSE]
    head_forward_dist = perpendicular_distance_to_line(nose, mid_sh, mid_hip)
    # Normalize by body scale: >0.15 * body_scale = high risk
    head_drift = min(head_forward_dist / (0.15 * body_scale), 1.0)

    # Component 3: Spine linearity — deviation of torso midpoint from straight line
    # Use the point midway between shoulders and hips as a proxy for mid-spine
    mid_torso = (mid_sh + mid_hip) / 2.0
    # In a straight spine, this point lies ON the shoulder-hip line
    # Deviation indicates curvature
    spine_dev = perpendicular_distance_to_line(mid_torso, mid_sh, mid_hip)
    spine_linearity = min(spine_dev / (0.05 * body_scale), 1.0)

    # Weighted combination
    risk_score = 0.4 * trunk_curl + 0.35 * head_drift + 0.25 * spine_linearity
    risk_score = min(max(risk_score, 0.0), 1.0)

    # Confidence based on landmark visibility and angle deviation magnitude
    confidence = min(0.5 + angle_deviation / 20.0, 1.0)

    rationale_parts = []
    if trunk_curl > 0.3:
        rationale_parts.append(f"trunk angle {current_angle:.0f}\u00b0 (baseline {baseline_torso_angle:.0f}\u00b0)")
    if head_drift > 0.3:
        rationale_parts.append(f"head drifted {head_forward_dist * 100:.0f}cm forward")
    if spine_linearity > 0.3:
        rationale_parts.append(f"mid-spine deviation detected")
    rationale = "; ".join(rationale_parts) if rationale_parts else "Within normal range"

    return RoundedBackAssessment(
        risk_score=risk_score,
        confidence=confidence,
        trunk_curl_component=trunk_curl,
        head_drift_component=head_drift,
        spine_linearity_component=spine_linearity,
        rationale=rationale,
    )
