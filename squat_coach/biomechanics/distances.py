"""Pairwise joint distance computation."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER, NOSE,
)


def compute_hip_depth_ratios(
    world_landmarks: NDArray[np.float64],
) -> tuple[float, float]:
    """Hip depth relative to knee and ankle (Y-axis).

    Negative values mean hip is above the reference point.
    Positive values mean hip is below (deeper squat).

    Returns:
        (hip_depth_vs_knee, hip_depth_vs_ankle) as normalized ratios.
    """
    mid_hip_y = (world_landmarks[LEFT_HIP][1] + world_landmarks[RIGHT_HIP][1]) / 2.0
    mid_knee_y = (world_landmarks[LEFT_KNEE][1] + world_landmarks[RIGHT_KNEE][1]) / 2.0
    mid_ankle_y = (world_landmarks[LEFT_ANKLE][1] + world_landmarks[RIGHT_ANKLE][1]) / 2.0

    # Normalize by leg length for body-size invariance
    leg_len = abs(mid_hip_y - mid_ankle_y)
    if leg_len < 0.01:
        return 0.0, 0.0

    hip_vs_knee = (mid_hip_y - mid_knee_y) / leg_len
    hip_vs_ankle = (mid_hip_y - mid_ankle_y) / leg_len
    return float(hip_vs_knee), float(hip_vs_ankle)


def compute_pairwise_distance_subset(
    world_landmarks: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 8 pairwise distances used in the D=42 feature vector.

    Pairs (from spec):
        L/R hip-knee, L/R knee-ankle, L/R shoulder-hip,
        hip_mid-shoulder_mid, nose-shoulder_mid
    """
    lm = world_landmarks
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0

    pairs = [
        (lm[LEFT_HIP], lm[LEFT_KNEE]),
        (lm[RIGHT_HIP], lm[RIGHT_KNEE]),
        (lm[LEFT_KNEE], lm[LEFT_ANKLE]),
        (lm[RIGHT_KNEE], lm[RIGHT_ANKLE]),
        (lm[LEFT_SHOULDER], lm[LEFT_HIP]),
        (lm[RIGHT_SHOULDER], lm[RIGHT_HIP]),
        (mid_hip, mid_sh),
        (lm[NOSE], mid_sh),
    ]

    distances = np.array(
        [float(np.linalg.norm(a - b)) for a, b in pairs], dtype=np.float64
    )
    return distances
