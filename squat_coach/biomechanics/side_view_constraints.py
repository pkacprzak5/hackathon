"""Side-view specific feature extraction."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX,
)
from squat_coach.utils.math_utils import angle_between_vectors, angle_at_joint


def compute_side_view_features(
    world_landmarks: NDArray[np.float64],
    trunk_stability_window: list[float],
) -> dict[str, float]:
    """Compute features specific to side-view squat analysis.

    Args:
        world_landmarks: (33, 3) world landmarks.
        trunk_stability_window: Recent torso_inclination values for variance.

    Returns:
        Dict of side-view specific features.
    """
    lm = world_landmarks
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0

    # Forward lean angle: trunk forward tilt in sagittal plane
    trunk_vec = mid_hip - mid_sh
    vertical = np.array([0.0, 1.0, 0.0])
    forward_lean = angle_between_vectors(trunk_vec, vertical)

    # Trunk stability: variance of frame-to-frame CHANGES in torso angle
    # This measures jitter/wobble, not the natural range of motion during a squat
    if len(trunk_stability_window) > 2:
        diffs = np.diff(trunk_stability_window)
        trunk_stability = float(np.var(diffs))
    else:
        trunk_stability = 0.0

    # Ankle/shin angle: dorsiflexion proxy
    ankle_shin_angle = angle_at_joint(lm[LEFT_KNEE], lm[LEFT_ANKLE], lm[LEFT_FOOT_INDEX])

    return {
        "forward_lean_angle": forward_lean,
        "trunk_stability": trunk_stability,
        "ankle_shin_angle": ankle_shin_angle,
    }
