"""Front-view specific feature extraction."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)


def compute_front_view_features(
    world_landmarks: NDArray[np.float64],
) -> dict[str, float]:
    """Compute features specific to front-view squat analysis.

    Args:
        world_landmarks: (33, 3) world landmarks.

    Returns:
        Dict of front-view specific features.
    """
    lm = world_landmarks

    # Knee valgus: inward collapse angle
    # Measure how much knees move inward relative to hips and ankles
    mid_hip_x = (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2.0
    l_knee_inward = max(0.0, lm[LEFT_KNEE][0] - lm[LEFT_HIP][0])   # positive = inward
    r_knee_inward = max(0.0, lm[RIGHT_HIP][0] - lm[RIGHT_KNEE][0])  # positive = inward
    knee_valgus_angle = float(np.degrees(np.arctan2(
        (l_knee_inward + r_knee_inward) / 2.0,
        abs(lm[LEFT_HIP][1] - lm[LEFT_KNEE][1]) + 1e-6
    )))

    # Stance width ratio: foot spread / hip width
    hip_width = abs(lm[LEFT_HIP][0] - lm[RIGHT_HIP][0])
    foot_width = abs(lm[LEFT_FOOT_INDEX][0] - lm[RIGHT_FOOT_INDEX][0])
    stance_width_ratio = foot_width / max(hip_width, 0.01)

    # Left-right symmetry: compare left vs right knee angles (proxy)
    l_knee_y = lm[LEFT_KNEE][1]
    r_knee_y = lm[RIGHT_KNEE][1]
    symmetry = 1.0 - min(abs(l_knee_y - r_knee_y) / 0.1, 1.0)

    # Hip lateral shift
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    mid_ankle = (lm[LEFT_ANKLE] + lm[RIGHT_ANKLE]) / 2.0
    hip_shift = abs(mid_hip[0] - mid_ankle[0])

    return {
        "knee_valgus_angle": knee_valgus_angle,
        "stance_width_ratio": stance_width_ratio,
        "left_right_symmetry": symmetry,
        "hip_shift_lateral": hip_shift,
    }
