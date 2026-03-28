"""Joint angle computation from 3D world landmarks.

All angles are computed in 3D using the full (x, y, z) coordinates
of BlazePose world landmarks. Angles are returned in degrees.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, NOSE,
)
from squat_coach.utils.math_utils import angle_at_joint, angle_between_vectors


def compute_joint_angles(world_landmarks: NDArray[np.float64]) -> dict[str, float]:
    """Compute all biomechanics joint angles from world landmarks.

    Args:
        world_landmarks: (33, 3) BlazePose world landmarks.

    Returns:
        Dictionary of named angles in degrees.
    """
    lm = world_landmarks

    # Knee angles: hip-knee-ankle
    left_knee_angle = angle_at_joint(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    right_knee_angle = angle_at_joint(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])

    # Hip angles: shoulder-hip-knee
    left_hip_angle = angle_at_joint(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE])
    right_hip_angle = angle_at_joint(lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_KNEE])

    # Ankle angle proxy: knee-ankle-foot_index
    ankle_angle = angle_at_joint(lm[LEFT_KNEE], lm[LEFT_ANKLE], lm[LEFT_FOOT_INDEX])

    # Torso inclination: angle of trunk vector vs vertical
    mid_shoulder = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    trunk_vec = mid_hip - mid_shoulder  # points downward when upright
    vertical = np.array([0.0, 1.0, 0.0])  # Y-down in BlazePose world coords
    torso_inclination = angle_between_vectors(trunk_vec, vertical)

    # Shoulder-hip line angle (same as torso but using the line, not vector direction)
    shoulder_hip_line_angle = torso_inclination

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "ankle_angle_proxy": ankle_angle,
        "torso_inclination_deg": torso_inclination,
        "shoulder_hip_line_angle": shoulder_hip_line_angle,
    }
