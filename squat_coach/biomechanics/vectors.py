"""Bone and trunk vector utilities."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE,
    LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.math_utils import normalize_vector


def compute_trunk_vector(world_landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mid-shoulder to mid-hip unit vector."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    return normalize_vector(mid_hip - mid_sh)


def compute_head_to_trunk_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Perpendicular distance from nose to the shoulder-hip trunk line."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    nose = world_landmarks[NOSE]
    from squat_coach.utils.math_utils import perpendicular_distance_to_line
    return perpendicular_distance_to_line(nose, mid_sh, mid_hip)


def compute_shoulder_hip_deltas(
    world_landmarks: NDArray[np.float64],
) -> tuple[float, float]:
    """Horizontal and vertical deltas between mid-shoulder and mid-hip.

    Returns:
        (horizontal_delta, vertical_delta) in meters.
    """
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    diff = mid_sh - mid_hip
    h_delta = float(np.sqrt(diff[0] ** 2 + diff[2] ** 2))  # XZ plane
    v_delta = float(abs(diff[1]))  # Y axis
    return h_delta, v_delta


def compute_nose_to_shoulder_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Horizontal offset of nose from mid-shoulder."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    nose = world_landmarks[NOSE]
    return float(np.sqrt((nose[0] - mid_sh[0]) ** 2 + (nose[2] - mid_sh[2]) ** 2))


def compute_neck_forward_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Forward offset of ear midpoint from shoulder midpoint (sagittal drift)."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_ear = (world_landmarks[LEFT_EAR] + world_landmarks[RIGHT_EAR]) / 2.0
    return float(np.sqrt((mid_ear[0] - mid_sh[0]) ** 2 + (mid_ear[2] - mid_sh[2]) ** 2))
