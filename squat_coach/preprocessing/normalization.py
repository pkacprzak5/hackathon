"""Hip-centered landmark normalization.

Translates world landmarks so the mid-hip point is at origin.
Optionally scales by a reference bone length for size invariance.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER

def normalize_to_hip_center(
    world_landmarks: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Center world landmarks at the mid-hip point.

    Args:
        world_landmarks: (33, 3) world landmarks in meters.

    Returns:
        Hip-centered landmarks, same shape.
    """
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    return world_landmarks - mid_hip

def compute_body_scale(world_landmarks: NDArray[np.float64]) -> float:
    """Compute a body scale factor from torso length (mid_shoulder to mid_hip).

    Used to normalize distance-based features across different body sizes.

    Returns:
        Torso length in meters. Returns 1.0 if landmarks are invalid.
    """
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    mid_shoulder = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    length = float(np.linalg.norm(mid_shoulder - mid_hip))
    return length if length > 0.01 else 1.0
