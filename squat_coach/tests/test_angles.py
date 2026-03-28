"""Tests for biomechanics angle computation."""
import numpy as np
from squat_coach.biomechanics.angles import compute_joint_angles
from squat_coach.pose.landmarks import LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

def test_compute_knee_angle_straight_leg():
    """Straight leg should give ~180 degrees."""
    landmarks = np.zeros((33, 3))
    landmarks[LEFT_HIP] = [0, -1, 0]
    landmarks[LEFT_KNEE] = [0, 0, 0]
    landmarks[LEFT_ANKLE] = [0, 1, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_knee_angle"] - 180.0) < 1.0

def test_compute_knee_angle_bent():
    """90-degree knee bend."""
    landmarks = np.zeros((33, 3))
    landmarks[LEFT_HIP] = [0, -1, 0]
    landmarks[LEFT_KNEE] = [0, 0, 0]
    landmarks[LEFT_ANKLE] = [1, 0, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_knee_angle"] - 90.0) < 1.0

def test_compute_hip_angle():
    landmarks = np.zeros((33, 3))
    # shoulder above hip, knee below hip at 90 deg
    landmarks[11] = [0, -1, 0]  # left shoulder
    landmarks[LEFT_HIP] = [0, 0, 0]
    landmarks[LEFT_KNEE] = [1, 0, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_hip_angle"] - 90.0) < 1.0

def test_torso_inclination_upright():
    landmarks = np.zeros((33, 3))
    landmarks[11] = [0, -0.4, 0]   # left shoulder
    landmarks[12] = [0, -0.4, 0]   # right shoulder
    landmarks[23] = [0, 0, 0]      # left hip
    landmarks[24] = [0, 0, 0]      # right hip
    angles = compute_joint_angles(landmarks)
    # Upright trunk: ~0 degrees from vertical (but our vertical is Y-down
    # so trunk vector points down = aligned with vertical)
    assert angles["torso_inclination_deg"] < 5.0
