"""Tests for calibration flow."""
import numpy as np
from squat_coach.preprocessing.calibration import CalibrationResult, Calibrator
from squat_coach.utils.enums import ViewType
from squat_coach.pose.base import PoseResult

def _make_side_view_pose(timestamp: float = 0.0) -> PoseResult:
    """Create a synthetic side-view standing pose.

    Side view: shoulders have similar X but different Z.
    """
    landmarks = np.zeros((33, 3))
    # Shoulders: same X, spread in Z (side view characteristic)
    landmarks[11] = [0.0, -0.4, -0.1]   # left shoulder
    landmarks[12] = [0.0, -0.4,  0.1]   # right shoulder
    # Hips
    landmarks[23] = [0.0, 0.0, -0.1]    # left hip
    landmarks[24] = [0.0, 0.0,  0.1]    # right hip
    # Head
    landmarks[0] = [0.0, -0.6, 0.0]     # nose
    landmarks[7] = [-0.05, -0.55, 0.0]  # left ear
    landmarks[8] = [0.05, -0.55, 0.0]   # right ear

    return PoseResult(
        timestamp=timestamp,
        world_landmarks=landmarks,
        image_landmarks=landmarks.copy(),
        visibility=np.ones(33) * 0.9,
        pose_confidence=0.9,
        detected=True,
    )

def _make_front_view_pose(timestamp: float = 0.0) -> PoseResult:
    """Front view: shoulders spread in X, similar Z."""
    landmarks = np.zeros((33, 3))
    landmarks[11] = [-0.2, -0.4, 0.0]   # left shoulder
    landmarks[12] = [ 0.2, -0.4, 0.0]   # right shoulder
    landmarks[23] = [-0.1, 0.0, 0.0]
    landmarks[24] = [ 0.1, 0.0, 0.0]
    landmarks[0]  = [0.0, -0.6, 0.0]
    landmarks[7]  = [-0.1, -0.55, 0.0]
    landmarks[8]  = [ 0.1, -0.55, 0.0]

    return PoseResult(
        timestamp=timestamp,
        world_landmarks=landmarks,
        image_landmarks=landmarks.copy(),
        visibility=np.ones(33) * 0.9,
        pose_confidence=0.9,
        detected=True,
    )

def test_calibrator_detects_side_view():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_side_view_pose(float(i)))
    result = cal.compute()
    assert result is not None
    assert result.view_type == ViewType.SIDE

def test_calibrator_detects_front_view():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_front_view_pose(float(i)))
    result = cal.compute()
    assert result is not None
    assert result.view_type == ViewType.FRONT

def test_calibrator_stores_baseline():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_side_view_pose(float(i)))
    result = cal.compute()
    assert result.baseline_torso_angle is not None
    assert result.body_scale > 0.0

def test_calibrator_not_ready_without_enough_frames():
    cal = Calibrator(num_frames=5)
    cal.add_frame(_make_side_view_pose(0.0))
    result = cal.compute()
    assert result is None
