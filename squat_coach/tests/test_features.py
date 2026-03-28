"""Tests for the full squat feature extraction pipeline."""
import numpy as np
from squat_coach.biomechanics.squat_features import SquatFeatureExtractor
from squat_coach.preprocessing.calibration import CalibrationResult
from squat_coach.utils.enums import ViewType

def _make_standing_landmarks() -> np.ndarray:
    """Create realistic standing pose landmarks."""
    lm = np.zeros((33, 3))
    lm[11] = [-0.15, -0.4, 0.0]  # left shoulder
    lm[12] = [ 0.15, -0.4, 0.0]  # right shoulder
    lm[23] = [-0.1,   0.0, 0.0]  # left hip
    lm[24] = [ 0.1,   0.0, 0.0]  # right hip
    lm[25] = [-0.1,   0.4, 0.0]  # left knee
    lm[26] = [ 0.1,   0.4, 0.0]  # right knee
    lm[27] = [-0.1,   0.8, 0.0]  # left ankle
    lm[28] = [ 0.1,   0.8, 0.0]  # right ankle
    lm[31] = [-0.1,   0.85, 0.1] # left foot
    lm[32] = [ 0.1,   0.85, 0.1] # right foot
    lm[0]  = [ 0.0,  -0.6, 0.0]  # nose
    lm[7]  = [-0.05, -0.55, 0.0] # left ear
    lm[8]  = [ 0.05, -0.55, 0.0] # right ear
    return lm

def _make_calibration() -> CalibrationResult:
    return CalibrationResult(
        view_type=ViewType.SIDE,
        baseline_torso_angle=5.0,
        baseline_head_offset=0.02,
        body_scale=0.4,
        dominant_side="left",
        baseline_landmarks=_make_standing_landmarks(),
    )

def test_extract_returns_model_features():
    """Model feature vector should be D=42."""
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    features = extractor.extract(_make_standing_landmarks(), np.ones(33) * 0.9)
    assert "model_features" in features
    assert len(features["model_features"]) == 42

def test_extract_returns_all_named_features():
    """Should include all named biomechanics features."""
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    features = extractor.extract(_make_standing_landmarks(), np.ones(33) * 0.9)
    required = ["left_knee_angle", "torso_inclination_deg", "rounded_back_risk"]
    for key in required:
        assert key in features, f"Missing feature: {key}"

def test_extract_visibility_features():
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    vis = np.ones(33) * 0.8
    features = extractor.extract(_make_standing_landmarks(), vis)
    assert abs(features["landmark_visibility_mean"] - 0.8) < 0.01
