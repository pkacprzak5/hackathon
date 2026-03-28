"""Personalized idealized squat reference from calibration."""
from dataclasses import dataclass
from squat_coach.preprocessing.calibration import CalibrationResult


@dataclass
class IdealReference:
    """Personalized target values for scoring comparison."""
    target_knee_angle: float        # Target depth angle (degrees)
    trunk_neutral_angle: float      # Calibrated neutral trunk angle
    trunk_tolerance: float          # Acceptable deviation (degrees)
    expected_descent_s: float       # Expected descent duration
    expected_ascent_s: float        # Expected ascent duration


def build_ideal_reference(
    calibration: CalibrationResult,
    target_knee_angle: float = 90.0,
    trunk_tolerance: float = 15.0,
) -> IdealReference:
    """Build personalized reference from calibration data."""
    return IdealReference(
        target_knee_angle=target_knee_angle,
        trunk_neutral_angle=calibration.baseline_torso_angle,
        trunk_tolerance=trunk_tolerance,
        expected_descent_s=1.0,
        expected_ascent_s=1.0,
    )
