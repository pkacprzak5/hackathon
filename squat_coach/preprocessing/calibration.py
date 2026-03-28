"""Calibration flow: detect view type, establish baseline posture.

At session start, the user stands upright for ~2 seconds. The calibrator
collects frames, detects whether the camera shows a side or front view,
and computes baseline measurements for scoring personalization.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from squat_coach.pose.base import PoseResult
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE, LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.enums import ViewType
from squat_coach.utils.math_utils import angle_between_vectors

logger = logging.getLogger("squat_coach.calibration")


@dataclass
class CalibrationResult:
    """Outputs from the calibration phase."""
    view_type: ViewType
    baseline_torso_angle: float         # degrees from vertical
    baseline_head_offset: float         # meters
    body_scale: float                   # torso length in meters
    dominant_side: str                  # "left" or "right"
    baseline_landmarks: NDArray[np.float64]  # averaged standing pose (33, 3)
    baseline_knee_angle: float = 170.0  # standing knee angle in degrees


class Calibrator:
    """Collects standing frames and computes calibration baseline."""

    def __init__(self, num_frames: int = 60) -> None:
        """
        Args:
            num_frames: Number of frames to collect before computing.
                        At 30fps, 60 frames = 2 seconds.
        """
        self._num_frames = num_frames
        self._frames: list[NDArray[np.float64]] = []

    def add_frame(self, pose: PoseResult) -> None:
        """Add a detected pose frame to the calibration buffer."""
        if pose.detected and pose.world_landmarks is not None:
            self._frames.append(pose.world_landmarks.copy())

    @property
    def frame_count(self) -> int:
        """Number of frames collected so far."""
        return len(self._frames)

    @property
    def num_frames(self) -> int:
        """Total frames required for calibration."""
        return self._num_frames

    @property
    def is_ready(self) -> bool:
        return len(self._frames) >= self._num_frames

    def compute(self) -> Optional[CalibrationResult]:
        """Compute calibration from collected frames. Returns None if not enough data."""
        if len(self._frames) < self._num_frames:
            return None

        # Average landmarks across frames for stability
        avg_landmarks = np.mean(self._frames, axis=0)

        view_type = self._detect_view(avg_landmarks)
        torso_angle = self._compute_torso_angle(avg_landmarks)
        head_offset = self._compute_head_offset(avg_landmarks)
        body_scale = self._compute_body_scale(avg_landmarks)
        dominant_side = self._detect_dominant_side(avg_landmarks, view_type)
        knee_angle = self._compute_knee_angle(avg_landmarks)

        logger.info(
            "Calibration complete: view=%s, torso=%.1f\u00b0, scale=%.3fm, knee=%.1f\u00b0",
            view_type.value, torso_angle, body_scale, knee_angle,
        )

        return CalibrationResult(
            view_type=view_type,
            baseline_torso_angle=torso_angle,
            baseline_head_offset=head_offset,
            body_scale=body_scale,
            dominant_side=dominant_side,
            baseline_landmarks=avg_landmarks,
            baseline_knee_angle=knee_angle,
        )

    def _detect_view(self, landmarks: NDArray[np.float64]) -> ViewType:
        """Detect side vs front view from shoulder geometry.

        Side view: shoulders appear close in X (image horizontal), spread in Z (depth).
        Front view: shoulders spread in X, similar Z.
        """
        l_sh = landmarks[LEFT_SHOULDER]
        r_sh = landmarks[RIGHT_SHOULDER]
        dx = abs(l_sh[0] - r_sh[0])  # horizontal spread
        dz = abs(l_sh[2] - r_sh[2])  # depth spread

        # If shoulders are wider in X than Z, it's a front view
        if dx > dz * 1.5:
            return ViewType.FRONT
        return ViewType.SIDE

    def _compute_torso_angle(self, landmarks: NDArray[np.float64]) -> float:
        """Angle of mid_shoulder->mid_hip vector vs vertical (Y-down)."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        mid_hip = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
        trunk_vec = mid_hip - mid_sh
        vertical = np.array([0.0, 1.0, 0.0])  # Y-down in world coords
        return angle_between_vectors(trunk_vec, vertical)

    def _compute_head_offset(self, landmarks: NDArray[np.float64]) -> float:
        """Forward offset of nose relative to mid-shoulder."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        nose = landmarks[NOSE]
        # In side view, forward is primarily along X or Z axis
        return float(np.linalg.norm(nose[:2] - mid_sh[:2]))

    def _compute_body_scale(self, landmarks: NDArray[np.float64]) -> float:
        """Torso length: mid_shoulder to mid_hip distance."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        mid_hip = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
        length = float(np.linalg.norm(mid_sh - mid_hip))
        return max(length, 0.01)

    def _detect_dominant_side(self, landmarks: NDArray[np.float64], view: ViewType) -> str:
        """Detect which side is closer to camera (higher visibility expected)."""
        if view == ViewType.FRONT:
            return "both"
        # In side view, the side with higher Z value is closer to camera
        left_z = landmarks[LEFT_SHOULDER][2]
        right_z = landmarks[RIGHT_SHOULDER][2]
        return "left" if left_z > right_z else "right"

    def _compute_knee_angle(self, landmarks: NDArray[np.float64]) -> float:
        """Average knee angle during standing (both knees)."""
        from squat_coach.utils.math_utils import angle_at_joint
        from squat_coach.pose.landmarks import (
            LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
        )
        left = angle_at_joint(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
        right = angle_at_joint(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
        return (left + right) / 2.0
