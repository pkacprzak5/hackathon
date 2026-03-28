"""MediaPipe Pose Landmarker wrapper using BlazePose 3D world landmarks.

This is the primary pose estimation backend. It uses MediaPipe's Pose Landmarker
Tasks API to detect 33 BlazePose landmarks in both image coordinates and 3D world
coordinates (meters, hip-centered coordinate system).

Requires the pose_landmarker model file (.task) — downloaded automatically or
placed alongside this file.
"""
import logging
from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from squat_coach.pose.base import PoseEstimator, PoseResult
from squat_coach.pose.landmarks import NUM_LANDMARKS

logger = logging.getLogger("squat_coach.pose")

# Default model path — heavy model for best accuracy
_DEFAULT_MODEL_PATH = str(Path(__file__).parent / "pose_landmarker_heavy.task")


class MediaPipeBlazePose3D(PoseEstimator):
    """MediaPipe Pose Landmarker with BlazePose 3D world landmarks.

    Uses the Tasks API (mediapipe >= 0.10.9) with PoseLandmarker.
    Produces both image-space (normalized) and world-space (meters) landmarks
    for all 33 BlazePose body points. World landmarks are in a hip-centered
    coordinate system suitable for view-invariant biomechanics computation.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Pose landmarker model not found at {model_path}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                "pose_landmarker_heavy.task"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._frame_count = 0

        logger.info(
            "MediaPipe PoseLandmarker initialized (model=%s, det_conf=%.2f, track_conf=%.2f)",
            Path(model_path).name,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def estimate(self, frame_bgr: NDArray[np.uint8], timestamp: float) -> PoseResult:
        """Run PoseLandmarker on a BGR frame, return world + image landmarks."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # MediaPipe Tasks API requires mediapipe.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp in milliseconds (must be monotonically increasing)
        self._frame_count += 1
        timestamp_ms = int(self._frame_count * (1000.0 / 30.0))

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_world_landmarks or not result.pose_landmarks:
            return PoseResult(timestamp=timestamp, detected=False)

        # First detected pose (single user)
        world_lm = result.pose_world_landmarks[0]
        image_lm = result.pose_landmarks[0]

        # Extract world landmarks: 3D in meters, hip-centered
        world_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_lm], dtype=np.float64
        )

        # Extract image landmarks: normalized [0,1] x, y + z depth proxy
        image_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in image_lm], dtype=np.float64
        )

        # Extract per-landmark visibility
        vis = np.array([lm.visibility for lm in world_lm], dtype=np.float64)

        # Overall confidence: mean visibility of detected landmarks
        pose_confidence = float(np.mean(vis))

        return PoseResult(
            timestamp=timestamp,
            world_landmarks=world_arr,
            image_landmarks=image_arr,
            visibility=vis,
            pose_confidence=pose_confidence,
            detected=True,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("MediaPipe PoseLandmarker closed")
