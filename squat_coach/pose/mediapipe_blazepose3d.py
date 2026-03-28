"""MediaPipe Pose Landmarker wrapper using BlazePose 3D world landmarks.

This is the primary pose estimation backend. It uses MediaPipe's Pose Landmarker
task API to detect 33 BlazePose landmarks in both image coordinates and 3D world
coordinates (meters, hip-centered coordinate system).
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from squat_coach.pose.base import PoseEstimator, PoseResult
from squat_coach.pose.landmarks import NUM_LANDMARKS

logger = logging.getLogger("squat_coach.pose")


class MediaPipeBlazePose3D(PoseEstimator):
    """MediaPipe Pose Landmarker with BlazePose 3D world landmarks.

    Produces both image-space (normalized) and world-space (meters) landmarks
    for all 33 BlazePose body points. World landmarks are in a hip-centered
    coordinate system suitable for view-invariant biomechanics computation.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        # Use the legacy Pose solution for broad compatibility
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "MediaPipe BlazePose initialized (complexity=%d, det_conf=%.2f, track_conf=%.2f)",
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def estimate(self, frame_bgr: NDArray[np.uint8], timestamp: float) -> PoseResult:
        """Run BlazePose on a BGR frame, return world + image landmarks."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)

        if results.pose_world_landmarks is None or results.pose_landmarks is None:
            return PoseResult(timestamp=timestamp, detected=False)

        # Extract world landmarks: 3D in meters, hip-centered
        world_lm = results.pose_world_landmarks.landmark
        world_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_lm], dtype=np.float64
        )

        # Extract image landmarks: normalized [0,1] x, y + z depth proxy
        image_lm = results.pose_landmarks.landmark
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
        self._pose.close()
        logger.info("MediaPipe BlazePose closed")
