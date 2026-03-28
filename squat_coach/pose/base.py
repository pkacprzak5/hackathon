"""Pose estimator interface and result dataclass."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass
class PoseResult:
    """Result from a single pose estimation frame.

    Attributes:
        timestamp: Frame timestamp in seconds.
        world_landmarks: 3D world landmarks (33, 3) in meters, hip-centered.
        image_landmarks: 2D image landmarks (33, 3) as normalized x, y, visibility.
        visibility: Per-landmark visibility scores (33,), range [0, 1].
        pose_confidence: Overall detection confidence [0, 1].
        detected: Whether a pose was successfully detected.
    """
    timestamp: float
    world_landmarks: Optional[NDArray[np.float64]] = None   # (33, 3)
    image_landmarks: Optional[NDArray[np.float64]] = None   # (33, 3)
    visibility: Optional[NDArray[np.float64]] = None        # (33,)
    pose_confidence: float = 0.0
    detected: bool = False

class PoseEstimator(ABC):
    """Abstract base class for pose estimation backends."""

    @abstractmethod
    def estimate(self, frame_bgr: NDArray[np.uint8], timestamp: float) -> PoseResult:
        """Run pose estimation on a single BGR frame."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
