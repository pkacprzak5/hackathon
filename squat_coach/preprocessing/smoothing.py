"""Exponential moving average (EMA) landmark smoother.

Smooths 3D landmark positions across frames to reduce jitter from
pose estimation noise. Higher alpha = less smoothing, lower latency.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional

class EMALandmarkSmoother:
    """EMA smoother for (N, 3) landmark arrays."""

    def __init__(self, alpha: float = 0.4) -> None:
        """
        Args:
            alpha: Smoothing factor in [0, 1]. Higher = less smoothing.
                   0.4 is a good default for 30fps pose data.
        """
        self._alpha = alpha
        self._prev: Optional[NDArray[np.float64]] = None

    def smooth(self, landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply EMA to a new frame of landmarks.

        Args:
            landmarks: Shape (N, 3) landmark positions.

        Returns:
            Smoothed landmarks, same shape.
        """
        if self._prev is None:
            self._prev = landmarks.copy()
            return landmarks.copy()

        smoothed = self._alpha * landmarks + (1.0 - self._alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        """Reset smoother state (e.g., after detection loss)."""
        self._prev = None
