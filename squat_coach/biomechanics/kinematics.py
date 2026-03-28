"""Velocity and acceleration computation via finite differences.

Computes first and second derivatives of key features across frames.
Requires at least 2 frames for velocity, 3 for acceleration.
"""
from typing import Optional
import numpy as np


class KinematicsTracker:
    """Track velocities and accelerations of scalar features over frames."""

    def __init__(self, fps: float = 30.0) -> None:
        self._dt = 1.0 / fps
        self._prev_values: Optional[dict[str, float]] = None
        self._prev_velocities: Optional[dict[str, float]] = None

    def update(self, values: dict[str, float]) -> dict[str, float]:
        """Compute velocities and accelerations from current values.

        Args:
            values: Dict with keys like 'hip_y', 'trunk_angle', 'knee_angle', 'hip_angle'.

        Returns:
            Dict with velocity and acceleration for each tracked value.
        """
        result: dict[str, float] = {}
        velocities: dict[str, float] = {}

        for key, val in values.items():
            vel_key = f"{key}_velocity"
            accel_key = f"{key}_acceleration"

            # Velocity = finite difference
            if self._prev_values is not None and key in self._prev_values:
                vel = (val - self._prev_values[key]) / self._dt
            else:
                vel = 0.0
            velocities[key] = vel
            result[vel_key] = vel

            # Acceleration = finite difference of velocity
            if self._prev_velocities is not None and key in self._prev_velocities:
                accel = (vel - self._prev_velocities[key]) / self._dt
            else:
                accel = 0.0
            result[accel_key] = accel

        self._prev_values = dict(values)
        self._prev_velocities = dict(velocities)
        return result

    def reset(self) -> None:
        self._prev_values = None
        self._prev_velocities = None
