"""Auto-label squat phases from hip vertical trajectory.

Uses hip Y-coordinate over time to determine:
- DESCENT: hip moving downward (negative velocity, sustained)
- BOTTOM: hip at local minimum (lowest point)
- ASCENT: hip moving upward (positive velocity, sustained)
- TOP: hip at or near starting height

Uses hysteresis to avoid noisy label flipping.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase


def label_phases_from_hip_trajectory(
    hip_y_sequence: NDArray[np.float64],
    fps: float = 30.0,
    velocity_threshold: float = 0.005,
    hysteresis: float = 0.002,
) -> list[Phase]:
    """Label each frame with a squat phase based on hip Y position.

    Args:
        hip_y_sequence: (T,) array of hip Y positions over time.
            In BlazePose world coords, Y increases downward.
        fps: Frame rate.
        velocity_threshold: Min velocity magnitude to classify as moving.
        hysteresis: Buffer to prevent oscillation at boundaries.

    Returns:
        List of Phase labels, one per frame.
    """
    T = len(hip_y_sequence)
    if T < 3:
        return [Phase.TOP] * T

    # Compute velocity via finite differences
    dt = 1.0 / fps
    velocity = np.gradient(hip_y_sequence, dt)

    # Smooth velocity
    kernel_size = max(3, int(fps * 0.1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    velocity_smooth = np.convolve(velocity, kernel, mode="same")

    labels: list[Phase] = []
    current_phase = Phase.TOP

    # Find global min/max for normalization context
    y_range = np.max(hip_y_sequence) - np.min(hip_y_sequence)
    if y_range < 0.01:
        return [Phase.TOP] * T

    for i in range(T):
        vel = velocity_smooth[i]

        if current_phase == Phase.TOP:
            if vel > velocity_threshold + hysteresis:
                current_phase = Phase.DESCENT
        elif current_phase == Phase.DESCENT:
            if vel < velocity_threshold - hysteresis and vel > -velocity_threshold + hysteresis:
                current_phase = Phase.BOTTOM
        elif current_phase == Phase.BOTTOM:
            if vel < -velocity_threshold - hysteresis:
                current_phase = Phase.ASCENT
        elif current_phase == Phase.ASCENT:
            if vel > -velocity_threshold + hysteresis and vel < velocity_threshold - hysteresis:
                current_phase = Phase.TOP

        labels.append(current_phase)

    return labels
