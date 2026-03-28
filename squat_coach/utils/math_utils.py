"""Core math utilities for geometry and vector operations."""
import numpy as np
from numpy.typing import NDArray

def angle_between_vectors(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """Angle in degrees between two vectors. Returns 0-180."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def angle_at_joint(
    point_a: NDArray[np.float64],
    joint: NDArray[np.float64],
    point_c: NDArray[np.float64],
) -> float:
    """Angle at joint formed by point_a--joint--point_c, in degrees (0-180)."""
    v1 = point_a - joint
    v2 = point_c - joint
    return angle_between_vectors(v1, v2)

def vector_from_points(
    start: NDArray[np.float64], end: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Vector from start to end."""
    return end - start

def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vector. Returns zero vector if input is near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm

def perpendicular_distance_to_line(
    point: NDArray[np.float64],
    line_start: NDArray[np.float64],
    line_end: NDArray[np.float64],
) -> float:
    """Perpendicular distance from point to the line defined by line_start->line_end."""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        return float(np.linalg.norm(point - line_start))
    # Cross product magnitude / line length = perpendicular distance
    cross = np.cross(line_vec, line_start - point)
    return float(np.linalg.norm(cross) / line_len)
