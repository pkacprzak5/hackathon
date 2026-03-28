"""Tests for math utilities."""
import numpy as np
import pytest
from squat_coach.utils.math_utils import (
    angle_between_vectors,
    angle_at_joint,
    vector_from_points,
    normalize_vector,
    perpendicular_distance_to_line,
)

def test_angle_between_vectors_perpendicular():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 90.0) < 0.1

def test_angle_between_vectors_parallel():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2)) < 0.1

def test_angle_between_vectors_opposite():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 180.0) < 0.1

def test_angle_at_joint_straight():
    # Straight leg: hip-knee-ankle in a line
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])  # joint
    c = np.array([0.0, -1.0, 0.0])
    assert abs(angle_at_joint(a, b, c) - 180.0) < 0.1

def test_angle_at_joint_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(angle_at_joint(a, b, c) - 90.0) < 0.1

def test_vector_from_points():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = vector_from_points(a, b)
    np.testing.assert_array_equal(result, np.array([3.0, 3.0, 3.0]))

def test_normalize_vector():
    v = np.array([3.0, 4.0, 0.0])
    result = normalize_vector(v)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6

def test_normalize_zero_vector():
    v = np.array([0.0, 0.0, 0.0])
    result = normalize_vector(v)
    np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))

def test_perpendicular_distance_to_line():
    # Point directly above the midpoint of a horizontal line
    line_start = np.array([0.0, 0.0, 0.0])
    line_end = np.array([2.0, 0.0, 0.0])
    point = np.array([1.0, 1.0, 0.0])
    dist = perpendicular_distance_to_line(point, line_start, line_end)
    assert abs(dist - 1.0) < 0.01
