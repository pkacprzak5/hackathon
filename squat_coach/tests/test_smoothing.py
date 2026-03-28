"""Tests for EMA landmark smoothing."""
import numpy as np
from squat_coach.preprocessing.smoothing import EMALandmarkSmoother

def test_smoother_first_frame_passthrough():
    """First frame should pass through unchanged."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    landmarks = np.ones((33, 3))
    result = smoother.smooth(landmarks)
    np.testing.assert_array_equal(result, landmarks)

def test_smoother_reduces_jitter():
    """EMA should smooth out a spike."""
    smoother = EMALandmarkSmoother(alpha=0.3)
    base = np.zeros((33, 3))
    smoother.smooth(base.copy())  # Initialize

    # Spike frame
    spike = np.ones((33, 3)) * 10.0
    result = smoother.smooth(spike)

    # Should be pulled toward base, not at the spike value
    assert np.all(result < 5.0)

def test_smoother_converges():
    """Repeated same input should converge to that input."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    target = np.ones((33, 3)) * 5.0
    for _ in range(50):
        result = smoother.smooth(target.copy())
    np.testing.assert_allclose(result, target, atol=0.01)

def test_smoother_reset():
    """After reset, next frame should pass through."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    smoother.smooth(np.ones((33, 3)))
    smoother.reset()
    new_val = np.ones((33, 3)) * 3.0
    result = smoother.smooth(new_val)
    np.testing.assert_array_equal(result, new_val)
