"""Tests for rolling sequence buffer."""
import numpy as np
from squat_coach.preprocessing.sequence_buffer import SequenceBuffer

def test_buffer_not_ready_initially():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    assert not buf.is_ready

def test_buffer_ready_after_filling():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    for i in range(4):
        buf.push(np.ones(3) * i)
    assert buf.is_ready

def test_buffer_get_sequence_shape():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    for i in range(4):
        buf.push(np.ones(3) * i)
    seq = buf.get_sequence()
    assert seq.shape == (4, 3)

def test_buffer_rolling_window():
    buf = SequenceBuffer(seq_len=3, feature_dim=2)
    for i in range(5):
        buf.push(np.array([float(i), float(i)]))
    seq = buf.get_sequence()
    # Should contain last 3: [2,2], [3,3], [4,4]
    np.testing.assert_array_equal(seq[0], [2.0, 2.0])
    np.testing.assert_array_equal(seq[2], [4.0, 4.0])

def test_buffer_padded_when_partial():
    buf = SequenceBuffer(seq_len=4, feature_dim=2)
    buf.push(np.array([1.0, 2.0]))
    buf.push(np.array([3.0, 4.0]))
    seq = buf.get_sequence_padded()
    assert seq.shape == (4, 2)
    # First 2 rows should be zero-padded
    np.testing.assert_array_equal(seq[0], [0.0, 0.0])
    np.testing.assert_array_equal(seq[2], [1.0, 2.0])
    np.testing.assert_array_equal(seq[3], [3.0, 4.0])

def test_buffer_reset():
    buf = SequenceBuffer(seq_len=3, feature_dim=2)
    buf.push(np.array([1.0, 2.0]))
    buf.reset()
    assert not buf.is_ready
    assert len(buf) == 0
