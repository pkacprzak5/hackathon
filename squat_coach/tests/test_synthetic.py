"""Tests for synthetic data generation."""
import numpy as np
from squat_coach.training.synthetic_generator import generate_synthetic_squat
from squat_coach.utils.enums import FaultType

def test_synthetic_shapes():
    features, phases, faults, quality = generate_synthetic_squat(seq_len=60, feature_dim=42)
    assert features.shape == (60, 42)
    assert phases.shape == (60,)
    assert faults.shape == (6,)
    assert 0.0 <= quality <= 1.0

def test_synthetic_with_fault():
    features, phases, faults, quality = generate_synthetic_squat(
        inject_fault=FaultType.INSUFFICIENT_DEPTH
    )
    assert faults[0] == 1.0  # depth fault active
    assert quality < 0.9

def test_synthetic_good_form():
    features, phases, faults, quality = generate_synthetic_squat(inject_fault=None)
    assert np.sum(faults) == 0.0
    assert quality >= 0.8

def test_synthetic_phase_labels_valid():
    _, phases, _, _ = generate_synthetic_squat()
    assert all(p in [0, 1, 2, 3] for p in phases)
