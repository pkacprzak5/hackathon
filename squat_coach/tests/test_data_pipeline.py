"""Tests for data pipeline."""
import tempfile
import numpy as np
from squat_coach.training.data_pipeline import generate_synthetic_dataset, compute_normalization_stats

def test_synthetic_dataset_shapes():
    features, phases, faults, quality = generate_synthetic_dataset(num_samples=10, seq_len=30)
    assert features.shape == (10, 30, 42)
    assert phases.shape == (10, 30)
    assert faults.shape == (10, 6)
    assert quality.shape == (10,)

def test_synthetic_dataset_caching(tmp_path):
    cache = str(tmp_path / "test_cache.npz")
    f1, _, _, _ = generate_synthetic_dataset(num_samples=5, cache_path=cache)
    f2, _, _, _ = generate_synthetic_dataset(num_samples=5, cache_path=cache)
    np.testing.assert_array_equal(f1, f2)

def test_normalization_stats(tmp_path):
    features = np.random.randn(10, 30, 42)
    stats_path = str(tmp_path / "stats.json")
    compute_normalization_stats(features, stats_path)
    import json
    with open(stats_path) as f:
        data = json.load(f)
    assert len(data["mean"]) == 42
    assert len(data["std"]) == 42
