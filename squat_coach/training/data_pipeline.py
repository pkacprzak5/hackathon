"""Data loading, preprocessing, and caching pipeline.

Handles ALEX-GYM-1, Zenodo, and synthetic data sources.
Produces cached .npz files ready for training.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import NDArray

from squat_coach.training.synthetic_generator import generate_synthetic_squat
from squat_coach.utils.enums import FaultType

logger = logging.getLogger("squat_coach.training")

# Faults to generate synthetically
SYNTHETIC_FAULTS = [
    None,  # good form
    FaultType.INSUFFICIENT_DEPTH,
    FaultType.EXCESSIVE_FORWARD_LEAN,
    FaultType.ROUNDED_BACK_RISK,
    FaultType.HEEL_FAULT,
    FaultType.UNSTABLE_TORSO,
    FaultType.INCONSISTENT_TEMPO,
]


def generate_synthetic_dataset(
    num_samples: int = 2000,
    seq_len: int = 60,
    feature_dim: int = 42,
    cache_path: Optional[str] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a full synthetic training dataset.

    Returns:
        features: (N, seq_len, feature_dim)
        phase_labels: (N, seq_len)
        fault_labels: (N, 6)
        quality_labels: (N,)
    """
    if cache_path and Path(cache_path).exists():
        logger.info("Loading cached synthetic data from %s", cache_path)
        data = np.load(cache_path)
        return data["features"], data["phase_labels"], data["fault_labels"], data["quality_labels"]

    logger.info("Generating %d synthetic sequences...", num_samples)
    rng = np.random.default_rng(42)

    all_features = []
    all_phases = []
    all_faults = []
    all_quality = []

    for i in range(num_samples):
        fault = rng.choice(SYNTHETIC_FAULTS)
        severity = rng.uniform(0.3, 1.0) if fault else 0.0
        noise = rng.uniform(0.01, 0.05)

        features, phases, faults, quality = generate_synthetic_squat(
            seq_len=seq_len,
            feature_dim=feature_dim,
            inject_fault=fault,
            fault_severity=severity,
            noise_level=noise,
            rng=rng,
        )
        all_features.append(features)
        all_phases.append(phases)
        all_faults.append(faults)
        all_quality.append(quality)

    features_arr = np.array(all_features)
    phases_arr = np.array(all_phases)
    faults_arr = np.array(all_faults)
    quality_arr = np.array(all_quality)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            features=features_arr,
            phase_labels=phases_arr,
            fault_labels=faults_arr,
            quality_labels=quality_arr,
        )
        logger.info("Cached synthetic data to %s", cache_path)

    return features_arr, phases_arr, faults_arr, quality_arr


def compute_normalization_stats(
    features: NDArray[np.float64], stats_path: str
) -> None:
    """Compute and save per-feature mean and std for z-score normalization."""
    # features: (N, seq_len, D) -> flatten to (N*seq_len, D)
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0).tolist()
    std = flat.std(axis=0).tolist()
    # Avoid zero std
    std = [max(s, 1e-8) for s in std]

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    logger.info("Saved normalization stats to %s", stats_path)
