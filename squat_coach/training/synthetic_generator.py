"""Synthetic squat sequence generator for training data augmentation.

Generates artificial squat feature sequences with known labels.
Used to fill gaps in real training data (e.g., for faults not
covered by ALEX-GYM-1 or Zenodo datasets).
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase, FaultType
from typing import Optional


def generate_synthetic_squat(
    seq_len: int = 60,
    feature_dim: int = 42,
    fps: float = 30.0,
    inject_fault: Optional[FaultType] = None,
    fault_severity: float = 0.5,
    noise_level: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], float]:
    """Generate one synthetic squat sequence.

    Returns:
        features: (seq_len, feature_dim) feature array
        phase_labels: (seq_len,) phase label indices (0=top, 1=desc, 2=bottom, 3=ascent)
        fault_labels: (6,) binary fault labels for the sequence
        quality: scalar quality score [0-1]
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 2 * np.pi, seq_len)

    # Base squat trajectory: sinusoidal knee angle curve
    # Knee goes from ~170 (standing) to ~85 (deep squat) and back
    base_knee = 127.5 - 42.5 * np.cos(t)  # range ~85-170
    base_hip = 120 - 35 * np.cos(t)       # range ~85-155
    base_torso = 5 + 15 * (1 - np.cos(t)) / 2  # range ~5-20 degrees

    # Hip Y trajectory (Y-down in BlazePose: higher = deeper)
    hip_y = 0.3 * np.cos(t)  # oscillates

    # Phase labels from trajectory shape
    phase_labels = np.zeros(seq_len, dtype=np.int64)
    quarter = seq_len // 4
    phase_labels[:quarter] = 0           # TOP -> start of descent
    phase_labels[quarter:2*quarter] = 1  # DESCENT
    phase_labels[2*quarter:3*quarter] = 2  # BOTTOM
    phase_labels[3*quarter:] = 3         # ASCENT

    # Fault injection
    fault_labels = np.zeros(6, dtype=np.float64)
    quality = 0.9  # default good

    if inject_fault == FaultType.INSUFFICIENT_DEPTH:
        base_knee += 20  # Don't go as deep
        fault_labels[0] = 1.0
        quality -= 0.3 * fault_severity

    elif inject_fault == FaultType.EXCESSIVE_FORWARD_LEAN:
        base_torso += 20 * fault_severity
        fault_labels[1] = 1.0
        quality -= 0.25 * fault_severity

    elif inject_fault == FaultType.ROUNDED_BACK_RISK:
        base_torso += 15 * fault_severity
        fault_labels[2] = 1.0
        quality -= 0.3 * fault_severity

    elif inject_fault == FaultType.HEEL_FAULT:
        fault_labels[3] = 1.0
        quality -= 0.2 * fault_severity

    elif inject_fault == FaultType.UNSTABLE_TORSO:
        base_torso += rng.normal(0, 5 * fault_severity, seq_len)
        fault_labels[4] = 1.0
        quality -= 0.2 * fault_severity

    elif inject_fault == FaultType.INCONSISTENT_TEMPO:
        # Distort the time axis
        t_distorted = t + 0.3 * fault_severity * np.sin(3 * t)
        base_knee = 127.5 - 42.5 * np.cos(t_distorted)
        fault_labels[5] = 1.0
        quality -= 0.15 * fault_severity

    quality = max(0.0, min(1.0, quality))

    # Build feature vector (simplified — fills D=42 slots)
    features = np.zeros((seq_len, feature_dim))
    features[:, 0] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # left knee
    features[:, 1] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # right knee
    features[:, 2] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # primary knee
    features[:, 3] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # left hip
    features[:, 4] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # right hip
    features[:, 5] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # primary hip
    features[:, 6] = 90 + rng.normal(0, noise_level * 3, seq_len)         # ankle proxy
    features[:, 7] = base_torso + rng.normal(0, noise_level * 3, seq_len) # torso incl
    features[:, 8] = base_torso + rng.normal(0, noise_level * 3, seq_len) # sh-hip angle

    # Offsets (small values)
    for i in range(9, 16):
        features[:, i] = rng.normal(0, noise_level, seq_len)

    # View-specific (16-19)
    features[:, 16] = base_torso  # forward lean / knee valgus
    features[:, 17] = rng.uniform(0, 0.3, seq_len)  # back risk / stance
    features[:, 18] = rng.uniform(0, 5, seq_len)     # stability / symmetry
    features[:, 19] = 80 + rng.normal(0, 3, seq_len) # ankle / hip shift

    # Kinematics (20-27): derive from finite differences
    dt = 1.0 / fps
    features[:, 20] = np.gradient(hip_y, dt)
    features[:, 21] = np.gradient(features[:, 20], dt)
    features[:, 22] = np.gradient(base_torso, dt)
    features[:, 23] = np.gradient(features[:, 22], dt)
    features[:, 24] = np.gradient(base_knee, dt)
    features[:, 25] = np.gradient(features[:, 24], dt)
    features[:, 26] = np.gradient(base_hip, dt)
    features[:, 27] = np.gradient(features[:, 26], dt)

    # Quality features (28-33)
    features[:, 28] = 0.9 + rng.normal(0, 0.02, seq_len)  # vis mean
    features[:, 29] = 0.85 + rng.normal(0, 0.03, seq_len)  # lower body vis
    features[:, 30] = 0.9 + rng.normal(0, 0.02, seq_len)  # torso vis
    features[:, 31] = 0.85 + rng.normal(0, 0.02, seq_len)  # reliability
    features[:, 32] = 1.0   # view validity
    features[:, 33] = 0.1 + rng.normal(0, 0.02, seq_len)   # occlusion risk

    # Pairwise distances (34-41)
    for i in range(34, 42):
        features[:, i] = 0.3 + rng.normal(0, 0.02, seq_len)

    return features, phase_labels, fault_labels, quality
