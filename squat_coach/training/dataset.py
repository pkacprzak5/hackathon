"""PyTorch Dataset for squat training sequences."""
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


class SquatSequenceDataset(Dataset):
    """Dataset of squat feature sequences with multi-task labels."""

    def __init__(
        self,
        features: NDArray[np.float64],     # (N, seq_len, feature_dim)
        phase_labels: NDArray[np.int64],    # (N, seq_len)
        fault_labels: NDArray[np.float64],  # (N, 6)
        quality_labels: NDArray[np.float64], # (N,)
    ) -> None:
        self._features = features
        self._phases = phase_labels
        self._faults = fault_labels
        self._quality = quality_labels

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self._features[idx], dtype=torch.float32),
            "phase_labels": torch.tensor(self._phases[idx, -1], dtype=torch.long),  # last frame label
            "fault_labels": torch.tensor(self._faults[idx], dtype=torch.float32),
            "quality_label": torch.tensor(self._quality[idx], dtype=torch.float32),
        }
