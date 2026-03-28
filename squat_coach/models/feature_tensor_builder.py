"""Convert feature dicts to model input tensors with z-score normalization."""
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from numpy.typing import NDArray


class FeatureTensorBuilder:
    """Builds normalized model input tensors from raw feature vectors.

    Applies z-score normalization using pre-computed training set statistics.
    """

    def __init__(self, stats_path: Optional[str] = None) -> None:
        """
        Args:
            stats_path: Path to JSON file with 'mean' and 'std' arrays (each length 42).
                        If None, no normalization is applied (raw features passed through).
        """
        self._mean: Optional[NDArray[np.float64]] = None
        self._std: Optional[NDArray[np.float64]] = None

        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                data = json.load(f)
            self._mean = np.array(data["mean"], dtype=np.float64)
            self._std = np.array(data["std"], dtype=np.float64)
            # Avoid division by zero
            self._std = np.where(self._std < 1e-8, 1.0, self._std)

    def normalize(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score normalize a feature vector or sequence."""
        if self._mean is None:
            return features
        return (features - self._mean) / self._std

    def to_tensor(self, sequence: NDArray[np.float64]) -> torch.Tensor:
        """Convert a (seq_len, 42) numpy array to a (1, seq_len, 42) tensor."""
        normalized = self.normalize(sequence)
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
