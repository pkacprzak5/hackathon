"""Rolling sequence buffer for temporal model input.

Maintains a fixed-length sliding window of feature vectors.
Supports both ready (full) and padded (partial) extraction.
"""
from collections import deque
import numpy as np
from numpy.typing import NDArray

class SequenceBuffer:
    """Rolling window buffer for fixed-length sequence extraction."""

    def __init__(self, seq_len: int = 60, feature_dim: int = 42) -> None:
        """
        Args:
            seq_len: Number of frames in the window.
            feature_dim: Dimensionality of each feature vector.
        """
        self._seq_len = seq_len
        self._feature_dim = feature_dim
        self._buffer: deque[NDArray[np.float64]] = deque(maxlen=seq_len)

    def push(self, features: NDArray[np.float64]) -> None:
        """Add a feature vector to the buffer."""
        self._buffer.append(features.copy())

    @property
    def is_ready(self) -> bool:
        """True when buffer has seq_len frames."""
        return len(self._buffer) >= self._seq_len

    def get_sequence(self) -> NDArray[np.float64]:
        """Get the current window as (seq_len, feature_dim). Requires is_ready."""
        if not self.is_ready:
            raise ValueError("Buffer not full. Use get_sequence_padded() instead.")
        return np.array(list(self._buffer), dtype=np.float64)

    def get_sequence_padded(self) -> NDArray[np.float64]:
        """Get the window zero-padded at the front if not full."""
        current = list(self._buffer)
        pad_count = self._seq_len - len(current)
        if pad_count > 0:
            padding = [np.zeros(self._feature_dim) for _ in range(pad_count)]
            current = padding + current
        return np.array(current, dtype=np.float64)

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
