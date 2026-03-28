"""Base class for temporal sequence models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TemporalModelOutput:
    """Standardized output from all temporal models."""
    phase_probs: torch.Tensor       # (batch, 4) softmax probabilities
    fault_probs: torch.Tensor       # (batch, 6) sigmoid probabilities
    quality_score: torch.Tensor     # (batch, 1) regression [0-1]

class TemporalModelBase(nn.Module, ABC):
    """Abstract base class for temporal squat analysis models.

    All temporal models receive input of shape (batch, seq_len, feature_dim)
    and produce TemporalModelOutput with phase, fault, and quality predictions.
    """

    def __init__(self, feature_dim: int = 42, seq_len: int = 60) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._seq_len = seq_len

        # Shared output heads (subclasses set self._hidden_dim before calling _build_heads)
        self._hidden_dim: int = 0  # Must be set by subclass

    def _build_heads(self) -> None:
        """Build output heads. Call after setting self._hidden_dim."""
        self.phase_head = nn.Linear(self._hidden_dim, 4)
        self.fault_head = nn.Linear(self._hidden_dim, 6)
        self.quality_head = nn.Linear(self._hidden_dim, 1)

    def _apply_heads(self, hidden: torch.Tensor) -> TemporalModelOutput:
        """Apply output heads to hidden representation.

        Note: phase_probs returns RAW LOGITS during training (for CrossEntropyLoss).
        Call .softmax() on phase_probs at inference time for probabilities.
        fault_probs and quality_score use sigmoid since BCE/MSE expect [0,1].
        """
        phase_logits = self.phase_head(hidden)
        return TemporalModelOutput(
            phase_probs=phase_logits,  # Raw logits -- CE loss applies softmax internally
            fault_probs=torch.sigmoid(self.fault_head(hidden)),
            quality_score=torch.sigmoid(self.quality_head(hidden)),
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """Forward pass. x shape: (batch, seq_len, feature_dim)."""
        ...
