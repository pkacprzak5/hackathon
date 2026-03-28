"""Gated Recurrent Unit model for squat temporal analysis.

Lightweight RNN. ~1ms inference per window. Good at capturing
sequential motion dynamics in squat phase transitions.
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import register_model


@register_model("gru")
class TemporalGRU(TemporalModelBase):
    """GRU model for squat temporal analysis."""

    def __init__(
        self,
        feature_dim: int = 42,
        seq_len: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(feature_dim, seq_len)
        self._hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """x: (batch, seq_len, feature_dim) -> TemporalModelOutput."""
        output, _ = self.gru(x)
        # Take last timestep
        hidden = output[:, -1, :]
        return self._apply_heads(hidden)
