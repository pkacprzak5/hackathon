"""Temporal Convolutional Network for squat analysis.

Uses causal dilated convolutions to capture temporal patterns.
Fastest inference of all temporal models (~0.5ms per window).
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import register_model


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self._padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self._padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding (causal)
        if self._padding > 0:
            out = out[:, :, :-self._padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with causal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.residual(x))


@register_model("tcn")
class TemporalTCN(TemporalModelBase):
    """TCN model for squat temporal analysis."""

    def __init__(
        self,
        feature_dim: int = 42,
        seq_len: int = 60,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(feature_dim, seq_len)
        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        in_ch = feature_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self._hidden_dim = num_channels[-1]
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """x: (batch, seq_len, feature_dim) -> TemporalModelOutput."""
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Take last timestep
        hidden = out[:, :, -1]
        return self._apply_heads(hidden)
