"""ST-GCN scaffold -- extension point for future skeleton-graph model.

This is NOT trained or used in production. It defines the interface
and placeholder architecture for a future Spatial-Temporal Graph
Convolutional Network that would operate on the skeleton graph directly.

To activate: implement the graph convolution layers and register the model.
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput

# NOT registered in factory -- scaffold only
class STGCNScaffold(TemporalModelBase):
    """Placeholder for future ST-GCN implementation.

    Expected input: (batch, seq_len, num_joints, feature_dim_per_joint)
    The adjacency matrix would define the skeleton graph connectivity.
    """

    # Skeleton adjacency matrix (33 joints, BlazePose topology)
    # This defines which joints are connected in the body graph
    # TODO: Fill in actual graph convolution layers

    def __init__(self, feature_dim: int = 42, seq_len: int = 60) -> None:
        super().__init__(feature_dim, seq_len)
        self._hidden_dim = 128
        # Placeholder linear layer
        self.placeholder = nn.Linear(feature_dim * seq_len, self._hidden_dim)
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """Placeholder forward pass."""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        hidden = torch.relu(self.placeholder(flat))
        return self._apply_heads(hidden)
