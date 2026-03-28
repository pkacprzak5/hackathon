"""Tests for temporal models and ensemble fusion."""
import torch
from squat_coach.models.temporal_tcn import TemporalTCN
from squat_coach.models.temporal_gru import TemporalGRU
from squat_coach.models.model_factory import create_model

def test_tcn_output_shapes():
    model = TemporalTCN(feature_dim=42, seq_len=60)
    model.eval()
    x = torch.randn(2, 60, 42)
    out = model(x)
    assert out.phase_probs.shape == (2, 4)
    assert out.fault_probs.shape == (2, 6)
    assert out.quality_score.shape == (2, 1)

def test_tcn_phase_logits_valid():
    model = TemporalTCN()
    model.eval()
    x = torch.randn(1, 60, 42)
    out = model(x)
    # phase_probs are raw logits -- softmax should sum to 1
    probs = torch.softmax(out.phase_probs, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 0.01

def test_factory_creates_tcn():
    model = create_model("tcn", feature_dim=42, seq_len=60)
    assert isinstance(model, TemporalTCN)

def test_gru_output_shapes():
    model = TemporalGRU(feature_dim=42, seq_len=60)
    model.eval()
    x = torch.randn(2, 60, 42)
    out = model(x)
    assert out.phase_probs.shape == (2, 4)
    assert out.fault_probs.shape == (2, 6)
    assert out.quality_score.shape == (2, 1)

def test_factory_creates_gru():
    model = create_model("gru", feature_dim=42, seq_len=60)
    assert isinstance(model, TemporalGRU)
