"""Confidence-weighted ensemble fusion of temporal model outputs.

Combines outputs from TCN and GRU (and optionally other models) into
a single fused prediction. Supports per-head weight configuration
and graceful fallback when a model is unavailable.
"""
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from squat_coach.models.temporal_base import TemporalModelOutput


@dataclass
class FusedOutput:
    """Result of ensemble fusion."""
    phase_probs: np.ndarray         # (4,) fused phase probabilities
    fault_probs: np.ndarray         # (6,) fused fault probabilities
    quality_score: float            # Fused quality [0-1]
    confidence: float               # Assessment confidence [0-1]
    model_agreement: float          # How much models agree [0-1]


class EnsembleFusion:
    """Fuse outputs from multiple temporal models."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        per_head_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Args:
            weights: Default per-model weights, e.g. {'tcn': 0.5, 'gru': 0.5}.
            per_head_weights: Optional per-head overrides.
        """
        self._weights = weights or {"tcn": 0.5, "gru": 0.5}
        self._per_head = per_head_weights or {}

    def fuse(self, outputs: dict[str, TemporalModelOutput]) -> FusedOutput:
        """Fuse model outputs into a single prediction.

        Args:
            outputs: Dict mapping model name to its TemporalModelOutput.

        Returns:
            FusedOutput with weighted-average predictions and confidence.
        """
        if not outputs:
            return self._empty_output()

        # Get weights for available models, re-normalize
        available = {k: v for k, v in self._weights.items() if k in outputs}
        if not available:
            # Fallback: equal weight for whatever we have
            available = {k: 1.0 / len(outputs) for k in outputs}
        total_w = sum(available.values())
        norm_w = {k: v / total_w for k, v in available.items()}

        # Fuse each head
        phase = self._fuse_head(outputs, norm_w, "phase")
        fault = self._fuse_head(outputs, norm_w, "fault")
        quality = self._fuse_head(outputs, norm_w, "quality")

        # Confidence from model agreement and phase entropy
        agreement = self._compute_agreement(outputs, "phase")
        phase_entropy = -np.sum(phase * np.log(phase + 1e-8))
        max_entropy = -np.log(1.0 / 4.0)
        confidence = agreement * (1.0 - phase_entropy / max_entropy)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return FusedOutput(
            phase_probs=phase,
            fault_probs=fault,
            quality_score=float(quality.item()) if quality.size > 0 else 0.5,
            confidence=confidence,
            model_agreement=agreement,
        )

    def _fuse_head(
        self,
        outputs: dict[str, TemporalModelOutput],
        default_weights: dict[str, float],
        head: str,
    ) -> np.ndarray:
        """Weighted average of a specific output head."""
        head_weights = self._per_head.get(head, default_weights)
        # Re-normalize for available models
        available_w = {k: v for k, v in head_weights.items() if k in outputs}
        if not available_w:
            available_w = default_weights
        total = sum(available_w.values())

        result = None
        for name, weight in available_w.items():
            if name not in outputs:
                continue
            out = outputs[name]
            if head == "phase":
                # phase_probs are raw logits from model -- apply softmax for probabilities
                arr = torch.softmax(out.phase_probs, dim=-1).detach().cpu().numpy().squeeze()
            elif head == "fault":
                arr = out.fault_probs.detach().cpu().numpy().squeeze()
            else:
                arr = out.quality_score.detach().cpu().numpy().squeeze()

            weighted = arr * (weight / total)
            result = weighted if result is None else result + weighted

        return result if result is not None else np.zeros(4)

    def _compute_agreement(self, outputs: dict[str, TemporalModelOutput], head: str) -> float:
        """How much models agree on phase prediction (0=disagree, 1=agree)."""
        if len(outputs) < 2:
            return 1.0
        predictions = []
        for out in outputs.values():
            if head == "phase":
                predictions.append(torch.softmax(out.phase_probs, dim=-1).detach().cpu().numpy().squeeze())
        # Agreement = 1 - mean pairwise L1 distance
        dists = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                dists.append(np.mean(np.abs(predictions[i] - predictions[j])))
        mean_dist = np.mean(dists) if dists else 0.0
        return float(1.0 - min(mean_dist, 1.0))

    def _empty_output(self) -> FusedOutput:
        return FusedOutput(
            phase_probs=np.array([1.0, 0.0, 0.0, 0.0]),
            fault_probs=np.zeros(6),
            quality_score=0.5,
            confidence=0.0,
            model_agreement=0.0,
        )
