"""Manage multi-model temporal inference.

Loads model checkpoints, runs inference on sequence windows,
and fuses outputs via the ensemble layer.
"""
import logging
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from numpy.typing import NDArray

from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import create_model
from squat_coach.models.ensemble_fusion import EnsembleFusion, FusedOutput
from squat_coach.models.feature_tensor_builder import FeatureTensorBuilder

logger = logging.getLogger("squat_coach.inference")


class InferenceManager:
    """Load models and run ensemble inference on feature sequences."""

    def __init__(
        self,
        model_configs: dict,
        ensemble_config: dict,
        checkpoint_dir: str = "squat_coach/models/checkpoints",
        stats_path: Optional[str] = None,
        device: str = "cpu",
        view: str = "side",
    ) -> None:
        self._device = torch.device(device)
        self._models: dict[str, TemporalModelBase] = {}
        self._tensor_builder = FeatureTensorBuilder(stats_path)
        self._fusion = EnsembleFusion(
            weights=ensemble_config.get("weights"),
            per_head_weights=ensemble_config.get("per_head_weights"),
        )

        # Load enabled models
        for name, cfg in model_configs.items():
            if not cfg.get("enabled", False):
                continue
            try:
                model = create_model(name, **{k: v for k, v in cfg.items() if k != "enabled"})
                # Try loading checkpoint
                ckpt_path = Path(checkpoint_dir) / f"{name}_{view}_best.pt"
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, map_location=self._device, weights_only=True)
                    model.load_state_dict(state)
                    logger.info("Loaded checkpoint: %s", ckpt_path)
                else:
                    logger.warning("No checkpoint found at %s, using random weights", ckpt_path)

                model.to(self._device)
                model.eval()
                self._models[name] = model
                logger.info("Model loaded: %s (device=%s)", name, self._device)
            except Exception as e:
                logger.error("Failed to load model %s: %s", name, e)

    def infer(self, sequence: NDArray[np.float64]) -> FusedOutput:
        """Run ensemble inference on a feature sequence.

        Args:
            sequence: (seq_len, 42) feature array.

        Returns:
            FusedOutput with fused predictions.
        """
        if not self._models:
            return self._fusion.fuse({})

        tensor = self._tensor_builder.to_tensor(sequence).to(self._device)

        outputs: dict[str, TemporalModelOutput] = {}
        with torch.no_grad():
            for name, model in self._models.items():
                outputs[name] = model(tensor)

        return self._fusion.fuse(outputs)

    @property
    def has_models(self) -> bool:
        return len(self._models) > 0
