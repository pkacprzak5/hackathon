"""Unified training loop for temporal models. Supports MPS and CPU."""
import logging
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from squat_coach.models.temporal_base import TemporalModelBase

logger = logging.getLogger("squat_coach.training")


class Trainer:
    """Train a temporal model with multi-task loss."""

    def __init__(
        self,
        model: TemporalModelBase,
        device: str = "auto",
        lr: float = 0.001,
        loss_weights: dict[str, float] | None = None,
        checkpoint_dir: str = "squat_coach/models/checkpoints",
        model_name: str = "model",
        view: str = "side",
    ) -> None:
        self._device = self._resolve_device(device)
        self._model = model.to(self._device)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._loss_weights = loss_weights or {"phase": 1.0, "fault": 1.0, "quality": 0.5}
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._view = view

        self._phase_loss = nn.CrossEntropyLoss()
        self._fault_loss = nn.BCELoss()
        self._quality_loss = nn.MSELoss()

        logger.info("Trainer initialized: model=%s, device=%s", model_name, self._device)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
        elif device in ("mps", "cuda"):
            return torch.device(device)
        return torch.device("cpu")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        patience: int = 10,
    ) -> dict:
        """Train the model with early stopping.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(max_epochs):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | time=%.1fs",
                epoch + 1, max_epochs, train_loss, val_loss, elapsed,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(f"{self._model_name}_{self._view}_best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        count = 0

        for batch in loader:
            features = batch["features"].to(self._device)
            phase_labels = batch["phase_labels"].to(self._device)
            fault_labels = batch["fault_labels"].to(self._device)
            quality_labels = batch["quality_label"].to(self._device)

            self._optimizer.zero_grad()
            output = self._model(features)

            loss = self._compute_loss(output, phase_labels, fault_labels, quality_labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    def _validate(self, loader: DataLoader) -> float:
        self._model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self._device)
                phase_labels = batch["phase_labels"].to(self._device)
                fault_labels = batch["fault_labels"].to(self._device)
                quality_labels = batch["quality_label"].to(self._device)

                output = self._model(features)
                loss = self._compute_loss(output, phase_labels, fault_labels, quality_labels)
                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def _compute_loss(self, output, phase_labels, fault_labels, quality_labels) -> torch.Tensor:
        w = self._loss_weights
        l_phase = self._phase_loss(output.phase_probs, phase_labels)
        l_fault = self._fault_loss(output.fault_probs, fault_labels)
        l_quality = self._quality_loss(output.quality_score.squeeze(), quality_labels)
        return w["phase"] * l_phase + w["fault"] * l_fault + w["quality"] * l_quality

    def _save_checkpoint(self, filename: str) -> None:
        path = self._checkpoint_dir / filename
        torch.save(self._model.state_dict(), path)
        logger.info("Saved checkpoint: %s", path)
