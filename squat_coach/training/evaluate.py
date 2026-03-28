"""Evaluation metrics for trained models."""
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from squat_coach.models.temporal_base import TemporalModelBase

logger = logging.getLogger("squat_coach.training")


def evaluate_model(
    model: TemporalModelBase,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model on test data.

    Returns:
        Dict with phase_accuracy, fault_f1, quality_mae.
    """
    model.eval()
    all_phase_pred = []
    all_phase_true = []
    all_fault_pred = []
    all_fault_true = []
    all_quality_pred = []
    all_quality_true = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            output = model(features)

            all_phase_pred.extend(output.phase_probs.argmax(dim=-1).cpu().numpy())
            all_phase_true.extend(batch["phase_labels"].numpy())
            all_fault_pred.extend((output.fault_probs > 0.5).cpu().numpy())
            all_fault_true.extend(batch["fault_labels"].numpy())
            all_quality_pred.extend(output.quality_score.squeeze().cpu().numpy())
            all_quality_true.extend(batch["quality_label"].numpy())

    phase_acc = np.mean(np.array(all_phase_pred) == np.array(all_phase_true))

    # Fault F1
    fault_pred = np.array(all_fault_pred)
    fault_true = np.array(all_fault_true)
    tp = np.sum((fault_pred == 1) & (fault_true == 1))
    fp = np.sum((fault_pred == 1) & (fault_true == 0))
    fn = np.sum((fault_pred == 0) & (fault_true == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fault_f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    quality_mae = np.mean(np.abs(np.array(all_quality_pred) - np.array(all_quality_true)))

    results = {
        "phase_accuracy": float(phase_acc),
        "fault_f1": float(fault_f1),
        "quality_mae": float(quality_mae),
    }
    logger.info("Evaluation: %s", results)
    return results
