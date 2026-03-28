"""End-to-end training script: generate data, train all models, evaluate."""
import logging
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch

from squat_coach.training.data_pipeline import generate_synthetic_dataset, compute_normalization_stats
from squat_coach.training.dataset import SquatSequenceDataset
from squat_coach.training.trainer import Trainer
from squat_coach.training.evaluate import evaluate_model
from squat_coach.models.model_factory import create_model
# Import models to register them
import squat_coach.models.temporal_tcn  # noqa: F401
import squat_coach.models.temporal_gru  # noqa: F401

logger = logging.getLogger("squat_coach.training")


def train_all(config_path: str = "squat_coach/config/model.yaml") -> None:
    """Train all enabled models end-to-end."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seq_len = config["sequence"]["length"]
    feature_dim = config["sequence"]["feature_dim"]
    training_cfg = config["training"]
    checkpoint_dir = config["checkpoints"]["dir"]

    # Generate synthetic training data
    cache_path = str(Path(checkpoint_dir) / "synthetic_data.npz")
    stats_path = str(Path(checkpoint_dir) / "feature_stats.json")

    features, phase_labels, fault_labels, quality_labels = generate_synthetic_dataset(
        num_samples=3000,
        seq_len=seq_len,
        feature_dim=feature_dim,
        cache_path=cache_path,
    )

    # Compute normalization stats
    compute_normalization_stats(features, stats_path)

    # Normalize features
    import json
    import numpy as np
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    features_norm = (features - mean) / std

    # Create dataset
    dataset = SquatSequenceDataset(features_norm, phase_labels, fault_labels, quality_labels)

    # Split: 70/15/15
    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=training_cfg["batch_size"])

    # Train each enabled model
    models_to_train = ["tcn", "gru"]
    for model_name in models_to_train:
        model_cfg = config["models"].get(model_name, {})
        if not model_cfg.get("enabled", False):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        logger.info("=" * 60)
        logger.info("Training model: %s", model_name)
        logger.info("=" * 60)

        # Create model
        model_params = {k: v for k, v in model_cfg.items() if k != "enabled"}
        model_params["feature_dim"] = feature_dim
        model_params["seq_len"] = seq_len
        model = create_model(model_name, **model_params)

        # Train
        trainer = Trainer(
            model=model,
            lr=training_cfg["learning_rate"],
            loss_weights=training_cfg["loss_weights"],
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=training_cfg["max_epochs"],
            patience=training_cfg["patience"],
        )

        # Evaluate
        device = trainer._device
        # Reload best checkpoint
        ckpt = Path(checkpoint_dir) / f"{model_name}_side_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)

        results = evaluate_model(model, test_loader, device)
        logger.info("Model %s test results: %s", model_name, results)

    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
