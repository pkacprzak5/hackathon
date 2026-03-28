"""Combine component scores into rep quality and overall form scores."""


def compute_rep_quality_score(
    component_scores: dict[str, float],
    model_quality: float = 0.5,
    weights: dict[str, float] | None = None,
    model_weight: float = 0.2,
) -> float:
    """Compute weighted rep quality score (0-100).

    Args:
        component_scores: Dict with keys: depth, trunk_control, posture_stability, movement_consistency.
        model_quality: Model's quality_score output [0-1].
        weights: Per-component weights (default: equal).
        model_weight: Weight for model quality (default 0.2).
    """
    if weights is None:
        weights = {"depth": 0.25, "trunk_control": 0.25, "posture_stability": 0.25, "movement_consistency": 0.25}

    component_total = sum(
        component_scores.get(k, 50.0) * w for k, w in weights.items()
    )
    # Model contributes model_weight, components contribute (1 - model_weight)
    return (1.0 - model_weight) * component_total + model_weight * (model_quality * 100.0)
