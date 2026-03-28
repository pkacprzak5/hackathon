"""Aggregate evidence from rules and model outputs to detect faults."""
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.faults.fault_rules import FAULT_RULES
from squat_coach.faults.confidence_gating import apply_confidence_gate


class EvidenceEngine:
    """Run all fault rules and aggregate results."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        self._conf_threshold = confidence_threshold

    def evaluate(
        self, features: dict, config: dict
    ) -> list[FaultDetection]:
        """Evaluate all fault rules against current features.

        Args:
            features: Named features from SquatFeatureExtractor.
            config: Fault threshold config (from scoring.yaml).

        Returns:
            List of detected faults (confidence-gated).
        """
        detections: list[FaultDetection] = []
        for rule_fn in FAULT_RULES:
            result = rule_fn(features, config)
            if result is not None:
                detections.append(result)

        return apply_confidence_gate(detections, self._conf_threshold)
