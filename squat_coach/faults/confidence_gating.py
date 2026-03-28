"""Suppress low-confidence fault detections."""
from squat_coach.faults.fault_types import FaultDetection


def apply_confidence_gate(
    detections: list[FaultDetection], threshold: float = 0.3
) -> list[FaultDetection]:
    """Filter out faults below confidence threshold."""
    return [d for d in detections if d.confidence >= threshold]
