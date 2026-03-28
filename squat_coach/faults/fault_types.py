"""Fault type definitions and dataclasses."""
from dataclasses import dataclass, field
from squat_coach.utils.enums import FaultType

# Coaching cue templates for each fault type
FAULT_CUES: dict[FaultType, str] = {
    FaultType.INSUFFICIENT_DEPTH: "Try to squat deeper",
    FaultType.EXCESSIVE_FORWARD_LEAN: "Keep your chest up",
    FaultType.ROUNDED_BACK_RISK: "Keep your chest up at the bottom",
    FaultType.UNSTABLE_TORSO: "Keep your torso steady",
    FaultType.HEEL_FAULT: "Keep your heels on the ground",
    FaultType.KNEE_VALGUS: "Push your knees out over your toes",
    FaultType.INCONSISTENT_TEMPO: "Try to maintain a steady pace",
    FaultType.POOR_TRUNK_CONTROL: "Control your trunk through the movement",
    FaultType.LOW_CONFIDENCE: "Move to a better position for the camera",
    FaultType.INVALID_VIEW: "Adjust camera angle",
}

@dataclass
class FaultDetection:
    """A detected fault with evidence."""
    fault_type: FaultType
    severity: float             # 0.0-1.0
    confidence: float           # 0.0-1.0
    evidence: list[str]         # What data supports this detection
    explanation_token: str      # Short coaching cue
    affects_overlay: bool = True
    affects_gemini: bool = True
