"""Shared enumerations."""
from enum import Enum

class Phase(Enum):
    TOP = "top"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"

class ViewType(Enum):
    SIDE = "side"
    FRONT = "front"
    UNKNOWN = "unknown"

class FaultType(Enum):
    INSUFFICIENT_DEPTH = "insufficient_depth"
    EXCESSIVE_FORWARD_LEAN = "excessive_forward_lean"
    ROUNDED_BACK_RISK = "rounded_back_risk"
    UNSTABLE_TORSO = "unstable_torso"
    HEEL_FAULT = "heel_fault"
    KNEE_VALGUS = "knee_valgus"
    INCONSISTENT_TEMPO = "inconsistent_tempo"
    POOR_TRUNK_CONTROL = "poor_trunk_control"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_VIEW = "invalid_view"
