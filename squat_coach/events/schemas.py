"""Event dataclasses for the squat coach event system."""
from dataclasses import dataclass, field
from typing import Any
from squat_coach.utils.enums import Phase, FaultType


@dataclass
class SquatEvent:
    """Base event type."""
    event_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepSummaryEvent(SquatEvent):
    """Emitted when a rep is completed."""
    event_type: str = "rep_completed"

    # Populated by event_builder
    rep_index: int = 0
    phase_durations: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    coaching_cue: str = ""
