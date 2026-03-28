"""Session-level state tracking."""
from dataclasses import dataclass, field
from squat_coach.utils.enums import Phase, ViewType


@dataclass
class SessionState:
    """Mutable state for the current session."""
    is_calibrated: bool = False
    current_phase: Phase = Phase.TOP
    rep_count: int = 0
    current_score: float = 0.0
    overall_score: float = 0.0
    current_cue: str = ""
    view_type: ViewType = ViewType.UNKNOWN
    frame_index: int = 0
    dropped_frame_count: int = 0
