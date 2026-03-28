"""Squat phase state machine.

Valid transitions:
    TOP -> DESCENT -> BOTTOM -> ASCENT -> TOP

Any other transition is rejected (the state machine stays in current phase).
"""
from squat_coach.utils.enums import Phase

# Allowed transitions
_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.TOP: {Phase.DESCENT},
    Phase.DESCENT: {Phase.BOTTOM},
    Phase.BOTTOM: {Phase.ASCENT},
    Phase.ASCENT: {Phase.TOP},
}


def is_valid_transition(current: Phase, proposed: Phase) -> bool:
    """Check if a phase transition is valid."""
    if current == proposed:
        return True  # staying in same phase is always valid
    return proposed in _TRANSITIONS.get(current, set())
