"""Coaching cue arbitration -- select the most important cue to show."""
import time
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.utils.enums import FaultType


class CoachingPrioritizer:
    """Select the single most important coaching cue.

    Priority = severity * confidence * persistence * (1 / recency)
    Where recency = seconds since this fault type was last displayed.
    """

    def __init__(self, suppress_repeat_s: float = 5.0) -> None:
        self._suppress_s = suppress_repeat_s
        self._last_shown: dict[FaultType, float] = {}  # fault_type -> timestamp

    def select_cue(self, faults: list[FaultDetection]) -> str:
        """Select the highest-priority coaching cue.

        Returns:
            Coaching cue string, or empty string if no faults.
        """
        if not faults:
            return ""

        now = time.monotonic()
        best_priority = -1.0
        best_cue = ""

        for fault in faults:
            last = self._last_shown.get(fault.fault_type, 0.0)
            recency = max(now - last, 1.0)

            # Suppress recently shown cues
            if now - last < self._suppress_s and last > 0:
                continue

            priority = fault.severity * fault.confidence * (1.0 / recency)
            if priority > best_priority:
                best_priority = priority
                best_cue = fault.explanation_token

        if best_cue:
            # Record that we're showing this cue now
            for fault in faults:
                if fault.explanation_token == best_cue:
                    self._last_shown[fault.fault_type] = now
                    break

        return best_cue
