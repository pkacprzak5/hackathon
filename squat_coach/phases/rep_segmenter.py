"""Rep boundary detection and validation."""
from dataclasses import dataclass
from typing import Optional
from squat_coach.utils.enums import Phase


@dataclass
class RepResult:
    """Result of a completed rep."""
    rep_index: int
    start_time: float
    bottom_time: float
    end_time: float
    descent_duration: float
    bottom_duration: float
    ascent_duration: float
    valid: bool
    rejection_reason: str = ""


class RepSegmenter:
    """Segment squat reps from phase transitions.

    Tracks the phase state machine: TOP → DESCENT → BOTTOM → ASCENT → TOP.
    A rep is complete when we return to TOP after visiting BOTTOM.
    """

    def __init__(
        self,
        min_rep_duration_s: float = 0.5,
        cooldown_s: float = 0.3,
        fps: float = 30.0,
    ) -> None:
        self._min_rep_duration = min_rep_duration_s
        self._cooldown = cooldown_s
        self._rep_count = 0
        self._in_rep = False
        self._hit_bottom = False
        self._rep_start_time = 0.0
        self._descent_start_time = 0.0
        self._bottom_start_time = 0.0
        self._ascent_start_time = 0.0
        self._last_rep_end_time = -999.0
        self._prev_phase = Phase.TOP

    def update(self, phase: Phase, timestamp: float) -> Optional[RepResult]:
        """Update with current phase. Returns RepResult when a rep completes."""
        result = None

        if phase != self._prev_phase:
            # Start of a new rep
            if phase == Phase.DESCENT and not self._in_rep:
                if timestamp - self._last_rep_end_time >= self._cooldown:
                    self._in_rep = True
                    self._hit_bottom = False
                    self._rep_start_time = timestamp
                    self._descent_start_time = timestamp

            elif phase == Phase.BOTTOM and self._in_rep:
                self._hit_bottom = True
                self._bottom_start_time = timestamp

            elif phase == Phase.ASCENT and self._in_rep:
                self._ascent_start_time = timestamp

            elif phase == Phase.TOP and self._in_rep and self._hit_bottom:
                # Rep complete — we went down and came back up
                result = self._complete_rep(timestamp)

        self._prev_phase = phase
        return result

    def _complete_rep(self, timestamp: float) -> RepResult:
        self._rep_count += 1
        duration = timestamp - self._rep_start_time
        valid = duration >= self._min_rep_duration

        # Safe duration calculations
        descent_dur = max(0, self._bottom_start_time - self._descent_start_time)
        bottom_dur = max(0, self._ascent_start_time - self._bottom_start_time) if self._ascent_start_time > self._bottom_start_time else 0.1
        ascent_dur = max(0, timestamp - self._ascent_start_time) if self._ascent_start_time > self._bottom_start_time else 0.1

        result = RepResult(
            rep_index=self._rep_count,
            start_time=self._rep_start_time,
            bottom_time=self._bottom_start_time,
            end_time=timestamp,
            descent_duration=descent_dur,
            bottom_duration=bottom_dur,
            ascent_duration=ascent_dur,
            valid=valid,
            rejection_reason="" if valid else f"Too short: {duration:.2f}s",
        )

        self._in_rep = False
        self._hit_bottom = False
        self._last_rep_end_time = timestamp
        return result

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def reset(self) -> None:
        self._rep_count = 0
        self._in_rep = False
        self._hit_bottom = False
        self._prev_phase = Phase.TOP
