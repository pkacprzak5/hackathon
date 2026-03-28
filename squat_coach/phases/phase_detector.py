"""Phase detection using knee angle direction tracking with timeout recovery.

Tracks the smoothed knee angle and its direction of change. Uses relative
thresholds (how much the knee has changed) rather than absolute angle values,
which makes it robust to different body types and camera angles.

Includes timeout recovery: if stuck in any phase for too long, resets to TOP.
"""
import logging
import numpy as np
from numpy.typing import NDArray
from collections import deque
from squat_coach.utils.enums import Phase

logger = logging.getLogger("squat_coach.phase")


class PhaseDetector:
    """Detect squat phase from knee angle direction changes."""

    def __init__(
        self,
        min_phase_duration_s: float = 0.1,
        fps: float = 30.0,
        confidence_threshold: float = 0.4,
        direction_window: int = 5,
        hysteresis_deg: float = 2.0,
        calibrated_knee_angle: float | None = None,
        phase_timeout_s: float = 8.0,
    ) -> None:
        self._min_frames = max(1, int(min_phase_duration_s * fps))
        self._conf_threshold = confidence_threshold
        self._direction_window = direction_window
        self._hysteresis = hysteresis_deg
        self._timeout_frames = int(phase_timeout_s * fps)

        self._current_phase = Phase.TOP
        self._frames_in_phase = self._min_frames

        # Knee angle from calibration
        self._cal_knee = calibrated_knee_angle or 165.0

        # Use relative thresholds: percentage of range from standing
        # These work regardless of actual standing angle
        self._descent_drop = 12.0    # knee must drop 12° to start descent
        self._bottom_drop = 25.0     # knee must drop 25° total for bottom territory
        self._ascent_rise = 10.0     # knee must rise 10° from minimum for ascent
        self._top_return = 12.0      # knee must be within 12° of standing for top

        # Tracking
        self._knee_buf: deque[float] = deque(maxlen=120)
        self._direction: str = "stable"
        self._rep_knee_min: float = 180.0
        self._rep_knee_max: float = 0.0

    def detect(
        self,
        phase_probs: NDArray[np.float64],
        hip_y: float,
        knee_angle: float | None = None,
    ) -> Phase:
        """Detect current phase."""
        self._frames_in_phase += 1

        if knee_angle is None:
            return self._current_phase

        self._knee_buf.append(knee_angle)

        # Need enough history for smoothing
        if len(self._knee_buf) < self._direction_window * 2:
            return self._current_phase

        # Timeout recovery: if stuck in any non-TOP phase too long, reset
        if self._current_phase != Phase.TOP and self._frames_in_phase > self._timeout_frames:
            logger.info("Phase timeout in %s (%d frames), resetting to TOP",
                        self._current_phase.value, self._frames_in_phase)
            self._current_phase = Phase.TOP
            self._frames_in_phase = 0
            self._rep_knee_min = 180.0
            self._rep_knee_max = 0.0
            return self._current_phase

        # Smoothed knee angle
        buf = list(self._knee_buf)
        smooth_now = np.mean(buf[-self._direction_window:])
        smooth_prev = np.mean(buf[-(self._direction_window * 2):-self._direction_window])
        delta = smooth_now - smooth_prev

        # Update direction
        if delta < -self._hysteresis:
            self._direction = "down"
        elif delta > self._hysteresis:
            self._direction = "up"

        # Track extremes during rep
        self._rep_knee_min = min(self._rep_knee_min, smooth_now)
        self._rep_knee_max = max(self._rep_knee_max, smooth_now)

        # How far from standing
        drop_from_standing = self._cal_knee - smooth_now  # positive = more bent
        rise_from_min = smooth_now - self._rep_knee_min   # positive = straightening

        proposed = self._propose_phase(smooth_now, drop_from_standing, rise_from_min)

        # Apply valid transitions + debounce
        if proposed != self._current_phase and self._frames_in_phase >= self._min_frames:
            valid_next = {
                Phase.TOP: {Phase.DESCENT},
                Phase.DESCENT: {Phase.BOTTOM},
                Phase.BOTTOM: {Phase.ASCENT},
                Phase.ASCENT: {Phase.TOP},
            }
            if proposed in valid_next.get(self._current_phase, set()):
                self._current_phase = proposed
                self._frames_in_phase = 0

                if proposed == Phase.DESCENT:
                    self._rep_knee_min = smooth_now
                    self._rep_knee_max = smooth_now
                elif proposed == Phase.TOP:
                    self._rep_knee_min = 180.0
                    self._rep_knee_max = 0.0

        return self._current_phase

    def _propose_phase(self, knee: float, drop: float, rise: float) -> Phase:
        """Propose phase from relative knee angle changes."""

        if self._current_phase == Phase.TOP:
            if self._direction == "down" and drop > self._descent_drop:
                return Phase.DESCENT

        elif self._current_phase == Phase.DESCENT:
            # Bottom when: deep enough AND (direction reverses OR knee stabilizes)
            if drop > self._bottom_drop:
                if self._direction != "down":
                    return Phase.BOTTOM
                # Also bottom if knee has been stable for a few frames
                recent = list(self._knee_buf)[-5:]
                if len(recent) >= 5 and max(recent) - min(recent) < 4.0:
                    return Phase.BOTTOM

        elif self._current_phase == Phase.BOTTOM:
            if self._direction == "up" and rise > self._ascent_rise:
                return Phase.ASCENT

        elif self._current_phase == Phase.ASCENT:
            if drop < self._top_return:
                return Phase.TOP

        return self._current_phase

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    def reset(self) -> None:
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0
        self._knee_buf.clear()
        self._direction = "stable"
        self._rep_knee_min = 180.0
        self._rep_knee_max = 0.0
