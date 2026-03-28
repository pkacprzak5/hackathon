"""Phase detection using knee angle direction tracking.

Simple, robust approach that doesn't depend on a fixed baseline:
- Track smoothed knee angle and its direction (bending vs straightening)
- Detect local peaks (standing) and valleys (bottom of squat)
- Use hysteresis to prevent jitter

Knee angle: ~160-175° standing, ~70-110° deep squat.
"""
import logging
import numpy as np
from numpy.typing import NDArray
from collections import deque
from squat_coach.utils.enums import Phase

logger = logging.getLogger("squat_coach.phase")


class PhaseDetector:
    """Detect squat phase from knee angle direction changes.

    Adapts to the user's actual knee angle range by learning from
    calibration data. Uses direction of knee angle change (bending
    vs straightening) with hysteresis to prevent jitter.
    """

    def __init__(
        self,
        min_phase_duration_s: float = 0.1,
        fps: float = 30.0,
        confidence_threshold: float = 0.4,
        direction_window: int = 6,
        hysteresis_deg: float = 3.0,
        calibrated_knee_angle: float | None = None,
    ) -> None:
        self._min_frames = max(1, int(min_phase_duration_s * fps))
        self._conf_threshold = confidence_threshold
        self._direction_window = direction_window
        self._hysteresis = hysteresis_deg

        self._current_phase = Phase.TOP
        self._frames_in_phase = self._min_frames

        # Adaptive thresholds — set from calibrated standing angle
        if calibrated_knee_angle and calibrated_knee_angle > 100:
            self._standing_angle = calibrated_knee_angle - 15.0
            self._squat_angle = calibrated_knee_angle - 30.0
        else:
            self._standing_angle = 145.0
            self._squat_angle = 125.0

        # Knee angle tracking
        self._knee_buf: deque[float] = deque(maxlen=90)
        self._direction: str = "stable"
        self._local_max: float = 0.0
        self._local_min: float = 180.0

    _PHASE_ORDER = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT]

    # Valid transitions
    _VALID = {
        Phase.TOP: {Phase.DESCENT},
        Phase.DESCENT: {Phase.BOTTOM},
        Phase.BOTTOM: {Phase.ASCENT},
        Phase.ASCENT: {Phase.TOP},
    }

    def detect(
        self,
        phase_probs: NDArray[np.float64],
        hip_y: float,
        knee_angle: float | None = None,
    ) -> Phase:
        """Detect current phase from knee angle.

        Args:
            phase_probs: Model output (used when confidence is high).
            hip_y: Not used in this version (kept for API compat).
            knee_angle: Primary knee angle in degrees.
        """
        self._frames_in_phase += 1

        if knee_angle is None:
            return self._current_phase

        self._knee_buf.append(knee_angle)
        if len(self._knee_buf) < self._direction_window + 2:
            return self._current_phase

        # Smoothed current value and recent direction
        smooth_now = np.mean(list(self._knee_buf)[-self._direction_window:])
        smooth_prev = np.mean(list(self._knee_buf)[-(self._direction_window * 2):-self._direction_window])
        delta = smooth_now - smooth_prev  # positive = straightening, negative = bending

        # Update direction with hysteresis
        if delta < -self._hysteresis:
            self._direction = "down"  # knee bending
        elif delta > self._hysteresis:
            self._direction = "up"    # knee straightening
        # else: keep previous direction (hysteresis deadband)

        # Track local extremes
        if smooth_now > self._local_max:
            self._local_max = smooth_now
        if smooth_now < self._local_min:
            self._local_min = smooth_now

        # Determine proposed phase
        proposed = self._propose_phase(smooth_now)

        # Apply transition rules + debounce
        if proposed != self._current_phase:
            if (
                proposed in self._VALID.get(self._current_phase, set())
                and self._frames_in_phase >= self._min_frames
            ):
                self._current_phase = proposed
                self._frames_in_phase = 0

                # Reset tracking on phase changes
                if proposed == Phase.DESCENT:
                    self._local_min = smooth_now
                elif proposed == Phase.TOP:
                    self._local_max = smooth_now
                    self._local_min = 180.0

        return self._current_phase

    def _propose_phase(self, knee: float) -> Phase:
        """Propose a phase based on knee angle and direction."""

        if self._current_phase == Phase.TOP:
            # Start descent: knee bending and below standing threshold
            if self._direction == "down" and knee < self._standing_angle:
                return Phase.DESCENT

        elif self._current_phase == Phase.DESCENT:
            # Bottom: knee stops going down (direction flips or stabilizes)
            if knee < self._squat_angle and self._direction != "down":
                return Phase.BOTTOM
            # Also bottom if we've been descending and velocity slows way down
            if knee < self._squat_angle:
                recent = list(self._knee_buf)[-4:]
                if len(recent) >= 4 and abs(recent[-1] - recent[0]) < 3.0:
                    return Phase.BOTTOM

        elif self._current_phase == Phase.BOTTOM:
            # Ascent: knee starts straightening
            if self._direction == "up" and knee > self._local_min + self._hysteresis * 2:
                return Phase.ASCENT

        elif self._current_phase == Phase.ASCENT:
            # Top: knee back above standing threshold
            if knee > self._standing_angle:
                return Phase.TOP
            # Also top if direction flips to stable/down near standing
            if knee > self._standing_angle - 10 and self._direction != "up":
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
        self._local_max = 0.0
        self._local_min = 180.0
