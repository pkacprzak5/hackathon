"""Phase detection from fused model outputs with fallback to knee angle kinematics.

Uses model phase_probs as primary signal when confidence is sufficient.
Falls back to knee angle tracking when model confidence is low — knee angle
is the most reliable indicator of squat phase:
  - Standing (TOP): knee ~160-180°
  - Going down (DESCENT): knee decreasing
  - Deep squat (BOTTOM): knee at minimum (~70-100°)
  - Coming up (ASCENT): knee increasing back toward standing
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase
from squat_coach.phases.state_machine import is_valid_transition


class PhaseDetector:
    """Determine current squat phase from model probabilities + knee angle."""

    def __init__(
        self,
        min_phase_duration_s: float = 0.15,
        fps: float = 30.0,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._min_frames = max(1, int(min_phase_duration_s * fps))
        self._conf_threshold = confidence_threshold
        self._current_phase = Phase.TOP
        self._frames_in_phase = self._min_frames

        # Knee angle tracking for kinematic fallback
        self._knee_history: list[float] = []
        self._knee_baseline: float | None = None  # Standing knee angle
        self._smoothing = 5  # frames to smooth over

        # Configurable thresholds (in degrees)
        self._descent_threshold = 15.0   # knee must drop this much from baseline to be "descending"
        self._bottom_threshold = 30.0    # knee must drop this much from baseline to be "bottom"
        self._ascent_recovery = 15.0     # knee must recover this much from min to be "ascending"
        self._top_recovery = 10.0        # within this much of baseline = back at top

        self._rep_knee_min: float = 180.0  # track deepest point in current rep

    _PHASE_ORDER = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT]

    def detect(
        self,
        phase_probs: NDArray[np.float64],
        hip_y: float,
        knee_angle: float | None = None,
    ) -> Phase:
        """Detect current phase.

        Args:
            phase_probs: (4,) logits or probabilities [top, descent, bottom, ascent].
            hip_y: Current mid-hip Y in image coords (0-1, increases downward).
            knee_angle: Primary knee angle in degrees (~180=straight, ~90=deep squat).
        """
        self._frames_in_phase += 1

        # Track knee angle
        if knee_angle is not None:
            self._knee_history.append(knee_angle)

        # Set baseline from first 20 frames of standing
        if self._knee_baseline is None and len(self._knee_history) >= 20:
            self._knee_baseline = np.mean(self._knee_history[:20])

        # Try model-based detection
        probs = np.array(phase_probs, dtype=np.float64)
        if abs(probs.sum() - 1.0) > 0.1:
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        max_prob = float(np.max(probs))
        proposed_idx = int(np.argmax(probs))
        proposed_model = self._PHASE_ORDER[proposed_idx]

        # Kinematic fallback from knee angle
        proposed_kinematic = self._detect_from_knee_angle()

        # Choose: model if confident, kinematic otherwise
        if max_prob >= self._conf_threshold:
            proposed = proposed_model
        else:
            proposed = proposed_kinematic

        # Apply state machine + debounce
        if proposed != self._current_phase:
            if (
                is_valid_transition(self._current_phase, proposed)
                and self._frames_in_phase >= self._min_frames
            ):
                self._current_phase = proposed
                self._frames_in_phase = 0

                # Reset rep tracking on new descent
                if proposed == Phase.DESCENT:
                    self._rep_knee_min = 180.0
                elif proposed == Phase.TOP:
                    self._rep_knee_min = 180.0

        # Track minimum knee angle during rep
        if knee_angle is not None and self._current_phase in (Phase.DESCENT, Phase.BOTTOM):
            self._rep_knee_min = min(self._rep_knee_min, knee_angle)

        return self._current_phase

    def _detect_from_knee_angle(self) -> Phase:
        """Fallback: detect phase from knee angle trajectory.

        Standing knee ~160-175°. Squat bottom knee ~70-110°.
        """
        if self._knee_baseline is None or len(self._knee_history) < self._smoothing + 1:
            return self._current_phase

        # Smoothed current knee angle
        current_knee = np.mean(self._knee_history[-self._smoothing:])
        # Knee angle drop from baseline (positive = more bent)
        drop = self._knee_baseline - current_knee

        # Velocity: positive = bending more, negative = straightening
        prev_knee = np.mean(self._knee_history[-(self._smoothing * 2):-self._smoothing]) \
            if len(self._knee_history) >= self._smoothing * 2 else current_knee
        velocity = prev_knee - current_knee  # positive = bending, negative = straightening

        if self._current_phase == Phase.TOP:
            # Start descent when knee drops significantly and is still going down
            if drop > self._descent_threshold and velocity > 0.5:
                return Phase.DESCENT

        elif self._current_phase == Phase.DESCENT:
            # Bottom when knee stops going down (velocity near zero or reversing) while deep
            if drop > self._bottom_threshold and velocity < 0.5:
                return Phase.BOTTOM

        elif self._current_phase == Phase.BOTTOM:
            # Ascent when knee clearly straightening
            recovery = current_knee - self._rep_knee_min
            if recovery > self._ascent_recovery and velocity < -0.5:
                return Phase.ASCENT

        elif self._current_phase == Phase.ASCENT:
            # Top when knee back near baseline
            if drop < self._top_recovery:
                return Phase.TOP

        return self._current_phase

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    def reset(self) -> None:
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0
        self._knee_history.clear()
        self._knee_baseline = None
        self._rep_knee_min = 180.0
