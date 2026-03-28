"""Phase detection from fused model outputs with fallback to kinematics.

Uses model phase_probs as primary signal when confidence is sufficient.
Falls back to hip vertical position + knee angle tracking when model
confidence is low (common with synthetic-only training data).
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase
from squat_coach.phases.state_machine import is_valid_transition


class PhaseDetector:
    """Determine current squat phase from model probabilities + kinematics.

    Two-signal approach:
    1. Model phase_probs (primary when confidence > threshold)
    2. Hip Y position + knee angle (fallback, always active as secondary)
    """

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

        # Hip tracking for kinematic fallback
        self._hip_y_history: list[float] = []
        self._hip_y_baseline: float | None = None
        self._hip_y_min: float = float('inf')  # Track lowest point in current rep
        self._smoothing_window = 5

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
            hip_y: Current mid-hip Y position (world coords, Y-down = deeper).
            knee_angle: Optional primary knee angle for additional signal.

        Returns:
            Current phase after applying state machine + debounce.
        """
        self._frames_in_phase += 1
        self._hip_y_history.append(hip_y)

        # Set baseline from first 15 frames of standing
        if self._hip_y_baseline is None and len(self._hip_y_history) >= 15:
            self._hip_y_baseline = np.mean(self._hip_y_history[:15])

        # Try model-based detection
        probs = np.array(phase_probs, dtype=np.float64)
        if abs(probs.sum() - 1.0) > 0.1:
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        max_prob = float(np.max(probs))
        proposed_idx = int(np.argmax(probs))
        proposed_model = self._PHASE_ORDER[proposed_idx]

        # Kinematic fallback
        proposed_kinematic = self._detect_from_kinematics(hip_y)

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

                # Track hip min for bottom detection
                if proposed == Phase.DESCENT:
                    self._hip_y_min = hip_y
                elif proposed == Phase.TOP:
                    self._hip_y_min = float('inf')

        # Update hip min during descent/bottom
        if self._current_phase in (Phase.DESCENT, Phase.BOTTOM):
            self._hip_y_min = min(self._hip_y_min, hip_y)

        return self._current_phase

    def _detect_from_kinematics(self, hip_y: float) -> Phase:
        """Fallback phase detection from hip vertical movement.

        In BlazePose world coords, Y increases downward (positive = deeper squat).
        We track the smoothed hip Y velocity and position relative to baseline.
        """
        if self._hip_y_baseline is None:
            return self._current_phase
        if len(self._hip_y_history) < self._smoothing_window + 2:
            return self._current_phase

        # Smoothed current and previous hip Y
        current_avg = np.mean(self._hip_y_history[-self._smoothing_window:])
        prev_avg = np.mean(self._hip_y_history[-(self._smoothing_window * 2):-self._smoothing_window])
        velocity = current_avg - prev_avg  # positive = going down, negative = going up

        # How deep relative to baseline
        depth = current_avg - self._hip_y_baseline

        # Adaptive thresholds based on body movement range
        # In world coords, a squat moves hip ~0.1-0.4m
        vel_threshold = 0.003   # m per window
        depth_down = 0.03       # 3cm below baseline = "going down"
        depth_up = 0.02         # within 2cm of baseline = "back up"

        if self._current_phase == Phase.TOP:
            if velocity > vel_threshold and depth > depth_down:
                return Phase.DESCENT

        elif self._current_phase == Phase.DESCENT:
            # Bottom = velocity reverses or slows significantly while deep
            if depth > depth_down and velocity < vel_threshold * 0.3:
                return Phase.BOTTOM

        elif self._current_phase == Phase.BOTTOM:
            # Ascent = hip clearly moving upward
            if velocity < -vel_threshold:
                return Phase.ASCENT

        elif self._current_phase == Phase.ASCENT:
            # Top = hip returns near baseline
            if abs(depth) < depth_up or velocity > -vel_threshold * 0.3:
                return Phase.TOP

        return self._current_phase

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    def reset(self) -> None:
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0
        self._hip_y_history.clear()
        self._hip_y_baseline = None
        self._hip_y_min = float('inf')
