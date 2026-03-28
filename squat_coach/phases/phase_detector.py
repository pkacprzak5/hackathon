"""Phase detection from fused model outputs with fallback to hip kinematics.

Uses model phase_probs as primary signal when confidence is sufficient.
Falls back to hip vertical position tracking when model confidence is low
(common with synthetic-only training data).
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase
from squat_coach.phases.state_machine import is_valid_transition


class PhaseDetector:
    """Determine current squat phase from model probabilities + hip kinematics.

    Two-signal approach:
    1. Model phase_probs (primary when confidence > threshold)
    2. Hip Y velocity tracking (fallback, always active as secondary)
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
        self._frames_in_phase = self._min_frames  # Start ready to transition
        self._fps = fps

        # Hip tracking for kinematic fallback
        self._hip_y_history: list[float] = []
        self._hip_y_baseline: float | None = None
        self._hip_y_window = 5  # frames to average for velocity

    _PHASE_ORDER = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT]

    def detect(self, phase_probs: NDArray[np.float64], hip_y: float) -> Phase:
        """Detect current phase.

        Args:
            phase_probs: (4,) logits or probabilities [top, descent, bottom, ascent].
            hip_y: Current mid-hip Y position (world coords, Y-down = deeper).

        Returns:
            Current phase after applying state machine + debounce.
        """
        self._frames_in_phase += 1
        self._hip_y_history.append(hip_y)

        # Set baseline from first frames
        if self._hip_y_baseline is None and len(self._hip_y_history) >= 10:
            self._hip_y_baseline = np.mean(self._hip_y_history[:10])

        # Try model-based detection first
        probs = np.array(phase_probs)
        # Apply softmax if these look like logits (not summing to ~1)
        if abs(probs.sum() - 1.0) > 0.1:
            exp_p = np.exp(probs - np.max(probs))
            probs = exp_p / exp_p.sum()

        max_prob = float(np.max(probs))
        proposed_idx = int(np.argmax(probs))
        proposed_model = self._PHASE_ORDER[proposed_idx]

        # Kinematic fallback: use hip velocity
        proposed_kinematic = self._detect_from_kinematics(hip_y)

        # Choose signal: model if confident, kinematic otherwise
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

        return self._current_phase

    def _detect_from_kinematics(self, hip_y: float) -> Phase:
        """Fallback phase detection from hip vertical movement.

        In BlazePose world coords, Y increases downward.
        So hip_y going UP (more positive) = squat going deeper.
        """
        if self._hip_y_baseline is None or len(self._hip_y_history) < self._hip_y_window + 1:
            return self._current_phase

        # Compute smoothed velocity (change in hip_y over recent window)
        recent = self._hip_y_history[-self._hip_y_window:]
        older = self._hip_y_history[-(self._hip_y_window + 1):-1]
        velocity = np.mean(recent) - np.mean(older)

        # How deep is the hip relative to baseline
        depth = hip_y - self._hip_y_baseline

        # Thresholds (in meters, world coords)
        vel_threshold = 0.002  # m/frame
        depth_threshold = 0.05  # meters below baseline to count as "down"

        if self._current_phase == Phase.TOP:
            # Transition to descent when hip starts moving down significantly
            if velocity > vel_threshold and depth > depth_threshold * 0.5:
                return Phase.DESCENT
        elif self._current_phase == Phase.DESCENT:
            # Transition to bottom when velocity slows near lowest point
            if abs(velocity) < vel_threshold and depth > depth_threshold:
                return Phase.BOTTOM
        elif self._current_phase == Phase.BOTTOM:
            # Transition to ascent when hip starts moving up
            if velocity < -vel_threshold:
                return Phase.ASCENT
        elif self._current_phase == Phase.ASCENT:
            # Transition to top when hip returns near baseline
            if depth < depth_threshold * 0.5 and abs(velocity) < vel_threshold:
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
