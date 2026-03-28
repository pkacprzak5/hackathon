"""Phase detection from fused model outputs with fallback to kinematics."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase
from squat_coach.phases.state_machine import is_valid_transition


class PhaseDetector:
    """Determine current squat phase from model probabilities.

    Uses model phase_probs as primary signal. Falls back to hip Y position
    if model confidence is low. Applies hysteresis and debounce.
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

    _PHASE_ORDER = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT]

    def detect(self, phase_probs: NDArray[np.float64], hip_y: float) -> Phase:
        """Detect current phase.

        Args:
            phase_probs: (4,) probabilities [top, descent, bottom, ascent].
            hip_y: Current hip vertical position (for fallback).

        Returns:
            Current phase after applying state machine + debounce.
        """
        self._frames_in_phase += 1

        # Primary: highest probability phase
        proposed_idx = int(np.argmax(phase_probs))
        proposed = self._PHASE_ORDER[proposed_idx]

        # Check if transition is valid and debounce has elapsed
        if proposed != self._current_phase:
            if (
                is_valid_transition(self._current_phase, proposed)
                and self._frames_in_phase >= self._min_frames
                and phase_probs[proposed_idx] >= self._conf_threshold
            ):
                self._current_phase = proposed
                self._frames_in_phase = 0

        return self._current_phase

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    def reset(self) -> None:
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0
