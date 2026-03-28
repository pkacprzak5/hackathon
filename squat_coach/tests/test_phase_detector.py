"""Tests for phase detection and rep segmentation."""
import numpy as np
from squat_coach.phases.phase_detector import PhaseDetector
from squat_coach.phases.rep_segmenter import RepSegmenter, RepResult
from squat_coach.utils.enums import Phase

def test_phase_from_model_probs():
    detector = PhaseDetector()
    probs = np.array([0.1, 0.7, 0.1, 0.1])  # descent most likely
    phase = detector.detect(probs, hip_y=0.0)
    assert phase == Phase.DESCENT

def test_phase_debounce():
    """Phase should not change faster than debounce threshold."""
    detector = PhaseDetector(min_phase_duration_s=0.15, fps=30.0)
    probs_descent = np.array([0.0, 0.9, 0.1, 0.0])
    probs_bottom = np.array([0.0, 0.1, 0.9, 0.0])

    # Start in TOP
    detector.detect(np.array([0.9, 0.0, 0.0, 0.1]), hip_y=0.0)

    # Switch to DESCENT
    for _ in range(3):
        detector.detect(probs_descent, hip_y=-0.1)

    # Try to switch too fast to BOTTOM (within debounce)
    phase = detector.detect(probs_bottom, hip_y=-0.3)
    # Should still be in DESCENT due to debounce
    assert phase == Phase.DESCENT

def test_rep_segmenter_counts_rep():
    seg = RepSegmenter(min_rep_duration_s=0.1, cooldown_s=0.1, fps=30.0)
    phases = (
        [Phase.TOP] * 5 +
        [Phase.DESCENT] * 10 +
        [Phase.BOTTOM] * 5 +
        [Phase.ASCENT] * 10 +
        [Phase.TOP] * 5
    )
    results = []
    for i, phase in enumerate(phases):
        r = seg.update(phase, timestamp=i / 30.0)
        if r is not None:
            results.append(r)

    assert len(results) == 1
    assert results[0].rep_index == 1
    assert results[0].valid

def test_rep_segmenter_rejects_short_rep():
    seg = RepSegmenter(min_rep_duration_s=1.0, cooldown_s=0.0, fps=30.0)
    # Very short "rep" -- 3 frames total
    phases = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.TOP]
    results = []
    for i, phase in enumerate(phases):
        r = seg.update(phase, timestamp=i / 30.0)
        if r is not None:
            results.append(r)
    # Should be rejected as too short
    assert all(not r.valid for r in results) or len(results) == 0
