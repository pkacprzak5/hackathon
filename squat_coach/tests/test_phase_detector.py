"""Tests for phase detection and rep segmentation."""
import numpy as np
from squat_coach.phases.phase_detector import PhaseDetector
from squat_coach.phases.rep_segmenter import RepSegmenter, RepResult
from squat_coach.utils.enums import Phase


def test_phase_detector_full_squat_cycle():
    """Simulate a full squat via knee angle: standing -> deep -> standing."""
    detector = PhaseDetector(min_phase_duration_s=0.05, fps=30.0)
    dummy_probs = np.array([0.25, 0.25, 0.25, 0.25])

    phases_seen = set()

    # Standing phase: knee at 170°
    for _ in range(20):
        p = detector.detect(dummy_probs, hip_y=0.0, knee_angle=170.0)
        phases_seen.add(p)

    assert Phase.TOP in phases_seen

    # Descent: knee going from 170 to 100
    for angle in np.linspace(170, 100, 30):
        p = detector.detect(dummy_probs, hip_y=0.0, knee_angle=float(angle))

    # Should be in DESCENT or BOTTOM by now
    assert p in (Phase.DESCENT, Phase.BOTTOM)

    # Bottom: knee at 90 for a bit
    for _ in range(15):
        p = detector.detect(dummy_probs, hip_y=0.0, knee_angle=90.0)

    assert p == Phase.BOTTOM

    # Ascent: knee going from 90 back to 170
    for angle in np.linspace(90, 170, 30):
        p = detector.detect(dummy_probs, hip_y=0.0, knee_angle=float(angle))

    # Should have reached TOP or ASCENT
    assert p in (Phase.ASCENT, Phase.TOP)


def test_phase_detector_stays_top_when_standing():
    """Standing still should stay in TOP phase."""
    detector = PhaseDetector()
    dummy_probs = np.array([0.25, 0.25, 0.25, 0.25])
    for _ in range(50):
        p = detector.detect(dummy_probs, hip_y=0.0, knee_angle=168.0)
    assert p == Phase.TOP


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
    phases = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.TOP]
    results = []
    for i, phase in enumerate(phases):
        r = seg.update(phase, timestamp=i / 30.0)
        if r is not None:
            results.append(r)
    assert all(not r.valid for r in results) or len(results) == 0


def test_rep_segmenter_requires_bottom():
    """Rep should not count if we never hit bottom (e.g. false transition)."""
    seg = RepSegmenter(min_rep_duration_s=0.0, cooldown_s=0.0)
    seg.update(Phase.TOP, 0.0)
    seg.update(Phase.DESCENT, 0.5)
    seg.update(Phase.ASCENT, 1.0)  # skipped BOTTOM
    r = seg.update(Phase.TOP, 1.5)
    assert r is None  # no rep should be counted
