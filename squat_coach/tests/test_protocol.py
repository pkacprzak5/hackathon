# squat_coach/tests/test_protocol.py
"""Tests for server protocol message schemas."""
import pytest
from squat_coach.server.protocol import FrameResult, RepData, CalibrationMessage


def test_frame_result_defaults():
    r = FrameResult(seq=0, timestamp=1.0)
    assert r.seq == 0
    assert r.landmarks is None
    assert r.phase is None
    assert r.rep is None


def test_frame_result_with_data():
    r = FrameResult(
        seq=5,
        timestamp=1.5,
        phase="DESCENT",
        knee_angle=95.2,
        hip_angle=78.0,
        torso_angle=12.5,
        score=82.0,
        confidence=0.92,
    )
    assert r.phase == "DESCENT"
    assert r.knee_angle == 95.2


def test_rep_data():
    rep = RepData(
        rep_index=3,
        scores={"total": 85, "depth": 90, "trunk_control": 80,
                "posture_stability": 85, "movement_consistency": 85},
        faults=["INSUFFICIENT_DEPTH"],
        coaching_text="Go deeper!",
    )
    assert rep.rep_index == 3
    assert len(rep.faults) == 1


def test_calibration_message():
    msg = CalibrationMessage(status="in_progress", progress=0.65)
    assert msg.status == "in_progress"
    assert msg.view_type is None

    msg2 = CalibrationMessage(status="complete", progress=1.0, view_type="side")
    assert msg2.view_type == "side"


def test_frame_result_to_dict():
    r = FrameResult(seq=1, timestamp=1.0, phase="TOP", knee_angle=170.0)
    d = r.to_dict()
    assert d["seq"] == 1
    assert d["phase"] == "TOP"
    assert d["knee_angle"] == 170.0
    # None fields should not appear
    assert "landmarks" not in d
    assert "rep" not in d
