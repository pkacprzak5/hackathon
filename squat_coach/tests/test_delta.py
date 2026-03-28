"""Tests for delta compression."""
import pytest
from squat_coach.server.delta import DeltaCompressor
from squat_coach.server.protocol import FrameResult


def test_first_frame_sends_everything():
    dc = DeltaCompressor()
    result = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0,
                         hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    delta = dc.compress(result)
    assert delta["seq"] == 0
    assert delta["phase"] == "TOP"
    assert delta["knee_angle"] == 170.0


def test_no_change_sends_only_seq():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0,
                     hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="TOP", knee_angle=170.0,
                     hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    delta = dc.compress(r2)
    assert delta == {"seq": 1}


def test_angle_change_above_threshold():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="TOP", knee_angle=168.5)
    delta = dc.compress(r2)
    assert "knee_angle" in delta
    assert delta["knee_angle"] == 168.5


def test_angle_change_below_threshold():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, knee_angle=170.0)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, knee_angle=170.3)
    delta = dc.compress(r2)
    assert "knee_angle" not in delta


def test_phase_change():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP")
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="DESCENT")
    delta = dc.compress(r2)
    assert delta["phase"] == "DESCENT"


def test_landmarks_change():
    dc = DeltaCompressor()
    lm1 = [[0.5, 0.3, -0.1, 0.9]] * 33
    r1 = FrameResult(seq=0, timestamp=0.0, landmarks=lm1)
    dc.compress(r1)

    lm2 = [[0.5, 0.3, -0.1, 0.9]] * 33
    lm2[0] = [0.52, 0.31, -0.1, 0.9]  # moved > 0.005
    r2 = FrameResult(seq=1, timestamp=0.033, landmarks=lm2)
    delta = dc.compress(r2)
    assert "landmarks" in delta


def test_landmarks_no_change():
    dc = DeltaCompressor()
    lm = [[0.5, 0.3, -0.1, 0.9]] * 33
    r1 = FrameResult(seq=0, timestamp=0.0, landmarks=lm)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, landmarks=lm)
    delta = dc.compress(r2)
    assert "landmarks" not in delta
