# squat_coach/tests/test_pipeline.py
"""Tests for headless SquatCoachPipeline."""
import numpy as np
import pytest
from squat_coach.server.pipeline import SquatCoachPipeline
from squat_coach.server.protocol import FrameResult


@pytest.fixture
def pipeline():
    return SquatCoachPipeline()


def test_pipeline_creates(pipeline):
    assert pipeline is not None
    assert pipeline.is_calibrated is False


def test_process_frame_returns_frame_result(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, timestamp=0.0)
    assert isinstance(result, FrameResult)
    assert result.seq == 0


def test_process_frame_increments_seq(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r1 = pipeline.process_frame(frame, 0.0)
    r2 = pipeline.process_frame(frame, 0.042)
    assert r1.seq == 0
    assert r2.seq == 1


def test_pipeline_cleanup(pipeline):
    pipeline.cleanup()
