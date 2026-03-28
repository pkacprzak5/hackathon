"""Tests for the scoring system."""
from squat_coach.scoring.score_components import compute_depth_score, compute_trunk_control_score
from squat_coach.scoring.score_fusion import compute_rep_quality_score
from squat_coach.scoring.trend_analysis import TrendTracker

def test_depth_score_perfect():
    score = compute_depth_score(knee_angle_min=85.0, target_angle=90.0)
    assert score >= 90

def test_depth_score_shallow():
    score = compute_depth_score(knee_angle_min=130.0, target_angle=90.0)
    assert score < 50

def test_trunk_control_good():
    score = compute_trunk_control_score(
        torso_variance=2.0, max_forward_lean=20.0, baseline_angle=10.0
    )
    assert score >= 70

def test_rep_quality_weighted():
    scores = {"depth": 80, "trunk_control": 60, "posture_stability": 70, "movement_consistency": 90}
    quality = compute_rep_quality_score(scores, model_quality=0.7)
    assert 60 <= quality <= 90

def test_trend_tracker():
    tracker = TrendTracker()
    tracker.update(70)
    tracker.update(75)
    tracker.update(80)
    trend = tracker.get_trend()
    assert trend > 0  # improving
