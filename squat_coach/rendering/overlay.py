"""Simple overlay compositor -- combines all drawing elements."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_pose import draw_skeleton
from squat_coach.rendering.draw_metrics import (
    draw_phase_label, draw_rep_count, draw_score, draw_angles, draw_score_panel,
)
from squat_coach.rendering.draw_feedback import draw_coaching_cue


def render_overlay(
    frame: NDArray[np.uint8],
    image_landmarks: NDArray[np.float64] | None,
    visibility: NDArray[np.float64] | None,
    phase: str,
    rep_count: int,
    score: float,
    coaching_cue: str,
    features: dict | None = None,
    last_score: float = 0.0,
    best_score: float = 0.0,
    avg_score: float = 0.0,
) -> NDArray[np.uint8]:
    """Render the full simple overlay on a video frame."""
    if image_landmarks is not None and visibility is not None:
        draw_skeleton(frame, image_landmarks, visibility)

    draw_phase_label(frame, phase)
    draw_rep_count(frame, rep_count)
    draw_score(frame, score)
    if features:
        draw_angles(frame, features)
    draw_score_panel(frame, last_score, best_score, avg_score, rep_count)
    draw_coaching_cue(frame, coaching_cue)

    return frame
