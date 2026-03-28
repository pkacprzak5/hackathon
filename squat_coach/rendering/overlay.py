"""Simple overlay compositor -- combines all drawing elements."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_pose import draw_skeleton
from squat_coach.rendering.draw_metrics import draw_phase_label, draw_rep_count, draw_score
from squat_coach.rendering.draw_feedback import draw_coaching_cue


def render_overlay(
    frame: NDArray[np.uint8],
    image_landmarks: NDArray[np.float64] | None,
    visibility: NDArray[np.float64] | None,
    phase: str,
    rep_count: int,
    score: float,
    coaching_cue: str,
) -> NDArray[np.uint8]:
    """Render the full simple overlay on a video frame.

    Args:
        frame: BGR frame (modified in-place and returned).
        image_landmarks: (33, 3) or None if no detection.
        visibility: (33,) or None.
        phase: Current phase name string.
        rep_count: Current rep count.
        score: Current form score (0-100).
        coaching_cue: Active coaching cue string.

    Returns:
        Frame with overlay drawn.
    """
    if image_landmarks is not None and visibility is not None:
        draw_skeleton(frame, image_landmarks, visibility)

    draw_phase_label(frame, phase)
    draw_rep_count(frame, rep_count)
    draw_score(frame, score)
    draw_coaching_cue(frame, coaching_cue)

    return frame
