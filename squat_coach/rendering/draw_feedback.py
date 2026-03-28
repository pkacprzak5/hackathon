"""Draw coaching cue on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_metrics import draw_text_with_bg


def draw_coaching_cue(
    frame: NDArray[np.uint8],
    cue: str,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw coaching cue bottom-center."""
    if not cue:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(cue, font, 0.8, 2)
    x = (w - tw) // 2
    draw_text_with_bg(frame, cue, (x, h - 50), font_scale=0.8, color=color)
