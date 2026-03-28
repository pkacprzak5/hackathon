"""Draw coaching cue on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_metrics import draw_text_with_bg


def draw_coaching_cue(
    frame: NDArray[np.uint8],
    cue: str,
    color: tuple[int, int, int] = (0, 220, 255),
) -> None:
    """Draw coaching cue bottom-center with high visibility."""
    if not cue:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(cue, font, font_scale, thickness)
    x = max(10, (w - tw) // 2)
    draw_text_with_bg(frame, cue, (x, h - 20), font_scale=font_scale, color=color)
