"""Draw score and angle text on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray


def draw_text_with_bg(
    frame: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """Draw text with a semi-transparent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_phase_label(frame: NDArray[np.uint8], phase: str) -> None:
    """Draw phase label top-left."""
    draw_text_with_bg(frame, f"Phase: {phase}", (10, 30))


def draw_rep_count(frame: NDArray[np.uint8], count: int) -> None:
    """Draw rep count top-right."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Rep: {count}", (w - 120, 30))


def draw_score(frame: NDArray[np.uint8], score: float) -> None:
    """Draw current score bottom-left."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Score: {score:.0f}", (10, h - 20))
