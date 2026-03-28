"""Draw score and angle text on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray


def draw_text_with_bg(
    frame: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.8,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    bg_alpha: float = 0.7,
) -> None:
    """Draw text with a semi-transparent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    pad = 6
    # Draw filled background rectangle directly (more visible than blending)
    cv2.rectangle(
        frame,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        bg_color,
        -1,
    )
    # Draw text on top
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_phase_label(frame: NDArray[np.uint8], phase: str) -> None:
    """Draw phase label top-left."""
    draw_text_with_bg(frame, f"Phase: {phase.upper()}", (10, 35), font_scale=0.9)


def draw_rep_count(frame: NDArray[np.uint8], count: int) -> None:
    """Draw rep count top-right."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Rep: {count}", (w - 150, 35), font_scale=0.9)


def draw_score(frame: NDArray[np.uint8], score: float) -> None:
    """Draw current score bottom-left."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Score: {score:.0f}", (10, h - 60), font_scale=0.9)


def draw_score_panel(
    frame: NDArray[np.uint8],
    last_score: float,
    best_score: float,
    avg_score: float,
    rep_count: int,
) -> None:
    """Draw score panel on the right side: last rep, best, average."""
    h, w = frame.shape[:2]
    x = w - 220
    y_start = 80

    if rep_count == 0:
        return

    # Color code the last score
    if last_score >= 80:
        score_color = (0, 255, 0)    # green
    elif last_score >= 60:
        score_color = (0, 220, 255)  # yellow
    else:
        score_color = (0, 100, 255)  # red-orange

    lines = [
        (f"Last: {last_score:.0f}", score_color),
        (f"Best: {best_score:.0f}", (200, 255, 200)),
        (f"Avg:  {avg_score:.0f}", (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(lines):
        draw_text_with_bg(
            frame, text, (x, y_start + i * 35),
            font_scale=0.65, color=color,
        )


def draw_angles(frame: NDArray[np.uint8], features: dict) -> None:
    """Draw key angles on the left side."""
    h, w = frame.shape[:2]
    y_start = 80
    angle_texts = [
        f"Knee: {features.get('primary_knee_angle', 0):.0f} deg",
        f"Hip: {features.get('primary_hip_angle', 0):.0f} deg",
        f"Torso: {features.get('torso_inclination_deg', 0):.0f} deg",
    ]
    for i, text in enumerate(angle_texts):
        draw_text_with_bg(
            frame, text, (10, y_start + i * 35),
            font_scale=0.6, color=(200, 255, 200),
        )
