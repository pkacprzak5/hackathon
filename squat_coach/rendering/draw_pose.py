"""Draw skeleton on video frame."""
import cv2
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import SKELETON_CONNECTIONS


def draw_skeleton(
    frame: NDArray[np.uint8],
    image_landmarks: NDArray[np.float64],
    visibility: NDArray[np.float64],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 4,
    vis_threshold: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw pose skeleton on frame.

    Args:
        frame: BGR frame to draw on.
        image_landmarks: (33, 3) normalized image landmarks.
        visibility: (33,) visibility scores.
        color: BGR color for bones.
    """
    h, w = frame.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in image_landmarks]

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if visibility[i] > vis_threshold and visibility[j] > vis_threshold:
            cv2.line(frame, pts[i], pts[j], color, thickness)

    # Draw landmarks
    for idx, (pt, vis) in enumerate(zip(pts, visibility)):
        if vis > vis_threshold:
            cv2.circle(frame, pt, radius, (0, 200, 255), -1)

    return frame
