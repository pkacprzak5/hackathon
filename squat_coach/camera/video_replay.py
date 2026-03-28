"""Video file replay source."""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from squat_coach.camera.base import VideoSource

class VideoReplay(VideoSource):
    """OpenCV video file reader for replay/debug mode."""

    def __init__(self, video_path: str) -> None:
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

    def read(self) -> tuple[bool, Optional[NDArray[np.uint8]]]:
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
