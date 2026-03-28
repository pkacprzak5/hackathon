"""Video source interface."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray

class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def read(self) -> tuple[bool, Optional[NDArray[np.uint8]]]:
        """Read next frame. Returns (success, frame_bgr)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release video source."""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is available."""
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Source FPS."""
        ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of frames."""
        ...
