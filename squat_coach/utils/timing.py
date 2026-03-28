"""Frame timing and FPS tracking."""
import time
from collections import deque

class FPSTracker:
    """Track rolling FPS over a window of recent frames."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record a frame timestamp."""
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Current rolling FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed < 1e-6:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
