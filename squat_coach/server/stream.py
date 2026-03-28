# squat_coach/server/stream.py
"""MJPEG streaming endpoint for rendered video frames."""
import asyncio
import threading
import time

from fastapi.responses import StreamingResponse


class FrameStore:
    """Thread-safe store for the latest rendered JPEG frame."""

    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._lock = threading.Lock()
        self._event = asyncio.Event()

    def put(self, jpeg_bytes: bytes) -> None:
        """Store a new rendered frame (called from processing thread)."""
        with self._lock:
            self._frame = jpeg_bytes
        # Signal async waiters — must be thread-safe
        try:
            self._event.set()
        except RuntimeError:
            pass

    def get(self) -> bytes | None:
        """Get the latest frame."""
        with self._lock:
            return self._frame

    async def wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        """Wait for a new frame asynchronously."""
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        return self.get()


# Singleton frame store shared between WS handler and MJPEG stream
frame_store = FrameStore()


async def _generate_mjpeg(store: FrameStore):
    """Async generator that yields MJPEG frames."""
    while True:
        frame = await store.wait_for_frame(timeout=2.0)
        if frame is None:
            # No frame yet — send a small delay and retry
            await asyncio.sleep(0.042)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
            b"\r\n" + frame + b"\r\n"
        )


def mjpeg_stream() -> StreamingResponse:
    """Create an MJPEG StreamingResponse."""
    return StreamingResponse(
        _generate_mjpeg(frame_store),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
