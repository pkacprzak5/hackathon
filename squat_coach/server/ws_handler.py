# squat_coach/server/ws_handler.py
"""WebSocket endpoint for real-time squat analysis."""
import asyncio
import logging
import time

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from squat_coach.server.pipeline import SquatCoachPipeline
from squat_coach.server.delta import DeltaCompressor
from squat_coach.server.stream import FrameStore

logger = logging.getLogger("squat_coach.ws")


async def session_handler(websocket: WebSocket, frame_store: FrameStore) -> None:
    """Handle one client session over WebSocket.

    Protocol:
    - Client sends: binary JPEG frames at ~24fps
    - Server sends: JSON text only (calibration, frame data, rep events, coaching)
    - Rendered frames (with skeleton) go to the MJPEG /stream endpoint via frame_store
    """
    await websocket.accept()
    logger.info("Client connected")

    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()
    loop = asyncio.get_event_loop()

    latest_frame: bytes | None = None
    frame_event = asyncio.Event()
    running = True

    async def receive_loop():
        """Continuously receive frames, keep only the latest."""
        nonlocal latest_frame, running
        try:
            while running:
                data = await websocket.receive_bytes()
                latest_frame = data
                frame_event.set()
        except WebSocketDisconnect:
            running = False
            frame_event.set()
        except Exception:
            running = False
            frame_event.set()

    async def process_loop():
        """Process latest frame, push rendered result to frame_store, send JSON data."""
        nonlocal latest_frame, running
        try:
            while running:
                await frame_event.wait()
                frame_event.clear()

                if not running:
                    break

                jpeg_bytes = latest_frame
                if jpeg_bytes is None:
                    continue

                frame = cv2.imdecode(
                    np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    continue

                timestamp = time.time()

                # Run CPU-bound processing in thread pool
                result = await loop.run_in_executor(
                    None, pipeline.process_frame, frame, timestamp
                )

                # Calibration phase — JSON only
                if result.calibration is not None:
                    await websocket.send_json(result.calibration.to_dict())
                    continue

                # Push rendered frame to MJPEG stream
                if result.rendered_jpeg is not None:
                    frame_store.put(result.rendered_jpeg)

                # Send frame data as JSON
                compressed = delta.compress(result)
                await websocket.send_json({"type": "frame", "data": compressed})

                # Rep event
                if result.rep is not None:
                    await websocket.send_json({
                        "type": "rep",
                        "rep_index": result.rep.rep_index,
                        "scores": result.rep.scores,
                        "faults": result.rep.faults,
                        "coaching_text": result.rep.coaching_text,
                    })

                # Gemini coaching (arrives async from background thread)
                if result.coaching_text is not None:
                    await websocket.send_json({
                        "type": "coaching",
                        "text": result.coaching_text,
                    })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("Process loop error: %s", e, exc_info=True)
        finally:
            running = False
            pipeline.cleanup()
            logger.info("Client disconnected")

    await asyncio.gather(
        receive_loop(),
        process_loop(),
    )
