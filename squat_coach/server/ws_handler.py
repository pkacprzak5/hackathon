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

logger = logging.getLogger("squat_coach.ws")


async def session_handler(websocket: WebSocket) -> None:
    """Handle one client session over WebSocket.

    Protocol:
    - Client sends: binary JPEG frames at ~24fps
    - Server sends: binary JPEG (skeleton rendered on frame) + JSON text (data)

    Processing is CPU-bound (MediaPipe), so we:
    1. Run process_frame in a thread pool (doesn't block event loop)
    2. Always process the latest frame (skip stale queued frames)
    3. Send results back immediately
    """
    await websocket.accept()
    logger.info("Client connected")

    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()
    loop = asyncio.get_event_loop()

    # Latest frame buffer — always overwritten, so we process the most recent
    latest_frame: bytes | None = None
    frame_event = asyncio.Event()
    running = True

    async def receive_loop():
        """Continuously receive frames and store the latest one."""
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
        """Process the latest frame and send results."""
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

                # Decode frame
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

                # Calibration phase — only JSON
                if result.calibration is not None:
                    await websocket.send_json(result.calibration.to_dict())
                    continue

                # Send rendered frame with skeleton as binary
                if result.rendered_jpeg is not None:
                    await websocket.send_bytes(result.rendered_jpeg)

                # Send frame data as JSON (for stats display)
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

                # Gemini coaching text (arrives async from background thread)
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

    # Run receive and process concurrently
    await asyncio.gather(
        receive_loop(),
        process_loop(),
    )
