# squat_coach/server/ws_handler.py
"""WebSocket endpoint for real-time squat analysis."""
import asyncio
import logging
import time
from collections import deque

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from squat_coach.server.pipeline import SquatCoachPipeline
from squat_coach.server.delta import DeltaCompressor

logger = logging.getLogger("squat_coach.ws")


async def session_handler(websocket: WebSocket) -> None:
    """Handle one client session over WebSocket.

    Each connection gets its own pipeline instance.

    Protocol:
    - Client sends: binary JPEG frames at ~24fps
    - Server sends: binary JPEG (rendered frame) + text JSON (data)
    """
    await websocket.accept()
    logger.info("Client connected")

    # Each client gets its own pipeline
    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()
    loop = asyncio.get_event_loop()
    frame_count = 0

    try:
        while True:
            # Receive a frame
            jpeg_bytes = await websocket.receive_bytes()
            frame_count += 1

            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                logger.warning("Failed to decode frame %d", frame_count)
                continue

            timestamp = time.time()

            # Process in thread pool so we don't block the event loop
            result = await loop.run_in_executor(
                None, pipeline.process_frame, frame, timestamp
            )

            # Calibration — JSON only, no rendered frame
            if result.calibration is not None:
                await websocket.send_json(result.calibration.to_dict())
                continue

            # Send rendered frame (skeleton on video) as binary
            if result.rendered_jpeg is not None:
                await websocket.send_bytes(result.rendered_jpeg)

            # Send data as JSON
            compressed = delta.compress(result)
            await websocket.send_json({"type": "frame", "data": compressed})

            # Rep completed
            if result.rep is not None:
                await websocket.send_json({
                    "type": "rep",
                    "rep_index": result.rep.rep_index,
                    "scores": result.rep.scores,
                    "faults": result.rep.faults,
                    "coaching_text": result.rep.coaching_text,
                })

            # Gemini coaching (async, arrives when ready)
            if result.coaching_text is not None:
                await websocket.send_json({
                    "type": "coaching",
                    "text": result.coaching_text,
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Session error after %d frames: %s", frame_count, e, exc_info=True)
    finally:
        pipeline.cleanup()
