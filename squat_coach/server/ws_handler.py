# squat_coach/server/ws_handler.py
"""WebSocket endpoint for real-time squat analysis."""
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
    - Client sends: binary JPEG frames
    - Server sends: binary JPEG (rendered frame with overlay) + JSON text (data)
      Binary messages = rendered frames for display
      Text messages = JSON data (calibration, frame data, rep events, coaching)
    """
    await websocket.accept()
    logger.info("Client connected")

    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            timestamp = time.time()
            result = pipeline.process_frame(frame, timestamp)

            # Calibration phase — only JSON, no rendered frame yet
            if result.calibration is not None:
                await websocket.send_json(result.calibration.to_dict())
                continue

            # Send rendered frame with overlay as binary
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

            # Gemini coaching text
            if result.coaching_text is not None:
                await websocket.send_json({
                    "type": "coaching",
                    "text": result.coaching_text,
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        pipeline.cleanup()
