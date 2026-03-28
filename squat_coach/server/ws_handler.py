# squat_coach/server/ws_handler.py
"""WebSocket endpoint for real-time squat analysis."""
import base64
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

    Each connection gets its own pipeline.

    Protocol — all messages are text (no binary from server):
    - Client sends: binary JPEG frames at ~25fps
    - Server sends: text messages only
      - type "calibration": calibration progress
      - type "frame_img": base64 JPEG of rendered frame (skeleton overlay)
      - type "frame": JSON data (angles, score, phase, confidence)
      - type "rep": rep completion event
      - type "coaching": Gemini coaching text
    """
    await websocket.accept()
    logger.info("Client connected")

    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()
    frame_count = 0

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            frame_count += 1

            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            timestamp = time.time()
            result = pipeline.process_frame(frame, timestamp)

            # Calibration — JSON only
            if result.calibration is not None:
                await websocket.send_json(result.calibration.to_dict())
                if frame_count % 10 == 0:
                    logger.info("Calibrating... frame %d, progress %.0f%%",
                                frame_count, result.calibration.progress * 100)
                continue

            # Send rendered frame as base64 text
            if result.rendered_jpeg is not None:
                b64 = base64.b64encode(result.rendered_jpeg).decode("ascii")
                await websocket.send_text(b64)

            # Send data as JSON
            compressed = delta.compress(result)
            await websocket.send_json({"type": "frame", "data": compressed})

            if result.rep is not None:
                await websocket.send_json({
                    "type": "rep",
                    "rep_index": result.rep.rep_index,
                    "scores": result.rep.scores,
                    "faults": result.rep.faults,
                    "coaching_text": result.rep.coaching_text,
                })

            if result.coaching_text is not None:
                await websocket.send_json({
                    "type": "coaching",
                    "text": result.coaching_text,
                })

            if frame_count % 25 == 0:
                logger.info("Processed %d frames", frame_count)

    except WebSocketDisconnect:
        logger.info("Client disconnected after %d frames", frame_count)
    except Exception as e:
        logger.error("Session error at frame %d: %s", frame_count, e, exc_info=True)
    finally:
        pipeline.cleanup()
