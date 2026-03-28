# squat_coach/server/main.py
"""FastAPI application for Squat Coach server."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from squat_coach.server.ws_handler import session_handler
from squat_coach.server.stream import mjpeg_stream, frame_store

logger = logging.getLogger("squat_coach.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info("Squat Coach server starting")
    yield
    logger.info("Squat Coach server shutting down")


app = FastAPI(title="Squat Coach", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stream")
async def stream():
    """MJPEG video stream — rendered frames with skeleton overlay.

    Use as: <img src="https://server:8000/stream" />
    Browser handles MJPEG natively, no JavaScript needed.
    """
    return mjpeg_stream()


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await session_handler(websocket, frame_store)
