"""Tests for WebSocket handler and FastAPI app."""
import pytest
from fastapi.testclient import TestClient
from squat_coach.server.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_websocket_connects(client):
    with client.websocket_connect("/ws/session") as ws:
        import cv2
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", frame)
        ws.send_bytes(jpeg.tobytes())

        data = ws.receive_json()
        assert "type" in data
        assert data["type"] in ("calibration", "frame")
