"""Delta compression for WebSocket frame responses."""
from squat_coach.server.protocol import FrameResult


class DeltaCompressor:
    """Compares current vs last-sent state, only includes changed fields."""

    NUMERIC_THRESHOLDS = {
        "knee_angle": 1.0,
        "hip_angle": 1.0,
        "torso_angle": 1.0,
        "score": 1.0,
        "confidence": 0.05,
    }
    LANDMARK_THRESHOLD = 0.005

    def __init__(self) -> None:
        self._last_sent: dict = {}

    def compress(self, current: FrameResult) -> dict:
        delta: dict = {"seq": current.seq}

        if current.phase is not None and current.phase != self._last_sent.get("phase"):
            delta["phase"] = current.phase

        for field, threshold in self.NUMERIC_THRESHOLDS.items():
            val = getattr(current, field, None)
            if val is None:
                continue
            last_val = self._last_sent.get(field)
            if last_val is None or abs(val - last_val) > threshold:
                delta[field] = round(val, 1) if isinstance(val, float) else val

        if current.landmarks is not None:
            if self._landmarks_changed(current.landmarks):
                delta["landmarks"] = current.landmarks

        self._last_sent.update(delta)
        return delta

    def _landmarks_changed(self, new_landmarks: list[list[float]]) -> bool:
        old = self._last_sent.get("landmarks")
        if old is None:
            return True
        for old_lm, new_lm in zip(old, new_landmarks):
            for i in range(3):
                if abs(old_lm[i] - new_lm[i]) > self.LANDMARK_THRESHOLD:
                    return True
        return False
