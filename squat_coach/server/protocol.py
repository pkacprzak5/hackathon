# squat_coach/server/protocol.py
"""Message schemas for WebSocket communication."""
from dataclasses import dataclass, field, asdict


@dataclass
class RepData:
    rep_index: int
    scores: dict
    faults: list[str]
    coaching_text: str


@dataclass
class CalibrationMessage:
    status: str  # "in_progress" | "complete"
    progress: float  # 0.0 - 1.0
    view_type: str | None = None

    def to_dict(self) -> dict:
        d = {"type": "calibration", "status": self.status, "progress": round(self.progress, 2)}
        if self.view_type is not None:
            d["view_type"] = self.view_type
        return d


@dataclass
class FrameResult:
    seq: int
    timestamp: float

    # Calibration (mutually exclusive with analysis fields)
    calibration: CalibrationMessage | None = None

    # Per-frame analysis
    landmarks: list[list[float]] | None = None  # 33 x [x, y, z, visibility]
    phase: str | None = None
    knee_angle: float | None = None
    hip_angle: float | None = None
    torso_angle: float | None = None
    score: float | None = None
    confidence: float | None = None

    # Rep event (only on rep completion)
    rep: RepData | None = None

    # Coaching (from Gemini, async)
    coaching_text: str | None = None

    def to_dict(self) -> dict:
        """Return dict with only non-None fields."""
        d: dict = {"seq": self.seq}
        for fld in ["landmarks", "phase", "knee_angle", "hip_angle",
                     "torso_angle", "score", "confidence"]:
            val = getattr(self, fld)
            if val is not None:
                d[fld] = val
        return d
