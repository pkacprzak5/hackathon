# Squat Coach Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the squat_coach Python backend to the Next.js frontend via FastAPI WebSocket, enabling real-time phone-based squat coaching with canvas overlay.

**Architecture:** Phone captures video and displays it locally at native framerate. Frames are sent at 24fps via WebSocket to a FastAPI server that runs the full squat_coach pipeline (MediaPipe, biomechanics, scoring). Server responds with delta-compressed JSON data at 24fps. Frontend overlays skeleton + metrics on a canvas over the video and uses Web Speech API for coaching TTS.

**Tech Stack:** Python (FastAPI, uvicorn, OpenCV, MediaPipe, PyTorch), TypeScript (Next.js 16, React 19, Canvas API, Web Speech API, WebSocket)

---

## File Structure

### New Server Files
| File | Responsibility |
|------|---------------|
| `squat_coach/server/__init__.py` | Package init |
| `squat_coach/server/protocol.py` | Pydantic message schemas (FrameResult, RepData, CalibrationMessage) |
| `squat_coach/server/pipeline.py` | Headless SquatCoachPipeline — processes frames, returns FrameResult |
| `squat_coach/server/delta.py` | DeltaCompressor — reduces payload size by omitting unchanged fields |
| `squat_coach/server/ws_handler.py` | WebSocket endpoint — receive frames, send JSON responses |
| `squat_coach/server/main.py` | FastAPI app with CORS and lifespan |

### New Frontend Files
| File | Responsibility |
|------|---------------|
| `frontend/lib/squat-types.ts` | TypeScript types for squat session (Landmark, FrameData, RepResult, etc.) |
| `frontend/lib/skeleton.ts` | Skeleton connection definitions + canvas drawing functions |
| `frontend/lib/interpolation.ts` | Landmark/angle lerp functions for smooth 60fps rendering |
| `frontend/lib/tts.ts` | Web Speech API wrapper |
| `frontend/hooks/use-squat-session.ts` | Core hook: WebSocket, frame capture, message handling, state |
| `frontend/components/session/overlay-canvas.tsx` | Canvas overlay component for skeleton + angles |
| `frontend/components/session/calibration-overlay.tsx` | Calibration progress UI |
| `frontend/components/session/session-hud.tsx` | Phase, rep count, score, coaching text display |

### Modified Files
| File | Change |
|------|--------|
| `squat_coach/requirements.txt` | Add fastapi, uvicorn[standard] |
| `frontend/lib/types.ts` | Add SquatRepResult type for results page bridge |
| `frontend/app/(player)/session/solo/page.tsx` | Integrate useSquatSession + overlay components |

---

## Task 1: Server Protocol & Message Schemas

**Files:**
- Create: `squat_coach/server/__init__.py`
- Create: `squat_coach/server/protocol.py`
- Test: `squat_coach/tests/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# squat_coach/tests/test_protocol.py
"""Tests for server protocol message schemas."""
import pytest
from squat_coach.server.protocol import FrameResult, RepData, CalibrationMessage


def test_frame_result_defaults():
    r = FrameResult(seq=0, timestamp=1.0)
    assert r.seq == 0
    assert r.landmarks is None
    assert r.phase is None
    assert r.rep is None


def test_frame_result_with_data():
    r = FrameResult(
        seq=5,
        timestamp=1.5,
        phase="DESCENT",
        knee_angle=95.2,
        hip_angle=78.0,
        torso_angle=12.5,
        score=82.0,
        confidence=0.92,
    )
    assert r.phase == "DESCENT"
    assert r.knee_angle == 95.2


def test_rep_data():
    rep = RepData(
        rep_index=3,
        scores={"total": 85, "depth": 90, "trunk_control": 80,
                "posture_stability": 85, "movement_consistency": 85},
        faults=["INSUFFICIENT_DEPTH"],
        coaching_text="Go deeper!",
    )
    assert rep.rep_index == 3
    assert len(rep.faults) == 1


def test_calibration_message():
    msg = CalibrationMessage(status="in_progress", progress=0.65)
    assert msg.status == "in_progress"
    assert msg.view_type is None

    msg2 = CalibrationMessage(status="complete", progress=1.0, view_type="side")
    assert msg2.view_type == "side"


def test_frame_result_to_dict():
    r = FrameResult(seq=1, timestamp=1.0, phase="TOP", knee_angle=170.0)
    d = r.to_dict()
    assert d["seq"] == 1
    assert d["phase"] == "TOP"
    assert d["knee_angle"] == 170.0
    # None fields should not appear
    assert "landmarks" not in d
    assert "rep" not in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'squat_coach.server'`

- [ ] **Step 3: Create package init and implement protocol**

```python
# squat_coach/server/__init__.py
```

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_protocol.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/server/__init__.py squat_coach/server/protocol.py squat_coach/tests/test_protocol.py
git commit -m "feat(server): add WebSocket protocol message schemas"
```

---

## Task 2: Delta Compressor

**Files:**
- Create: `squat_coach/server/delta.py`
- Test: `squat_coach/tests/test_delta.py`

- [ ] **Step 1: Write the failing test**

```python
# squat_coach/tests/test_delta.py
"""Tests for delta compression."""
import pytest
from squat_coach.server.delta import DeltaCompressor
from squat_coach.server.protocol import FrameResult


def test_first_frame_sends_everything():
    dc = DeltaCompressor()
    result = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0,
                         hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    delta = dc.compress(result)
    assert delta["seq"] == 0
    assert delta["phase"] == "TOP"
    assert delta["knee_angle"] == 170.0


def test_no_change_sends_only_seq():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0,
                     hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="TOP", knee_angle=170.0,
                     hip_angle=160.0, torso_angle=5.0, score=50.0, confidence=0.9)
    delta = dc.compress(r2)
    assert delta == {"seq": 1}


def test_angle_change_above_threshold():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP", knee_angle=170.0)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="TOP", knee_angle=168.5)
    delta = dc.compress(r2)
    assert "knee_angle" in delta
    assert delta["knee_angle"] == 168.5


def test_angle_change_below_threshold():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, knee_angle=170.0)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, knee_angle=170.3)
    delta = dc.compress(r2)
    assert "knee_angle" not in delta


def test_phase_change():
    dc = DeltaCompressor()
    r1 = FrameResult(seq=0, timestamp=0.0, phase="TOP")
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, phase="DESCENT")
    delta = dc.compress(r2)
    assert delta["phase"] == "DESCENT"


def test_landmarks_change():
    dc = DeltaCompressor()
    lm1 = [[0.5, 0.3, -0.1, 0.9]] * 33
    r1 = FrameResult(seq=0, timestamp=0.0, landmarks=lm1)
    dc.compress(r1)

    lm2 = [[0.5, 0.3, -0.1, 0.9]] * 33
    lm2[0] = [0.52, 0.31, -0.1, 0.9]  # moved > 0.005
    r2 = FrameResult(seq=1, timestamp=0.033, landmarks=lm2)
    delta = dc.compress(r2)
    assert "landmarks" in delta


def test_landmarks_no_change():
    dc = DeltaCompressor()
    lm = [[0.5, 0.3, -0.1, 0.9]] * 33
    r1 = FrameResult(seq=0, timestamp=0.0, landmarks=lm)
    dc.compress(r1)
    r2 = FrameResult(seq=1, timestamp=0.033, landmarks=lm)
    delta = dc.compress(r2)
    assert "landmarks" not in delta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_delta.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement delta compressor**

```python
# squat_coach/server/delta.py
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
    LANDMARK_THRESHOLD = 0.005  # normalized coordinate units

    def __init__(self) -> None:
        self._last_sent: dict = {}

    def compress(self, current: FrameResult) -> dict:
        delta: dict = {"seq": current.seq}

        # Phase: send on any change
        if current.phase is not None and current.phase != self._last_sent.get("phase"):
            delta["phase"] = current.phase

        # Numeric fields: send if changed beyond threshold
        for field, threshold in self.NUMERIC_THRESHOLDS.items():
            val = getattr(current, field, None)
            if val is None:
                continue
            last_val = self._last_sent.get(field)
            if last_val is None or abs(val - last_val) > threshold:
                delta[field] = round(val, 1) if isinstance(val, float) else val

        # Landmarks: send if any landmark moved significantly
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
            for i in range(3):  # x, y, z
                if abs(old_lm[i] - new_lm[i]) > self.LANDMARK_THRESHOLD:
                    return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_delta.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/server/delta.py squat_coach/tests/test_delta.py
git commit -m "feat(server): add delta compression for WebSocket responses"
```

---

## Task 3: Headless SquatCoachPipeline

**Files:**
- Create: `squat_coach/server/pipeline.py`
- Test: `squat_coach/tests/test_pipeline.py`

This is the core refactor: extract the processing logic from `app.py` into a headless class with no camera, no OpenCV window, no rendering. It receives a numpy frame and returns a `FrameResult`.

- [ ] **Step 1: Write the failing test**

```python
# squat_coach/tests/test_pipeline.py
"""Tests for headless SquatCoachPipeline."""
import numpy as np
import pytest
from squat_coach.server.pipeline import SquatCoachPipeline
from squat_coach.server.protocol import FrameResult


@pytest.fixture
def pipeline():
    return SquatCoachPipeline()


def test_pipeline_creates(pipeline):
    assert pipeline is not None
    assert pipeline.is_calibrated is False


def test_process_frame_returns_frame_result(pipeline):
    """Process a black frame — no pose detected, should return calibration status."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame, timestamp=0.0)
    assert isinstance(result, FrameResult)
    assert result.seq == 0


def test_process_frame_increments_seq(pipeline):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    r1 = pipeline.process_frame(frame, 0.0)
    r2 = pipeline.process_frame(frame, 0.042)
    assert r1.seq == 0
    assert r2.seq == 1


def test_pipeline_cleanup(pipeline):
    """cleanup() should not raise."""
    pipeline.cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement SquatCoachPipeline**

```python
# squat_coach/server/pipeline.py
"""Headless squat analysis pipeline — no camera, no display."""
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import yaml

from squat_coach.utils.enums import Phase
from squat_coach.pose.mediapipe_blazepose3d import MediaPipeBlazePose3D
from squat_coach.preprocessing.smoothing import EMALandmarkSmoother
from squat_coach.preprocessing.normalization import normalize_to_hip_center
from squat_coach.preprocessing.calibration import Calibrator
from squat_coach.preprocessing.sequence_buffer import SequenceBuffer
from squat_coach.biomechanics.squat_features import SquatFeatureExtractor
from squat_coach.models.inference_manager import InferenceManager
from squat_coach.phases.phase_detector import PhaseDetector
from squat_coach.phases.rep_segmenter import RepSegmenter
from squat_coach.faults.evidence_engine import EvidenceEngine
from squat_coach.scoring.ideal_reference import build_ideal_reference
from squat_coach.scoring.score_components import (
    compute_depth_score, compute_trunk_control_score,
    compute_posture_stability_score, compute_movement_consistency_score,
)
from squat_coach.scoring.score_fusion import compute_rep_quality_score
from squat_coach.scoring.rationale import build_rationale
from squat_coach.scoring.trend_analysis import TrendTracker
from squat_coach.events.event_builder import build_rep_summary
from squat_coach.events.gemini_payloads import (
    format_gemini_payload, send_to_gemini_async, _get_client,
)
from squat_coach.events.coaching_priority import CoachingPrioritizer
from squat_coach.session.session_state import SessionState
from squat_coach.server.protocol import FrameResult, RepData, CalibrationMessage

# Import models to register in factory
import squat_coach.models.temporal_tcn   # noqa: F401
import squat_coach.models.temporal_gru   # noqa: F401

logger = logging.getLogger("squat_coach.pipeline")


class SquatCoachPipeline:
    """Headless squat analysis — receives frames, returns structured results."""

    def __init__(self, config_dir: str = "squat_coach/config") -> None:
        self._config = self._load_config(f"{config_dir}/default.yaml")
        self._model_config = self._load_config(f"{config_dir}/model.yaml")
        self._scoring_config = self._load_config(f"{config_dir}/scoring.yaml")

        self._pose = MediaPipeBlazePose3D()
        self._smoother = EMALandmarkSmoother(
            alpha=self._config["preprocessing"]["ema_alpha"]
        )
        self._calibrator = Calibrator(
            num_frames=int(self._config["preprocessing"]["calibration_duration_s"] * 24)
        )
        self._session = SessionState()
        self._fault_engine = EvidenceEngine()
        self._coach = CoachingPrioritizer()
        self._trend_tracker = TrendTracker()
        self._seq_buf = SequenceBuffer(
            seq_len=self._model_config["sequence"]["length"],
            feature_dim=self._model_config["sequence"]["feature_dim"],
        )

        # Initialized after calibration
        self._feature_extractor = None
        self._inference_mgr = None
        self._phase_detector = None
        self._rep_segmenter = None
        self._ideal_ref = None
        self._cal_result = None

        # Frame counter
        self._seq = 0

        # Per-rep tracking
        self._rep_min_knee = 180.0
        self._rep_max_torso = 0.0
        self._rep_max_head_offset = 0.0
        self._rep_features_snapshot: dict = {}
        self._live_score_ema = 50.0

        # Score history
        self._all_rep_scores: list[float] = []
        self._last_rep_score = 0.0
        self._best_rep_score = 0.0

        # Coaching cue state
        self._last_cue = ""
        self._last_cue_time = 0.0
        self._cue_display_duration = 5.0

        # Gemini feedback callback
        self._pending_coaching: str | None = None
        self._coaching_lock = threading.Lock()

        # Pre-init Gemini client
        gemini_cfg = self._config.get("gemini", {})
        if gemini_cfg.get("enabled", False):
            key = gemini_cfg.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            if key:
                _get_client(key)

    @property
    def is_calibrated(self) -> bool:
        return self._session.is_calibrated

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        """Process one BGR frame, return structured result."""
        seq = self._seq
        self._seq += 1
        self._session.frame_index += 1

        # Pose estimation
        pose_result = self._pose.estimate(frame, timestamp)

        if not pose_result.detected:
            self._session.dropped_frame_count += 1
            # Return empty result during dropped frames
            if not self._session.is_calibrated:
                progress = self._calibrator.frame_count / max(self._calibrator.num_frames, 1)
                return FrameResult(
                    seq=seq, timestamp=timestamp,
                    calibration=CalibrationMessage(status="in_progress", progress=progress),
                )
            return FrameResult(seq=seq, timestamp=timestamp)

        self._session.dropped_frame_count = 0

        # Smooth + normalize
        smoothed = self._smoother.smooth(pose_result.world_landmarks)
        normalized = normalize_to_hip_center(smoothed)

        # Calibration phase
        if not self._session.is_calibrated:
            self._calibrator.add_frame(pose_result)
            progress = self._calibrator.frame_count / max(self._calibrator.num_frames, 1)

            if self._calibrator.is_ready:
                self._cal_result = self._calibrator.compute()
                if self._cal_result:
                    self._session.is_calibrated = True
                    self._session.view_type = self._cal_result.view_type
                    self._init_post_calibration()

                    return FrameResult(
                        seq=seq, timestamp=timestamp,
                        calibration=CalibrationMessage(
                            status="complete", progress=1.0,
                            view_type=self._cal_result.view_type.value,
                        ),
                    )

            return FrameResult(
                seq=seq, timestamp=timestamp,
                calibration=CalibrationMessage(status="in_progress", progress=progress),
            )

        # Feature extraction
        features = self._feature_extractor.extract(normalized, pose_result.visibility)
        model_features = features["model_features"]
        self._seq_buf.push(model_features)

        # Temporal inference
        fused = None
        if self._seq_buf.is_ready and self._inference_mgr and self._inference_mgr.has_models:
            fused = self._inference_mgr.infer(self._seq_buf.get_sequence())

        # Phase detection
        img_lm = pose_result.image_landmarks
        hip_y_img = float((img_lm[23][1] + img_lm[24][1]) / 2.0)
        knee_angle = features.get("primary_knee_angle", 170.0)

        if fused is not None:
            phase = self._phase_detector.detect(fused.phase_probs, hip_y_img, knee_angle)
        else:
            dummy_probs = np.array([0.25, 0.25, 0.25, 0.25])
            phase = self._phase_detector.detect(dummy_probs, hip_y_img, knee_angle)

        self._session.current_phase = phase

        # Track per-rep extremes during descent and bottom
        if phase in (Phase.DESCENT, Phase.BOTTOM):
            self._rep_min_knee = min(self._rep_min_knee, features.get("primary_knee_angle", 180))
            self._rep_max_torso = max(self._rep_max_torso, features.get("torso_inclination_deg", 0))
            self._rep_max_head_offset = max(self._rep_max_head_offset, features.get("head_to_trunk_offset", 0))
            self._rep_features_snapshot = dict(features)

        # Live score
        if phase != Phase.TOP:
            target = self._ideal_ref.target_knee_angle if self._ideal_ref else 90
            best_depth_score = compute_depth_score(self._rep_min_knee, target)
            if best_depth_score > self._live_score_ema:
                self._live_score_ema = best_depth_score
            self._session.current_score = self._live_score_ema

        # Fault detection
        fault_config = self._scoring_config.get("faults", {}).get("thresholds", {})
        fault_config["baseline_torso_angle"] = self._ideal_ref.trunk_neutral_angle if self._ideal_ref else 10.0
        faults = self._fault_engine.evaluate(features, fault_config)

        # Coaching cue
        new_cue = self._coach.select_cue(faults)
        now_cue = time.monotonic()
        if new_cue:
            self._last_cue = new_cue
            self._last_cue_time = now_cue
            self._session.current_cue = new_cue
        elif now_cue - self._last_cue_time < self._cue_display_duration:
            self._session.current_cue = self._last_cue
        else:
            self._session.current_cue = ""

        # Build frame result
        # Convert image landmarks to list format for JSON
        landmarks_list = [
            [round(float(lm[0]), 4), round(float(lm[1]), 4),
             round(float(lm[2]), 4), round(float(pose_result.visibility[i]), 2)]
            for i, lm in enumerate(pose_result.image_landmarks)
        ]

        result = FrameResult(
            seq=seq,
            timestamp=timestamp,
            landmarks=landmarks_list,
            phase=phase.value.upper(),
            knee_angle=features.get("primary_knee_angle"),
            hip_angle=features.get("primary_hip_angle"),
            torso_angle=features.get("torso_inclination_deg"),
            score=self._session.current_score,
            confidence=pose_result.pose_confidence,
        )

        # Rep segmentation
        rep_result = self._rep_segmenter.update(phase, timestamp)
        if rep_result and rep_result.valid:
            self._session.rep_count = rep_result.rep_index
            result.rep = self._score_rep(rep_result, features, faults, fused, pose_result)

        # Check for pending Gemini coaching
        with self._coaching_lock:
            if self._pending_coaching:
                result.coaching_text = self._pending_coaching
                self._pending_coaching = None

        return result

    def cleanup(self) -> None:
        """Release resources."""
        try:
            self._pose.close()
        except Exception:
            pass

    def _init_post_calibration(self) -> None:
        """Initialize subsystems that depend on calibration."""
        cal = self._cal_result
        scoring = self._scoring_config

        self._feature_extractor = SquatFeatureExtractor(cal, fps=24.0)
        self._ideal_ref = build_ideal_reference(
            cal, target_knee_angle=scoring["calibration"]["target_knee_angle_deg"],
        )
        self._phase_detector = PhaseDetector(
            min_phase_duration_s=scoring["calibration"]["min_phase_duration_s"],
            calibrated_knee_angle=cal.baseline_knee_angle,
        )
        self._rep_segmenter = RepSegmenter(
            min_rep_duration_s=scoring["calibration"]["min_rep_duration_s"],
            cooldown_s=scoring["calibration"]["rep_cooldown_s"],
        )

        ckpt_dir = self._model_config["checkpoints"]["dir"]
        stats_path = str(Path(ckpt_dir) / "feature_stats.json")
        self._inference_mgr = InferenceManager(
            model_configs=self._model_config["models"],
            ensemble_config=self._model_config["ensemble"],
            checkpoint_dir=ckpt_dir,
            stats_path=stats_path,
            view=cal.view_type.value,
        )

    def _score_rep(self, rep_result, features, faults, fused, pose_result) -> RepData:
        """Score a completed rep and return RepData."""
        ideal = self._ideal_ref
        cal = self._cal_result

        depth = compute_depth_score(
            self._rep_min_knee, ideal.target_knee_angle if ideal else 90
        )
        trunk = compute_trunk_control_score(
            features.get("trunk_stability", 0),
            self._rep_max_torso,
            ideal.trunk_neutral_angle if ideal else 10,
        )
        posture = compute_posture_stability_score(
            features.get("rounded_back_risk", 0),
            self._rep_max_head_offset,
            cal.body_scale if cal else 0.4,
        )
        consistency = compute_movement_consistency_score(
            rep_result.descent_duration, rep_result.ascent_duration,
        )

        scores = {
            "depth": round(depth, 1),
            "trunk_control": round(trunk, 1),
            "posture_stability": round(posture, 1),
            "movement_consistency": round(consistency, 1),
        }
        model_q = fused.quality_score if fused else 0.5
        rep_quality = compute_rep_quality_score(scores, model_quality=model_q)
        scores["total"] = round(rep_quality, 1)

        self._trend_tracker.update(rep_quality)
        self._session.current_score = rep_quality
        self._last_rep_score = rep_quality
        self._best_rep_score = max(self._best_rep_score, rep_quality)
        self._all_rep_scores.append(rep_quality)

        # Build rationale for coaching cue
        comparison = {
            "depth": f"{max(0, self._rep_min_knee - (ideal.target_knee_angle if ideal else 90)):.0f} deg from target",
            "trunk": f"{max(0, self._rep_max_torso - (ideal.trunk_neutral_angle if ideal else 10)):.0f} deg over baseline",
            "tempo": f"descent {rep_result.descent_duration:.1f}s / ascent {rep_result.ascent_duration:.1f}s",
        }
        trend_val = self._trend_tracker.get_trend()
        trend_str = f"{'up' if trend_val > 0 else 'down'}{abs(trend_val):.0f} over recent reps"
        rationale = build_rationale(rep_result.rep_index, scores, faults, comparison, trend_str)

        # Gemini coaching (async)
        confidence = fused.confidence if fused else 0.5
        rep_event = build_rep_summary(
            rep_result, rationale, faults, self._rep_features_snapshot,
            pose_result.pose_confidence, confidence,
        )
        gemini_cfg = self._config.get("gemini", {})
        if gemini_cfg.get("enabled", False):
            gemini_payload = format_gemini_payload(rep_event)
            key = gemini_cfg.get("api_key", "") or os.environ.get("GEMINI_API_KEY", "")
            model = gemini_cfg.get("model", "gemini-2.0-flash")

            def _on_feedback(fb: str) -> None:
                with self._coaching_lock:
                    self._pending_coaching = fb

            send_to_gemini_async(
                gemini_payload, api_key=key, model=model,
                speak_enabled=False,  # TTS happens on frontend
                on_feedback=_on_feedback,
            )

        fault_names = [f.fault_type.value for f in faults]

        # Reset per-rep tracking
        self._rep_min_knee = 180.0
        self._rep_max_torso = 0.0
        self._rep_max_head_offset = 0.0
        self._live_score_ema = 100.0

        return RepData(
            rep_index=rep_result.rep_index,
            scores=scores,
            faults=fault_names,
            coaching_text=rationale.coaching_cue,
        )

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/server/pipeline.py squat_coach/tests/test_pipeline.py
git commit -m "feat(server): add headless SquatCoachPipeline"
```

---

## Task 4: WebSocket Handler & FastAPI App

**Files:**
- Create: `squat_coach/server/ws_handler.py`
- Create: `squat_coach/server/main.py`
- Modify: `squat_coach/requirements.txt`
- Test: `squat_coach/tests/test_ws_handler.py`

- [ ] **Step 1: Update requirements.txt**

Add to end of `squat_coach/requirements.txt`:

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
```

- [ ] **Step 2: Write the failing test**

```python
# squat_coach/tests/test_ws_handler.py
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
        # Send a tiny black JPEG
        import cv2
        import numpy as np
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", frame)
        ws.send_bytes(jpeg.tobytes())

        # Should receive a JSON response
        data = ws.receive_json()
        assert "type" in data
        assert data["type"] in ("calibration", "frame")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_ws_handler.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement ws_handler.py**

```python
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
    """Handle one client session over WebSocket."""
    await websocket.accept()
    logger.info("Client connected")

    pipeline = SquatCoachPipeline()
    delta = DeltaCompressor()

    try:
        while True:
            # Receive binary JPEG frame
            jpeg_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(
                np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if frame is None:
                continue

            timestamp = time.time()
            result = pipeline.process_frame(frame, timestamp)

            # Calibration message
            if result.calibration is not None:
                await websocket.send_json(result.calibration.to_dict())
                continue

            # Frame data (delta-compressed)
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
```

- [ ] **Step 5: Implement main.py**

```python
# squat_coach/server/main.py
"""FastAPI application for Squat Coach server."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from squat_coach.server.ws_handler import session_handler

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


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    await session_handler(websocket)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/piotrkacprzak/programow/hackathon && pip install fastapi uvicorn[standard] && python -m pytest squat_coach/tests/test_ws_handler.py -v`
Expected: All 2 tests PASS

- [ ] **Step 7: Test server starts manually**

Run: `cd /Users/piotrkacprzak/programow/hackathon && timeout 5 python -m uvicorn squat_coach.server.main:app --host 0.0.0.0 --port 8000 || true`
Expected: Server starts, prints "Squat Coach server starting", then times out after 5s

- [ ] **Step 8: Commit**

```bash
git add squat_coach/server/ws_handler.py squat_coach/server/main.py squat_coach/requirements.txt squat_coach/tests/test_ws_handler.py
git commit -m "feat(server): add FastAPI WebSocket server with session handler"
```

---

## Task 5: Frontend Types & Utilities

**Files:**
- Create: `frontend/lib/squat-types.ts`
- Create: `frontend/lib/skeleton.ts`
- Create: `frontend/lib/interpolation.ts`
- Create: `frontend/lib/tts.ts`

- [ ] **Step 1: Create squat session types**

```typescript
// frontend/lib/squat-types.ts
export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export type SquatPhase = "TOP" | "DESCENT" | "BOTTOM" | "ASCENT";

export interface SquatAngles {
  knee: number;
  hip: number;
  torso: number;
}

export interface RepScores {
  total: number;
  depth: number;
  trunk_control: number;
  posture_stability: number;
  movement_consistency: number;
}

export interface SquatRepResult {
  rep_index: number;
  scores: RepScores;
  faults: string[];
  coaching_text: string;
}

export interface SquatSessionState {
  status: "idle" | "connecting" | "calibrating" | "active" | "ended";
  calibrationProgress: number;
  landmarks: Landmark[] | null;
  phase: SquatPhase;
  angles: SquatAngles;
  score: number;
  confidence: number;
  repCount: number;
  reps: SquatRepResult[];
  currentFaults: string[];
  coachingText: string | null;
}

// WebSocket incoming message types
export type CalibrationMessage = {
  type: "calibration";
  status: "in_progress" | "complete";
  progress: number;
  view_type?: string;
};

export type FrameMessage = {
  type: "frame";
  data: {
    seq: number;
    landmarks?: number[][];
    phase?: string;
    knee_angle?: number;
    hip_angle?: number;
    torso_angle?: number;
    score?: number;
    confidence?: number;
  };
};

export type RepMessage = {
  type: "rep";
  rep_index: number;
  scores: RepScores;
  faults: string[];
  coaching_text: string;
};

export type CoachingMessage = {
  type: "coaching";
  text: string;
};

export type SessionEndMessage = {
  type: "session_end";
  total_reps: number;
  avg_score: number;
  trend: string;
};

export type ServerMessage =
  | CalibrationMessage
  | FrameMessage
  | RepMessage
  | CoachingMessage
  | SessionEndMessage;
```

- [ ] **Step 2: Create skeleton drawing utilities**

```typescript
// frontend/lib/skeleton.ts
import type { Landmark } from "./squat-types";

// MediaPipe BlazePose skeleton connections (matching squat_coach/pose/landmarks.py)
export const SKELETON_CONNECTIONS: [number, number][] = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  [27, 29], [27, 31], // left foot
  [28, 30], [28, 32], // right foot
];

const MIN_VISIBILITY = 0.5;

function visibilityColor(v: number): string {
  if (v > 0.8) return "#22C55E"; // green
  if (v > 0.6) return "#F59E0B"; // yellow
  return "#EF4444"; // red
}

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number,
): void {
  // Draw connections
  for (const [i, j] of SKELETON_CONNECTIONS) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (a.visibility < MIN_VISIBILITY || b.visibility < MIN_VISIBILITY) continue;

    ctx.strokeStyle = visibilityColor(Math.min(a.visibility, b.visibility));
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  }

  // Draw landmark dots
  for (const lm of landmarks) {
    if (lm.visibility < MIN_VISIBILITY) continue;
    ctx.fillStyle = visibilityColor(lm.visibility);
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawAngleLabel(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  jointIndex: number,
  angle: number,
  width: number,
  height: number,
): void {
  const lm = landmarks[jointIndex];
  if (lm.visibility < MIN_VISIBILITY) return;

  const x = lm.x * width + 12;
  const y = lm.y * height - 8;

  ctx.font = "bold 12px monospace";
  ctx.fillStyle = "#FFFFFF";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 2;
  const text = `${Math.round(angle)}°`;
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}
```

- [ ] **Step 3: Create interpolation utilities**

```typescript
// frontend/lib/interpolation.ts
import type { Landmark } from "./squat-types";

export function lerpLandmarks(
  from: Landmark[],
  to: Landmark[],
  t: number,
): Landmark[] {
  return from.map((f, i) => ({
    x: f.x + (to[i].x - f.x) * t,
    y: f.y + (to[i].y - f.y) * t,
    z: f.z + (to[i].z - f.z) * t,
    visibility: to[i].visibility,
  }));
}

export function parseLandmarks(raw: number[][]): Landmark[] {
  return raw.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
}
```

- [ ] **Step 4: Create TTS wrapper**

```typescript
// frontend/lib/tts.ts
export function speakCoaching(text: string): void {
  if (typeof window === "undefined") return;
  if (!("speechSynthesis" in window)) return;

  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.pitch = 1.0;
  speechSynthesis.speak(utterance);
}
```

- [ ] **Step 5: Commit**

```bash
cd /Users/piotrkacprzak/programow/hackathon
git add frontend/lib/squat-types.ts frontend/lib/skeleton.ts frontend/lib/interpolation.ts frontend/lib/tts.ts
git commit -m "feat(frontend): add squat session types, skeleton drawing, interpolation, and TTS utilities"
```

---

## Task 6: useSquatSession Hook

**Files:**
- Create: `frontend/hooks/use-squat-session.ts`

- [ ] **Step 1: Implement the hook**

```typescript
// frontend/hooks/use-squat-session.ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { parseLandmarks } from "@/lib/interpolation";
import type {
  Landmark,
  SquatAngles,
  SquatPhase,
  SquatRepResult,
  SquatSessionState,
  ServerMessage,
} from "@/lib/squat-types";
import { speakCoaching } from "@/lib/tts";

const CAPTURE_INTERVAL = 42; // ms, ~24fps
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

interface AccumulatedState {
  landmarks: Landmark[] | null;
  prevLandmarks: Landmark[] | null;
  lastUpdateTime: number;
}

export function useSquatSession() {
  const [state, setState] = useState<SquatSessionState>({
    status: "idle",
    calibrationProgress: 0,
    landmarks: null,
    phase: "TOP",
    angles: { knee: 0, hip: 0, torso: 0 },
    score: 0,
    confidence: 0,
    repCount: 0,
    reps: [],
    currentFaults: [],
    coachingText: null,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const accRef = useRef<AccumulatedState>({
    landmarks: null,
    prevLandmarks: null,
    lastUpdateTime: 0,
  });

  const handleMessage = useCallback((event: MessageEvent) => {
    const msg: ServerMessage = JSON.parse(event.data);

    switch (msg.type) {
      case "calibration":
        setState((s) => ({
          ...s,
          status: msg.status === "complete" ? "active" : "calibrating",
          calibrationProgress: msg.progress,
        }));
        break;

      case "frame": {
        const data = msg.data;
        setState((s) => {
          const next = { ...s };
          if (data.phase) next.phase = data.phase as SquatPhase;
          if (data.knee_angle !== undefined) next.angles = { ...next.angles, knee: data.knee_angle };
          if (data.hip_angle !== undefined) next.angles = { ...next.angles, hip: data.hip_angle };
          if (data.torso_angle !== undefined) next.angles = { ...next.angles, torso: data.torso_angle };
          if (data.score !== undefined) next.score = data.score;
          if (data.confidence !== undefined) next.confidence = data.confidence;
          if (data.landmarks) {
            const parsed = parseLandmarks(data.landmarks);
            accRef.current.prevLandmarks = accRef.current.landmarks;
            accRef.current.landmarks = parsed;
            accRef.current.lastUpdateTime = performance.now();
            next.landmarks = parsed;
          }
          return next;
        });
        break;
      }

      case "rep":
        setState((s) => ({
          ...s,
          repCount: msg.rep_index,
          reps: [...s.reps, msg as SquatRepResult],
          currentFaults: msg.faults,
          coachingText: msg.coaching_text,
        }));
        speakCoaching(msg.coaching_text);
        break;

      case "coaching":
        setState((s) => ({ ...s, coachingText: msg.text }));
        speakCoaching(msg.text);
        break;

      case "session_end":
        setState((s) => ({ ...s, status: "ended" }));
        break;
    }
  }, []);

  const startCapture = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!video || !ws) return;

    if (!offscreenRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = CAPTURE_WIDTH;
      canvas.height = CAPTURE_HEIGHT;
      offscreenRef.current = canvas;
    }
    const offscreen = offscreenRef.current;
    const ctx = offscreen.getContext("2d")!;

    captureIntervalRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      offscreen.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }, CAPTURE_INTERVAL);
  }, []);

  const startSession = useCallback(async () => {
    setState((s) => ({ ...s, status: "connecting" }));

    // Request camera
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera access denied:", err);
      setState((s) => ({ ...s, status: "idle" }));
      return;
    }

    // Connect WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_ANALYSIS_WS_URL || "ws://localhost:8000/ws/session";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, status: "calibrating", calibrationProgress: 0 }));
      startCapture();
    };
    ws.onmessage = handleMessage;
    ws.onclose = () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    };
    ws.onerror = () => {
      ws.close();
      setState((s) => ({ ...s, status: "idle" }));
    };
  }, [handleMessage, startCapture]);

  const endSession = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // Stop camera
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    state,
    videoRef,
    canvasRef,
    accRef,
    startSession,
    endSession,
  };
}
```

- [ ] **Step 2: Commit**

```bash
cd /Users/piotrkacprzak/programow/hackathon
git add frontend/hooks/use-squat-session.ts
git commit -m "feat(frontend): add useSquatSession hook for real-time squat analysis"
```

---

## Task 7: Frontend Overlay Components

**Files:**
- Create: `frontend/components/session/overlay-canvas.tsx`
- Create: `frontend/components/session/calibration-overlay.tsx`
- Create: `frontend/components/session/session-hud.tsx`

- [ ] **Step 1: Create overlay canvas component**

```tsx
// frontend/components/session/overlay-canvas.tsx
"use client";

import { useEffect, useRef, type RefObject } from "react";

import { lerpLandmarks } from "@/lib/interpolation";
import { drawAngleLabel, drawSkeleton } from "@/lib/skeleton";
import type { Landmark, SquatAngles } from "@/lib/squat-types";

interface OverlayCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  accRef: RefObject<{
    landmarks: Landmark[] | null;
    prevLandmarks: Landmark[] | null;
    lastUpdateTime: number;
  }>;
  angles: SquatAngles;
  width: number;
  height: number;
}

export function OverlayCanvas({ canvasRef, accRef, angles, width, height }: OverlayCanvasProps) {
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;

    function render() {
      ctx.clearRect(0, 0, width, height);
      const acc = accRef.current;

      if (acc.landmarks) {
        // Interpolate between previous and current for smooth 60fps
        const t = acc.prevLandmarks
          ? Math.min((performance.now() - acc.lastUpdateTime) / 42, 1)
          : 1;
        const landmarks = acc.prevLandmarks
          ? lerpLandmarks(acc.prevLandmarks, acc.landmarks, t)
          : acc.landmarks;

        drawSkeleton(ctx, landmarks, width, height);

        // Knee angle labels (landmarks 25=left knee, 26=right knee)
        if (angles.knee > 0) {
          drawAngleLabel(ctx, landmarks, 25, angles.knee, width, height);
        }
        // Hip angle labels (landmarks 23=left hip, 24=right hip)
        if (angles.hip > 0) {
          drawAngleLabel(ctx, landmarks, 23, angles.hip, width, height);
        }
      }

      rafRef.current = requestAnimationFrame(render);
    }

    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [canvasRef, accRef, angles, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ width, height }}
    />
  );
}
```

- [ ] **Step 2: Create calibration overlay**

```tsx
// frontend/components/session/calibration-overlay.tsx
"use client";

interface CalibrationOverlayProps {
  progress: number; // 0.0 - 1.0
}

export function CalibrationOverlay({ progress }: CalibrationOverlayProps) {
  const percent = Math.round(progress * 100);

  return (
    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60">
      <div className="flex flex-col items-center gap-4 rounded-2xl bg-bg-card/90 p-8 backdrop-blur-sm">
        <div className="text-lg font-semibold text-text-primary">Stand Still</div>
        <p className="text-sm text-text-secondary">Calibrating your position...</p>

        {/* Progress bar */}
        <div className="h-2 w-48 overflow-hidden rounded-full bg-bg-surface">
          <div
            className="h-full rounded-full bg-gradient-to-r from-gradient-start to-gradient-end transition-all duration-200"
            style={{ width: `${percent}%` }}
          />
        </div>
        <p className="text-xs text-text-muted">{percent}%</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create session HUD**

```tsx
// frontend/components/session/session-hud.tsx
"use client";

import type { SquatPhase, SquatRepResult } from "@/lib/squat-types";

interface SessionHudProps {
  phase: SquatPhase;
  repCount: number;
  score: number;
  coachingText: string | null;
}

const PHASE_COLORS: Record<SquatPhase, string> = {
  TOP: "bg-blue-500",
  DESCENT: "bg-amber-500",
  BOTTOM: "bg-red-500",
  ASCENT: "bg-green-500",
};

export function SessionHud({ phase, repCount, score, coachingText }: SessionHudProps) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-20 flex flex-col gap-2 p-3">
      {/* Top bar: phase + rep + score */}
      <div className="flex items-center justify-between">
        <div className={`rounded-full px-3 py-1 text-xs font-bold text-white ${PHASE_COLORS[phase]}`}>
          {phase}
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold text-white backdrop-blur-sm">
            Rep {repCount}
          </div>
          <div className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold text-white backdrop-blur-sm">
            {Math.round(score)}
          </div>
        </div>
      </div>

      {/* Coaching banner */}
      {coachingText && (
        <div className="mx-auto max-w-xs rounded-xl bg-gradient-to-r from-gradient-start/80 to-gradient-end/80 px-4 py-2 text-center text-sm font-medium text-white backdrop-blur-sm">
          {coachingText}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Commit**

```bash
cd /Users/piotrkacprzak/programow/hackathon
git add frontend/components/session/overlay-canvas.tsx frontend/components/session/calibration-overlay.tsx frontend/components/session/session-hud.tsx
git commit -m "feat(frontend): add overlay canvas, calibration overlay, and session HUD components"
```

---

## Task 8: Integrate Solo Session Page

**Files:**
- Modify: `frontend/app/(player)/session/solo/page.tsx`

- [ ] **Step 1: Rewrite the solo session page**

Replace the full contents of `frontend/app/(player)/session/solo/page.tsx`:

```tsx
// frontend/app/(player)/session/solo/page.tsx
"use client";

import { ArrowLeft, Play, Square } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CalibrationOverlay } from "@/components/session/calibration-overlay";
import { OverlayCanvas } from "@/components/session/overlay-canvas";
import { SessionHud } from "@/components/session/session-hud";
import { useSquatSession } from "@/hooks/use-squat-session";

export default function SoloSessionPage() {
  const router = useRouter();
  const { state, videoRef, canvasRef, accRef, startSession, endSession } = useSquatSession();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Track container dimensions for canvas sizing
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleEnd = () => {
    endSession();
    router.push("/results");
  };

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => { endSession(); router.push("/home"); }}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <p className="text-sm font-semibold text-text-primary">Squat Coach</p>
        <div className="w-5" />
      </div>

      {/* Video + overlay container */}
      <div ref={containerRef} className="relative mx-4 flex-1 overflow-hidden rounded-xl bg-camera-bg">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />

        {/* Canvas overlay for skeleton */}
        {state.status === "active" && dimensions.width > 0 && (
          <OverlayCanvas
            canvasRef={canvasRef}
            accRef={accRef}
            angles={state.angles}
            width={dimensions.width}
            height={dimensions.height}
          />
        )}

        {/* Calibration overlay */}
        {state.status === "calibrating" && (
          <CalibrationOverlay progress={state.calibrationProgress} />
        )}

        {/* Session HUD */}
        {state.status === "active" && (
          <SessionHud
            phase={state.phase}
            repCount={state.repCount}
            score={state.score}
            coachingText={state.coachingText}
          />
        )}

        {/* Idle state: start button */}
        {state.status === "idle" && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <button
              onClick={startSession}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end px-6 py-3 text-sm font-semibold text-white"
            >
              <Play className="h-4 w-4" />
              Start Session
            </button>
          </div>
        )}
      </div>

      {/* Bottom controls */}
      <div className="flex items-center justify-center px-4 py-4">
        {state.status === "active" && (
          <button
            onClick={handleEnd}
            className="flex items-center gap-2 rounded-2xl bg-error px-6 py-3 text-sm font-semibold text-white"
          >
            <Square className="h-4 w-4" />
            End Session
          </button>
        )}

        {state.status === "idle" && (
          <p className="text-sm text-text-secondary">Tap Start to begin your squat session</p>
        )}

        {(state.status === "calibrating" || state.status === "connecting") && (
          <p className="text-sm text-text-secondary">
            {state.status === "connecting" ? "Connecting..." : "Calibrating..."}
          </p>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify the frontend builds**

Run: `cd /Users/piotrkacprzak/programow/hackathon/frontend && pnpm build`
Expected: Build succeeds with no TypeScript errors

- [ ] **Step 3: Commit**

```bash
cd /Users/piotrkacprzak/programow/hackathon
git add frontend/app/\(player\)/session/solo/page.tsx
git commit -m "feat(frontend): integrate useSquatSession into solo session page with live overlay"
```

---

## Task 9: Environment Config & End-to-End Verification

**Files:**
- Create: `frontend/.env.local` (if not exists)

- [ ] **Step 1: Create .env.local**

```bash
# frontend/.env.local
NEXT_PUBLIC_ANALYSIS_WS_URL=ws://localhost:8000/ws/session
```

- [ ] **Step 2: Verify server starts**

Run: `cd /Users/piotrkacprzak/programow/hackathon && pip install -r squat_coach/requirements.txt && timeout 5 python -m uvicorn squat_coach.server.main:app --host 0.0.0.0 --port 8000 2>&1 || true`
Expected: Server starts successfully, prints startup log

- [ ] **Step 3: Verify frontend builds and starts**

Run: `cd /Users/piotrkacprzak/programow/hackathon/frontend && pnpm build`
Expected: Build succeeds

- [ ] **Step 4: Run all backend tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_protocol.py squat_coach/tests/test_delta.py squat_coach/tests/test_pipeline.py squat_coach/tests/test_ws_handler.py -v`
Expected: All tests pass

- [ ] **Step 5: Commit .env.local**

```bash
cd /Users/piotrkacprzak/programow/hackathon
git add frontend/.env.local
git commit -m "chore: add analysis WebSocket URL to frontend env config"
```

- [ ] **Step 6: Add .env.local to .gitignore if not already there**

Check if `frontend/.gitignore` contains `.env.local`. If not, add it. If it does, skip this step.

---

## Summary

| Task | What it builds | Depends on |
|------|---------------|------------|
| 1 | Protocol message schemas | - |
| 2 | Delta compressor | Task 1 |
| 3 | Headless pipeline | Tasks 1-2 |
| 4 | FastAPI WebSocket server | Tasks 1-3 |
| 5 | Frontend types + utilities | - |
| 6 | useSquatSession hook | Task 5 |
| 7 | Overlay UI components | Tasks 5-6 |
| 8 | Solo session page integration | Tasks 6-7 |
| 9 | Environment config + verification | Tasks 4, 8 |

Tasks 1-4 (server) and 5-8 (frontend) are independent tracks that can be parallelized. Task 9 ties them together.
