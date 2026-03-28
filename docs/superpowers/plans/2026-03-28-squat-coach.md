# Squat Coach Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time squat analysis system that uses MediaPipe BlazePose 3D landmarks, TCN+GRU temporal models, and structured scoring to provide live coaching feedback with Gemini-ready event output.

**Architecture:** Layered pipeline: video capture → MediaPipe pose → preprocessing/smoothing → feature extraction (D=42 model input + ~50 total) → sequence buffer → TCN+GRU ensemble → phase detection + fault detection → scoring with rationale → terminal logging + simple overlay + Gemini payloads. Training pipeline uses ALEX-GYM-1 + Zenodo + synthetic data on MPS.

**Tech Stack:** Python 3.11+, OpenCV, MediaPipe, PyTorch, NumPy, PyYAML, dataclasses

**Spec:** `docs/superpowers/specs/2026-03-28-squat-coach-design.md`

---

## File Structure

```
squat_coach/
├── __init__.py
├── __main__.py                    # python -m squat_coach entry point
├── app.py                         # main pipeline orchestrator
├── config/
│   ├── default.yaml               # camera, FPS, debug, device, smoothing
│   ├── model.yaml                 # model arch, ensemble weights, seq length
│   ├── scoring.yaml               # score weights, fault thresholds, calibration
│   └── overlay.yaml               # display settings, colors, fonts
├── camera/
│   ├── __init__.py
│   ├── base.py                    # VideoSource ABC
│   ├── webcam_stream.py           # OpenCV webcam capture
│   └── video_replay.py            # OpenCV file replay
├── pose/
│   ├── __init__.py
│   ├── base.py                    # PoseEstimator ABC + PoseResult dataclass
│   ├── mediapipe_blazepose3d.py   # MediaPipe Pose Landmarker wrapper
│   └── landmarks.py               # landmark names, indices, skeleton connections
├── preprocessing/
│   ├── __init__.py
│   ├── smoothing.py               # EMA landmark smoother
│   ├── normalization.py           # hip-centered normalization
│   ├── sequence_buffer.py         # rolling window buffer
│   └── calibration.py             # calibration flow + view detection
├── biomechanics/
│   ├── __init__.py
│   ├── angles.py                  # 3-point angle computation
│   ├── vectors.py                 # bone/trunk vector utilities
│   ├── distances.py               # pairwise joint distances
│   ├── kinematics.py              # velocity/acceleration via finite diff
│   ├── squat_features.py          # orchestrator: full D=42 + extended features
│   ├── posture_analysis.py        # rounded_back_risk composite
│   ├── side_view_constraints.py   # side-view specific features
│   └── front_view_constraints.py  # front-view specific features
├── phases/
│   ├── __init__.py
│   ├── state_machine.py           # Phase enum + state machine
│   ├── phase_detector.py          # phase from model probs + fallback
│   └── rep_segmenter.py           # rep boundaries + validation
├── models/
│   ├── __init__.py
│   ├── temporal_base.py           # TemporalModelBase ABC
│   ├── temporal_tcn.py            # TCN implementation
│   ├── temporal_gru.py            # GRU implementation
│   ├── temporal_stgcn_scaffold.py # ST-GCN extension point (not trained)
│   ├── ensemble_fusion.py         # confidence-weighted fusion
│   ├── feature_tensor_builder.py  # dict→tensor conversion + z-score
│   ├── inference_manager.py       # load models, run inference
│   └── model_factory.py           # registry + factory
├── training/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset for squat sequences
│   ├── data_pipeline.py           # download/load/preprocess/cache data
│   ├── synthetic_generator.py     # synthetic squat sequence generator
│   ├── phase_labeler.py           # auto-label phases from hip kinematics
│   ├── trainer.py                 # unified training loop (MPS/CPU)
│   ├── evaluate.py                # metrics + model comparison
│   └── train_all.py               # end-to-end training script
├── scoring/
│   ├── __init__.py
│   ├── ideal_reference.py         # personalized idealized reference
│   ├── score_components.py        # individual score computations
│   ├── score_fusion.py            # combine into rep_quality + overall
│   ├── rationale.py               # rationale object builder
│   └── trend_analysis.py          # EMA trend across reps
├── faults/
│   ├── __init__.py
│   ├── fault_types.py             # FaultType enum + FaultDetection dataclass
│   ├── fault_rules.py             # per-fault rule definitions
│   ├── evidence_engine.py         # aggregate evidence + detect faults
│   └── confidence_gating.py       # suppress low-confidence faults
├── rendering/
│   ├── __init__.py
│   ├── overlay.py                 # simple overlay compositor
│   ├── draw_pose.py               # skeleton drawing on frame
│   ├── draw_metrics.py            # score/angle text
│   └── draw_feedback.py           # coaching cue text
├── events/
│   ├── __init__.py
│   ├── schemas.py                 # event dataclasses
│   ├── event_builder.py           # construct events from state
│   ├── formatter.py               # format for terminal logging
│   ├── gemini_payloads.py         # Gemini-ready payload formatter
│   └── coaching_priority.py       # cue arbitration
├── session/
│   ├── __init__.py
│   ├── session_state.py           # session-level state
│   ├── rep_history.py             # per-rep history
│   └── jsonl_logger.py            # JSONL file writer
├── utils/
│   ├── __init__.py
│   ├── math_utils.py              # angle_between_vectors, clamp, etc.
│   ├── logging_utils.py           # structured logging setup
│   ├── timing.py                  # FPS tracker
│   └── enums.py                   # Phase, ViewType, FaultType enums
├── tests/
│   ├── __init__.py
│   ├── test_math_utils.py
│   ├── test_angles.py
│   ├── test_features.py
│   ├── test_smoothing.py
│   ├── test_sequence_buffer.py
│   ├── test_phase_detector.py
│   ├── test_scoring.py
│   ├── test_ensemble.py
│   ├── test_synthetic.py
│   ├── test_fault_detection.py
│   ├── test_calibration.py
│   └── test_data_pipeline.py
├── requirements.txt
└── README.md
```

---

## Chunk 1: Foundation — Utils, Config, Camera, Pose

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `squat_coach/__init__.py`
- Create: `squat_coach/__main__.py`
- Create: `squat_coach/requirements.txt`
- Create: `squat_coach/utils/__init__.py`
- Create: `squat_coach/utils/enums.py`

- [ ] **Step 1: Create project root and requirements.txt**

```
squat_coach/requirements.txt:
```
```
opencv-python>=4.8.0
mediapipe>=0.10.9
torch>=2.1.0
numpy>=1.24.0
PyYAML>=6.0
```

- [ ] **Step 2: Create package init and entry point**

```python
# squat_coach/__init__.py
"""Squat Coach — Real-time squat analysis system."""

# squat_coach/__main__.py
"""Entry point for python -m squat_coach."""
import argparse
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Squat Coach")
    parser.add_argument("--mode", choices=["webcam", "replay", "train"], default="webcam")
    parser.add_argument("--video", type=str, help="Video file path for replay mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-features", action="store_true", help="Enable per-frame JSONL feature logging")
    args = parser.parse_args()

    from squat_coach.app import SquatCoachApp
    app = SquatCoachApp(
        mode=args.mode,
        video_path=args.video,
        debug=args.debug,
        log_features=args.log_features,
    )
    app.run()

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create enums**

```python
# squat_coach/utils/enums.py
"""Shared enumerations."""
from enum import Enum

class Phase(Enum):
    TOP = "top"
    DESCENT = "descent"
    BOTTOM = "bottom"
    ASCENT = "ascent"

class ViewType(Enum):
    SIDE = "side"
    FRONT = "front"
    UNKNOWN = "unknown"

class FaultType(Enum):
    INSUFFICIENT_DEPTH = "insufficient_depth"
    EXCESSIVE_FORWARD_LEAN = "excessive_forward_lean"
    ROUNDED_BACK_RISK = "rounded_back_risk"
    UNSTABLE_TORSO = "unstable_torso"
    HEEL_FAULT = "heel_fault"
    KNEE_VALGUS = "knee_valgus"
    INCONSISTENT_TEMPO = "inconsistent_tempo"
    POOR_TRUNK_CONTROL = "poor_trunk_control"
    LOW_CONFIDENCE = "low_confidence"
    INVALID_VIEW = "invalid_view"
```

- [ ] **Step 4: Commit**

```bash
git add squat_coach/
git commit -m "feat: project scaffolding with entry point, deps, enums"
```

---

### Task 2: Math utilities with tests

**Files:**
- Create: `squat_coach/utils/math_utils.py`
- Create: `squat_coach/tests/__init__.py`
- Create: `squat_coach/tests/test_math_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_math_utils.py
"""Tests for math utilities."""
import numpy as np
import pytest
from squat_coach.utils.math_utils import (
    angle_between_vectors,
    angle_at_joint,
    vector_from_points,
    normalize_vector,
    perpendicular_distance_to_line,
)

def test_angle_between_vectors_perpendicular():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 90.0) < 0.1

def test_angle_between_vectors_parallel():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2)) < 0.1

def test_angle_between_vectors_opposite():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([-1.0, 0.0, 0.0])
    assert abs(angle_between_vectors(v1, v2) - 180.0) < 0.1

def test_angle_at_joint_straight():
    # Straight leg: hip-knee-ankle in a line
    a = np.array([0.0, 1.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])  # joint
    c = np.array([0.0, -1.0, 0.0])
    assert abs(angle_at_joint(a, b, c) - 180.0) < 0.1

def test_angle_at_joint_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    assert abs(angle_at_joint(a, b, c) - 90.0) < 0.1

def test_vector_from_points():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = vector_from_points(a, b)
    np.testing.assert_array_equal(result, np.array([3.0, 3.0, 3.0]))

def test_normalize_vector():
    v = np.array([3.0, 4.0, 0.0])
    result = normalize_vector(v)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6

def test_normalize_zero_vector():
    v = np.array([0.0, 0.0, 0.0])
    result = normalize_vector(v)
    np.testing.assert_array_equal(result, np.array([0.0, 0.0, 0.0]))

def test_perpendicular_distance_to_line():
    # Point directly above the midpoint of a horizontal line
    line_start = np.array([0.0, 0.0, 0.0])
    line_end = np.array([2.0, 0.0, 0.0])
    point = np.array([1.0, 1.0, 0.0])
    dist = perpendicular_distance_to_line(point, line_start, line_end)
    assert abs(dist - 1.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_math_utils.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement math_utils**

```python
# squat_coach/utils/math_utils.py
"""Core math utilities for geometry and vector operations."""
import numpy as np
from numpy.typing import NDArray

def angle_between_vectors(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> float:
    """Angle in degrees between two vectors. Returns 0-180."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def angle_at_joint(
    point_a: NDArray[np.float64],
    joint: NDArray[np.float64],
    point_c: NDArray[np.float64],
) -> float:
    """Angle at joint formed by point_a—joint—point_c, in degrees (0-180)."""
    v1 = point_a - joint
    v2 = point_c - joint
    return angle_between_vectors(v1, v2)

def vector_from_points(
    start: NDArray[np.float64], end: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Vector from start to end."""
    return end - start

def normalize_vector(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vector. Returns zero vector if input is near-zero."""
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm

def perpendicular_distance_to_line(
    point: NDArray[np.float64],
    line_start: NDArray[np.float64],
    line_end: NDArray[np.float64],
) -> float:
    """Perpendicular distance from point to the line defined by line_start→line_end."""
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-8:
        return float(np.linalg.norm(point - line_start))
    # Cross product magnitude / line length = perpendicular distance
    cross = np.cross(line_vec, line_start - point)
    return float(np.linalg.norm(cross) / line_len)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_math_utils.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/utils/math_utils.py squat_coach/tests/
git commit -m "feat: math utilities with angle, vector, distance functions"
```

---

### Task 3: Logging and timing utilities

**Files:**
- Create: `squat_coach/utils/logging_utils.py`
- Create: `squat_coach/utils/timing.py`

- [ ] **Step 1: Implement logging setup**

```python
# squat_coach/utils/logging_utils.py
"""Structured logging configuration."""
import logging
import sys

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return the squat_coach logger."""
    logger = logging.getLogger("squat_coach")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    return logger
```

- [ ] **Step 2: Implement FPS tracker**

```python
# squat_coach/utils/timing.py
"""Frame timing and FPS tracking."""
import time
from collections import deque

class FPSTracker:
    """Track rolling FPS over a window of recent frames."""

    def __init__(self, window_size: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window_size)

    def tick(self) -> None:
        """Record a frame timestamp."""
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Current rolling FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed < 1e-6:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/utils/logging_utils.py squat_coach/utils/timing.py
git commit -m "feat: logging setup and FPS tracker utilities"
```

---

### Task 4: YAML config system

**Files:**
- Create: `squat_coach/config/default.yaml`
- Create: `squat_coach/config/model.yaml`
- Create: `squat_coach/config/scoring.yaml`
- Create: `squat_coach/config/overlay.yaml`

- [ ] **Step 1: Create default.yaml**

```yaml
# squat_coach/config/default.yaml
# General settings for the squat coach system

camera:
  device_id: 0
  width: 1280
  height: 720
  target_fps: 30
  min_resolution_w: 640
  min_resolution_h: 480

preprocessing:
  ema_alpha: 0.4               # EMA smoothing. Higher = less smoothing.
  max_dropped_frames: 5         # Hold last landmarks for this many frames
  calibration_duration_s: 2.0   # Seconds of standing for calibration

device: "auto"                  # "auto" detects MPS/CUDA/CPU

logging:
  terminal_throttle_hz: 5.0     # Max log lines per second for frame data
  debug: false

session:
  log_dir: "sessions"           # Directory for JSONL session logs
  log_features: false           # Per-frame JSONL feature logging
```

- [ ] **Step 2: Create model.yaml**

```yaml
# squat_coach/config/model.yaml
# Temporal model architecture and ensemble settings

sequence:
  length: 60                    # Frames per window (2s at 30fps)
  feature_dim: 42               # D=42 engineered features

models:
  tcn:
    enabled: true
    num_channels: [64, 64, 64]  # channels per TCN block
    kernel_size: 3
    dropout: 0.2

  gru:
    enabled: true
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2

  stgcn:
    enabled: false              # Scaffold only — not trained

output_heads:
  phase_classes: 4              # top, descent, bottom, ascent
  fault_classes: 6              # depth, forward_lean, rounded_back, heel, torso, tempo
  quality_dim: 1                # scalar quality regression

ensemble:
  weights:
    tcn: 0.5
    gru: 0.5
  per_head_weights:             # Optional per-head overrides
    phase:
      tcn: 0.5
      gru: 0.5
    fault:
      tcn: 0.5
      gru: 0.5
    quality:
      tcn: 0.5
      gru: 0.5

training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 50
  patience: 10                  # Early stopping patience
  loss_weights:
    phase: 1.0
    fault: 1.0
    quality: 0.5
  val_split: 0.15
  test_split: 0.15

checkpoints:
  dir: "squat_coach/models/checkpoints"
```

- [ ] **Step 3: Create scoring.yaml**

```yaml
# squat_coach/config/scoring.yaml
# Score weights, fault thresholds, calibration

scoring:
  weights:
    depth: 0.25
    trunk_control: 0.25
    posture_stability: 0.25
    movement_consistency: 0.25
  model_quality_weight: 0.2     # model quality_score weight in rep_quality

  overall_form_ema_alpha: 0.3   # EMA smoothing for session score

calibration:
  target_knee_angle_deg: 90.0   # Target depth (adjustable per user)
  trunk_tolerance_deg: 15.0     # Acceptable trunk deviation from baseline
  min_rep_duration_s: 0.8       # Reject reps shorter than this
  min_phase_duration_s: 0.15    # Debounce: min phase duration
  rep_cooldown_s: 0.5           # Min time between reps

faults:
  confidence_threshold: 0.5     # Suppress faults below this confidence
  persistence_frames: 5         # Fault must persist this many frames

  thresholds:
    insufficient_depth:
      knee_angle_max_deg: 110.0
    excessive_forward_lean:
      trunk_deviation_deg: 25.0
    rounded_back_risk:
      risk_threshold: 0.5
    unstable_torso:
      variance_threshold: 8.0
    heel_fault:
      displacement_threshold: 0.05
    knee_valgus:
      angle_threshold_deg: 10.0
    inconsistent_tempo:
      timing_deviation_ratio: 0.4
```

- [ ] **Step 4: Create overlay.yaml**

```yaml
# squat_coach/config/overlay.yaml
# Simple overlay display settings

overlay:
  skeleton_color: [0, 255, 0]       # BGR green
  skeleton_thickness: 2
  landmark_radius: 4
  landmark_color: [0, 200, 255]     # BGR yellow-orange

  text:
    font_scale: 0.7
    thickness: 2
    color: [255, 255, 255]          # White
    bg_color: [0, 0, 0]             # Black background
    bg_alpha: 0.6

  cue:
    display_duration_s: 3.0
    suppress_repeat_s: 5.0
    font_scale: 0.8
    color: [0, 200, 255]            # Yellow-orange
```

- [ ] **Step 5: Commit**

```bash
git add squat_coach/config/
git commit -m "feat: YAML config files for camera, models, scoring, overlay"
```

---

### Task 5: Camera subsystem

**Files:**
- Create: `squat_coach/camera/__init__.py`
- Create: `squat_coach/camera/base.py`
- Create: `squat_coach/camera/webcam_stream.py`
- Create: `squat_coach/camera/video_replay.py`

- [ ] **Step 1: Implement camera base and implementations**

```python
# squat_coach/camera/base.py
"""Video source interface."""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from numpy.typing import NDArray

class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def read(self) -> tuple[bool, Optional[NDArray[np.uint8]]]:
        """Read next frame. Returns (success, frame_bgr)."""
        ...

    @abstractmethod
    def release(self) -> None:
        """Release video source."""
        ...

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is available."""
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        """Source FPS."""
        ...

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of frames."""
        ...
```

```python
# squat_coach/camera/webcam_stream.py
"""Live webcam video source."""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from squat_coach.camera.base import VideoSource

class WebcamStream(VideoSource):
    """OpenCV webcam capture."""

    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720) -> None:
        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> tuple[bool, Optional[NDArray[np.uint8]]]:
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
```

```python
# squat_coach/camera/video_replay.py
"""Video file replay source."""
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from squat_coach.camera.base import VideoSource

class VideoReplay(VideoSource):
    """OpenCV video file reader for replay/debug mode."""

    def __init__(self, video_path: str) -> None:
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

    def read(self) -> tuple[bool, Optional[NDArray[np.uint8]]]:
        ret, frame = self._cap.read()
        return ret, frame if ret else None

    def release(self) -> None:
        self._cap.release()

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/camera/
git commit -m "feat: camera subsystem with webcam and video replay sources"
```

---

### Task 6: Pose subsystem — landmarks and base

**Files:**
- Create: `squat_coach/pose/__init__.py`
- Create: `squat_coach/pose/landmarks.py`
- Create: `squat_coach/pose/base.py`

- [ ] **Step 1: Implement landmark definitions**

```python
# squat_coach/pose/landmarks.py
"""MediaPipe BlazePose landmark names, indices, and skeleton connections.

BlazePose provides 33 landmarks in 3D world coordinates (meters, hip-centered).
See: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

# Landmark index constants — matches MediaPipe Pose Landmarker output order
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

NUM_LANDMARKS = 33

# Skeleton connections for drawing (pairs of landmark indices)
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_ANKLE, LEFT_FOOT_INDEX),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]

# Key joint groups for visibility checks
LOWER_BODY_LANDMARKS = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]
TORSO_LANDMARKS = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]

# Pairwise distance pairs for D=42 feature vector (indices 34-41)
PAIRWISE_DISTANCE_PAIRS: list[tuple[int, int]] = [
    (LEFT_HIP, LEFT_KNEE),
    (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    # hip_mid→shoulder_mid and nose→shoulder_mid computed from midpoints
]
```

- [ ] **Step 2: Implement PoseResult dataclass and base**

```python
# squat_coach/pose/base.py
"""Pose estimator interface and result dataclass."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass
class PoseResult:
    """Result from a single pose estimation frame.

    Attributes:
        timestamp: Frame timestamp in seconds.
        world_landmarks: 3D world landmarks (33, 3) in meters, hip-centered.
        image_landmarks: 2D image landmarks (33, 3) as normalized x, y, visibility.
        visibility: Per-landmark visibility scores (33,), range [0, 1].
        pose_confidence: Overall detection confidence [0, 1].
        detected: Whether a pose was successfully detected.
    """
    timestamp: float
    world_landmarks: Optional[NDArray[np.float64]] = None   # (33, 3)
    image_landmarks: Optional[NDArray[np.float64]] = None   # (33, 3)
    visibility: Optional[NDArray[np.float64]] = None        # (33,)
    pose_confidence: float = 0.0
    detected: bool = False

class PoseEstimator(ABC):
    """Abstract base class for pose estimation backends."""

    @abstractmethod
    def estimate(self, frame_bgr: NDArray[np.uint8], timestamp: float) -> PoseResult:
        """Run pose estimation on a single BGR frame."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
        ...
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/pose/
git commit -m "feat: pose subsystem with landmark definitions and base interface"
```

---

### Task 7: MediaPipe BlazePose 3D implementation

**Files:**
- Create: `squat_coach/pose/mediapipe_blazepose3d.py`

- [ ] **Step 1: Implement MediaPipe wrapper**

```python
# squat_coach/pose/mediapipe_blazepose3d.py
"""MediaPipe Pose Landmarker wrapper using BlazePose 3D world landmarks.

This is the primary pose estimation backend. It uses MediaPipe's Pose Landmarker
task API to detect 33 BlazePose landmarks in both image coordinates and 3D world
coordinates (meters, hip-centered coordinate system).
"""
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from squat_coach.pose.base import PoseEstimator, PoseResult
from squat_coach.pose.landmarks import NUM_LANDMARKS

logger = logging.getLogger("squat_coach.pose")


class MediaPipeBlazePose3D(PoseEstimator):
    """MediaPipe Pose Landmarker with BlazePose 3D world landmarks.

    Produces both image-space (normalized) and world-space (meters) landmarks
    for all 33 BlazePose body points. World landmarks are in a hip-centered
    coordinate system suitable for view-invariant biomechanics computation.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_poses: int = 1,
    ) -> None:
        # Use the legacy Pose solution for broad compatibility
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        logger.info(
            "MediaPipe BlazePose initialized (complexity=%d, det_conf=%.2f, track_conf=%.2f)",
            model_complexity,
            min_detection_confidence,
            min_tracking_confidence,
        )

    def estimate(self, frame_bgr: NDArray[np.uint8], timestamp: float) -> PoseResult:
        """Run BlazePose on a BGR frame, return world + image landmarks."""
        import cv2

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(frame_rgb)

        if results.pose_world_landmarks is None or results.pose_landmarks is None:
            return PoseResult(timestamp=timestamp, detected=False)

        # Extract world landmarks: 3D in meters, hip-centered
        world_lm = results.pose_world_landmarks.landmark
        world_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in world_lm], dtype=np.float64
        )

        # Extract image landmarks: normalized [0,1] x, y + z depth proxy
        image_lm = results.pose_landmarks.landmark
        image_arr = np.array(
            [[lm.x, lm.y, lm.z] for lm in image_lm], dtype=np.float64
        )

        # Extract per-landmark visibility
        vis = np.array([lm.visibility for lm in world_lm], dtype=np.float64)

        # Overall confidence: mean visibility of detected landmarks
        pose_confidence = float(np.mean(vis))

        return PoseResult(
            timestamp=timestamp,
            world_landmarks=world_arr,
            image_landmarks=image_arr,
            visibility=vis,
            pose_confidence=pose_confidence,
            detected=True,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        logger.info("MediaPipe BlazePose closed")
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/pose/mediapipe_blazepose3d.py
git commit -m "feat: MediaPipe BlazePose 3D pose estimator implementation"
```

---

### Task 8: Preprocessing — smoothing and normalization

**Files:**
- Create: `squat_coach/preprocessing/__init__.py`
- Create: `squat_coach/preprocessing/smoothing.py`
- Create: `squat_coach/preprocessing/normalization.py`
- Create: `squat_coach/tests/test_smoothing.py`

- [ ] **Step 1: Write failing tests for smoothing**

```python
# squat_coach/tests/test_smoothing.py
"""Tests for EMA landmark smoothing."""
import numpy as np
from squat_coach.preprocessing.smoothing import EMALandmarkSmoother

def test_smoother_first_frame_passthrough():
    """First frame should pass through unchanged."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    landmarks = np.ones((33, 3))
    result = smoother.smooth(landmarks)
    np.testing.assert_array_equal(result, landmarks)

def test_smoother_reduces_jitter():
    """EMA should smooth out a spike."""
    smoother = EMALandmarkSmoother(alpha=0.3)
    base = np.zeros((33, 3))
    smoother.smooth(base.copy())  # Initialize

    # Spike frame
    spike = np.ones((33, 3)) * 10.0
    result = smoother.smooth(spike)

    # Should be pulled toward base, not at the spike value
    assert np.all(result < 5.0)

def test_smoother_converges():
    """Repeated same input should converge to that input."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    target = np.ones((33, 3)) * 5.0
    for _ in range(50):
        result = smoother.smooth(target.copy())
    np.testing.assert_allclose(result, target, atol=0.01)

def test_smoother_reset():
    """After reset, next frame should pass through."""
    smoother = EMALandmarkSmoother(alpha=0.5)
    smoother.smooth(np.ones((33, 3)))
    smoother.reset()
    new_val = np.ones((33, 3)) * 3.0
    result = smoother.smooth(new_val)
    np.testing.assert_array_equal(result, new_val)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_smoothing.py -v`
Expected: FAIL

- [ ] **Step 3: Implement smoothing**

```python
# squat_coach/preprocessing/smoothing.py
"""Exponential moving average (EMA) landmark smoother.

Smooths 3D landmark positions across frames to reduce jitter from
pose estimation noise. Higher alpha = less smoothing, lower latency.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional

class EMALandmarkSmoother:
    """EMA smoother for (N, 3) landmark arrays."""

    def __init__(self, alpha: float = 0.4) -> None:
        """
        Args:
            alpha: Smoothing factor in [0, 1]. Higher = less smoothing.
                   0.4 is a good default for 30fps pose data.
        """
        self._alpha = alpha
        self._prev: Optional[NDArray[np.float64]] = None

    def smooth(self, landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply EMA to a new frame of landmarks.

        Args:
            landmarks: Shape (N, 3) landmark positions.

        Returns:
            Smoothed landmarks, same shape.
        """
        if self._prev is None:
            self._prev = landmarks.copy()
            return landmarks.copy()

        smoothed = self._alpha * landmarks + (1.0 - self._alpha) * self._prev
        self._prev = smoothed.copy()
        return smoothed

    def reset(self) -> None:
        """Reset smoother state (e.g., after detection loss)."""
        self._prev = None
```

- [ ] **Step 4: Implement normalization**

```python
# squat_coach/preprocessing/normalization.py
"""Hip-centered landmark normalization.

Translates world landmarks so the mid-hip point is at origin.
Optionally scales by a reference bone length for size invariance.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import LEFT_HIP, RIGHT_HIP, LEFT_SHOULDER, RIGHT_SHOULDER

def normalize_to_hip_center(
    world_landmarks: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Center world landmarks at the mid-hip point.

    Args:
        world_landmarks: (33, 3) world landmarks in meters.

    Returns:
        Hip-centered landmarks, same shape.
    """
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    return world_landmarks - mid_hip

def compute_body_scale(world_landmarks: NDArray[np.float64]) -> float:
    """Compute a body scale factor from torso length (mid_shoulder to mid_hip).

    Used to normalize distance-based features across different body sizes.

    Returns:
        Torso length in meters. Returns 1.0 if landmarks are invalid.
    """
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    mid_shoulder = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    length = float(np.linalg.norm(mid_shoulder - mid_hip))
    return length if length > 0.01 else 1.0
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_smoothing.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add squat_coach/preprocessing/ squat_coach/tests/test_smoothing.py
git commit -m "feat: EMA smoothing and hip-centered normalization"
```

---

### Task 9: Sequence buffer

**Files:**
- Create: `squat_coach/preprocessing/sequence_buffer.py`
- Create: `squat_coach/tests/test_sequence_buffer.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_sequence_buffer.py
"""Tests for rolling sequence buffer."""
import numpy as np
from squat_coach.preprocessing.sequence_buffer import SequenceBuffer

def test_buffer_not_ready_initially():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    assert not buf.is_ready

def test_buffer_ready_after_filling():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    for i in range(4):
        buf.push(np.ones(3) * i)
    assert buf.is_ready

def test_buffer_get_sequence_shape():
    buf = SequenceBuffer(seq_len=4, feature_dim=3)
    for i in range(4):
        buf.push(np.ones(3) * i)
    seq = buf.get_sequence()
    assert seq.shape == (4, 3)

def test_buffer_rolling_window():
    buf = SequenceBuffer(seq_len=3, feature_dim=2)
    for i in range(5):
        buf.push(np.array([float(i), float(i)]))
    seq = buf.get_sequence()
    # Should contain last 3: [2,2], [3,3], [4,4]
    np.testing.assert_array_equal(seq[0], [2.0, 2.0])
    np.testing.assert_array_equal(seq[2], [4.0, 4.0])

def test_buffer_padded_when_partial():
    buf = SequenceBuffer(seq_len=4, feature_dim=2)
    buf.push(np.array([1.0, 2.0]))
    buf.push(np.array([3.0, 4.0]))
    seq = buf.get_sequence_padded()
    assert seq.shape == (4, 2)
    # First 2 rows should be zero-padded
    np.testing.assert_array_equal(seq[0], [0.0, 0.0])
    np.testing.assert_array_equal(seq[2], [1.0, 2.0])
    np.testing.assert_array_equal(seq[3], [3.0, 4.0])

def test_buffer_reset():
    buf = SequenceBuffer(seq_len=3, feature_dim=2)
    buf.push(np.array([1.0, 2.0]))
    buf.reset()
    assert not buf.is_ready
    assert len(buf) == 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_sequence_buffer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement sequence buffer**

```python
# squat_coach/preprocessing/sequence_buffer.py
"""Rolling sequence buffer for temporal model input.

Maintains a fixed-length sliding window of feature vectors.
Supports both ready (full) and padded (partial) extraction.
"""
from collections import deque
import numpy as np
from numpy.typing import NDArray

class SequenceBuffer:
    """Rolling window buffer for fixed-length sequence extraction."""

    def __init__(self, seq_len: int = 60, feature_dim: int = 42) -> None:
        """
        Args:
            seq_len: Number of frames in the window.
            feature_dim: Dimensionality of each feature vector.
        """
        self._seq_len = seq_len
        self._feature_dim = feature_dim
        self._buffer: deque[NDArray[np.float64]] = deque(maxlen=seq_len)

    def push(self, features: NDArray[np.float64]) -> None:
        """Add a feature vector to the buffer."""
        self._buffer.append(features.copy())

    @property
    def is_ready(self) -> bool:
        """True when buffer has seq_len frames."""
        return len(self._buffer) >= self._seq_len

    def get_sequence(self) -> NDArray[np.float64]:
        """Get the current window as (seq_len, feature_dim). Requires is_ready."""
        if not self.is_ready:
            raise ValueError("Buffer not full. Use get_sequence_padded() instead.")
        return np.array(list(self._buffer), dtype=np.float64)

    def get_sequence_padded(self) -> NDArray[np.float64]:
        """Get the window zero-padded at the front if not full."""
        current = list(self._buffer)
        pad_count = self._seq_len - len(current)
        if pad_count > 0:
            padding = [np.zeros(self._feature_dim) for _ in range(pad_count)]
            current = padding + current
        return np.array(current, dtype=np.float64)

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_sequence_buffer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/preprocessing/sequence_buffer.py squat_coach/tests/test_sequence_buffer.py
git commit -m "feat: rolling sequence buffer with padding support"
```

---

### Task 10: Calibration

**Files:**
- Create: `squat_coach/preprocessing/calibration.py`
- Create: `squat_coach/tests/test_calibration.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_calibration.py
"""Tests for calibration flow."""
import numpy as np
from squat_coach.preprocessing.calibration import CalibrationResult, Calibrator
from squat_coach.utils.enums import ViewType
from squat_coach.pose.base import PoseResult

def _make_side_view_pose(timestamp: float = 0.0) -> PoseResult:
    """Create a synthetic side-view standing pose.

    Side view: shoulders have similar X but different Z.
    """
    landmarks = np.zeros((33, 3))
    # Shoulders: same X, spread in Z (side view characteristic)
    landmarks[11] = [0.0, -0.4, -0.1]   # left shoulder
    landmarks[12] = [0.0, -0.4,  0.1]   # right shoulder
    # Hips
    landmarks[23] = [0.0, 0.0, -0.1]    # left hip
    landmarks[24] = [0.0, 0.0,  0.1]    # right hip
    # Head
    landmarks[0] = [0.0, -0.6, 0.0]     # nose
    landmarks[7] = [-0.05, -0.55, 0.0]  # left ear
    landmarks[8] = [0.05, -0.55, 0.0]   # right ear

    return PoseResult(
        timestamp=timestamp,
        world_landmarks=landmarks,
        image_landmarks=landmarks.copy(),
        visibility=np.ones(33) * 0.9,
        pose_confidence=0.9,
        detected=True,
    )

def _make_front_view_pose(timestamp: float = 0.0) -> PoseResult:
    """Front view: shoulders spread in X, similar Z."""
    landmarks = np.zeros((33, 3))
    landmarks[11] = [-0.2, -0.4, 0.0]   # left shoulder
    landmarks[12] = [ 0.2, -0.4, 0.0]   # right shoulder
    landmarks[23] = [-0.1, 0.0, 0.0]
    landmarks[24] = [ 0.1, 0.0, 0.0]
    landmarks[0]  = [0.0, -0.6, 0.0]
    landmarks[7]  = [-0.1, -0.55, 0.0]
    landmarks[8]  = [ 0.1, -0.55, 0.0]

    return PoseResult(
        timestamp=timestamp,
        world_landmarks=landmarks,
        image_landmarks=landmarks.copy(),
        visibility=np.ones(33) * 0.9,
        pose_confidence=0.9,
        detected=True,
    )

def test_calibrator_detects_side_view():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_side_view_pose(float(i)))
    result = cal.compute()
    assert result is not None
    assert result.view_type == ViewType.SIDE

def test_calibrator_detects_front_view():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_front_view_pose(float(i)))
    result = cal.compute()
    assert result is not None
    assert result.view_type == ViewType.FRONT

def test_calibrator_stores_baseline():
    cal = Calibrator(num_frames=3)
    for i in range(3):
        cal.add_frame(_make_side_view_pose(float(i)))
    result = cal.compute()
    assert result.baseline_torso_angle is not None
    assert result.body_scale > 0.0

def test_calibrator_not_ready_without_enough_frames():
    cal = Calibrator(num_frames=5)
    cal.add_frame(_make_side_view_pose(0.0))
    result = cal.compute()
    assert result is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_calibration.py -v`
Expected: FAIL

- [ ] **Step 3: Implement calibration**

```python
# squat_coach/preprocessing/calibration.py
"""Calibration flow: detect view type, establish baseline posture.

At session start, the user stands upright for ~2 seconds. The calibrator
collects frames, detects whether the camera shows a side or front view,
and computes baseline measurements for scoring personalization.
"""
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from squat_coach.pose.base import PoseResult
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE, LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.enums import ViewType
from squat_coach.utils.math_utils import angle_between_vectors

logger = logging.getLogger("squat_coach.calibration")


@dataclass
class CalibrationResult:
    """Outputs from the calibration phase."""
    view_type: ViewType
    baseline_torso_angle: float         # degrees from vertical
    baseline_head_offset: float         # meters
    body_scale: float                   # torso length in meters
    dominant_side: str                  # "left" or "right"
    baseline_landmarks: NDArray[np.float64]  # averaged standing pose (33, 3)


class Calibrator:
    """Collects standing frames and computes calibration baseline."""

    def __init__(self, num_frames: int = 60) -> None:
        """
        Args:
            num_frames: Number of frames to collect before computing.
                        At 30fps, 60 frames = 2 seconds.
        """
        self._num_frames = num_frames
        self._frames: list[NDArray[np.float64]] = []

    def add_frame(self, pose: PoseResult) -> None:
        """Add a detected pose frame to the calibration buffer."""
        if pose.detected and pose.world_landmarks is not None:
            self._frames.append(pose.world_landmarks.copy())

    @property
    def is_ready(self) -> bool:
        return len(self._frames) >= self._num_frames

    def compute(self) -> Optional[CalibrationResult]:
        """Compute calibration from collected frames. Returns None if not enough data."""
        if len(self._frames) < self._num_frames:
            return None

        # Average landmarks across frames for stability
        avg_landmarks = np.mean(self._frames, axis=0)

        view_type = self._detect_view(avg_landmarks)
        torso_angle = self._compute_torso_angle(avg_landmarks)
        head_offset = self._compute_head_offset(avg_landmarks)
        body_scale = self._compute_body_scale(avg_landmarks)
        dominant_side = self._detect_dominant_side(avg_landmarks, view_type)

        logger.info(
            "Calibration complete: view=%s, torso_angle=%.1f°, scale=%.3fm",
            view_type.value, torso_angle, body_scale,
        )

        return CalibrationResult(
            view_type=view_type,
            baseline_torso_angle=torso_angle,
            baseline_head_offset=head_offset,
            body_scale=body_scale,
            dominant_side=dominant_side,
            baseline_landmarks=avg_landmarks,
        )

    def _detect_view(self, landmarks: NDArray[np.float64]) -> ViewType:
        """Detect side vs front view from shoulder geometry.

        Side view: shoulders appear close in X (image horizontal), spread in Z (depth).
        Front view: shoulders spread in X, similar Z.
        """
        l_sh = landmarks[LEFT_SHOULDER]
        r_sh = landmarks[RIGHT_SHOULDER]
        dx = abs(l_sh[0] - r_sh[0])  # horizontal spread
        dz = abs(l_sh[2] - r_sh[2])  # depth spread

        # If shoulders are wider in X than Z, it's a front view
        if dx > dz * 1.5:
            return ViewType.FRONT
        return ViewType.SIDE

    def _compute_torso_angle(self, landmarks: NDArray[np.float64]) -> float:
        """Angle of mid_shoulder→mid_hip vector vs vertical (Y-down)."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        mid_hip = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
        trunk_vec = mid_hip - mid_sh
        vertical = np.array([0.0, 1.0, 0.0])  # Y-down in world coords
        return angle_between_vectors(trunk_vec, vertical)

    def _compute_head_offset(self, landmarks: NDArray[np.float64]) -> float:
        """Forward offset of nose relative to mid-shoulder."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        nose = landmarks[NOSE]
        # In side view, forward is primarily along X or Z axis
        return float(np.linalg.norm(nose[:2] - mid_sh[:2]))

    def _compute_body_scale(self, landmarks: NDArray[np.float64]) -> float:
        """Torso length: mid_shoulder to mid_hip distance."""
        mid_sh = (landmarks[LEFT_SHOULDER] + landmarks[RIGHT_SHOULDER]) / 2.0
        mid_hip = (landmarks[LEFT_HIP] + landmarks[RIGHT_HIP]) / 2.0
        length = float(np.linalg.norm(mid_sh - mid_hip))
        return max(length, 0.01)

    def _detect_dominant_side(self, landmarks: NDArray[np.float64], view: ViewType) -> str:
        """Detect which side is closer to camera (higher visibility expected)."""
        if view == ViewType.FRONT:
            return "both"
        # In side view, the side with higher Z value is closer to camera
        left_z = landmarks[LEFT_SHOULDER][2]
        right_z = landmarks[RIGHT_SHOULDER][2]
        return "left" if left_z > right_z else "right"
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_calibration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/preprocessing/calibration.py squat_coach/tests/test_calibration.py
git commit -m "feat: calibration with view detection and baseline measurement"
```

---

## Chunk 2: Biomechanics Feature Extraction

### Task 11: Angle computation

**Files:**
- Create: `squat_coach/biomechanics/__init__.py`
- Create: `squat_coach/biomechanics/angles.py`
- Create: `squat_coach/tests/test_angles.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_angles.py
"""Tests for biomechanics angle computation."""
import numpy as np
from squat_coach.biomechanics.angles import compute_joint_angles
from squat_coach.pose.landmarks import LEFT_HIP, LEFT_KNEE, LEFT_ANKLE

def test_compute_knee_angle_straight_leg():
    """Straight leg should give ~180 degrees."""
    landmarks = np.zeros((33, 3))
    landmarks[LEFT_HIP] = [0, -1, 0]
    landmarks[LEFT_KNEE] = [0, 0, 0]
    landmarks[LEFT_ANKLE] = [0, 1, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_knee_angle"] - 180.0) < 1.0

def test_compute_knee_angle_bent():
    """90-degree knee bend."""
    landmarks = np.zeros((33, 3))
    landmarks[LEFT_HIP] = [0, -1, 0]
    landmarks[LEFT_KNEE] = [0, 0, 0]
    landmarks[LEFT_ANKLE] = [1, 0, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_knee_angle"] - 90.0) < 1.0

def test_compute_hip_angle():
    landmarks = np.zeros((33, 3))
    # shoulder above hip, knee below hip at 90 deg
    landmarks[11] = [0, -1, 0]  # left shoulder
    landmarks[LEFT_HIP] = [0, 0, 0]
    landmarks[LEFT_KNEE] = [1, 0, 0]
    angles = compute_joint_angles(landmarks)
    assert abs(angles["left_hip_angle"] - 90.0) < 1.0

def test_torso_inclination_upright():
    landmarks = np.zeros((33, 3))
    landmarks[11] = [0, -0.4, 0]   # left shoulder
    landmarks[12] = [0, -0.4, 0]   # right shoulder
    landmarks[23] = [0, 0, 0]      # left hip
    landmarks[24] = [0, 0, 0]      # right hip
    angles = compute_joint_angles(landmarks)
    # Upright trunk: ~0 degrees from vertical (but our vertical is Y-down
    # so trunk vector points down = aligned with vertical)
    assert angles["torso_inclination_deg"] < 5.0
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_angles.py -v`

- [ ] **Step 3: Implement angles**

```python
# squat_coach/biomechanics/angles.py
"""Joint angle computation from 3D world landmarks.

All angles are computed in 3D using the full (x, y, z) coordinates
of BlazePose world landmarks. Angles are returned in degrees.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE,
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, NOSE,
)
from squat_coach.utils.math_utils import angle_at_joint, angle_between_vectors


def compute_joint_angles(world_landmarks: NDArray[np.float64]) -> dict[str, float]:
    """Compute all biomechanics joint angles from world landmarks.

    Args:
        world_landmarks: (33, 3) BlazePose world landmarks.

    Returns:
        Dictionary of named angles in degrees.
    """
    lm = world_landmarks

    # Knee angles: hip-knee-ankle
    left_knee_angle = angle_at_joint(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE])
    right_knee_angle = angle_at_joint(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE])

    # Hip angles: shoulder-hip-knee
    left_hip_angle = angle_at_joint(lm[LEFT_SHOULDER], lm[LEFT_HIP], lm[LEFT_KNEE])
    right_hip_angle = angle_at_joint(lm[RIGHT_SHOULDER], lm[RIGHT_HIP], lm[RIGHT_KNEE])

    # Ankle angle proxy: knee-ankle-foot_index
    ankle_angle = angle_at_joint(lm[LEFT_KNEE], lm[LEFT_ANKLE], lm[LEFT_FOOT_INDEX])

    # Torso inclination: angle of trunk vector vs vertical
    mid_shoulder = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    trunk_vec = mid_hip - mid_shoulder  # points downward when upright
    vertical = np.array([0.0, 1.0, 0.0])  # Y-down in BlazePose world coords
    torso_inclination = angle_between_vectors(trunk_vec, vertical)

    # Shoulder-hip line angle (same as torso but using the line, not vector direction)
    shoulder_hip_line_angle = torso_inclination

    return {
        "left_knee_angle": left_knee_angle,
        "right_knee_angle": right_knee_angle,
        "left_hip_angle": left_hip_angle,
        "right_hip_angle": right_hip_angle,
        "ankle_angle_proxy": ankle_angle,
        "torso_inclination_deg": torso_inclination,
        "shoulder_hip_line_angle": shoulder_hip_line_angle,
    }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_angles.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/biomechanics/ squat_coach/tests/test_angles.py
git commit -m "feat: joint angle computation from 3D landmarks"
```

---

### Task 12: Vectors, distances, kinematics

**Files:**
- Create: `squat_coach/biomechanics/vectors.py`
- Create: `squat_coach/biomechanics/distances.py`
- Create: `squat_coach/biomechanics/kinematics.py`

- [ ] **Step 1: Implement vectors**

```python
# squat_coach/biomechanics/vectors.py
"""Bone and trunk vector utilities."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE,
    LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.math_utils import normalize_vector


def compute_trunk_vector(world_landmarks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mid-shoulder to mid-hip unit vector."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    return normalize_vector(mid_hip - mid_sh)


def compute_head_to_trunk_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Perpendicular distance from nose to the shoulder-hip trunk line."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    nose = world_landmarks[NOSE]
    from squat_coach.utils.math_utils import perpendicular_distance_to_line
    return perpendicular_distance_to_line(nose, mid_sh, mid_hip)


def compute_shoulder_hip_deltas(
    world_landmarks: NDArray[np.float64],
) -> tuple[float, float]:
    """Horizontal and vertical deltas between mid-shoulder and mid-hip.

    Returns:
        (horizontal_delta, vertical_delta) in meters.
    """
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_hip = (world_landmarks[LEFT_HIP] + world_landmarks[RIGHT_HIP]) / 2.0
    diff = mid_sh - mid_hip
    h_delta = float(np.sqrt(diff[0] ** 2 + diff[2] ** 2))  # XZ plane
    v_delta = float(abs(diff[1]))  # Y axis
    return h_delta, v_delta


def compute_nose_to_shoulder_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Horizontal offset of nose from mid-shoulder."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    nose = world_landmarks[NOSE]
    return float(np.sqrt((nose[0] - mid_sh[0]) ** 2 + (nose[2] - mid_sh[2]) ** 2))


def compute_neck_forward_offset(world_landmarks: NDArray[np.float64]) -> float:
    """Forward offset of ear midpoint from shoulder midpoint (sagittal drift)."""
    mid_sh = (world_landmarks[LEFT_SHOULDER] + world_landmarks[RIGHT_SHOULDER]) / 2.0
    mid_ear = (world_landmarks[LEFT_EAR] + world_landmarks[RIGHT_EAR]) / 2.0
    return float(np.sqrt((mid_ear[0] - mid_sh[0]) ** 2 + (mid_ear[2] - mid_sh[2]) ** 2))
```

- [ ] **Step 2: Implement distances**

```python
# squat_coach/biomechanics/distances.py
"""Pairwise joint distance computation."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_SHOULDER, RIGHT_SHOULDER, NOSE,
)


def compute_hip_depth_ratios(
    world_landmarks: NDArray[np.float64],
) -> tuple[float, float]:
    """Hip depth relative to knee and ankle (Y-axis).

    Negative values mean hip is above the reference point.
    Positive values mean hip is below (deeper squat).

    Returns:
        (hip_depth_vs_knee, hip_depth_vs_ankle) as normalized ratios.
    """
    mid_hip_y = (world_landmarks[LEFT_HIP][1] + world_landmarks[RIGHT_HIP][1]) / 2.0
    mid_knee_y = (world_landmarks[LEFT_KNEE][1] + world_landmarks[RIGHT_KNEE][1]) / 2.0
    mid_ankle_y = (world_landmarks[LEFT_ANKLE][1] + world_landmarks[RIGHT_ANKLE][1]) / 2.0

    # Normalize by leg length for body-size invariance
    leg_len = abs(mid_hip_y - mid_ankle_y)
    if leg_len < 0.01:
        return 0.0, 0.0

    hip_vs_knee = (mid_hip_y - mid_knee_y) / leg_len
    hip_vs_ankle = (mid_hip_y - mid_ankle_y) / leg_len
    return float(hip_vs_knee), float(hip_vs_ankle)


def compute_pairwise_distance_subset(
    world_landmarks: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the 8 pairwise distances used in the D=42 feature vector.

    Pairs (from spec):
        L/R hip-knee, L/R knee-ankle, L/R shoulder-hip,
        hip_mid-shoulder_mid, nose-shoulder_mid
    """
    lm = world_landmarks
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0

    pairs = [
        (lm[LEFT_HIP], lm[LEFT_KNEE]),
        (lm[RIGHT_HIP], lm[RIGHT_KNEE]),
        (lm[LEFT_KNEE], lm[LEFT_ANKLE]),
        (lm[RIGHT_KNEE], lm[RIGHT_ANKLE]),
        (lm[LEFT_SHOULDER], lm[LEFT_HIP]),
        (lm[RIGHT_SHOULDER], lm[RIGHT_HIP]),
        (mid_hip, mid_sh),
        (lm[NOSE], mid_sh),
    ]

    distances = np.array(
        [float(np.linalg.norm(a - b)) for a, b in pairs], dtype=np.float64
    )
    return distances
```

- [ ] **Step 3: Implement kinematics**

```python
# squat_coach/biomechanics/kinematics.py
"""Velocity and acceleration computation via finite differences.

Computes first and second derivatives of key features across frames.
Requires at least 2 frames for velocity, 3 for acceleration.
"""
from typing import Optional
import numpy as np


class KinematicsTracker:
    """Track velocities and accelerations of scalar features over frames."""

    def __init__(self, fps: float = 30.0) -> None:
        self._dt = 1.0 / fps
        self._prev_values: Optional[dict[str, float]] = None
        self._prev_velocities: Optional[dict[str, float]] = None

    def update(self, values: dict[str, float]) -> dict[str, float]:
        """Compute velocities and accelerations from current values.

        Args:
            values: Dict with keys like 'hip_y', 'trunk_angle', 'knee_angle', 'hip_angle'.

        Returns:
            Dict with velocity and acceleration for each tracked value.
        """
        result: dict[str, float] = {}
        velocities: dict[str, float] = {}

        for key, val in values.items():
            vel_key = f"{key}_velocity"
            accel_key = f"{key}_acceleration"

            # Velocity = finite difference
            if self._prev_values is not None and key in self._prev_values:
                vel = (val - self._prev_values[key]) / self._dt
            else:
                vel = 0.0
            velocities[key] = vel
            result[vel_key] = vel

            # Acceleration = finite difference of velocity
            if self._prev_velocities is not None and key in self._prev_velocities:
                accel = (vel - self._prev_velocities[key]) / self._dt
            else:
                accel = 0.0
            result[accel_key] = accel

        self._prev_values = dict(values)
        self._prev_velocities = dict(velocities)
        return result

    def reset(self) -> None:
        self._prev_values = None
        self._prev_velocities = None
```

- [ ] **Step 4: Commit**

```bash
git add squat_coach/biomechanics/vectors.py squat_coach/biomechanics/distances.py squat_coach/biomechanics/kinematics.py
git commit -m "feat: vectors, pairwise distances, and kinematics tracker"
```

---

### Task 13: Posture analysis and view-specific features

**Files:**
- Create: `squat_coach/biomechanics/posture_analysis.py`
- Create: `squat_coach/biomechanics/side_view_constraints.py`
- Create: `squat_coach/biomechanics/front_view_constraints.py`

- [ ] **Step 1: Implement posture analysis (rounded_back_risk)**

```python
# squat_coach/biomechanics/posture_analysis.py
"""Rounded back risk and posture proxy analysis.

IMPORTANT LIMITATIONS:
- This estimates rounded back RISK from surface landmarks only.
- It CANNOT detect actual vertebral flexion or spinal curvature.
- It measures visible postural indicators correlated with back rounding.
- Should NOT be used for medical assessment.
- Works best from true lateral (side) view; degrades at oblique angles.
"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE, LEFT_EAR, RIGHT_EAR,
)
from squat_coach.utils.math_utils import angle_between_vectors, perpendicular_distance_to_line


@dataclass
class RoundedBackAssessment:
    """Result of rounded back risk estimation."""
    risk_score: float               # 0.0-1.0
    confidence: float               # 0.0-1.0
    trunk_curl_component: float     # 0.0-1.0
    head_drift_component: float     # 0.0-1.0
    spine_linearity_component: float  # 0.0-1.0
    rationale: str
    limitations: str = "Estimated from surface landmarks only. Not a medical assessment."


def compute_rounded_back_risk(
    world_landmarks: NDArray[np.float64],
    baseline_torso_angle: float,
    body_scale: float,
) -> RoundedBackAssessment:
    """Compute rounded back risk from side-view landmarks.

    Three components:
    1. Trunk curl proxy: trunk angle deviation from calibrated baseline.
    2. Head/neck drift: forward displacement of head relative to shoulders.
    3. Spine linearity: how much the mid-torso deviates from a straight
       shoulder-hip line (proxy for thoracic rounding).

    Args:
        world_landmarks: (33, 3) world landmarks.
        baseline_torso_angle: Calibrated upright trunk angle in degrees.
        body_scale: Torso length in meters (from calibration).
    """
    lm = world_landmarks
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0

    # Component 1: Trunk curl — deviation of current trunk angle from baseline
    trunk_vec = mid_hip - mid_sh
    vertical = np.array([0.0, 1.0, 0.0])
    current_angle = angle_between_vectors(trunk_vec, vertical)
    angle_deviation = max(0.0, current_angle - baseline_torso_angle)
    # Normalize: 30+ degrees deviation = risk 1.0
    trunk_curl = min(angle_deviation / 30.0, 1.0)

    # Component 2: Head drift — forward displacement of nose relative to shoulders
    nose = lm[NOSE]
    head_forward_dist = perpendicular_distance_to_line(nose, mid_sh, mid_hip)
    # Normalize by body scale: >0.15 * body_scale = high risk
    head_drift = min(head_forward_dist / (0.15 * body_scale), 1.0)

    # Component 3: Spine linearity — deviation of torso midpoint from straight line
    # Use the point midway between shoulders and hips as a proxy for mid-spine
    mid_torso = (mid_sh + mid_hip) / 2.0
    # In a straight spine, this point lies ON the shoulder-hip line
    # Deviation indicates curvature
    spine_dev = perpendicular_distance_to_line(mid_torso, mid_sh, mid_hip)
    spine_linearity = min(spine_dev / (0.05 * body_scale), 1.0)

    # Weighted combination
    risk_score = 0.4 * trunk_curl + 0.35 * head_drift + 0.25 * spine_linearity
    risk_score = min(max(risk_score, 0.0), 1.0)

    # Confidence based on landmark visibility and angle deviation magnitude
    confidence = min(0.5 + angle_deviation / 20.0, 1.0)

    rationale_parts = []
    if trunk_curl > 0.3:
        rationale_parts.append(f"trunk angle {current_angle:.0f}° (baseline {baseline_torso_angle:.0f}°)")
    if head_drift > 0.3:
        rationale_parts.append(f"head drifted {head_forward_dist * 100:.0f}cm forward")
    if spine_linearity > 0.3:
        rationale_parts.append(f"mid-spine deviation detected")
    rationale = "; ".join(rationale_parts) if rationale_parts else "Within normal range"

    return RoundedBackAssessment(
        risk_score=risk_score,
        confidence=confidence,
        trunk_curl_component=trunk_curl,
        head_drift_component=head_drift,
        spine_linearity_component=spine_linearity,
        rationale=rationale,
    )
```

- [ ] **Step 2: Implement side-view and front-view features**

```python
# squat_coach/biomechanics/side_view_constraints.py
"""Side-view specific feature extraction."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX,
)
from squat_coach.utils.math_utils import angle_between_vectors, angle_at_joint


def compute_side_view_features(
    world_landmarks: NDArray[np.float64],
    trunk_stability_window: list[float],
) -> dict[str, float]:
    """Compute features specific to side-view squat analysis.

    Args:
        world_landmarks: (33, 3) world landmarks.
        trunk_stability_window: Recent torso_inclination values for variance.

    Returns:
        Dict of side-view specific features.
    """
    lm = world_landmarks
    mid_sh = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0

    # Forward lean angle: trunk forward tilt in sagittal plane
    trunk_vec = mid_hip - mid_sh
    vertical = np.array([0.0, 1.0, 0.0])
    forward_lean = angle_between_vectors(trunk_vec, vertical)

    # Trunk stability: variance of torso angle over recent window
    trunk_stability = float(np.var(trunk_stability_window)) if len(trunk_stability_window) > 1 else 0.0

    # Ankle/shin angle: dorsiflexion proxy
    ankle_shin_angle = angle_at_joint(lm[LEFT_KNEE], lm[LEFT_ANKLE], lm[LEFT_FOOT_INDEX])

    return {
        "forward_lean_angle": forward_lean,
        "trunk_stability": trunk_stability,
        "ankle_shin_angle": ankle_shin_angle,
    }
```

```python
# squat_coach/biomechanics/front_view_constraints.py
"""Front-view specific feature extraction."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import (
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
)


def compute_front_view_features(
    world_landmarks: NDArray[np.float64],
) -> dict[str, float]:
    """Compute features specific to front-view squat analysis.

    Args:
        world_landmarks: (33, 3) world landmarks.

    Returns:
        Dict of front-view specific features.
    """
    lm = world_landmarks

    # Knee valgus: inward collapse angle
    # Measure how much knees move inward relative to hips and ankles
    mid_hip_x = (lm[LEFT_HIP][0] + lm[RIGHT_HIP][0]) / 2.0
    l_knee_inward = max(0.0, lm[LEFT_KNEE][0] - lm[LEFT_HIP][0])   # positive = inward
    r_knee_inward = max(0.0, lm[RIGHT_HIP][0] - lm[RIGHT_KNEE][0])  # positive = inward
    knee_valgus_angle = float(np.degrees(np.arctan2(
        (l_knee_inward + r_knee_inward) / 2.0,
        abs(lm[LEFT_HIP][1] - lm[LEFT_KNEE][1]) + 1e-6
    )))

    # Stance width ratio: foot spread / hip width
    hip_width = abs(lm[LEFT_HIP][0] - lm[RIGHT_HIP][0])
    foot_width = abs(lm[LEFT_FOOT_INDEX][0] - lm[RIGHT_FOOT_INDEX][0])
    stance_width_ratio = foot_width / max(hip_width, 0.01)

    # Left-right symmetry: compare left vs right knee angles (proxy)
    l_knee_y = lm[LEFT_KNEE][1]
    r_knee_y = lm[RIGHT_KNEE][1]
    symmetry = 1.0 - min(abs(l_knee_y - r_knee_y) / 0.1, 1.0)

    # Hip lateral shift
    mid_hip = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2.0
    mid_ankle = (lm[LEFT_ANKLE] + lm[RIGHT_ANKLE]) / 2.0
    hip_shift = abs(mid_hip[0] - mid_ankle[0])

    return {
        "knee_valgus_angle": knee_valgus_angle,
        "stance_width_ratio": stance_width_ratio,
        "left_right_symmetry": symmetry,
        "hip_shift_lateral": hip_shift,
    }
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/biomechanics/posture_analysis.py squat_coach/biomechanics/side_view_constraints.py squat_coach/biomechanics/front_view_constraints.py
git commit -m "feat: posture analysis, side-view and front-view features"
```

---

### Task 14: Full squat feature orchestrator

**Files:**
- Create: `squat_coach/biomechanics/squat_features.py`
- Create: `squat_coach/tests/test_features.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_features.py
"""Tests for the full squat feature extraction pipeline."""
import numpy as np
from squat_coach.biomechanics.squat_features import SquatFeatureExtractor
from squat_coach.preprocessing.calibration import CalibrationResult
from squat_coach.utils.enums import ViewType

def _make_standing_landmarks() -> np.ndarray:
    """Create realistic standing pose landmarks."""
    lm = np.zeros((33, 3))
    lm[11] = [-0.15, -0.4, 0.0]  # left shoulder
    lm[12] = [ 0.15, -0.4, 0.0]  # right shoulder
    lm[23] = [-0.1,   0.0, 0.0]  # left hip
    lm[24] = [ 0.1,   0.0, 0.0]  # right hip
    lm[25] = [-0.1,   0.4, 0.0]  # left knee
    lm[26] = [ 0.1,   0.4, 0.0]  # right knee
    lm[27] = [-0.1,   0.8, 0.0]  # left ankle
    lm[28] = [ 0.1,   0.8, 0.0]  # right ankle
    lm[31] = [-0.1,   0.85, 0.1] # left foot
    lm[32] = [ 0.1,   0.85, 0.1] # right foot
    lm[0]  = [ 0.0,  -0.6, 0.0]  # nose
    lm[7]  = [-0.05, -0.55, 0.0] # left ear
    lm[8]  = [ 0.05, -0.55, 0.0] # right ear
    return lm

def _make_calibration() -> CalibrationResult:
    return CalibrationResult(
        view_type=ViewType.SIDE,
        baseline_torso_angle=5.0,
        baseline_head_offset=0.02,
        body_scale=0.4,
        dominant_side="left",
        baseline_landmarks=_make_standing_landmarks(),
    )

def test_extract_returns_model_features():
    """Model feature vector should be D=42."""
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    features = extractor.extract(_make_standing_landmarks(), np.ones(33) * 0.9)
    assert "model_features" in features
    assert len(features["model_features"]) == 42

def test_extract_returns_all_named_features():
    """Should include all named biomechanics features."""
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    features = extractor.extract(_make_standing_landmarks(), np.ones(33) * 0.9)
    required = ["left_knee_angle", "torso_inclination_deg", "rounded_back_risk"]
    for key in required:
        assert key in features, f"Missing feature: {key}"

def test_extract_visibility_features():
    extractor = SquatFeatureExtractor(_make_calibration(), fps=30.0)
    vis = np.ones(33) * 0.8
    features = extractor.extract(_make_standing_landmarks(), vis)
    assert abs(features["landmark_visibility_mean"] - 0.8) < 0.01
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_features.py -v`

- [ ] **Step 3: Implement squat feature orchestrator**

```python
# squat_coach/biomechanics/squat_features.py
"""Full squat feature extraction orchestrator.

Computes all biomechanics features from a single frame of world landmarks.
Produces both the D=42 model input vector and extended features for
rule-based scoring, fault detection, and terminal logging.
"""
import numpy as np
from numpy.typing import NDArray
from collections import deque

from squat_coach.preprocessing.calibration import CalibrationResult
from squat_coach.biomechanics.angles import compute_joint_angles
from squat_coach.biomechanics.vectors import (
    compute_head_to_trunk_offset,
    compute_shoulder_hip_deltas,
    compute_nose_to_shoulder_offset,
    compute_neck_forward_offset,
)
from squat_coach.biomechanics.distances import (
    compute_hip_depth_ratios,
    compute_pairwise_distance_subset,
)
from squat_coach.biomechanics.kinematics import KinematicsTracker
from squat_coach.biomechanics.posture_analysis import compute_rounded_back_risk
from squat_coach.biomechanics.side_view_constraints import compute_side_view_features
from squat_coach.biomechanics.front_view_constraints import compute_front_view_features
from squat_coach.pose.landmarks import LOWER_BODY_LANDMARKS, TORSO_LANDMARKS
from squat_coach.utils.enums import ViewType


class SquatFeatureExtractor:
    """Extracts full biomechanics feature set from world landmarks.

    Produces:
        - 'model_features': NDArray of shape (42,) for temporal model input
        - Named features dict for scoring, faults, logging
    """

    # Rolling window for trunk stability computation
    _TRUNK_WINDOW_SIZE = 30

    def __init__(self, calibration: CalibrationResult, fps: float = 30.0) -> None:
        self._cal = calibration
        self._kinematics = KinematicsTracker(fps=fps)
        self._trunk_angle_window: deque[float] = deque(maxlen=self._TRUNK_WINDOW_SIZE)

    def extract(
        self,
        world_landmarks: NDArray[np.float64],
        visibility: NDArray[np.float64],
    ) -> dict:
        """Extract all features from a single frame.

        Args:
            world_landmarks: (33, 3) world landmarks.
            visibility: (33,) per-landmark visibility.

        Returns:
            Dict containing:
                'model_features': np.ndarray of shape (42,)
                Plus all named scalar features.
        """
        # A. Core joint angles
        angles = compute_joint_angles(world_landmarks)

        # Primary angles (dominant side from calibration)
        if self._cal.dominant_side == "left" or self._cal.dominant_side == "both":
            primary_knee = angles["left_knee_angle"]
            primary_hip = angles["left_hip_angle"]
        else:
            primary_knee = angles["right_knee_angle"]
            primary_hip = angles["right_hip_angle"]

        # B. Trunk and head offsets
        head_offset = compute_head_to_trunk_offset(world_landmarks)
        sh_h_delta, sh_v_delta = compute_shoulder_hip_deltas(world_landmarks)
        nose_offset = compute_nose_to_shoulder_offset(world_landmarks)
        neck_offset = compute_neck_forward_offset(world_landmarks)

        # C. Hip depth
        hip_vs_knee, hip_vs_ankle = compute_hip_depth_ratios(world_landmarks)

        # D. Trunk stability window
        self._trunk_angle_window.append(angles["torso_inclination_deg"])

        # E. View-specific features (indices 16-19 in model vector)
        if self._cal.view_type == ViewType.SIDE:
            view_feats = compute_side_view_features(
                world_landmarks, list(self._trunk_angle_window)
            )
            back_risk = compute_rounded_back_risk(
                world_landmarks, self._cal.baseline_torso_angle, self._cal.body_scale
            )
            feat_16 = view_feats["forward_lean_angle"]
            feat_17 = back_risk.risk_score
            feat_18 = view_feats["trunk_stability"]
            feat_19 = view_feats["ankle_shin_angle"]
        else:
            view_feats = compute_front_view_features(world_landmarks)
            back_risk = None
            feat_16 = view_feats["knee_valgus_angle"]
            feat_17 = view_feats["stance_width_ratio"]
            feat_18 = view_feats["left_right_symmetry"]
            feat_19 = view_feats["hip_shift_lateral"]

        # F. Kinematics
        mid_hip_y = float((world_landmarks[23][1] + world_landmarks[24][1]) / 2.0)
        kin_input = {
            "hip_vertical": mid_hip_y,
            "trunk_angle": angles["torso_inclination_deg"],
            "knee_angle": primary_knee,
            "hip_angle": primary_hip,
        }
        kin = self._kinematics.update(kin_input)

        # G. Visibility / confidence features
        vis_mean = float(np.mean(visibility))
        lower_vis = float(np.mean(visibility[LOWER_BODY_LANDMARKS]))
        torso_vis = float(np.mean(visibility[TORSO_LANDMARKS]))
        reliability = vis_mean * min(lower_vis, torso_vis)
        view_validity = 1.0 if self._cal.view_type != ViewType.UNKNOWN else 0.0
        occlusion_risk = 1.0 - min(lower_vis, torso_vis)

        # H. Pairwise distances
        pw_dists = compute_pairwise_distance_subset(world_landmarks)

        # Build D=42 model feature vector
        model_features = np.array([
            # 0-5: angles
            angles["left_knee_angle"], angles["right_knee_angle"], primary_knee,
            angles["left_hip_angle"], angles["right_hip_angle"], primary_hip,
            # 6: ankle
            angles["ankle_angle_proxy"],
            # 7-8: torso
            angles["torso_inclination_deg"], angles["shoulder_hip_line_angle"],
            # 9-11: offsets
            head_offset, sh_h_delta, sh_v_delta,
            # 12-13: hip depth
            hip_vs_knee, hip_vs_ankle,
            # 14-15: nose/neck
            nose_offset, neck_offset,
            # 16-19: view-dependent
            feat_16, feat_17, feat_18, feat_19,
            # 20-27: kinematics
            kin.get("hip_vertical_velocity", 0.0),
            kin.get("hip_vertical_acceleration", 0.0),
            kin.get("trunk_angle_velocity", 0.0),
            kin.get("trunk_angle_acceleration", 0.0),
            kin.get("knee_angle_velocity", 0.0),
            kin.get("knee_angle_acceleration", 0.0),
            kin.get("hip_angle_velocity", 0.0),
            kin.get("hip_angle_acceleration", 0.0),
            # 28-33: quality
            vis_mean, lower_vis, torso_vis, reliability, view_validity, occlusion_risk,
            # 34-41: pairwise distances
            *pw_dists,
        ], dtype=np.float64)

        # Build full named features dict
        features: dict = {
            **angles,
            "primary_knee_angle": primary_knee,
            "primary_hip_angle": primary_hip,
            "head_to_trunk_offset": head_offset,
            "shoulder_to_hip_h_delta": sh_h_delta,
            "shoulder_to_hip_v_delta": sh_v_delta,
            "hip_depth_vs_knee": hip_vs_knee,
            "hip_depth_vs_ankle": hip_vs_ankle,
            "nose_to_shoulder_offset": nose_offset,
            "neck_forward_offset": neck_offset,
            **view_feats,
            "rounded_back_risk": back_risk.risk_score if back_risk else 0.0,
            **kin,
            "landmark_visibility_mean": vis_mean,
            "lower_body_visibility": lower_vis,
            "torso_visibility": torso_vis,
            "frame_reliability_score": reliability,
            "view_validity_score": view_validity,
            "occlusion_risk_score": occlusion_risk,
            "model_features": model_features,
        }
        if back_risk:
            features["rounded_back_assessment"] = back_risk

        return features

    def reset(self) -> None:
        """Reset temporal state (kinematics, windows)."""
        self._kinematics.reset()
        self._trunk_angle_window.clear()
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_features.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/biomechanics/squat_features.py squat_coach/tests/test_features.py
git commit -m "feat: full squat feature extractor producing D=42 model input"
```

---

## Chunk 3: Temporal Models, Ensemble, and Phase Detection

### Task 15: Temporal model base class and factory

**Files:**
- Create: `squat_coach/models/__init__.py`
- Create: `squat_coach/models/temporal_base.py`
- Create: `squat_coach/models/model_factory.py`

- [ ] **Step 1: Implement base class**

```python
# squat_coach/models/temporal_base.py
"""Base class for temporal sequence models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TemporalModelOutput:
    """Standardized output from all temporal models."""
    phase_probs: torch.Tensor       # (batch, 4) softmax probabilities
    fault_probs: torch.Tensor       # (batch, 6) sigmoid probabilities
    quality_score: torch.Tensor     # (batch, 1) regression [0-1]

class TemporalModelBase(nn.Module, ABC):
    """Abstract base class for temporal squat analysis models.

    All temporal models receive input of shape (batch, seq_len, feature_dim)
    and produce TemporalModelOutput with phase, fault, and quality predictions.
    """

    def __init__(self, feature_dim: int = 42, seq_len: int = 60) -> None:
        super().__init__()
        self._feature_dim = feature_dim
        self._seq_len = seq_len

        # Shared output heads (subclasses set self._hidden_dim before calling _build_heads)
        self._hidden_dim: int = 0  # Must be set by subclass

    def _build_heads(self) -> None:
        """Build output heads. Call after setting self._hidden_dim."""
        self.phase_head = nn.Linear(self._hidden_dim, 4)
        self.fault_head = nn.Linear(self._hidden_dim, 6)
        self.quality_head = nn.Linear(self._hidden_dim, 1)

    def _apply_heads(self, hidden: torch.Tensor) -> TemporalModelOutput:
        """Apply output heads to hidden representation.

        Note: phase_probs returns RAW LOGITS during training (for CrossEntropyLoss).
        Call .softmax() on phase_probs at inference time for probabilities.
        fault_probs and quality_score use sigmoid since BCE/MSE expect [0,1].
        """
        phase_logits = self.phase_head(hidden)
        return TemporalModelOutput(
            phase_probs=phase_logits,  # Raw logits — CE loss applies softmax internally
            fault_probs=torch.sigmoid(self.fault_head(hidden)),
            quality_score=torch.sigmoid(self.quality_head(hidden)),
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """Forward pass. x shape: (batch, seq_len, feature_dim)."""
        ...
```

- [ ] **Step 2: Implement model factory**

```python
# squat_coach/models/model_factory.py
"""Model registry and factory."""
from typing import Type
from squat_coach.models.temporal_base import TemporalModelBase

_REGISTRY: dict[str, Type[TemporalModelBase]] = {}

def register_model(name: str):
    """Decorator to register a temporal model class."""
    def decorator(cls: Type[TemporalModelBase]) -> Type[TemporalModelBase]:
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> TemporalModelBase:
    """Create a model by name from the registry."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)

def available_models() -> list[str]:
    return list(_REGISTRY.keys())
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/models/
git commit -m "feat: temporal model base class and factory registry"
```

---

### Task 16: TCN model

**Files:**
- Create: `squat_coach/models/temporal_tcn.py`
- Create: `squat_coach/tests/test_ensemble.py`

- [ ] **Step 1: Implement TCN**

```python
# squat_coach/models/temporal_tcn.py
"""Temporal Convolutional Network for squat analysis.

Uses causal dilated convolutions to capture temporal patterns.
Fastest inference of all temporal models (~0.5ms per window).
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import register_model


class CausalConv1d(nn.Module):
    """Causal 1D convolution with dilation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()
        self._padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self._padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding (causal)
        if self._padding > 0:
            out = out[:, :, :-self._padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN residual block with causal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.residual(x))


@register_model("tcn")
class TemporalTCN(TemporalModelBase):
    """TCN model for squat temporal analysis."""

    def __init__(
        self,
        feature_dim: int = 42,
        seq_len: int = 60,
        num_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(feature_dim, seq_len)
        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        in_ch = feature_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        self._hidden_dim = num_channels[-1]
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """x: (batch, seq_len, feature_dim) -> TemporalModelOutput."""
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Take last timestep
        hidden = out[:, :, -1]
        return self._apply_heads(hidden)
```

- [ ] **Step 2: Write tests for model shapes**

```python
# squat_coach/tests/test_ensemble.py
"""Tests for temporal models and ensemble fusion."""
import torch
from squat_coach.models.temporal_tcn import TemporalTCN
from squat_coach.models.model_factory import create_model

def test_tcn_output_shapes():
    model = TemporalTCN(feature_dim=42, seq_len=60)
    model.eval()
    x = torch.randn(2, 60, 42)
    out = model(x)
    assert out.phase_probs.shape == (2, 4)
    assert out.fault_probs.shape == (2, 6)
    assert out.quality_score.shape == (2, 1)

def test_tcn_phase_logits_valid():
    model = TemporalTCN()
    model.eval()
    x = torch.randn(1, 60, 42)
    out = model(x)
    # phase_probs are raw logits — softmax should sum to 1
    probs = torch.softmax(out.phase_probs, dim=-1)
    assert abs(probs.sum().item() - 1.0) < 0.01

def test_factory_creates_tcn():
    model = create_model("tcn", feature_dim=42, seq_len=60)
    assert isinstance(model, TemporalTCN)

# NOTE: GRU tests are in Task 17 after GRU implementation
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/models/temporal_tcn.py squat_coach/tests/test_ensemble.py
git commit -m "feat: TCN temporal model with causal dilated convolutions"
```

---

### Task 17: GRU model and ST-GCN scaffold

**Files:**
- Create: `squat_coach/models/temporal_gru.py`
- Create: `squat_coach/models/temporal_stgcn_scaffold.py`

- [ ] **Step 1: Implement GRU**

```python
# squat_coach/models/temporal_gru.py
"""Gated Recurrent Unit model for squat temporal analysis.

Lightweight RNN. ~1ms inference per window. Good at capturing
sequential motion dynamics in squat phase transitions.
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import register_model


@register_model("gru")
class TemporalGRU(TemporalModelBase):
    """GRU model for squat temporal analysis."""

    def __init__(
        self,
        feature_dim: int = 42,
        seq_len: int = 60,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(feature_dim, seq_len)
        self._hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """x: (batch, seq_len, feature_dim) -> TemporalModelOutput."""
        output, _ = self.gru(x)
        # Take last timestep
        hidden = output[:, -1, :]
        return self._apply_heads(hidden)
```

- [ ] **Step 2: Implement ST-GCN scaffold**

```python
# squat_coach/models/temporal_stgcn_scaffold.py
"""ST-GCN scaffold — extension point for future skeleton-graph model.

This is NOT trained or used in production. It defines the interface
and placeholder architecture for a future Spatial-Temporal Graph
Convolutional Network that would operate on the skeleton graph directly.

To activate: implement the graph convolution layers and register the model.
"""
import torch
import torch.nn as nn
from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput

# NOT registered in factory — scaffold only
class STGCNScaffold(TemporalModelBase):
    """Placeholder for future ST-GCN implementation.

    Expected input: (batch, seq_len, num_joints, feature_dim_per_joint)
    The adjacency matrix would define the skeleton graph connectivity.
    """

    # Skeleton adjacency matrix (33 joints, BlazePose topology)
    # This defines which joints are connected in the body graph
    # TODO: Fill in actual graph convolution layers

    def __init__(self, feature_dim: int = 42, seq_len: int = 60) -> None:
        super().__init__(feature_dim, seq_len)
        self._hidden_dim = 128
        # Placeholder linear layer
        self.placeholder = nn.Linear(feature_dim * seq_len, self._hidden_dim)
        self._build_heads()

    def forward(self, x: torch.Tensor) -> TemporalModelOutput:
        """Placeholder forward pass."""
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        hidden = torch.relu(self.placeholder(flat))
        return self._apply_heads(hidden)
```

- [ ] **Step 3: Add GRU tests to test_ensemble.py**

Append to `squat_coach/tests/test_ensemble.py`:
```python
from squat_coach.models.temporal_gru import TemporalGRU

def test_gru_output_shapes():
    model = TemporalGRU(feature_dim=42, seq_len=60)
    model.eval()
    x = torch.randn(2, 60, 42)
    out = model(x)
    assert out.phase_probs.shape == (2, 4)
    assert out.fault_probs.shape == (2, 6)
    assert out.quality_score.shape == (2, 1)

def test_factory_creates_gru():
    model = create_model("gru", feature_dim=42, seq_len=60)
    assert isinstance(model, TemporalGRU)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_ensemble.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/models/temporal_gru.py squat_coach/models/temporal_stgcn_scaffold.py squat_coach/tests/test_ensemble.py
git commit -m "feat: GRU temporal model and ST-GCN scaffold"
```

---

### Task 18: Feature tensor builder and ensemble fusion

**Files:**
- Create: `squat_coach/models/feature_tensor_builder.py`
- Create: `squat_coach/models/ensemble_fusion.py`

- [ ] **Step 1: Implement feature tensor builder**

```python
# squat_coach/models/feature_tensor_builder.py
"""Convert feature dicts to model input tensors with z-score normalization."""
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from numpy.typing import NDArray


class FeatureTensorBuilder:
    """Builds normalized model input tensors from raw feature vectors.

    Applies z-score normalization using pre-computed training set statistics.
    """

    def __init__(self, stats_path: Optional[str] = None) -> None:
        """
        Args:
            stats_path: Path to JSON file with 'mean' and 'std' arrays (each length 42).
                        If None, no normalization is applied (raw features passed through).
        """
        self._mean: Optional[NDArray[np.float64]] = None
        self._std: Optional[NDArray[np.float64]] = None

        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                data = json.load(f)
            self._mean = np.array(data["mean"], dtype=np.float64)
            self._std = np.array(data["std"], dtype=np.float64)
            # Avoid division by zero
            self._std = np.where(self._std < 1e-8, 1.0, self._std)

    def normalize(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Z-score normalize a feature vector or sequence."""
        if self._mean is None:
            return features
        return (features - self._mean) / self._std

    def to_tensor(self, sequence: NDArray[np.float64]) -> torch.Tensor:
        """Convert a (seq_len, 42) numpy array to a (1, seq_len, 42) tensor."""
        normalized = self.normalize(sequence)
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
```

- [ ] **Step 2: Implement ensemble fusion**

```python
# squat_coach/models/ensemble_fusion.py
"""Confidence-weighted ensemble fusion of temporal model outputs.

Combines outputs from TCN and GRU (and optionally other models) into
a single fused prediction. Supports per-head weight configuration
and graceful fallback when a model is unavailable.
"""
from dataclasses import dataclass
from typing import Optional
import torch
import numpy as np
from squat_coach.models.temporal_base import TemporalModelOutput


@dataclass
class FusedOutput:
    """Result of ensemble fusion."""
    phase_probs: np.ndarray         # (4,) fused phase probabilities
    fault_probs: np.ndarray         # (6,) fused fault probabilities
    quality_score: float            # Fused quality [0-1]
    confidence: float               # Assessment confidence [0-1]
    model_agreement: float          # How much models agree [0-1]


class EnsembleFusion:
    """Fuse outputs from multiple temporal models."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        per_head_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Args:
            weights: Default per-model weights, e.g. {'tcn': 0.5, 'gru': 0.5}.
            per_head_weights: Optional per-head overrides.
        """
        self._weights = weights or {"tcn": 0.5, "gru": 0.5}
        self._per_head = per_head_weights or {}

    def fuse(self, outputs: dict[str, TemporalModelOutput]) -> FusedOutput:
        """Fuse model outputs into a single prediction.

        Args:
            outputs: Dict mapping model name to its TemporalModelOutput.

        Returns:
            FusedOutput with weighted-average predictions and confidence.
        """
        if not outputs:
            return self._empty_output()

        # Get weights for available models, re-normalize
        available = {k: v for k, v in self._weights.items() if k in outputs}
        if not available:
            # Fallback: equal weight for whatever we have
            available = {k: 1.0 / len(outputs) for k in outputs}
        total_w = sum(available.values())
        norm_w = {k: v / total_w for k, v in available.items()}

        # Fuse each head
        phase = self._fuse_head(outputs, norm_w, "phase")
        fault = self._fuse_head(outputs, norm_w, "fault")
        quality = self._fuse_head(outputs, norm_w, "quality")

        # Confidence from model agreement and phase entropy
        agreement = self._compute_agreement(outputs, "phase")
        phase_entropy = -np.sum(phase * np.log(phase + 1e-8))
        max_entropy = -np.log(1.0 / 4.0)
        confidence = agreement * (1.0 - phase_entropy / max_entropy)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return FusedOutput(
            phase_probs=phase,
            fault_probs=fault,
            quality_score=float(quality[0]) if quality.size > 0 else 0.5,
            confidence=confidence,
            model_agreement=agreement,
        )

    def _fuse_head(
        self,
        outputs: dict[str, TemporalModelOutput],
        default_weights: dict[str, float],
        head: str,
    ) -> np.ndarray:
        """Weighted average of a specific output head."""
        head_weights = self._per_head.get(head, default_weights)
        # Re-normalize for available models
        available_w = {k: v for k, v in head_weights.items() if k in outputs}
        if not available_w:
            available_w = default_weights
        total = sum(available_w.values())

        result = None
        for name, weight in available_w.items():
            if name not in outputs:
                continue
            out = outputs[name]
            if head == "phase":
                # phase_probs are raw logits from model — apply softmax for probabilities
                arr = torch.softmax(out.phase_probs, dim=-1).detach().cpu().numpy().squeeze()
            elif head == "fault":
                arr = out.fault_probs.detach().cpu().numpy().squeeze()
            else:
                arr = out.quality_score.detach().cpu().numpy().squeeze()

            weighted = arr * (weight / total)
            result = weighted if result is None else result + weighted

        return result if result is not None else np.zeros(4)

    def _compute_agreement(self, outputs: dict[str, TemporalModelOutput], head: str) -> float:
        """How much models agree on phase prediction (0=disagree, 1=agree)."""
        if len(outputs) < 2:
            return 1.0
        predictions = []
        for out in outputs.values():
            if head == "phase":
                predictions.append(torch.softmax(out.phase_probs, dim=-1).detach().cpu().numpy().squeeze())
        # Agreement = 1 - mean pairwise L1 distance
        dists = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                dists.append(np.mean(np.abs(predictions[i] - predictions[j])))
        mean_dist = np.mean(dists) if dists else 0.0
        return float(1.0 - min(mean_dist, 1.0))

    def _empty_output(self) -> FusedOutput:
        return FusedOutput(
            phase_probs=np.array([1.0, 0.0, 0.0, 0.0]),
            fault_probs=np.zeros(6),
            quality_score=0.5,
            confidence=0.0,
            model_agreement=0.0,
        )
```

- [ ] **Step 3: Commit**

```bash
git add squat_coach/models/feature_tensor_builder.py squat_coach/models/ensemble_fusion.py
git commit -m "feat: feature tensor builder and ensemble fusion layer"
```

---

### Task 19: Inference manager

**Files:**
- Create: `squat_coach/models/inference_manager.py`

- [ ] **Step 1: Implement inference manager**

```python
# squat_coach/models/inference_manager.py
"""Manage multi-model temporal inference.

Loads model checkpoints, runs inference on sequence windows,
and fuses outputs via the ensemble layer.
"""
import logging
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from numpy.typing import NDArray

from squat_coach.models.temporal_base import TemporalModelBase, TemporalModelOutput
from squat_coach.models.model_factory import create_model
from squat_coach.models.ensemble_fusion import EnsembleFusion, FusedOutput
from squat_coach.models.feature_tensor_builder import FeatureTensorBuilder

logger = logging.getLogger("squat_coach.inference")


class InferenceManager:
    """Load models and run ensemble inference on feature sequences."""

    def __init__(
        self,
        model_configs: dict,
        ensemble_config: dict,
        checkpoint_dir: str = "squat_coach/models/checkpoints",
        stats_path: Optional[str] = None,
        device: str = "cpu",
        view: str = "side",
    ) -> None:
        self._device = torch.device(device)
        self._models: dict[str, TemporalModelBase] = {}
        self._tensor_builder = FeatureTensorBuilder(stats_path)
        self._fusion = EnsembleFusion(
            weights=ensemble_config.get("weights"),
            per_head_weights=ensemble_config.get("per_head_weights"),
        )

        # Load enabled models
        for name, cfg in model_configs.items():
            if not cfg.get("enabled", False):
                continue
            try:
                model = create_model(name, **{k: v for k, v in cfg.items() if k != "enabled"})
                # Try loading checkpoint
                ckpt_path = Path(checkpoint_dir) / f"{name}_{view}_best.pt"
                if ckpt_path.exists():
                    state = torch.load(ckpt_path, map_location=self._device, weights_only=True)
                    model.load_state_dict(state)
                    logger.info("Loaded checkpoint: %s", ckpt_path)
                else:
                    logger.warning("No checkpoint found at %s, using random weights", ckpt_path)

                model.to(self._device)
                model.eval()
                self._models[name] = model
                logger.info("Model loaded: %s (device=%s)", name, self._device)
            except Exception as e:
                logger.error("Failed to load model %s: %s", name, e)

    def infer(self, sequence: NDArray[np.float64]) -> FusedOutput:
        """Run ensemble inference on a feature sequence.

        Args:
            sequence: (seq_len, 42) feature array.

        Returns:
            FusedOutput with fused predictions.
        """
        if not self._models:
            return self._fusion.fuse({})

        tensor = self._tensor_builder.to_tensor(sequence).to(self._device)

        outputs: dict[str, TemporalModelOutput] = {}
        with torch.no_grad():
            for name, model in self._models.items():
                outputs[name] = model(tensor)

        return self._fusion.fuse(outputs)

    @property
    def has_models(self) -> bool:
        return len(self._models) > 0
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/models/inference_manager.py
git commit -m "feat: inference manager with checkpoint loading and ensemble"
```

---

### Task 20: Phase detection and rep segmentation

**Files:**
- Create: `squat_coach/phases/__init__.py`
- Create: `squat_coach/phases/state_machine.py`
- Create: `squat_coach/phases/phase_detector.py`
- Create: `squat_coach/phases/rep_segmenter.py`
- Create: `squat_coach/tests/test_phase_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_phase_detector.py
"""Tests for phase detection and rep segmentation."""
import numpy as np
from squat_coach.phases.phase_detector import PhaseDetector
from squat_coach.phases.rep_segmenter import RepSegmenter, RepResult
from squat_coach.utils.enums import Phase

def test_phase_from_model_probs():
    detector = PhaseDetector()
    probs = np.array([0.1, 0.7, 0.1, 0.1])  # descent most likely
    phase = detector.detect(probs, hip_y=0.0)
    assert phase == Phase.DESCENT

def test_phase_debounce():
    """Phase should not change faster than debounce threshold."""
    detector = PhaseDetector(min_phase_duration_s=0.15, fps=30.0)
    probs_descent = np.array([0.0, 0.9, 0.1, 0.0])
    probs_bottom = np.array([0.0, 0.1, 0.9, 0.0])

    # Start in TOP
    detector.detect(np.array([0.9, 0.0, 0.0, 0.1]), hip_y=0.0)

    # Switch to DESCENT
    for _ in range(3):
        detector.detect(probs_descent, hip_y=-0.1)

    # Try to switch too fast to BOTTOM (within debounce)
    phase = detector.detect(probs_bottom, hip_y=-0.3)
    # Should still be in DESCENT due to debounce
    assert phase == Phase.DESCENT

def test_rep_segmenter_counts_rep():
    seg = RepSegmenter(min_rep_duration_s=0.1, cooldown_s=0.1, fps=30.0)
    phases = (
        [Phase.TOP] * 5 +
        [Phase.DESCENT] * 10 +
        [Phase.BOTTOM] * 5 +
        [Phase.ASCENT] * 10 +
        [Phase.TOP] * 5
    )
    results = []
    for i, phase in enumerate(phases):
        r = seg.update(phase, timestamp=i / 30.0)
        if r is not None:
            results.append(r)

    assert len(results) == 1
    assert results[0].rep_index == 1
    assert results[0].valid

def test_rep_segmenter_rejects_short_rep():
    seg = RepSegmenter(min_rep_duration_s=1.0, cooldown_s=0.0, fps=30.0)
    # Very short "rep" — 3 frames total
    phases = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT, Phase.TOP]
    results = []
    for i, phase in enumerate(phases):
        r = seg.update(phase, timestamp=i / 30.0)
        if r is not None:
            results.append(r)
    # Should be rejected as too short
    assert all(not r.valid for r in results) or len(results) == 0
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_phase_detector.py -v`

- [ ] **Step 3: Implement state machine**

```python
# squat_coach/phases/state_machine.py
"""Squat phase state machine.

Valid transitions:
    TOP → DESCENT → BOTTOM → ASCENT → TOP

Any other transition is rejected (the state machine stays in current phase).
"""
from squat_coach.utils.enums import Phase

# Allowed transitions
_TRANSITIONS: dict[Phase, set[Phase]] = {
    Phase.TOP: {Phase.DESCENT},
    Phase.DESCENT: {Phase.BOTTOM},
    Phase.BOTTOM: {Phase.ASCENT},
    Phase.ASCENT: {Phase.TOP},
}


def is_valid_transition(current: Phase, proposed: Phase) -> bool:
    """Check if a phase transition is valid."""
    if current == proposed:
        return True  # staying in same phase is always valid
    return proposed in _TRANSITIONS.get(current, set())
```

- [ ] **Step 4: Implement phase detector**

```python
# squat_coach/phases/phase_detector.py
"""Phase detection from fused model outputs with fallback to kinematics."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase
from squat_coach.phases.state_machine import is_valid_transition


class PhaseDetector:
    """Determine current squat phase from model probabilities.

    Uses model phase_probs as primary signal. Falls back to hip Y position
    if model confidence is low. Applies hysteresis and debounce.
    """

    def __init__(
        self,
        min_phase_duration_s: float = 0.15,
        fps: float = 30.0,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._min_frames = max(1, int(min_phase_duration_s * fps))
        self._conf_threshold = confidence_threshold
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0

    _PHASE_ORDER = [Phase.TOP, Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT]

    def detect(self, phase_probs: NDArray[np.float64], hip_y: float) -> Phase:
        """Detect current phase.

        Args:
            phase_probs: (4,) probabilities [top, descent, bottom, ascent].
            hip_y: Current hip vertical position (for fallback).

        Returns:
            Current phase after applying state machine + debounce.
        """
        self._frames_in_phase += 1

        # Primary: highest probability phase
        proposed_idx = int(np.argmax(phase_probs))
        proposed = self._PHASE_ORDER[proposed_idx]

        # Check if transition is valid and debounce has elapsed
        if proposed != self._current_phase:
            if (
                is_valid_transition(self._current_phase, proposed)
                and self._frames_in_phase >= self._min_frames
                and phase_probs[proposed_idx] >= self._conf_threshold
            ):
                self._current_phase = proposed
                self._frames_in_phase = 0

        return self._current_phase

    @property
    def current_phase(self) -> Phase:
        return self._current_phase

    def reset(self) -> None:
        self._current_phase = Phase.TOP
        self._frames_in_phase = 0
```

- [ ] **Step 5: Implement rep segmenter**

```python
# squat_coach/phases/rep_segmenter.py
"""Rep boundary detection and validation."""
from dataclasses import dataclass, field
from typing import Optional
from squat_coach.utils.enums import Phase


@dataclass
class RepResult:
    """Result of a completed rep."""
    rep_index: int
    start_time: float
    bottom_time: float
    end_time: float
    descent_duration: float
    bottom_duration: float
    ascent_duration: float
    valid: bool
    rejection_reason: str = ""


class RepSegmenter:
    """Segment squat reps from phase transitions."""

    def __init__(
        self,
        min_rep_duration_s: float = 0.8,
        cooldown_s: float = 0.5,
        fps: float = 30.0,
    ) -> None:
        self._min_rep_duration = min_rep_duration_s
        self._cooldown = cooldown_s
        self._fps = fps
        self._rep_count = 0
        self._in_rep = False
        self._rep_start_time = 0.0
        self._descent_start_time = 0.0
        self._bottom_start_time = 0.0
        self._ascent_start_time = 0.0
        self._last_rep_end_time = -999.0
        self._prev_phase = Phase.TOP

    def update(self, phase: Phase, timestamp: float) -> Optional[RepResult]:
        """Update with current phase. Returns RepResult when a rep completes."""
        result = None

        # Detect transitions
        if phase != self._prev_phase:
            if phase == Phase.DESCENT and self._prev_phase == Phase.TOP:
                # Rep started
                if timestamp - self._last_rep_end_time >= self._cooldown:
                    self._in_rep = True
                    self._rep_start_time = timestamp
                    self._descent_start_time = timestamp

            elif phase == Phase.BOTTOM and self._in_rep:
                self._bottom_start_time = timestamp

            elif phase == Phase.ASCENT and self._in_rep:
                self._ascent_start_time = timestamp

            elif phase == Phase.TOP and self._prev_phase == Phase.ASCENT and self._in_rep:
                # Rep completed
                self._rep_count += 1
                duration = timestamp - self._rep_start_time
                valid = duration >= self._min_rep_duration

                result = RepResult(
                    rep_index=self._rep_count,
                    start_time=self._rep_start_time,
                    bottom_time=self._bottom_start_time,
                    end_time=timestamp,
                    descent_duration=self._bottom_start_time - self._descent_start_time,
                    bottom_duration=self._ascent_start_time - self._bottom_start_time,
                    ascent_duration=timestamp - self._ascent_start_time,
                    valid=valid,
                    rejection_reason="" if valid else f"Too short: {duration:.2f}s",
                )

                self._in_rep = False
                self._last_rep_end_time = timestamp

        self._prev_phase = phase
        return result

    @property
    def rep_count(self) -> int:
        return self._rep_count

    def reset(self) -> None:
        self._rep_count = 0
        self._in_rep = False
        self._prev_phase = Phase.TOP
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_phase_detector.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add squat_coach/phases/ squat_coach/tests/test_phase_detector.py
git commit -m "feat: phase detector with state machine and rep segmenter"
```

---

## Chunk 4: Faults, Scoring, Events, Rendering, and Session

### Task 21: Fault detection system

**Files:**
- Create: `squat_coach/faults/__init__.py`
- Create: `squat_coach/faults/fault_types.py`
- Create: `squat_coach/faults/fault_rules.py`
- Create: `squat_coach/faults/evidence_engine.py`
- Create: `squat_coach/faults/confidence_gating.py`
- Create: `squat_coach/tests/test_fault_detection.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_fault_detection.py
"""Tests for fault detection."""
import numpy as np
from squat_coach.faults.evidence_engine import EvidenceEngine
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.utils.enums import FaultType

def test_detects_insufficient_depth():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 120.0, "hip_depth_vs_knee": -0.1}
    faults = engine.evaluate(features, {})
    fault_types = [f.fault_type for f in faults]
    assert FaultType.INSUFFICIENT_DEPTH in fault_types

def test_no_depth_fault_when_deep():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 80.0, "hip_depth_vs_knee": 0.1}
    faults = engine.evaluate(features, {})
    fault_types = [f.fault_type for f in faults]
    assert FaultType.INSUFFICIENT_DEPTH not in fault_types

def test_detects_forward_lean():
    engine = EvidenceEngine()
    features = {"torso_inclination_deg": 45.0}
    config = {"baseline_torso_angle": 10.0}
    faults = engine.evaluate(features, config)
    fault_types = [f.fault_type for f in faults]
    assert FaultType.EXCESSIVE_FORWARD_LEAN in fault_types

def test_fault_has_required_fields():
    engine = EvidenceEngine()
    features = {"primary_knee_angle": 120.0, "hip_depth_vs_knee": -0.1}
    faults = engine.evaluate(features, {})
    for fault in faults:
        assert 0.0 <= fault.severity <= 1.0
        assert 0.0 <= fault.confidence <= 1.0
        assert len(fault.evidence) > 0
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_fault_detection.py -v`

- [ ] **Step 3: Implement fault types**

```python
# squat_coach/faults/fault_types.py
"""Fault type definitions and dataclasses."""
from dataclasses import dataclass, field
from squat_coach.utils.enums import FaultType

# Coaching cue templates for each fault type
FAULT_CUES: dict[FaultType, str] = {
    FaultType.INSUFFICIENT_DEPTH: "Try to squat deeper",
    FaultType.EXCESSIVE_FORWARD_LEAN: "Keep your chest up",
    FaultType.ROUNDED_BACK_RISK: "Keep your chest up at the bottom",
    FaultType.UNSTABLE_TORSO: "Keep your torso steady",
    FaultType.HEEL_FAULT: "Keep your heels on the ground",
    FaultType.KNEE_VALGUS: "Push your knees out over your toes",
    FaultType.INCONSISTENT_TEMPO: "Try to maintain a steady pace",
    FaultType.POOR_TRUNK_CONTROL: "Control your trunk through the movement",
    FaultType.LOW_CONFIDENCE: "Move to a better position for the camera",
    FaultType.INVALID_VIEW: "Adjust camera angle",
}

@dataclass
class FaultDetection:
    """A detected fault with evidence."""
    fault_type: FaultType
    severity: float             # 0.0-1.0
    confidence: float           # 0.0-1.0
    evidence: list[str]         # What data supports this detection
    explanation_token: str      # Short coaching cue
    affects_overlay: bool = True
    affects_gemini: bool = True
```

- [ ] **Step 4: Implement fault rules and evidence engine**

```python
# squat_coach/faults/fault_rules.py
"""Per-fault rule definitions with configurable thresholds."""
from squat_coach.faults.fault_types import FaultDetection, FAULT_CUES
from squat_coach.utils.enums import FaultType
from typing import Optional


def check_insufficient_depth(features: dict, config: dict) -> Optional[FaultDetection]:
    knee = features.get("primary_knee_angle", 180.0)
    threshold = config.get("knee_angle_max_deg", 110.0)
    if knee > threshold:
        severity = min((knee - threshold) / 40.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.INSUFFICIENT_DEPTH,
            severity=severity,
            confidence=0.8,
            evidence=[f"Knee angle {knee:.0f}° > {threshold:.0f}° target"],
            explanation_token=FAULT_CUES[FaultType.INSUFFICIENT_DEPTH],
        )
    return None


def check_excessive_forward_lean(features: dict, config: dict) -> Optional[FaultDetection]:
    torso = features.get("torso_inclination_deg", 0.0)
    baseline = config.get("baseline_torso_angle", 10.0)
    threshold = config.get("trunk_deviation_deg", 25.0)
    deviation = torso - baseline
    if deviation > threshold:
        severity = min(deviation / 40.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.EXCESSIVE_FORWARD_LEAN,
            severity=severity,
            confidence=0.75,
            evidence=[f"Trunk angle {torso:.0f}° ({deviation:.0f}° over baseline)"],
            explanation_token=FAULT_CUES[FaultType.EXCESSIVE_FORWARD_LEAN],
        )
    return None


def check_rounded_back(features: dict, config: dict) -> Optional[FaultDetection]:
    risk = features.get("rounded_back_risk", 0.0)
    threshold = config.get("risk_threshold", 0.5)
    if risk > threshold:
        return FaultDetection(
            fault_type=FaultType.ROUNDED_BACK_RISK,
            severity=min(risk, 1.0),
            confidence=risk * 0.9,
            evidence=[f"Rounded back risk score: {risk:.2f}"],
            explanation_token=FAULT_CUES[FaultType.ROUNDED_BACK_RISK],
        )
    return None


def check_unstable_torso(features: dict, config: dict) -> Optional[FaultDetection]:
    stability = features.get("trunk_stability", 0.0)
    threshold = config.get("variance_threshold", 8.0)
    if stability > threshold:
        severity = min(stability / 20.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.UNSTABLE_TORSO,
            severity=severity,
            confidence=0.7,
            evidence=[f"Trunk variance: {stability:.1f}"],
            explanation_token=FAULT_CUES[FaultType.UNSTABLE_TORSO],
        )
    return None


def check_heel_fault(features: dict, config: dict) -> Optional[FaultDetection]:
    threshold = config.get("displacement_threshold", 0.05)
    # Proxy: ankle/foot displacement from expected position
    ankle_shin = features.get("ankle_shin_angle", 90.0)
    if ankle_shin < 60.0:  # Excessive dorsiflexion or heel lift proxy
        severity = min((60.0 - ankle_shin) / 30.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.HEEL_FAULT,
            severity=severity,
            confidence=0.6,
            evidence=[f"Ankle/shin angle {ankle_shin:.0f}° indicates possible heel issue"],
            explanation_token=FAULT_CUES[FaultType.HEEL_FAULT],
        )
    return None


def check_knee_valgus(features: dict, config: dict) -> Optional[FaultDetection]:
    angle = features.get("knee_valgus_angle", 0.0)
    threshold = config.get("angle_threshold_deg", 10.0)
    if angle > threshold:
        severity = min(angle / 20.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.KNEE_VALGUS,
            severity=severity,
            confidence=0.7,
            evidence=[f"Knee valgus angle {angle:.1f}° > {threshold:.0f}° threshold"],
            explanation_token=FAULT_CUES[FaultType.KNEE_VALGUS],
        )
    return None


def check_inconsistent_tempo(features: dict, config: dict) -> Optional[FaultDetection]:
    # This is evaluated at rep level, not per-frame; uses rep timing data passed in features
    descent_s = features.get("last_descent_duration", 0.0)
    ascent_s = features.get("last_ascent_duration", 0.0)
    if descent_s > 0 and ascent_s > 0:
        ratio = min(descent_s, ascent_s) / max(descent_s, ascent_s)
        threshold = config.get("timing_deviation_ratio", 0.4)
        if ratio < threshold:
            severity = min((threshold - ratio) / threshold, 1.0)
            return FaultDetection(
                fault_type=FaultType.INCONSISTENT_TEMPO,
                severity=severity,
                confidence=0.65,
                evidence=[f"Descent/ascent ratio {ratio:.2f} (threshold {threshold:.2f})"],
                explanation_token=FAULT_CUES[FaultType.INCONSISTENT_TEMPO],
            )
    return None


def check_poor_trunk_control(features: dict, config: dict) -> Optional[FaultDetection]:
    torso = features.get("torso_inclination_deg", 0.0)
    baseline = config.get("baseline_torso_angle", 10.0)
    trunk_stability = features.get("trunk_stability", 0.0)
    # Combined: high variance AND deviation from baseline at bottom
    deviation = max(0, torso - baseline)
    if trunk_stability > 5.0 and deviation > 15.0:
        severity = min((trunk_stability / 15.0 + deviation / 30.0) / 2.0, 1.0)
        return FaultDetection(
            fault_type=FaultType.POOR_TRUNK_CONTROL,
            severity=severity,
            confidence=0.7,
            evidence=[f"Trunk variance {trunk_stability:.1f} + deviation {deviation:.0f}°"],
            explanation_token=FAULT_CUES[FaultType.POOR_TRUNK_CONTROL],
        )
    return None


def check_low_confidence(features: dict, config: dict) -> Optional[FaultDetection]:
    reliability = features.get("frame_reliability_score", 1.0)
    if reliability < 0.5:
        return FaultDetection(
            fault_type=FaultType.LOW_CONFIDENCE,
            severity=1.0 - reliability,
            confidence=0.9,
            evidence=[f"Frame reliability {reliability:.2f}"],
            explanation_token=FAULT_CUES[FaultType.LOW_CONFIDENCE],
            affects_overlay=True,
            affects_gemini=False,
        )
    return None


def check_invalid_view(features: dict, config: dict) -> Optional[FaultDetection]:
    validity = features.get("view_validity_score", 1.0)
    if validity < 0.5:
        return FaultDetection(
            fault_type=FaultType.INVALID_VIEW,
            severity=1.0 - validity,
            confidence=0.9,
            evidence=[f"View validity {validity:.2f}"],
            explanation_token=FAULT_CUES[FaultType.INVALID_VIEW],
            affects_overlay=True,
            affects_gemini=False,
        )
    return None


# Registry of all rule functions
FAULT_RULES = [
    check_insufficient_depth,
    check_excessive_forward_lean,
    check_rounded_back,
    check_unstable_torso,
    check_heel_fault,
    check_knee_valgus,
    check_inconsistent_tempo,
    check_poor_trunk_control,
    check_low_confidence,
    check_invalid_view,
]
```

```python
# squat_coach/faults/evidence_engine.py
"""Aggregate evidence from rules and model outputs to detect faults."""
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.faults.fault_rules import FAULT_RULES
from squat_coach.faults.confidence_gating import apply_confidence_gate


class EvidenceEngine:
    """Run all fault rules and aggregate results."""

    def __init__(self, confidence_threshold: float = 0.3) -> None:
        self._conf_threshold = confidence_threshold

    def evaluate(
        self, features: dict, config: dict
    ) -> list[FaultDetection]:
        """Evaluate all fault rules against current features.

        Args:
            features: Named features from SquatFeatureExtractor.
            config: Fault threshold config (from scoring.yaml).

        Returns:
            List of detected faults (confidence-gated).
        """
        detections: list[FaultDetection] = []
        for rule_fn in FAULT_RULES:
            result = rule_fn(features, config)
            if result is not None:
                detections.append(result)

        return apply_confidence_gate(detections, self._conf_threshold)
```

```python
# squat_coach/faults/confidence_gating.py
"""Suppress low-confidence fault detections."""
from squat_coach.faults.fault_types import FaultDetection


def apply_confidence_gate(
    detections: list[FaultDetection], threshold: float = 0.3
) -> list[FaultDetection]:
    """Filter out faults below confidence threshold."""
    return [d for d in detections if d.confidence >= threshold]
```

- [ ] **Step 5: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_fault_detection.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add squat_coach/faults/ squat_coach/tests/test_fault_detection.py
git commit -m "feat: fault detection with evidence engine and confidence gating"
```

---

### Task 22: Scoring system

**Files:**
- Create: `squat_coach/scoring/__init__.py`
- Create: `squat_coach/scoring/ideal_reference.py`
- Create: `squat_coach/scoring/score_components.py`
- Create: `squat_coach/scoring/score_fusion.py`
- Create: `squat_coach/scoring/rationale.py`
- Create: `squat_coach/scoring/trend_analysis.py`
- Create: `squat_coach/tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# squat_coach/tests/test_scoring.py
"""Tests for the scoring system."""
from squat_coach.scoring.score_components import compute_depth_score, compute_trunk_control_score
from squat_coach.scoring.score_fusion import compute_rep_quality_score
from squat_coach.scoring.trend_analysis import TrendTracker

def test_depth_score_perfect():
    score = compute_depth_score(knee_angle_min=85.0, target_angle=90.0)
    assert score >= 90

def test_depth_score_shallow():
    score = compute_depth_score(knee_angle_min=130.0, target_angle=90.0)
    assert score < 50

def test_trunk_control_good():
    score = compute_trunk_control_score(
        torso_variance=2.0, max_forward_lean=20.0, baseline_angle=10.0
    )
    assert score >= 70

def test_rep_quality_weighted():
    scores = {"depth": 80, "trunk_control": 60, "posture_stability": 70, "movement_consistency": 90}
    quality = compute_rep_quality_score(scores, model_quality=0.7)
    assert 60 <= quality <= 90

def test_trend_tracker():
    tracker = TrendTracker()
    tracker.update(70)
    tracker.update(75)
    tracker.update(80)
    trend = tracker.get_trend()
    assert trend > 0  # improving
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_scoring.py -v`

- [ ] **Step 3: Implement scoring components**

```python
# squat_coach/scoring/ideal_reference.py
"""Personalized idealized squat reference from calibration."""
from dataclasses import dataclass
from squat_coach.preprocessing.calibration import CalibrationResult


@dataclass
class IdealReference:
    """Personalized target values for scoring comparison."""
    target_knee_angle: float        # Target depth angle (degrees)
    trunk_neutral_angle: float      # Calibrated neutral trunk angle
    trunk_tolerance: float          # Acceptable deviation (degrees)
    expected_descent_s: float       # Expected descent duration
    expected_ascent_s: float        # Expected ascent duration


def build_ideal_reference(
    calibration: CalibrationResult,
    target_knee_angle: float = 90.0,
    trunk_tolerance: float = 15.0,
) -> IdealReference:
    """Build personalized reference from calibration data."""
    return IdealReference(
        target_knee_angle=target_knee_angle,
        trunk_neutral_angle=calibration.baseline_torso_angle,
        trunk_tolerance=trunk_tolerance,
        expected_descent_s=1.0,
        expected_ascent_s=1.0,
    )
```

```python
# squat_coach/scoring/score_components.py
"""Individual score component computations. All scores are 0-100."""
import numpy as np


def compute_depth_score(knee_angle_min: float, target_angle: float = 90.0) -> float:
    """Score how close the minimum knee angle was to target depth.

    Score 100 = at or below target angle. Decreases linearly above target.
    """
    if knee_angle_min <= target_angle:
        return 100.0
    gap = knee_angle_min - target_angle
    # 50 degrees short of target = 0 score
    return max(0.0, 100.0 - (gap / 50.0) * 100.0)


def compute_trunk_control_score(
    torso_variance: float,
    max_forward_lean: float,
    baseline_angle: float,
) -> float:
    """Score trunk stability through the rep.

    Based on: how much torso angle varied and max deviation from baseline.
    """
    # Variance penalty: 0 variance = perfect, >15 = 0 points
    var_score = max(0.0, 100.0 - (torso_variance / 15.0) * 100.0)

    # Lean penalty: deviation from baseline
    lean_deviation = max(0.0, max_forward_lean - baseline_angle)
    lean_score = max(0.0, 100.0 - (lean_deviation / 30.0) * 100.0)

    return 0.5 * var_score + 0.5 * lean_score


def compute_posture_stability_score(
    rounded_back_risk: float,
    head_drift: float,
    body_scale: float,
) -> float:
    """Score posture quality based on back rounding and head drift."""
    back_score = max(0.0, 100.0 - rounded_back_risk * 100.0)
    drift_normalized = head_drift / max(body_scale, 0.01)
    drift_score = max(0.0, 100.0 - (drift_normalized / 0.3) * 100.0)
    return 0.6 * back_score + 0.4 * drift_score


def compute_movement_consistency_score(
    descent_duration: float,
    ascent_duration: float,
    expected_descent: float = 1.0,
    expected_ascent: float = 1.0,
) -> float:
    """Score movement smoothness and tempo symmetry."""
    # Timing symmetry: how similar descent and ascent durations are
    if descent_duration + ascent_duration < 0.01:
        return 50.0
    ratio = min(descent_duration, ascent_duration) / max(descent_duration, ascent_duration)
    symmetry_score = ratio * 100.0

    # Tempo match: how close to expected durations
    desc_match = max(0.0, 1.0 - abs(descent_duration - expected_descent) / expected_descent)
    asc_match = max(0.0, 1.0 - abs(ascent_duration - expected_ascent) / expected_ascent)
    tempo_score = (desc_match + asc_match) / 2.0 * 100.0

    return 0.5 * symmetry_score + 0.5 * tempo_score
```

```python
# squat_coach/scoring/score_fusion.py
"""Combine component scores into rep quality and overall form scores."""


def compute_rep_quality_score(
    component_scores: dict[str, float],
    model_quality: float = 0.5,
    weights: dict[str, float] | None = None,
    model_weight: float = 0.2,
) -> float:
    """Compute weighted rep quality score (0-100).

    Args:
        component_scores: Dict with keys: depth, trunk_control, posture_stability, movement_consistency.
        model_quality: Model's quality_score output [0-1].
        weights: Per-component weights (default: equal).
        model_weight: Weight for model quality (default 0.2).
    """
    if weights is None:
        weights = {"depth": 0.25, "trunk_control": 0.25, "posture_stability": 0.25, "movement_consistency": 0.25}

    component_total = sum(
        component_scores.get(k, 50.0) * w for k, w in weights.items()
    )
    # Model contributes model_weight, components contribute (1 - model_weight)
    return (1.0 - model_weight) * component_total + model_weight * (model_quality * 100.0)
```

```python
# squat_coach/scoring/rationale.py
"""Build structured rationale objects for rep scoring."""
from dataclasses import dataclass, field
from squat_coach.faults.fault_types import FaultDetection


@dataclass
class RepRationale:
    """Structured explanation of a rep's score."""
    rep_index: int
    scores: dict[str, float]
    dominant_fault: str
    fault_evidence: str
    comparison_to_ideal: dict[str, str]
    trend: str
    coaching_cue: str


def build_rationale(
    rep_index: int,
    scores: dict[str, float],
    faults: list[FaultDetection],
    ideal_comparison: dict[str, str],
    trend_str: str,
) -> RepRationale:
    """Build a rationale object for a completed rep."""
    # Dominant fault = highest severity
    if faults:
        dominant = max(faults, key=lambda f: f.severity)
        dominant_name = dominant.fault_type.value
        evidence_str = "; ".join(dominant.evidence)
        cue = dominant.explanation_token
    else:
        dominant_name = "none"
        evidence_str = "No significant faults detected"
        cue = "Good form, keep it up!"

    return RepRationale(
        rep_index=rep_index,
        scores=scores,
        dominant_fault=dominant_name,
        fault_evidence=evidence_str,
        comparison_to_ideal=ideal_comparison,
        trend=trend_str,
        coaching_cue=cue,
    )
```

```python
# squat_coach/scoring/trend_analysis.py
"""Rolling trend analysis across reps."""

class TrendTracker:
    """Track score trends with EMA and delta tracking."""

    def __init__(self, ema_alpha: float = 0.3, window: int = 5) -> None:
        self._alpha = ema_alpha
        self._window = window
        self._scores: list[float] = []
        self._ema: float = 50.0

    def update(self, score: float) -> None:
        """Add a new rep score."""
        self._scores.append(score)
        self._ema = self._alpha * score + (1.0 - self._alpha) * self._ema

    def get_trend(self) -> float:
        """Get trend direction. Positive = improving, negative = declining."""
        if len(self._scores) < 2:
            return 0.0
        recent = self._scores[-min(self._window, len(self._scores)):]
        if len(recent) < 2:
            return 0.0
        # Simple linear slope
        n = len(recent)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(recent) / n
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, recent))
        den = sum((xi - x_mean) ** 2 for xi in x)
        return num / den if den > 0 else 0.0

    @property
    def ema_score(self) -> float:
        return self._ema

    @property
    def last_score(self) -> float:
        return self._scores[-1] if self._scores else 50.0
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add squat_coach/scoring/ squat_coach/tests/test_scoring.py
git commit -m "feat: scoring system with components, fusion, rationale, trends"
```

---

### Task 23: Event system and Gemini payloads

**Files:**
- Create: `squat_coach/events/__init__.py`
- Create: `squat_coach/events/schemas.py`
- Create: `squat_coach/events/event_builder.py`
- Create: `squat_coach/events/formatter.py`
- Create: `squat_coach/events/gemini_payloads.py`
- Create: `squat_coach/events/coaching_priority.py`

- [ ] **Step 1: Implement all event modules**

```python
# squat_coach/events/schemas.py
"""Event dataclasses for the squat coach event system."""
from dataclasses import dataclass, field
from typing import Any
from squat_coach.utils.enums import Phase, FaultType


@dataclass
class SquatEvent:
    """Base event type."""
    event_type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepSummaryEvent(SquatEvent):
    """Emitted when a rep is completed."""
    event_type: str = "rep_completed"

    # Populated by event_builder
    rep_index: int = 0
    phase_durations: dict[str, float] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    coaching_cue: str = ""
```

```python
# squat_coach/events/event_builder.py
"""Construct events from system state."""
from squat_coach.events.schemas import SquatEvent, RepSummaryEvent
from squat_coach.phases.rep_segmenter import RepResult
from squat_coach.scoring.rationale import RepRationale
from squat_coach.faults.fault_types import FaultDetection


def build_rep_summary(
    rep_result: RepResult,
    rationale: RepRationale,
    faults: list[FaultDetection],
    features_snapshot: dict[str, float],
    pose_confidence: float,
    assessment_confidence: float,
) -> RepSummaryEvent:
    """Build a complete rep summary event."""
    return RepSummaryEvent(
        timestamp=rep_result.end_time,
        rep_index=rep_result.rep_index,
        phase_durations={
            "descent_s": round(rep_result.descent_duration, 2),
            "bottom_s": round(rep_result.bottom_duration, 2),
            "ascent_s": round(rep_result.ascent_duration, 2),
        },
        features={
            "knee_angle_min_deg": round(features_snapshot.get("primary_knee_angle", 0), 1),
            "hip_angle_min_deg": round(features_snapshot.get("primary_hip_angle", 0), 1),
            "torso_inclination_peak_deg": round(features_snapshot.get("torso_inclination_deg", 0), 1),
            "rounded_back_risk": round(features_snapshot.get("rounded_back_risk", 0), 2),
        },
        faults=[
            {
                "type": f.fault_type.value,
                "severity": round(f.severity, 2),
                "confidence": round(f.confidence, 2),
            }
            for f in faults
        ],
        scores=rationale.scores,
        confidence={
            "pose_confidence": round(pose_confidence, 2),
            "assessment_confidence": round(assessment_confidence, 2),
        },
        coaching_cue=rationale.coaching_cue,
    )


def build_phase_transition_event(
    timestamp: float, from_phase: str, to_phase: str
) -> SquatEvent:
    return SquatEvent(
        event_type="phase_transition",
        timestamp=timestamp,
        data={"from": from_phase, "to": to_phase},
    )
```

```python
# squat_coach/events/formatter.py
"""Format events for terminal logging."""
from squat_coach.events.schemas import SquatEvent, RepSummaryEvent


def format_frame_log(
    frame_idx: int,
    phase: str,
    features: dict,
    confidence: float,
) -> str:
    """Format a single frame log line for terminal output."""
    return (
        f"FRAME {frame_idx:>5d} | phase={phase:<8s} | "
        f"knee={features.get('primary_knee_angle', 0):>5.1f}° | "
        f"hip={features.get('primary_hip_angle', 0):>5.1f}° | "
        f"torso={features.get('torso_inclination_deg', 0):>4.1f}° | "
        f"depth={features.get('hip_depth_vs_knee', 0):>5.2f} | "
        f"back_risk={features.get('rounded_back_risk', 0):>4.2f} | "
        f"conf={confidence:.2f}"
    )


def format_rep_summary(event: RepSummaryEvent) -> str:
    """Format a rep summary for terminal output."""
    s = event.scores
    lines = [
        "═" * 60,
        f"  REP {event.rep_index} COMPLETE | quality={s.get('rep_quality', 0):.0f} | "
        f"depth={s.get('depth', 0):.0f} | trunk={s.get('trunk_control', 0):.0f} | "
        f"consistency={s.get('movement_consistency', 0):.0f} | "
        f"posture={s.get('posture_stability', 0):.0f}",
    ]
    for fault in event.faults:
        lines.append(
            f"  FAULT: {fault['type']} (severity={fault['severity']:.2f}, conf={fault['confidence']:.2f})"
        )
    lines.append(f'  CUE: "{event.coaching_cue}"')
    lines.append("═" * 60)
    return "\n".join(lines)
```

```python
# squat_coach/events/gemini_payloads.py
"""Gemini-ready compact payload formatter.

This module formats structured rep summaries into compact payloads
designed for the Gemini Live API. The actual API integration is a
future extension — this module produces the payload format.

FUTURE: Replace the placeholder adapter with actual Gemini Live API calls.
See: https://ai.google.dev/gemini-api/docs/live (when available)
"""
import json
from squat_coach.events.schemas import RepSummaryEvent


def format_gemini_payload(event: RepSummaryEvent) -> dict:
    """Format a rep summary into a Gemini-ready payload.

    This produces a compact, semantic summary suitable for
    natural language generation — NOT raw frame data.

    Returns:
        Dict ready for JSON serialization and Gemini API submission.
    """
    return {
        "exercise": "squat",
        "rep_index": event.rep_index,
        "phase_durations": event.phase_durations,
        "key_features": event.features,
        "faults": event.faults,
        "scores": event.scores,
        "confidence": event.confidence,
        "primary_coaching_cue": event.coaching_cue,
    }


def format_gemini_text_prompt(payload: dict) -> str:
    """Format payload as a text prompt for Gemini language generation.

    FUTURE: This would be sent to Gemini Live API for spoken feedback.
    """
    scores = payload["scores"]
    cue = payload["primary_coaching_cue"]

    prompt = (
        f"The user just completed rep {payload['rep_index']} of squats. "
        f"Overall score: {scores.get('rep_quality', 50):.0f}/100. "
    )
    if payload["faults"]:
        top_fault = payload["faults"][0]
        prompt += f"Main issue: {top_fault['type'].replace('_', ' ')} "
        prompt += f"(severity {top_fault['severity']:.0%}). "
    prompt += f"Coaching cue: {cue}"

    return prompt


# PLACEHOLDER: Future Gemini Live API adapter
# async def send_to_gemini(payload: dict) -> None:
#     """Send payload to Gemini Live API for spoken feedback.
#
#     This will use the Gemini Live streaming API to generate
#     real-time spoken coaching feedback from the structured payload.
#     """
#     # TODO: Implement when Gemini Live API is available
#     # client = genai.Client()
#     # session = await client.live.connect(model="gemini-2.0-flash-live")
#     # await session.send(format_gemini_text_prompt(payload))
#     pass
```

```python
# squat_coach/events/coaching_priority.py
"""Coaching cue arbitration — select the most important cue to show."""
import time
from squat_coach.faults.fault_types import FaultDetection
from squat_coach.utils.enums import FaultType


class CoachingPrioritizer:
    """Select the single most important coaching cue.

    Priority = severity * confidence * persistence * (1 / recency)
    Where recency = seconds since this fault type was last displayed.
    """

    def __init__(self, suppress_repeat_s: float = 5.0) -> None:
        self._suppress_s = suppress_repeat_s
        self._last_shown: dict[FaultType, float] = {}  # fault_type -> timestamp

    def select_cue(self, faults: list[FaultDetection]) -> str:
        """Select the highest-priority coaching cue.

        Returns:
            Coaching cue string, or empty string if no faults.
        """
        if not faults:
            return ""

        now = time.monotonic()
        best_priority = -1.0
        best_cue = ""

        for fault in faults:
            last = self._last_shown.get(fault.fault_type, 0.0)
            recency = max(now - last, 1.0)

            # Suppress recently shown cues
            if now - last < self._suppress_s and last > 0:
                continue

            priority = fault.severity * fault.confidence * (1.0 / recency)
            if priority > best_priority:
                best_priority = priority
                best_cue = fault.explanation_token

        if best_cue:
            # Record that we're showing this cue now
            for fault in faults:
                if fault.explanation_token == best_cue:
                    self._last_shown[fault.fault_type] = now
                    break

        return best_cue
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/events/
git commit -m "feat: event system with schemas, Gemini payloads, coaching priority"
```

---

### Task 24: Rendering — simple overlay

**Files:**
- Create: `squat_coach/rendering/__init__.py`
- Create: `squat_coach/rendering/draw_pose.py`
- Create: `squat_coach/rendering/draw_metrics.py`
- Create: `squat_coach/rendering/draw_feedback.py`
- Create: `squat_coach/rendering/overlay.py`

- [ ] **Step 1: Implement all rendering modules**

```python
# squat_coach/rendering/draw_pose.py
"""Draw skeleton on video frame."""
import cv2
import numpy as np
from numpy.typing import NDArray
from squat_coach.pose.landmarks import SKELETON_CONNECTIONS


def draw_skeleton(
    frame: NDArray[np.uint8],
    image_landmarks: NDArray[np.float64],
    visibility: NDArray[np.float64],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    radius: int = 4,
    vis_threshold: float = 0.5,
) -> NDArray[np.uint8]:
    """Draw pose skeleton on frame.

    Args:
        frame: BGR frame to draw on.
        image_landmarks: (33, 3) normalized image landmarks.
        visibility: (33,) visibility scores.
        color: BGR color for bones.
    """
    h, w = frame.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in image_landmarks]

    # Draw connections
    for i, j in SKELETON_CONNECTIONS:
        if visibility[i] > vis_threshold and visibility[j] > vis_threshold:
            cv2.line(frame, pts[i], pts[j], color, thickness)

    # Draw landmarks
    for idx, (pt, vis) in enumerate(zip(pts, visibility)):
        if vis > vis_threshold:
            cv2.circle(frame, pt, radius, (0, 200, 255), -1)

    return frame
```

```python
# squat_coach/rendering/draw_metrics.py
"""Draw score and angle text on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray


def draw_text_with_bg(
    frame: NDArray[np.uint8],
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.7,
    color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    """Draw text with a semi-transparent background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 4), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_phase_label(frame: NDArray[np.uint8], phase: str) -> None:
    """Draw phase label top-left."""
    draw_text_with_bg(frame, f"Phase: {phase}", (10, 30))


def draw_rep_count(frame: NDArray[np.uint8], count: int) -> None:
    """Draw rep count top-right."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Rep: {count}", (w - 120, 30))


def draw_score(frame: NDArray[np.uint8], score: float) -> None:
    """Draw current score bottom-left."""
    h, w = frame.shape[:2]
    draw_text_with_bg(frame, f"Score: {score:.0f}", (10, h - 20))
```

```python
# squat_coach/rendering/draw_feedback.py
"""Draw coaching cue on frame."""
import cv2
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_metrics import draw_text_with_bg


def draw_coaching_cue(
    frame: NDArray[np.uint8],
    cue: str,
    color: tuple[int, int, int] = (0, 200, 255),
) -> None:
    """Draw coaching cue bottom-center."""
    if not cue:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(cue, font, 0.8, 2)
    x = (w - tw) // 2
    draw_text_with_bg(frame, cue, (x, h - 50), font_scale=0.8, color=color)
```

```python
# squat_coach/rendering/overlay.py
"""Simple overlay compositor — combines all drawing elements."""
import numpy as np
from numpy.typing import NDArray
from squat_coach.rendering.draw_pose import draw_skeleton
from squat_coach.rendering.draw_metrics import draw_phase_label, draw_rep_count, draw_score
from squat_coach.rendering.draw_feedback import draw_coaching_cue


def render_overlay(
    frame: NDArray[np.uint8],
    image_landmarks: NDArray[np.float64] | None,
    visibility: NDArray[np.float64] | None,
    phase: str,
    rep_count: int,
    score: float,
    coaching_cue: str,
) -> NDArray[np.uint8]:
    """Render the full simple overlay on a video frame.

    Args:
        frame: BGR frame (modified in-place and returned).
        image_landmarks: (33, 3) or None if no detection.
        visibility: (33,) or None.
        phase: Current phase name string.
        rep_count: Current rep count.
        score: Current form score (0-100).
        coaching_cue: Active coaching cue string.

    Returns:
        Frame with overlay drawn.
    """
    if image_landmarks is not None and visibility is not None:
        draw_skeleton(frame, image_landmarks, visibility)

    draw_phase_label(frame, phase)
    draw_rep_count(frame, rep_count)
    draw_score(frame, score)
    draw_coaching_cue(frame, coaching_cue)

    return frame
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/rendering/
git commit -m "feat: simple overlay with skeleton, metrics, and coaching cue"
```

---

### Task 25: Session state and JSONL logging

**Files:**
- Create: `squat_coach/session/__init__.py`
- Create: `squat_coach/session/session_state.py`
- Create: `squat_coach/session/rep_history.py`
- Create: `squat_coach/session/jsonl_logger.py`

- [ ] **Step 1: Implement session modules**

```python
# squat_coach/session/session_state.py
"""Session-level state tracking."""
from dataclasses import dataclass, field
from squat_coach.utils.enums import Phase, ViewType


@dataclass
class SessionState:
    """Mutable state for the current session."""
    is_calibrated: bool = False
    current_phase: Phase = Phase.TOP
    rep_count: int = 0
    current_score: float = 0.0
    overall_score: float = 0.0
    current_cue: str = ""
    view_type: ViewType = ViewType.UNKNOWN
    frame_index: int = 0
    dropped_frame_count: int = 0
```

```python
# squat_coach/session/rep_history.py
"""Per-rep history storage for trend analysis."""
from dataclasses import dataclass
from squat_coach.events.schemas import RepSummaryEvent


class RepHistory:
    """Store and query rep summaries."""

    def __init__(self) -> None:
        self._reps: list[RepSummaryEvent] = []

    def add(self, rep: RepSummaryEvent) -> None:
        self._reps.append(rep)

    @property
    def count(self) -> int:
        return len(self._reps)

    def get_last(self, n: int = 1) -> list[RepSummaryEvent]:
        return self._reps[-n:]

    def get_all(self) -> list[RepSummaryEvent]:
        return list(self._reps)
```

```python
# squat_coach/session/jsonl_logger.py
"""JSONL session file writer."""
import json
import logging
from datetime import datetime
from pathlib import Path
from squat_coach.events.schemas import RepSummaryEvent
from dataclasses import asdict

logger = logging.getLogger("squat_coach.session")


class JSONLLogger:
    """Write rep summaries as JSONL lines to a session file."""

    def __init__(self, log_dir: str = "sessions") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._path = self._dir / f"{ts}.jsonl"
        self._file = open(self._path, "a")
        logger.info("Session log: %s", self._path)

    def log_rep(self, event: RepSummaryEvent) -> None:
        """Write a rep summary as one JSON line."""
        data = asdict(event)
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/session/
git commit -m "feat: session state, rep history, and JSONL logger"
```

---

## Chunk 5: Training Pipeline

### Task 26: Training data pipeline and synthetic generator

**Files:**
- Create: `squat_coach/training/__init__.py`
- Create: `squat_coach/training/data_pipeline.py`
- Create: `squat_coach/training/phase_labeler.py`
- Create: `squat_coach/training/synthetic_generator.py`
- Create: `squat_coach/training/dataset.py`
- Create: `squat_coach/tests/test_synthetic.py`
- Create: `squat_coach/tests/test_data_pipeline.py`

- [ ] **Step 1: Implement phase labeler**

```python
# squat_coach/training/phase_labeler.py
"""Auto-label squat phases from hip vertical trajectory.

Uses hip Y-coordinate over time to determine:
- DESCENT: hip moving downward (negative velocity, sustained)
- BOTTOM: hip at local minimum (lowest point)
- ASCENT: hip moving upward (positive velocity, sustained)
- TOP: hip at or near starting height

Uses hysteresis to avoid noisy label flipping.
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase


def label_phases_from_hip_trajectory(
    hip_y_sequence: NDArray[np.float64],
    fps: float = 30.0,
    velocity_threshold: float = 0.005,
    hysteresis: float = 0.002,
) -> list[Phase]:
    """Label each frame with a squat phase based on hip Y position.

    Args:
        hip_y_sequence: (T,) array of hip Y positions over time.
            In BlazePose world coords, Y increases downward.
        fps: Frame rate.
        velocity_threshold: Min velocity magnitude to classify as moving.
        hysteresis: Buffer to prevent oscillation at boundaries.

    Returns:
        List of Phase labels, one per frame.
    """
    T = len(hip_y_sequence)
    if T < 3:
        return [Phase.TOP] * T

    # Compute velocity via finite differences
    dt = 1.0 / fps
    velocity = np.gradient(hip_y_sequence, dt)

    # Smooth velocity
    kernel_size = max(3, int(fps * 0.1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    velocity_smooth = np.convolve(velocity, kernel, mode="same")

    labels: list[Phase] = []
    current_phase = Phase.TOP

    # Find global min/max for normalization context
    y_range = np.max(hip_y_sequence) - np.min(hip_y_sequence)
    if y_range < 0.01:
        return [Phase.TOP] * T

    for i in range(T):
        vel = velocity_smooth[i]

        if current_phase == Phase.TOP:
            if vel > velocity_threshold + hysteresis:
                current_phase = Phase.DESCENT
        elif current_phase == Phase.DESCENT:
            if vel < velocity_threshold - hysteresis and vel > -velocity_threshold + hysteresis:
                current_phase = Phase.BOTTOM
        elif current_phase == Phase.BOTTOM:
            if vel < -velocity_threshold - hysteresis:
                current_phase = Phase.ASCENT
        elif current_phase == Phase.ASCENT:
            if vel > -velocity_threshold + hysteresis and vel < velocity_threshold - hysteresis:
                current_phase = Phase.TOP

        labels.append(current_phase)

    return labels
```

- [ ] **Step 2: Implement synthetic generator**

```python
# squat_coach/training/synthetic_generator.py
"""Synthetic squat sequence generator for training data augmentation.

Generates artificial squat feature sequences with known labels.
Used to fill gaps in real training data (e.g., for faults not
covered by ALEX-GYM-1 or Zenodo datasets).
"""
import numpy as np
from numpy.typing import NDArray
from squat_coach.utils.enums import Phase, FaultType
from typing import Optional


def generate_synthetic_squat(
    seq_len: int = 60,
    feature_dim: int = 42,
    fps: float = 30.0,
    inject_fault: Optional[FaultType] = None,
    fault_severity: float = 0.5,
    noise_level: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], float]:
    """Generate one synthetic squat sequence.

    Returns:
        features: (seq_len, feature_dim) feature array
        phase_labels: (seq_len,) phase label indices (0=top, 1=desc, 2=bottom, 3=ascent)
        fault_labels: (6,) binary fault labels for the sequence
        quality: scalar quality score [0-1]
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0, 2 * np.pi, seq_len)

    # Base squat trajectory: sinusoidal knee angle curve
    # Knee goes from ~170 (standing) to ~85 (deep squat) and back
    base_knee = 127.5 - 42.5 * np.cos(t)  # range ~85-170
    base_hip = 120 - 35 * np.cos(t)       # range ~85-155
    base_torso = 5 + 15 * (1 - np.cos(t)) / 2  # range ~5-20 degrees

    # Hip Y trajectory (Y-down in BlazePose: higher = deeper)
    hip_y = 0.3 * np.cos(t)  # oscillates

    # Phase labels from trajectory shape
    phase_labels = np.zeros(seq_len, dtype=np.int64)
    quarter = seq_len // 4
    phase_labels[:quarter] = 0           # TOP -> start of descent
    phase_labels[quarter:2*quarter] = 1  # DESCENT
    phase_labels[2*quarter:3*quarter] = 2  # BOTTOM
    phase_labels[3*quarter:] = 3         # ASCENT

    # Fault injection
    fault_labels = np.zeros(6, dtype=np.float64)
    quality = 0.9  # default good

    if inject_fault == FaultType.INSUFFICIENT_DEPTH:
        base_knee += 20  # Don't go as deep
        fault_labels[0] = 1.0
        quality -= 0.3 * fault_severity

    elif inject_fault == FaultType.EXCESSIVE_FORWARD_LEAN:
        base_torso += 20 * fault_severity
        fault_labels[1] = 1.0
        quality -= 0.25 * fault_severity

    elif inject_fault == FaultType.ROUNDED_BACK_RISK:
        base_torso += 15 * fault_severity
        fault_labels[2] = 1.0
        quality -= 0.3 * fault_severity

    elif inject_fault == FaultType.HEEL_FAULT:
        fault_labels[3] = 1.0
        quality -= 0.2 * fault_severity

    elif inject_fault == FaultType.UNSTABLE_TORSO:
        base_torso += rng.normal(0, 5 * fault_severity, seq_len)
        fault_labels[4] = 1.0
        quality -= 0.2 * fault_severity

    elif inject_fault == FaultType.INCONSISTENT_TEMPO:
        # Distort the time axis
        t_distorted = t + 0.3 * fault_severity * np.sin(3 * t)
        base_knee = 127.5 - 42.5 * np.cos(t_distorted)
        fault_labels[5] = 1.0
        quality -= 0.15 * fault_severity

    quality = max(0.0, min(1.0, quality))

    # Build feature vector (simplified — fills D=42 slots)
    features = np.zeros((seq_len, feature_dim))
    features[:, 0] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # left knee
    features[:, 1] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # right knee
    features[:, 2] = base_knee + rng.normal(0, noise_level * 5, seq_len)  # primary knee
    features[:, 3] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # left hip
    features[:, 4] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # right hip
    features[:, 5] = base_hip + rng.normal(0, noise_level * 5, seq_len)   # primary hip
    features[:, 6] = 90 + rng.normal(0, noise_level * 3, seq_len)         # ankle proxy
    features[:, 7] = base_torso + rng.normal(0, noise_level * 3, seq_len) # torso incl
    features[:, 8] = base_torso + rng.normal(0, noise_level * 3, seq_len) # sh-hip angle

    # Offsets (small values)
    for i in range(9, 16):
        features[:, i] = rng.normal(0, noise_level, seq_len)

    # View-specific (16-19)
    features[:, 16] = base_torso  # forward lean / knee valgus
    features[:, 17] = rng.uniform(0, 0.3, seq_len)  # back risk / stance
    features[:, 18] = rng.uniform(0, 5, seq_len)     # stability / symmetry
    features[:, 19] = 80 + rng.normal(0, 3, seq_len) # ankle / hip shift

    # Kinematics (20-27): derive from finite differences
    dt = 1.0 / fps
    features[:, 20] = np.gradient(hip_y, dt)
    features[:, 21] = np.gradient(features[:, 20], dt)
    features[:, 22] = np.gradient(base_torso, dt)
    features[:, 23] = np.gradient(features[:, 22], dt)
    features[:, 24] = np.gradient(base_knee, dt)
    features[:, 25] = np.gradient(features[:, 24], dt)
    features[:, 26] = np.gradient(base_hip, dt)
    features[:, 27] = np.gradient(features[:, 26], dt)

    # Quality features (28-33)
    features[:, 28] = 0.9 + rng.normal(0, 0.02, seq_len)  # vis mean
    features[:, 29] = 0.85 + rng.normal(0, 0.03, seq_len)  # lower body vis
    features[:, 30] = 0.9 + rng.normal(0, 0.02, seq_len)  # torso vis
    features[:, 31] = 0.85 + rng.normal(0, 0.02, seq_len)  # reliability
    features[:, 32] = 1.0   # view validity
    features[:, 33] = 0.1 + rng.normal(0, 0.02, seq_len)   # occlusion risk

    # Pairwise distances (34-41)
    for i in range(34, 42):
        features[:, i] = 0.3 + rng.normal(0, 0.02, seq_len)

    return features, phase_labels, fault_labels, quality
```

- [ ] **Step 3: Write tests for synthetic generator**

```python
# squat_coach/tests/test_synthetic.py
"""Tests for synthetic data generation."""
import numpy as np
from squat_coach.training.synthetic_generator import generate_synthetic_squat
from squat_coach.utils.enums import FaultType

def test_synthetic_shapes():
    features, phases, faults, quality = generate_synthetic_squat(seq_len=60, feature_dim=42)
    assert features.shape == (60, 42)
    assert phases.shape == (60,)
    assert faults.shape == (6,)
    assert 0.0 <= quality <= 1.0

def test_synthetic_with_fault():
    features, phases, faults, quality = generate_synthetic_squat(
        inject_fault=FaultType.INSUFFICIENT_DEPTH
    )
    assert faults[0] == 1.0  # depth fault active
    assert quality < 0.9

def test_synthetic_good_form():
    features, phases, faults, quality = generate_synthetic_squat(inject_fault=None)
    assert np.sum(faults) == 0.0
    assert quality >= 0.8

def test_synthetic_phase_labels_valid():
    _, phases, _, _ = generate_synthetic_squat()
    assert all(p in [0, 1, 2, 3] for p in phases)
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_synthetic.py -v`
Expected: All PASS

- [ ] **Step 5: Implement dataset and data pipeline**

```python
# squat_coach/training/dataset.py
"""PyTorch Dataset for squat training sequences."""
import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


class SquatSequenceDataset(Dataset):
    """Dataset of squat feature sequences with multi-task labels."""

    def __init__(
        self,
        features: NDArray[np.float64],     # (N, seq_len, feature_dim)
        phase_labels: NDArray[np.int64],    # (N, seq_len)
        fault_labels: NDArray[np.float64],  # (N, 6)
        quality_labels: NDArray[np.float64], # (N,)
    ) -> None:
        self._features = features
        self._phases = phase_labels
        self._faults = fault_labels
        self._quality = quality_labels

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.tensor(self._features[idx], dtype=torch.float32),
            "phase_labels": torch.tensor(self._phases[idx, -1], dtype=torch.long),  # last frame label
            "fault_labels": torch.tensor(self._faults[idx], dtype=torch.float32),
            "quality_label": torch.tensor(self._quality[idx], dtype=torch.float32),
        }
```

```python
# squat_coach/training/data_pipeline.py
"""Data loading, preprocessing, and caching pipeline.

Handles ALEX-GYM-1, Zenodo, and synthetic data sources.
Produces cached .npz files ready for training.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional
from numpy.typing import NDArray

from squat_coach.training.synthetic_generator import generate_synthetic_squat
from squat_coach.utils.enums import FaultType

logger = logging.getLogger("squat_coach.training")

# Faults to generate synthetically
SYNTHETIC_FAULTS = [
    None,  # good form
    FaultType.INSUFFICIENT_DEPTH,
    FaultType.EXCESSIVE_FORWARD_LEAN,
    FaultType.ROUNDED_BACK_RISK,
    FaultType.HEEL_FAULT,
    FaultType.UNSTABLE_TORSO,
    FaultType.INCONSISTENT_TEMPO,
]


def generate_synthetic_dataset(
    num_samples: int = 2000,
    seq_len: int = 60,
    feature_dim: int = 42,
    cache_path: Optional[str] = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Generate a full synthetic training dataset.

    Returns:
        features: (N, seq_len, feature_dim)
        phase_labels: (N, seq_len)
        fault_labels: (N, 6)
        quality_labels: (N,)
    """
    if cache_path and Path(cache_path).exists():
        logger.info("Loading cached synthetic data from %s", cache_path)
        data = np.load(cache_path)
        return data["features"], data["phase_labels"], data["fault_labels"], data["quality_labels"]

    logger.info("Generating %d synthetic sequences...", num_samples)
    rng = np.random.default_rng(42)

    all_features = []
    all_phases = []
    all_faults = []
    all_quality = []

    for i in range(num_samples):
        fault = rng.choice(SYNTHETIC_FAULTS)
        severity = rng.uniform(0.3, 1.0) if fault else 0.0
        noise = rng.uniform(0.01, 0.05)

        features, phases, faults, quality = generate_synthetic_squat(
            seq_len=seq_len,
            feature_dim=feature_dim,
            inject_fault=fault,
            fault_severity=severity,
            noise_level=noise,
            rng=rng,
        )
        all_features.append(features)
        all_phases.append(phases)
        all_faults.append(faults)
        all_quality.append(quality)

    features_arr = np.array(all_features)
    phases_arr = np.array(all_phases)
    faults_arr = np.array(all_faults)
    quality_arr = np.array(all_quality)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            features=features_arr,
            phase_labels=phases_arr,
            fault_labels=faults_arr,
            quality_labels=quality_arr,
        )
        logger.info("Cached synthetic data to %s", cache_path)

    return features_arr, phases_arr, faults_arr, quality_arr


def compute_normalization_stats(
    features: NDArray[np.float64], stats_path: str
) -> None:
    """Compute and save per-feature mean and std for z-score normalization."""
    # features: (N, seq_len, D) -> flatten to (N*seq_len, D)
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0).tolist()
    std = flat.std(axis=0).tolist()
    # Avoid zero std
    std = [max(s, 1e-8) for s in std]

    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    logger.info("Saved normalization stats to %s", stats_path)
```

- [ ] **Step 6: Write data pipeline test**

```python
# squat_coach/tests/test_data_pipeline.py
"""Tests for data pipeline."""
import tempfile
import numpy as np
from squat_coach.training.data_pipeline import generate_synthetic_dataset, compute_normalization_stats

def test_synthetic_dataset_shapes():
    features, phases, faults, quality = generate_synthetic_dataset(num_samples=10, seq_len=30)
    assert features.shape == (10, 30, 42)
    assert phases.shape == (10, 30)
    assert faults.shape == (10, 6)
    assert quality.shape == (10,)

def test_synthetic_dataset_caching(tmp_path):
    cache = str(tmp_path / "test_cache.npz")
    f1, _, _, _ = generate_synthetic_dataset(num_samples=5, cache_path=cache)
    f2, _, _, _ = generate_synthetic_dataset(num_samples=5, cache_path=cache)
    np.testing.assert_array_equal(f1, f2)

def test_normalization_stats(tmp_path):
    features = np.random.randn(10, 30, 42)
    stats_path = str(tmp_path / "stats.json")
    compute_normalization_stats(features, stats_path)
    import json
    with open(stats_path) as f:
        data = json.load(f)
    assert len(data["mean"]) == 42
    assert len(data["std"]) == 42
```

- [ ] **Step 7: Run tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/test_data_pipeline.py squat_coach/tests/test_synthetic.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add squat_coach/training/ squat_coach/tests/test_synthetic.py squat_coach/tests/test_data_pipeline.py
git commit -m "feat: training data pipeline with synthetic generator and caching"
```

---

### Task 27: Trainer and train_all script

**Files:**
- Create: `squat_coach/training/trainer.py`
- Create: `squat_coach/training/evaluate.py`
- Create: `squat_coach/training/train_all.py`

- [ ] **Step 1: Implement trainer**

```python
# squat_coach/training/trainer.py
"""Unified training loop for temporal models. Supports MPS and CPU."""
import logging
import time
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from squat_coach.models.temporal_base import TemporalModelBase

logger = logging.getLogger("squat_coach.training")


class Trainer:
    """Train a temporal model with multi-task loss."""

    def __init__(
        self,
        model: TemporalModelBase,
        device: str = "auto",
        lr: float = 0.001,
        loss_weights: dict[str, float] | None = None,
        checkpoint_dir: str = "squat_coach/models/checkpoints",
        model_name: str = "model",
        view: str = "side",
    ) -> None:
        self._device = self._resolve_device(device)
        self._model = model.to(self._device)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._loss_weights = loss_weights or {"phase": 1.0, "fault": 1.0, "quality": 0.5}
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model_name
        self._view = view

        self._phase_loss = nn.CrossEntropyLoss()
        self._fault_loss = nn.BCELoss()
        self._quality_loss = nn.MSELoss()

        logger.info("Trainer initialized: model=%s, device=%s", model_name, self._device)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
        elif device in ("mps", "cuda"):
            return torch.device(device)
        return torch.device("cpu")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 50,
        patience: int = 10,
    ) -> dict:
        """Train the model with early stopping.

        Returns:
            Dict with training history (train_loss, val_loss per epoch).
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(max_epochs):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | time=%.1fs",
                epoch + 1, max_epochs, train_loss, val_loss, elapsed,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self._save_checkpoint(f"{self._model_name}_{self._view}_best.pt")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self._model.train()
        total_loss = 0.0
        count = 0

        for batch in loader:
            features = batch["features"].to(self._device)
            phase_labels = batch["phase_labels"].to(self._device)
            fault_labels = batch["fault_labels"].to(self._device)
            quality_labels = batch["quality_label"].to(self._device)

            self._optimizer.zero_grad()
            output = self._model(features)

            loss = self._compute_loss(output, phase_labels, fault_labels, quality_labels)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)

    def _validate(self, loader: DataLoader) -> float:
        self._model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self._device)
                phase_labels = batch["phase_labels"].to(self._device)
                fault_labels = batch["fault_labels"].to(self._device)
                quality_labels = batch["quality_label"].to(self._device)

                output = self._model(features)
                loss = self._compute_loss(output, phase_labels, fault_labels, quality_labels)
                total_loss += loss.item()
                count += 1

        return total_loss / max(count, 1)

    def _compute_loss(self, output, phase_labels, fault_labels, quality_labels) -> torch.Tensor:
        w = self._loss_weights
        l_phase = self._phase_loss(output.phase_probs, phase_labels)
        l_fault = self._fault_loss(output.fault_probs, fault_labels)
        l_quality = self._quality_loss(output.quality_score.squeeze(), quality_labels)
        return w["phase"] * l_phase + w["fault"] * l_fault + w["quality"] * l_quality

    def _save_checkpoint(self, filename: str) -> None:
        path = self._checkpoint_dir / filename
        torch.save(self._model.state_dict(), path)
        logger.info("Saved checkpoint: %s", path)
```

- [ ] **Step 2: Implement evaluator**

```python
# squat_coach/training/evaluate.py
"""Evaluation metrics for trained models."""
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from squat_coach.models.temporal_base import TemporalModelBase

logger = logging.getLogger("squat_coach.training")


def evaluate_model(
    model: TemporalModelBase,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate a model on test data.

    Returns:
        Dict with phase_accuracy, fault_f1, quality_mae.
    """
    model.eval()
    all_phase_pred = []
    all_phase_true = []
    all_fault_pred = []
    all_fault_true = []
    all_quality_pred = []
    all_quality_true = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            output = model(features)

            all_phase_pred.extend(output.phase_probs.argmax(dim=-1).cpu().numpy())
            all_phase_true.extend(batch["phase_labels"].numpy())
            all_fault_pred.extend((output.fault_probs > 0.5).cpu().numpy())
            all_fault_true.extend(batch["fault_labels"].numpy())
            all_quality_pred.extend(output.quality_score.squeeze().cpu().numpy())
            all_quality_true.extend(batch["quality_label"].numpy())

    phase_acc = np.mean(np.array(all_phase_pred) == np.array(all_phase_true))

    # Fault F1
    fault_pred = np.array(all_fault_pred)
    fault_true = np.array(all_fault_true)
    tp = np.sum((fault_pred == 1) & (fault_true == 1))
    fp = np.sum((fault_pred == 1) & (fault_true == 0))
    fn = np.sum((fault_pred == 0) & (fault_true == 1))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fault_f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    quality_mae = np.mean(np.abs(np.array(all_quality_pred) - np.array(all_quality_true)))

    results = {
        "phase_accuracy": float(phase_acc),
        "fault_f1": float(fault_f1),
        "quality_mae": float(quality_mae),
    }
    logger.info("Evaluation: %s", results)
    return results
```

- [ ] **Step 3: Implement train_all script**

```python
# squat_coach/training/train_all.py
"""End-to-end training script: generate data, train all models, evaluate."""
import logging
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, random_split
import torch

from squat_coach.training.data_pipeline import generate_synthetic_dataset, compute_normalization_stats
from squat_coach.training.dataset import SquatSequenceDataset
from squat_coach.training.trainer import Trainer
from squat_coach.training.evaluate import evaluate_model
from squat_coach.models.model_factory import create_model
# Import models to register them
import squat_coach.models.temporal_tcn  # noqa: F401
import squat_coach.models.temporal_gru  # noqa: F401

logger = logging.getLogger("squat_coach.training")


def train_all(config_path: str = "squat_coach/config/model.yaml") -> None:
    """Train all enabled models end-to-end."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seq_len = config["sequence"]["length"]
    feature_dim = config["sequence"]["feature_dim"]
    training_cfg = config["training"]
    checkpoint_dir = config["checkpoints"]["dir"]

    # Generate synthetic training data
    cache_path = str(Path(checkpoint_dir) / "synthetic_data.npz")
    stats_path = str(Path(checkpoint_dir) / "feature_stats.json")

    features, phase_labels, fault_labels, quality_labels = generate_synthetic_dataset(
        num_samples=3000,
        seq_len=seq_len,
        feature_dim=feature_dim,
        cache_path=cache_path,
    )

    # Compute normalization stats
    compute_normalization_stats(features, stats_path)

    # Normalize features
    import json
    import numpy as np
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    features_norm = (features - mean) / std

    # Create dataset
    dataset = SquatSequenceDataset(features_norm, phase_labels, fault_labels, quality_labels)

    # Split: 70/15/15
    n = len(dataset)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=training_cfg["batch_size"])

    # Train each enabled model
    models_to_train = ["tcn", "gru"]
    for model_name in models_to_train:
        model_cfg = config["models"].get(model_name, {})
        if not model_cfg.get("enabled", False):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        logger.info("=" * 60)
        logger.info("Training model: %s", model_name)
        logger.info("=" * 60)

        # Create model
        model_params = {k: v for k, v in model_cfg.items() if k != "enabled"}
        model_params["feature_dim"] = feature_dim
        model_params["seq_len"] = seq_len
        model = create_model(model_name, **model_params)

        # Train
        trainer = Trainer(
            model=model,
            lr=training_cfg["learning_rate"],
            loss_weights=training_cfg["loss_weights"],
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=training_cfg["max_epochs"],
            patience=training_cfg["patience"],
        )

        # Evaluate
        device = trainer._device
        # Reload best checkpoint
        ckpt = Path(checkpoint_dir) / f"{model_name}_side_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)

        results = evaluate_model(model, test_loader, device)
        logger.info("Model %s test results: %s", model_name, results)

    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
```

- [ ] **Step 4: Commit**

```bash
git add squat_coach/training/trainer.py squat_coach/training/evaluate.py squat_coach/training/train_all.py
git commit -m "feat: training loop with early stopping, evaluation, and train_all script"
```

---

## Chunk 6: App Orchestrator and Integration

### Task 28: Main application orchestrator

**Files:**
- Create: `squat_coach/app.py`

- [ ] **Step 1: Implement the main app**

```python
# squat_coach/app.py
"""Main application orchestrator — ties all subsystems together."""
import logging
import time
import cv2
import yaml
import numpy as np
from pathlib import Path

from squat_coach.utils.logging_utils import setup_logging
from squat_coach.utils.timing import FPSTracker
from squat_coach.utils.enums import Phase
from squat_coach.camera.webcam_stream import WebcamStream
from squat_coach.camera.video_replay import VideoReplay
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
from squat_coach.events.formatter import format_frame_log, format_rep_summary
from squat_coach.events.gemini_payloads import format_gemini_payload
from squat_coach.events.coaching_priority import CoachingPrioritizer
from squat_coach.rendering.overlay import render_overlay
from squat_coach.session.session_state import SessionState
from squat_coach.session.rep_history import RepHistory
from squat_coach.session.jsonl_logger import JSONLLogger

# Import models to register in factory
import squat_coach.models.temporal_tcn   # noqa: F401
import squat_coach.models.temporal_gru   # noqa: F401

logger = logging.getLogger("squat_coach")


class SquatCoachApp:
    """Main application: video → pose → features → models → scoring → output."""

    def __init__(
        self,
        mode: str = "webcam",
        video_path: str | None = None,
        debug: bool = False,
        log_features: bool = False,
    ) -> None:
        self._mode = mode
        self._video_path = video_path
        self._debug = debug
        self._log_features = log_features

        setup_logging(debug=debug)

        # Load configs
        self._config = self._load_config("squat_coach/config/default.yaml")
        self._model_config = self._load_config("squat_coach/config/model.yaml")
        self._scoring_config = self._load_config("squat_coach/config/scoring.yaml")

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def run(self) -> None:
        """Main loop."""
        if self._mode == "train":
            from squat_coach.training.train_all import train_all
            train_all()
            return

        # Initialize subsystems
        camera = self._create_camera()
        pose = MediaPipeBlazePose3D()
        smoother = EMALandmarkSmoother(alpha=self._config["preprocessing"]["ema_alpha"])
        calibrator = Calibrator(
            num_frames=int(self._config["preprocessing"]["calibration_duration_s"] * 30)
        )
        session = SessionState()
        fps_tracker = FPSTracker()
        jsonl_logger = JSONLLogger(self._config["session"]["log_dir"])
        rep_history = RepHistory()
        trend_tracker = TrendTracker()
        coach = CoachingPrioritizer()

        # These are initialized after calibration
        feature_extractor = None
        inference_mgr = None
        phase_detector = None
        rep_segmenter = None
        fault_engine = EvidenceEngine()
        ideal_ref = None
        cal_result = None

        seq_buf = SequenceBuffer(
            seq_len=self._model_config["sequence"]["length"],
            feature_dim=self._model_config["sequence"]["feature_dim"],
        )

        log_throttle_interval = 1.0 / self._config["logging"]["terminal_throttle_hz"]
        last_log_time = 0.0

        # Track min values within a rep for scoring
        rep_min_knee = 180.0
        rep_max_torso = 0.0
        rep_max_head_offset = 0.0
        rep_features_snapshot: dict = {}

        logger.info("Starting Squat Coach in %s mode", self._mode)

        try:
            while camera.is_opened():
                ret, frame = camera.read()
                if not ret or frame is None:
                    if self._mode == "replay":
                        logger.info("Video replay complete")
                        break
                    continue

                fps_tracker.tick()
                session.frame_index += 1
                timestamp = session.frame_index / 30.0

                # Pose estimation
                pose_result = pose.estimate(frame, timestamp)

                if not pose_result.detected:
                    session.dropped_frame_count += 1
                    if session.dropped_frame_count > self._config["preprocessing"]["max_dropped_frames"]:
                        session.current_cue = "Repositioning needed"
                    # Still render overlay with last known state
                    frame = render_overlay(
                        frame, None, None,
                        session.current_phase.value,
                        session.rep_count,
                        session.current_score,
                        session.current_cue,
                    )
                    cv2.imshow("Squat Coach", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                session.dropped_frame_count = 0

                # Smooth landmarks
                smoothed = smoother.smooth(pose_result.world_landmarks)
                normalized = normalize_to_hip_center(smoothed)

                # Calibration phase
                if not session.is_calibrated:
                    calibrator.add_frame(pose_result)
                    if calibrator.is_ready:
                        cal_result = calibrator.compute()
                        if cal_result:
                            session.is_calibrated = True
                            session.view_type = cal_result.view_type
                            logger.info("Calibration complete: %s view", cal_result.view_type.value)

                            # Initialize post-calibration subsystems
                            feature_extractor = SquatFeatureExtractor(cal_result, fps=30.0)
                            ideal_ref = build_ideal_reference(
                                cal_result,
                                target_knee_angle=self._scoring_config["calibration"]["target_knee_angle_deg"],
                            )
                            phase_detector = PhaseDetector(
                                min_phase_duration_s=self._scoring_config["calibration"]["min_phase_duration_s"],
                            )
                            rep_segmenter = RepSegmenter(
                                min_rep_duration_s=self._scoring_config["calibration"]["min_rep_duration_s"],
                                cooldown_s=self._scoring_config["calibration"]["rep_cooldown_s"],
                            )

                            ckpt_dir = self._model_config["checkpoints"]["dir"]
                            stats_path = str(Path(ckpt_dir) / "feature_stats.json")
                            inference_mgr = InferenceManager(
                                model_configs=self._model_config["models"],
                                ensemble_config=self._model_config["ensemble"],
                                checkpoint_dir=ckpt_dir,
                                stats_path=stats_path,
                                view=cal_result.view_type.value,
                            )
                    else:
                        # Show calibration prompt
                        cv2.putText(
                            frame, "Stand still for calibration...",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
                        )
                        cv2.imshow("Squat Coach", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        continue

                # Feature extraction
                features = feature_extractor.extract(normalized, pose_result.visibility)
                model_features = features["model_features"]

                # Sequence buffer
                seq_buf.push(model_features)

                # Temporal inference
                fused = None
                if seq_buf.is_ready and inference_mgr and inference_mgr.has_models:
                    fused = inference_mgr.infer(seq_buf.get_sequence())

                # Phase detection
                if fused is not None:
                    hip_y = float((normalized[23][1] + normalized[24][1]) / 2.0)
                    phase = phase_detector.detect(fused.phase_probs, hip_y)
                else:
                    phase = session.current_phase

                prev_phase = session.current_phase
                session.current_phase = phase

                # Track per-rep extremes
                if phase in (Phase.DESCENT, Phase.BOTTOM, Phase.ASCENT):
                    rep_min_knee = min(rep_min_knee, features.get("primary_knee_angle", 180))
                    rep_max_torso = max(rep_max_torso, features.get("torso_inclination_deg", 0))
                    rep_max_head_offset = max(rep_max_head_offset, features.get("head_to_trunk_offset", 0))
                    rep_features_snapshot = dict(features)

                # Fault detection
                fault_config = self._scoring_config.get("faults", {}).get("thresholds", {})
                fault_config["baseline_torso_angle"] = ideal_ref.trunk_neutral_angle if ideal_ref else 10.0
                faults = fault_engine.evaluate(features, fault_config)

                # Coaching cue
                session.current_cue = coach.select_cue(faults)

                # Rep segmentation
                rep_result = rep_segmenter.update(phase, timestamp)
                if rep_result and rep_result.valid:
                    session.rep_count = rep_result.rep_index

                    # Score the rep
                    depth = compute_depth_score(rep_min_knee, ideal_ref.target_knee_angle if ideal_ref else 90)
                    trunk = compute_trunk_control_score(
                        features.get("trunk_stability", 0),
                        rep_max_torso,
                        ideal_ref.trunk_neutral_angle if ideal_ref else 10,
                    )
                    posture = compute_posture_stability_score(
                        features.get("rounded_back_risk", 0),
                        rep_max_head_offset,
                        cal_result.body_scale if cal_result else 0.4,  # body_scale in meters
                    )
                    consistency = compute_movement_consistency_score(
                        rep_result.descent_duration,
                        rep_result.ascent_duration,
                    )

                    scores = {
                        "depth": depth,
                        "trunk_control": trunk,
                        "posture_stability": posture,
                        "movement_consistency": consistency,
                    }
                    model_q = fused.quality_score if fused else 0.5
                    rep_quality = compute_rep_quality_score(scores, model_quality=model_q)
                    scores["rep_quality"] = rep_quality

                    trend_tracker.update(rep_quality)
                    scores["overall_form"] = trend_tracker.ema_score

                    session.current_score = rep_quality

                    # Build rationale
                    comparison = {
                        "depth": f"{max(0, rep_min_knee - (ideal_ref.target_knee_angle if ideal_ref else 90)):.0f}° from target",
                        "trunk": f"{max(0, rep_max_torso - (ideal_ref.trunk_neutral_angle if ideal_ref else 10)):.0f}° over baseline",
                        "tempo": f"descent {rep_result.descent_duration:.1f}s / ascent {rep_result.ascent_duration:.1f}s",
                    }
                    trend_val = trend_tracker.get_trend()
                    trend_str = f"{'↑' if trend_val > 0 else '↓'}{abs(trend_val):.0f} over recent reps"

                    rationale = build_rationale(
                        rep_result.rep_index, scores, faults, comparison, trend_str
                    )

                    # Build events
                    confidence = fused.confidence if fused else 0.5
                    rep_event = build_rep_summary(
                        rep_result, rationale, faults, rep_features_snapshot,
                        pose_result.pose_confidence, confidence,
                    )

                    # Log to terminal
                    logger.info("\n%s", format_rep_summary(rep_event))

                    # Log to JSONL
                    jsonl_logger.log_rep(rep_event)
                    rep_history.add(rep_event)

                    # Gemini payload (placeholder — log for now)
                    gemini = format_gemini_payload(rep_event)
                    logger.debug("Gemini payload: %s", gemini)

                    # Reset per-rep tracking
                    rep_min_knee = 180.0
                    rep_max_torso = 0.0
                    rep_max_head_offset = 0.0

                # Terminal frame logging (throttled)
                now = time.monotonic()
                if now - last_log_time >= log_throttle_interval:
                    confidence = fused.confidence if fused else 0.0
                    logger.info(format_frame_log(
                        session.frame_index, phase.value, features, confidence
                    ))
                    last_log_time = now

                # Render overlay
                frame = render_overlay(
                    frame,
                    pose_result.image_landmarks,
                    pose_result.visibility,
                    phase.value,
                    session.rep_count,
                    session.current_score,
                    session.current_cue,
                )

                cv2.imshow("Squat Coach", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            camera.release()
            pose.close()
            jsonl_logger.close()
            cv2.destroyAllWindows()
            logger.info("Session ended. %d reps completed.", session.rep_count)

    def _create_camera(self):
        cam_cfg = self._config["camera"]
        if self._mode == "replay" and self._video_path:
            return VideoReplay(self._video_path)
        return WebcamStream(
            device_id=cam_cfg["device_id"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
        )
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/app.py
git commit -m "feat: main app orchestrator connecting all subsystems"
```

---

### Task 29: README

**Files:**
- Create: `squat_coach/README.md`

- [ ] **Step 1: Write README**

```markdown
# Squat Coach

Real-time squat analysis system using MediaPipe BlazePose 3D landmarks, TCN+GRU temporal models, and structured scoring.

## Setup

```bash
cd squat_coach
pip install -r requirements.txt
```

## Train Models

```bash
python -m squat_coach --mode train
```

This generates synthetic training data, trains TCN and GRU models, and saves checkpoints.

## Run Live Analysis

```bash
python -m squat_coach --mode webcam
```

Stand still for 2 seconds for calibration, then start squatting.

## Replay a Video

```bash
python -m squat_coach --mode replay --video path/to/squat_video.mp4
```

## Debug Mode

```bash
python -m squat_coach --mode webcam --debug
```

## Controls

- Press `q` to quit

## Architecture

Video → MediaPipe Pose → Feature Extraction (D=42) → TCN+GRU Ensemble → Phase Detection + Fault Detection → Scoring → Terminal Logging + Overlay + Gemini Payloads

See `docs/superpowers/specs/2026-03-28-squat-coach-design.md` for the full design specification.
```

- [ ] **Step 2: Commit**

```bash
git add squat_coach/README.md
git commit -m "feat: README with setup and usage instructions"
```

---

### Task 30: Final integration test — run training and verify

- [ ] **Step 1: Install dependencies**

Run: `cd /Users/piotrkacprzak/programow/hackathon && pip install -r squat_coach/requirements.txt`

- [ ] **Step 2: Run all unit tests**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m pytest squat_coach/tests/ -v`
Expected: All PASS

- [ ] **Step 3: Run training**

Run: `cd /Users/piotrkacprzak/programow/hackathon && python -m squat_coach --mode train`
Expected: Trains TCN and GRU, outputs loss per epoch, saves checkpoints

- [ ] **Step 4: Verify checkpoints exist**

Run: `ls squat_coach/models/checkpoints/`
Expected: `tcn_side_best.pt`, `gru_side_best.pt`, `feature_stats.json`, `synthetic_data.npz`

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete squat coach system — ready for live testing"
```
