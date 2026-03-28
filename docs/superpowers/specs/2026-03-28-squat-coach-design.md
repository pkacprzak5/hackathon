# Squat Coach — Real-Time Squat Analysis System

## Overview

A real-time squat analysis system that processes live webcam video (or recorded video replay), estimates 3D pose via MediaPipe BlazePose, extracts biomechanics features, runs trained temporal models, fuses their outputs, scores form against an idealized reference, detects faults, renders a simple live overlay, logs everything to terminal, and emits structured per-rep payloads for future Gemini Live API spoken feedback.

**Scope**: Bodyweight / back squat. Single user. Dual-view: side-view primary, front-view supported. Auto-detected at calibration.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pose backend | MediaPipe Pose Landmarker, BlazePose 3D world landmarks | Explicit requirement; 33 joints × 3D in body-centered meters |
| Temporal models | TCN + GRU (production ensemble) | Lowest inference latency; BiLSTM + Transformer available but off by default |
| Training device | MPS (Apple Silicon) | User is on Mac; CPU fallback |
| Training data | ALEX-GYM-1 (primary) + Zenodo squat (supplementary) + synthetic augmentation | Best squat coverage; lateral + frontal poses; per-criteria labels |
| View support | Side-view primary, front-view supported | ALEX-GYM-1 has both views; 3D world landmarks are partially view-invariant |
| Output: overlay | Simple skeleton + phase + rep count + score + one cue | Minimal, not a complex UI |
| Output: terminal | Rich real-time logging of all features, phases, scores, faults | Primary debugging/monitoring interface |
| Output: Gemini | One structured payload per rep, sent at rep_completed (ascent→top) | Full rep data available for scoring |
| Config | YAML files | Easy tuning without code edits |
| Language | Python only | OpenCV, MediaPipe, PyTorch, NumPy |

## Architecture

```
LIVE CAMERA / VIDEO FILE
        │
        ▼
┌─────────────────┐
│ Video Acquisition│  ← webcam or file replay
└────────┬────────┘
         ▼
┌─────────────────┐
│ MediaPipe Pose   │  ← BlazePose 3D world landmarks (33 joints × 3D)
│ Landmarker       │    + image landmarks + visibility + confidence
└────────┬────────┘
         ▼
┌─────────────────┐
│ View Detection   │  ← auto-detect front vs side from shoulder geometry
│ & Calibration    │    estimate body scale, baseline posture, dominant side
└────────┬────────┘
         ▼
┌─────────────────┐
│ Preprocessing    │  ← EMA smoothing, hip-centered normalization,
│                  │    landmark stabilization, dropped-frame handling
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────┐
│ Feature Extraction (per-frame, ~50+ feats)    │
│ (D=42 → models; rest → rules/scoring/logging)│
│  A. Core geometry (18+)                      │
│  B. Kinematics (8+)                          │
│  C. Skeleton structure                       │
│  D. View-specific features                   │
│  E. Quality/confidence (5+)                  │
└────────┬────────────────────────────────────┘
         ▼
┌─────────────────┐
│ Sequence Buffer  │  ← rolling window (e.g. 60 frames / 2s)
└────────┬────────┘
         ▼
┌──────────────────────────────────────┐
│ Multi-Model Temporal Inference        │
│  TCN ──┐                             │
│  GRU ──┤→ Ensemble Fusion            │
│  (BiLSTM, Transformer: optional)     │
│  (ST-GCN: scaffold)                  │
└────────┬─────────────────────────────┘
         ▼
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ Phase  │ │  Fault   │
│Detector│ │Detection │
│+ Rep   │ │+ Evidence│
│Segment.│ │+ Gating  │
└───┬────┘ └────┬─────┘
    └─────┬─────┘
          ▼
┌─────────────────────────┐
│ Scoring Engine           │
│ depth / trunk_control /  │
│ posture_stability /      │
│ movement_consistency /   │
│ rep_quality /            │
│ overall_form (0-100)     │
│ + rationale object       │
└────────┬────────────────┘
         │
    ┌────┴─────────┐
    ▼              ▼
┌────────┐  ┌────────────────┐
│Terminal │  │ Simple Overlay  │
│Logging │  │ + Event System  │
│(rich)  │  │ + Gemini Payload│
└────────┘  └────────────────┘
```

## Data Flow

Frame → MediaPipe (3D landmarks) → View detection (once at calibration) → Smoothing + normalization → Feature extraction (all ~50+ features every frame) → Sequence buffer (rolling window) → Temporal models (TCN + GRU) → Ensemble fusion → Phase detection + Fault detection → Scoring with rationale → Terminal logging (every frame, throttled) + Simple overlay (every frame) + Gemini payload (per rep at rep_completed).

## Technical Specifications

### Video Requirements

- **Target FPS**: 30 fps
- **Minimum resolution**: 640×480 (720p preferred)
- **Camera suitability check** (during calibration): validates resolution ≥ 640×480, stable FPS ≥ 20, landmark detection success rate ≥ 80% over calibration frames, adequate lighting (mean frame brightness > 40)

### Dual-View Semantics

Dual-view means a **single camera** that the user positions in either side or front orientation. **One view per session**, auto-detected at calibration. This is NOT simultaneous dual-camera. The user chooses their camera angle; the system detects which view it is and selects the appropriate feature extraction path and fault set.

### Model Input Tensor Contract

All temporal models receive tensors of shape: **[batch, seq_len, D]** where:
- `seq_len` = 60 (default, 2 seconds at 30 FPS, configurable in model.yaml)
- `D` = 42 (engineered feature vector dimension)

**Feature vector layout (D=42), concatenated in this order:**

| Index | Feature | Count |
|-------|---------|-------|
| 0-5 | left/right/primary knee angle, left/right/primary hip angle | 6 |
| 6 | ankle_angle_proxy | 1 |
| 7-8 | torso_inclination_deg, shoulder_hip_line_angle | 2 |
| 9-11 | head_to_trunk_offset, shoulder_to_hip_h_delta, shoulder_to_hip_v_delta | 3 |
| 12-13 | hip_depth_vs_knee, hip_depth_vs_ankle | 2 |
| 14-15 | nose_to_shoulder_offset, neck_forward_offset | 2 |
| 16-19 | forward_lean_angle/knee_valgus_angle, rounded_back_risk/stance_width_ratio, trunk_stability/left_right_symmetry, ankle_shin_angle/hip_shift_lateral (view-dependent) | 4 |
| 20-27 | hip_vert_vel, hip_vert_accel, trunk_ang_vel, trunk_ang_accel, knee_ang_vel, knee_ang_accel, hip_ang_vel, hip_ang_accel | 8 |
| 28-33 | landmark_visibility_mean, lower_body_visibility, torso_visibility, frame_reliability_score, view_validity_score, occlusion_risk_score | 6 |
| 34-41 | pairwise_distance_subset (8 key joint pairs: L/R hip-knee, L/R knee-ankle, L/R shoulder-hip, hip_mid-shoulder_mid, nose-shoulder_mid) | 8 |

View-dependent features (indices 16-19): side-view features are used when view=side, front-view features when view=front. The slot positions are the same; the semantics change based on detected view. **Separate model weights are trained per view** — `{model_type}_side_best.pt` and `{model_type}_front_best.pt`. The inference manager loads the correct checkpoint based on the calibrated view. This avoids ambiguity in view-dependent slot interpretation.

**D=42 is the model input subset.** The full feature extraction pipeline computes ~50+ features per frame (including bone vectors, joint velocities, full normalized landmarks, etc.). Features beyond D=42 are used by the rule-based scoring engine, fault evidence engine, and terminal logging — they do NOT feed the temporal models.

**Input normalization**: All D=42 features are z-score standardized using per-feature mean and standard deviation computed on the training set. These statistics are saved alongside model checkpoints and applied identically at inference time. This handles the mixed-scale nature of the features (degrees, meters, ratios, velocities).

### Preprocessing Parameters

- **EMA smoothing alpha**: 0.4 (default, configurable in default.yaml). Higher = less smoothing, lower latency. Lower = more smoothing, more lag.
- **Dropped frame handling**: Hold last valid landmarks for up to 5 frames. After 5 consecutive drops, mark sequence segment as invalid and suppress scoring/faults. Display "Repositioning needed" on overlay.
- **No-detection state**: When MediaPipe returns no landmarks, increment drop counter. Reset on successful detection.

### Gemini Payload Trigger

Gemini payload is sent at **`rep_completed`** (ascent→top confirmed), NOT at bottom→ascent. This ensures the full rep data (including ascent quality) is available for scoring. The earlier mention of "bottom→ascent" in the Key Decisions table is corrected here — all scoring and Gemini dispatch happens at rep completion.

## Training Pipeline

### Data Sources

**ALEX-GYM-1 (primary)**
- 295 squat videos with lateral + frontal pose keypoints (33 landmarks × 3D per frame)
- Per-criteria quality ratings in squat.xlsx
- Auto-label phases from hip vertical trajectory (descent = hip dropping, bottom = local min, ascent = hip rising, top = local max)
- Map Excel criteria columns → fault labels (see label mapping below)

**Zenodo Squat Dataset (supplementary)**
- Side-view squat images, ~824 MB
- Labels: Good / Bad_Back / Bad_Heel
- Run MediaPipe on images → extract landmarks → compute features
- Map to per-frame fault supervision (Good→no_fault, Bad_Back→rounded_back, Bad_Heel→heel_fault)
- **Usage in temporal training**: Zenodo provides single images, not sequences. These are used ONLY for per-frame fault classification pre-training. They are NOT assembled into pseudo-sequences. The per-frame fault head can be pre-trained on Zenodo, then fine-tuned jointly with temporal training on ALEX-GYM-1 sequences.

### ALEX-GYM-1 Label Mapping

The squat.xlsx file contains per-criteria rating columns ending in "F" (frontal) and "L" (lateral). Mapping to our fault labels:

| Excel Column Pattern | Our Fault Label | Binarization |
|---------------------|-----------------|--------------|
| Depth-related columns (L) | insufficient_depth | score < 0.5 → fault present |
| Back/spine columns (L) | rounded_back_risk | score < 0.5 → fault present |
| Forward lean columns (L) | excessive_forward_lean | score < 0.5 → fault present |
| Heel/foot columns (L) | heel_fault | score < 0.5 → fault present |
| Stability columns (L/F) | unstable_torso | score < 0.5 → fault present |
| "class" column | overall quality | 0=good, 1+=has errors |

Exact column name discovery happens during data pipeline preprocessing (columns vary per dataset version). The pipeline logs the discovered mapping for verification. Lateral ("L") columns are preferred for side-view training; frontal ("F") columns for front-view.

For faults not covered by ALEX-GYM-1 labels (inconsistent_tempo, poor_trunk_control), synthetic augmentation provides labeled training data.

**Synthetic Augmentation (gap-filling)**
- Generate smooth squat trajectories (sinusoidal hip path + joint angle curves)
- Inject known faults at controlled severities
- Add realistic noise + jitter
- Label phases + faults + quality scores
- Generate both front-view and side-view variants

### Preprocessing (run once, cache to disk)

1. Load ALEX-GYM-1 lateral + frontal pose sequences
2. Compute full biomechanics feature vector per frame (same pipeline as inference)
3. Auto-label phases from hip kinematics with hysteresis
4. Map quality ratings → fault labels
5. Run MediaPipe on Zenodo images → landmarks → features → fault labels
6. Generate synthetic sequences with known labels
7. Cache all as .npz feature tensors with labels

### Model Training

All models share the same feature tensors, train/val/test splits (70/15/15), and output heads.

**Model outputs (shared interface):**
- `phase_probs`: [top, descent, bottom, ascent] — softmax
- `fault_probs`: [depth, forward_lean, rounded_back, heel_fault, unstable_torso, tempo] — sigmoid per fault
- `quality_score`: scalar [0-1] — regression

Note: `confidence` is NOT a model output head. It is computed post-hoc from: (a) ensemble disagreement between models, (b) mean landmark visibility of the input window, (c) phase probability entropy. This avoids the need for confidence ground truth labels.

**Models trained:**
1. TCN (production) — causal temporal convolutions
2. GRU (production) — single-direction gated RNN
3. BiLSTM (optional) — bidirectional LSTM
4. Transformer (optional) — encoder-only with positional encoding

**Loss function:**
- Phase head: cross-entropy loss (4-class classification)
- Fault head: binary cross-entropy loss (per-fault sigmoid, multi-label)
- Quality head: MSE loss (regression to [0-1])
- **Combined loss**: `L = 1.0 * L_phase + 1.0 * L_fault + 0.5 * L_quality`
- Weights configurable in model.yaml. Phase and fault are equally weighted as primary tasks; quality is secondary.

**Training config:**
- Device: MPS primary, CPU fallback
- Early stopping on validation loss (patience=10 epochs)
- ~20-50 epochs target
- Moderate model sizes to keep inference fast
- Train/val/test split: 70/15/15, **video-level splitting** (all frames from one video go to the same split, no data leakage). Stratified by fault distribution.

### Model Checkpointing

- Checkpoints saved to: `squat_coach/models/checkpoints/{model_type}_{timestamp}.pt`
- Best model (lowest val loss) symlinked as: `squat_coach/models/checkpoints/{model_type}_best.pt`
- Ensemble config in model.yaml references checkpoint paths
- At inference, `inference_manager.py` loads `*_best.pt` for each enabled model

**Ensemble calibration:**
- After all models trained, calibrate ensemble weights on validation set
- Default: TCN + GRU with confidence-weighted fusion
- Per-head fusion (phase weights may differ from fault weights)
- Calibration procedure: grid search over weight combinations [0.0, 0.25, 0.5, 0.75, 1.0] per model per head, selecting the combination that minimizes validation loss. Fast since only 2 production models.

### Best 2 Models for Production

**TCN**: Pure convolutions, fastest inference (~0.5ms per window), fully parallelizable, no sequential dependency. Excellent for real-time.

**GRU**: Lightweight RNN, ~1ms per window, proven on motion sequence data, captures temporal dynamics efficiently.

BiLSTM and Transformer remain available as optional ensemble members for experimentation.

## Feature Schema

### Core Geometry (view-agnostic, from 3D world landmarks)

| Feature | Formula | Unit |
|---------|---------|------|
| left_knee_angle | angle(hip_L, knee_L, ankle_L) | deg |
| right_knee_angle | angle(hip_R, knee_R, ankle_R) | deg |
| primary_knee_angle | selected by dominant visible side | deg |
| left_hip_angle | angle(shoulder_L, hip_L, knee_L) | deg |
| right_hip_angle | angle(shoulder_R, hip_R, knee_R) | deg |
| primary_hip_angle | selected by dominant visible side | deg |
| ankle_angle_proxy | angle(knee, ankle, foot_index) | deg |
| torso_inclination_deg | angle of mid_shoulder→mid_hip vector vs vertical | deg |
| shoulder_hip_line_angle | angle of shoulder→hip line vs vertical | deg |
| head_to_trunk_offset | perpendicular distance from nose to trunk line | m |
| shoulder_to_hip_h_delta | horizontal distance shoulder→hip | m |
| shoulder_to_hip_v_delta | vertical distance shoulder→hip | m |
| hip_depth_vs_knee | (hip.y - knee.y) normalized | ratio |
| hip_depth_vs_ankle | (hip.y - ankle.y) normalized | ratio |
| nose_to_shoulder_offset | nose horizontal offset from mid_shoulder | m |
| neck_forward_offset | ear midpoint offset from shoulder midpoint (sagittal) | m |

### Side-View Specific

| Feature | Description |
|---------|-------------|
| forward_lean_angle | trunk forward angle from vertical in sagittal plane |
| rounded_back_risk | composite: trunk curl proxy + head drift + mid-spine deviation from straight line. Score 0-1 with confidence and rationale. |
| trunk_stability | rolling variance of torso_inclination_deg over window |
| ankle_shin_angle | dorsiflexion proxy from shin-to-foot angle |

### Front-View Specific

| Feature | Description |
|---------|-------------|
| knee_valgus_angle | knee inward collapse angle (left + right) |
| stance_width_ratio | foot spread / hip width |
| left_right_symmetry | bilateral comparison score (angles, depths) |
| hip_shift_lateral | lateral hip displacement from center |

### Kinematics (finite differences)

| Feature | Derivation |
|---------|------------|
| hip_vertical_velocity | Δ(hip.y) / Δt |
| hip_vertical_acceleration | Δ(hip_vertical_velocity) / Δt |
| trunk_angle_velocity | Δ(torso_inclination) / Δt |
| trunk_angle_acceleration | Δ(trunk_angle_velocity) / Δt |
| knee_angle_velocity | Δ(primary_knee_angle) / Δt |
| knee_angle_acceleration | Δ(knee_angle_velocity) / Δt |
| hip_angle_velocity | Δ(primary_hip_angle) / Δt |
| hip_angle_acceleration | Δ(hip_angle_velocity) / Δt |

### Skeleton Structure

- normalized_world_landmarks: hip-centered 3D (33×3)
- bone_vectors: key bone direction unit vectors
- bone_lengths: normalized bone segment lengths
- pairwise_distance_subset: reduced joint distance set (configurable)
- joint_velocity_vectors: per-joint 3D velocity
- joint_acceleration_vectors: per-joint 3D acceleration

### Quality / Confidence

| Feature | Description |
|---------|-------------|
| landmark_visibility_mean | mean visibility across all 33 landmarks |
| lower_body_visibility | mean visibility of hip/knee/ankle landmarks |
| torso_visibility | mean visibility of shoulder/hip landmarks |
| frame_reliability_score | composite quality: visibility × stability × detection confidence |
| view_validity_score | how suitable current view is (front or side) |
| occlusion_risk_score | risk that key landmarks are occluded |

## Rounded Back Risk Subsystem

The `rounded_back_risk` feature is a composite proxy score (0.0-1.0) estimated from side-view landmarks. It does NOT claim to measure actual spinal curvature — it measures visible postural indicators correlated with back rounding.

**Components:**
1. **Trunk curl proxy**: deviation of mid-shoulder→mid-hip vector from the calibrated baseline angle. Higher deviation at bottom = higher risk.
2. **Head/neck drift**: forward displacement of nose/ear relative to shoulder line compared to calibrated standing. Excessive forward drift suggests upper back rounding.
3. **Mid-spine straightness proxy**: linearity of the shoulder→hip line compared to where a "mid-back" point would be expected. Uses shoulder and hip landmarks as endpoints; deviation of the torso midpoint from the straight line estimates curvature.

**Output:**
```python
@dataclass
class RoundedBackAssessment:
    risk_score: float       # 0.0-1.0
    confidence: float       # 0.0-1.0
    trunk_curl_component: float
    head_drift_component: float
    spine_linearity_component: float
    rationale: str          # human-readable explanation
    limitations: str        # "Estimated from surface landmarks only..."
```

**Limitations (documented in code):**
- Cannot detect actual vertebral flexion — surface proxy only
- Thick clothing or hair can affect landmark placement
- Works best from true lateral view; degrades at oblique angles
- Should not be used for medical assessment

## Calibration

At session start, user stands upright for ~2 seconds. System captures:
- Baseline torso inclination angle
- Neutral head-to-trunk position
- Dominant visible side (left/right)
- View type (front/side) from shoulder geometry
- Body scaling values (limb proportions from landmarks)
- Camera suitability check

Calibration personalizes the idealized reference: target depth is adjusted for estimated mobility, trunk baseline sets the "neutral" for scoring deviations, body scale normalizes distance-based features.

## Phase Detection & Rep Segmentation

**Phases:** TOP → DESCENT → BOTTOM → ASCENT → TOP

**Detection uses:**
- Fused temporal model phase probabilities (primary)
- Hip vertical position thresholds (secondary/fallback)
- Hysteresis bands to prevent oscillation
- Debounce: minimum phase duration (e.g. 150ms)
- Cooldown: minimum time between reps (e.g. 500ms)

**Rep segmenter emits:**
- rep_started (top→descent confirmed)
- rep_bottom_reached (bottom phase entered)
- rep_completed (ascent→top confirmed)
- phase durations per rep
- rep validity (rejected if too short, low confidence, or incomplete)

## Fault Detection

Structured evidence-based system, not if-else blocks.

**Fault types:**
| Fault | View | Detection Method |
|-------|------|-----------------|
| insufficient_depth | both | min knee angle vs target, hip-below-knee check |
| excessive_forward_lean | side | trunk angle vs calibrated baseline at bottom |
| rounded_back_risk | side | composite proxy (see subsystem above) |
| unstable_torso | both | trunk angle variance through rep |
| heel_fault | side | ankle/foot landmark displacement |
| knee_valgus | front | knee inward collapse angle |
| inconsistent_tempo | both | phase timing deviation from recent average |
| poor_trunk_control | both | trunk angle change bottom vs setup |
| low_confidence_assessment | both | when pose confidence too low to assess |
| invalid_view_setup | both | when view detection fails quality check |

**Each fault produces:**
```python
@dataclass
class FaultDetection:
    fault_type: str
    severity: float          # 0.0-1.0
    confidence: float        # 0.0-1.0
    evidence: list[str]      # what data supports this
    explanation_token: str   # short coaching cue key
    affects_overlay: bool
    affects_gemini: bool
```

**Evidence engine:**
- Aggregates multiple signals per fault
- Weighted rules with configurable thresholds (YAML)
- Confidence gating: suppress faults below confidence threshold
- Persistence tracking: fault must persist across frames/reps to trigger

## Scoring

### Score Components (all 0-100, per rep)

| Score | Based On |
|-------|----------|
| depth_score | min knee angle vs target, hip depth ratio |
| trunk_control_score | torso angle variance, max forward lean vs baseline |
| posture_stability_score | rounded_back_risk, head drift, trunk collapse at bottom |
| movement_consistency_score | velocity smoothness, phase timing symmetry, jerk |
| rep_quality_score | weighted combination of above 4 + model quality_score output (scaled ×100, weight 0.2) |
| overall_form_score | EMA of rep_quality_scores across session |

The model's `quality_score` output [0-1] contributes as one input (weight 0.2) to the `rep_quality_score`. The scoring engine's own decomposed scores (depth, trunk, posture, consistency) contribute the remaining 0.8. This way the model's learned quality estimate informs but does not dominate the explainable scoring pipeline.

### Rationale Object (per rep)

```json
{
  "rep_index": 4,
  "scores": {
    "depth_score": 76,
    "trunk_control_score": 64,
    "posture_stability_score": 68,
    "movement_consistency_score": 81,
    "rep_quality_score": 72,
    "overall_form_score": 74
  },
  "dominant_fault": "rounded_back_risk",
  "fault_evidence": "trunk inclination peaked at 42° (baseline: 25°), head drifted 8cm forward at bottom",
  "comparison_to_ideal": {
    "depth": "3° short of target",
    "trunk": "17° more forward lean than baseline",
    "tempo": "descent 0.3s faster than ascent"
  },
  "trend": "depth improving (+4 over last 3 reps), trunk control declining (-6)",
  "coaching_cue": "Keep your chest up at the bottom"
}
```

### Idealized Reference

Not a fixed "perfect human" constant. Personalized from calibration:
- Target depth: based on estimated mobility (calibrated standing angles)
- Expected trunk range: calibrated neutral ± configurable tolerance
- Expected phase timing: learned from first 2-3 reps as personal baseline
- Motion smoothness target: based on velocity profiles of "good" training data

Configurable via YAML. Scoring compares actual rep against this personalized reference.

## Coaching Arbitration

Selects the single most important cue for overlay and Gemini.

**Priority ranking:** severity × confidence × persistence × (1 / recency)

Where `recency` = seconds since this fault type was last displayed as a coaching cue. A fault shown 2 seconds ago has recency=2, so its priority is halved compared to one shown 4 seconds ago. Minimum recency clamped to 1.0 to avoid division by zero.

- Only one cue displayed at a time on overlay
- Suppress cues shown in last N seconds (configurable, default=5s)
- Suppress low-confidence faults
- Prioritize novel faults over repeated ones
- Gemini payload includes top-3 ranked cues with full rationale

## Output Channels

### Terminal Logging (continuous, throttled ~5Hz)

```
[12:04:03.142] FRAME 847 | phase=DESCENT | knee=112.3° | hip=98.1°
  | torso=28.4° | depth_ratio=0.72 | back_risk=0.31 | conf=0.91
[12:04:03.341] FRAME 849 | phase=BOTTOM  | knee=84.2° | hip=67.8°
  | torso=35.1° | depth_ratio=0.95 | back_risk=0.58 | conf=0.88

═══════════════════════════════════════════════════════════════
  REP 4 COMPLETE | quality=72 | depth=76 | trunk=64 |
  consistency=81 | posture=68
  FAULT: rounded_back_risk (severity=0.63, conf=0.78)
  CUE: "Keep your chest up at the bottom"
  TREND: depth ↑4 | trunk ↓6 | consistency ↑2
═══════════════════════════════════════════════════════════════
```

### Simple Video Overlay

- Skeleton lines drawn on detected pose (colored by confidence)
- Phase label top-left (e.g. "DESCENT")
- Rep count top-right (e.g. "Rep: 4")
- Current score bottom-left (e.g. "Score: 72")
- One coaching cue bottom-center (e.g. "Keep chest up")

### Gemini Payload (per rep, at rep_completed)

Structured semantic summary with all scores, faults, rationale, trend, and prioritized coaching cue. Formatted for natural language generation, not raw numbers.

### JSONL Session Log

One JSON line per rep summary, written to `sessions/{YYYY-MM-DD_HH-MM-SS}.jsonl` (path configurable in default.yaml). For replay analysis and debugging.

## Event System

Typed events emitted for:
- `calibration_complete` — baseline established
- `phase_transition` — phase changed (e.g. descent→bottom)
- `rep_started` — top→descent confirmed
- `rep_bottom_reached` — bottom phase entered
- `rep_completed` — ascent→top confirmed, triggers scoring + Gemini payload
- `fault_triggered` — fault detected with evidence
- `score_updated` — scores computed for completed rep
- `session_trend_updated` — rolling averages updated

All events are dataclasses/Pydantic models with timestamps and structured payloads.

## Project Structure

```
squat_coach/
  app.py                          # main application orchestrator
  main.py                         # CLI entry point
  config/
    default.yaml                  # general settings
    model.yaml                    # temporal model settings + ensemble weights
    scoring.yaml                  # score weights + thresholds
    overlay.yaml                  # overlay display settings
  camera/
    webcam_stream.py              # live webcam capture
    video_replay.py               # local video file replay
  pose/
    base.py                       # pose estimator interface
    mediapipe_blazepose3d.py      # MediaPipe Pose Landmarker implementation
    landmarks.py                  # landmark names, indices, constants
  preprocessing/
    smoothing.py                  # EMA landmark smoothing
    normalization.py              # hip-centered normalization
    sequence_buffer.py            # rolling window buffer
    calibration.py                # calibration flow + view detection
  biomechanics/
    angles.py                     # joint angle computation
    vectors.py                    # bone/trunk vector utilities
    distances.py                  # pairwise distance computation
    kinematics.py                 # velocity/acceleration from finite differences
    squat_features.py             # full feature extraction orchestrator
    posture_analysis.py           # rounded_back_risk + posture proxies
    side_view_constraints.py      # side-view specific features
    front_view_constraints.py     # front-view specific features
  phases/
    phase_detector.py             # phase classification from fused outputs
    rep_segmenter.py              # rep boundary detection + validation
    state_machine.py              # TOP→DESCENT→BOTTOM→ASCENT state machine
  models/
    temporal_base.py              # base class for all temporal models
    temporal_tcn.py               # TCN implementation
    temporal_gru.py               # GRU implementation
    temporal_bilstm.py            # BiLSTM implementation
    temporal_transformer.py       # Transformer encoder implementation
    temporal_stgcn_scaffold.py    # ST-GCN scaffold (extension point only, not trained or used in production)
    ensemble_fusion.py            # confidence-weighted ensemble fusion
    feature_tensor_builder.py     # build model input tensors from features
    inference_manager.py          # manage multi-model inference
    model_factory.py              # registry + factory for model creation
  training/
    dataset.py                    # PyTorch dataset for squat sequences
    data_pipeline.py              # download, preprocess, cache training data
    synthetic_generator.py        # synthetic squat sequence generator
    phase_labeler.py              # auto-label phases from hip kinematics
    trainer.py                    # unified training loop (MPS/CPU)
    evaluate.py                   # evaluation metrics + model comparison
    train_all.py                  # train all models end-to-end script
  scoring/
    ideal_reference.py            # personalized idealized squat reference
    score_components.py           # individual score computations
    score_fusion.py               # combine component scores
    rationale.py                  # rationale object builder
    trend_analysis.py             # rolling trend across reps
  faults/
    evidence_engine.py            # aggregate evidence for faults
    fault_rules.py                # weighted rule definitions
    fault_types.py                # fault type definitions + dataclasses
    confidence_gating.py          # suppress low-confidence faults
  rendering/
    overlay.py                    # simple overlay compositor
    draw_pose.py                  # skeleton drawing
    draw_metrics.py               # score/angle text rendering
    draw_feedback.py              # coaching cue rendering
  events/
    schemas.py                    # event dataclasses/models
    event_builder.py              # construct events from system state
    formatter.py                  # format events for logging
    gemini_payloads.py            # Gemini-ready compact payload formatter
    coaching_priority.py          # coaching cue arbitration
  session/
    session_state.py              # session-level state tracking
    rep_history.py                # per-rep history storage
    jsonl_logger.py               # JSONL session file writer
  utils/
    math_utils.py                 # angle computation, vector ops
    logging_utils.py              # structured logging setup
    timing.py                     # frame timing + FPS tracking
    enums.py                      # shared enums (Phase, ViewType, etc.)
  tests/
    test_angles.py
    test_features.py
    test_phase_detector.py
    test_scoring.py
    test_ensemble.py
    test_synthetic.py
requirements.txt
README.md
```

## Config System

YAML files for all tunable parameters:

- **default.yaml**: camera settings, FPS, debug mode, log throttle rate, device
- **model.yaml**: model architectures (hidden dims, layers, dropout), ensemble weights, which models enabled, sequence length, feature count
- **scoring.yaml**: score weights, fault thresholds, calibration tolerances, idealized reference defaults, trend smoothing
- **overlay.yaml**: display settings, cue display duration, colors, font sizes

## Modes

- `python -m squat_coach --mode webcam` — live analysis
- `python -m squat_coach --mode replay --video path/to/video.mp4` — replay mode
- `python -m squat_coach --mode train` — run training pipeline
- `--debug` flag enables verbose terminal logging + additional overlay info
- `--log-features` flag enables per-frame JSONL feature logging

## Extension Points

1. **New exercises**: Add feature extractors and fault rules per exercise; temporal models and scoring are exercise-agnostic
2. **Gemini Live integration**: Replace placeholder in `gemini_payloads.py` with actual API calls
3. **New temporal models**: Implement `TemporalModelBase`, register in factory
4. **New faults**: Add fault type + rule + evidence in `faults/`
5. **New views**: Add view-specific feature module (e.g. rear view)
6. **Real training data**: Drop labeled sequences into the data pipeline, retrain
7. **ST-GCN expansion**: Fill in the scaffold with actual graph convolution layers
