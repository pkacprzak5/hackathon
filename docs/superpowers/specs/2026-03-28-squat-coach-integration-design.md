# Squat Coach Integration Design

**Date:** 2026-03-28
**Goal:** Merge the `squat_coach` Python backend with the Next.js `frontend` so users can connect from a phone, get real-time AI-powered squat coaching with video overlay.

---

## Architecture Overview

```
PHONE (Browser)                         SERVER (FastAPI)
┌─────────────────────────┐             ┌─────────────────────────┐
│ getUserMedia → <video>  │             │ WebSocket endpoint      │
│ (native fps display)    │             │   /ws/session           │
│                         │             │                         │
│ Canvas encoder ─────────┼── JPEG ────►│ Frame Decoder           │
│ (640x480, q=0.7)        │  (binary)   │   (JPEG → numpy)       │
│                         │             │                         │
│                         │             │ SquatCoachPipeline      │
│                         │             │   MediaPipe → Features  │
│ Overlay <canvas> ◄──────┼── JSON ◄───│   → Models → Phase     │
│ React HUD components    │  (24fps)    │   → Faults → Scoring   │
│                         │             │                         │
│ Web Speech API ◄────────┼── text ◄───│ Gemini async coaching   │
└─────────────────────────┘             └─────────────────────────┘
         Single WebSocket connection
```

### Key Decisions

- **Single WebSocket** carries binary JPEG frames upstream, JSON data downstream
- **Video displays locally** at device-native framerate — zero display latency
- **Server processes every frame** at 24fps, sends data back at 24fps
- **Delta compression** reduces payload size (not frequency) — every frame gets a response, but only changed fields are included
- **Frontend interpolates** between 24fps data and 60fps canvas rendering (16ms gap)
- **TTS via Web Speech API** in browser — server sends coaching text only
- **Existing frontend pages** (coach, multi, history, leaderboard, profile) stay untouched for future use
- **No multi-connection support** in this phase — single player solo session only

---

## Server-Side Design

### New Module: `squat_coach/server/`

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, CORS config, lifespan startup/shutdown |
| `ws_handler.py` | WebSocket endpoint — frame receive loop, response sender |
| `pipeline.py` | `SquatCoachPipeline` — headless version of `SquatCoachApp` (no camera, no OpenCV window, no rendering) |
| `delta.py` | Delta compressor — compares current vs last-sent state, strips unchanged fields |
| `protocol.py` | Pydantic message schemas for all JSON message types |

### Pipeline Refactor

Extract processing logic from `SquatCoachApp` into a headless `SquatCoachPipeline`:

```python
class SquatCoachPipeline:
    """Headless squat analysis — no camera, no display."""

    def __init__(self, config: dict):
        self.pose_estimator = BlazePose3D()
        self.smoother = EMASmoother(alpha=config["preprocessing"]["ema_alpha"])
        self.calibrator = Calibrator(duration=config["preprocessing"]["calibration_duration_s"])
        self.normalizer = Normalizer()
        self.feature_extractor = SquatFeatureExtractor()
        self.sequence_buffer = SequenceBuffer(seq_len=60)
        self.inference_manager = InferenceManager(config)
        self.phase_detector = None       # initialized after calibration
        self.rep_segmenter = None        # initialized after calibration
        self.fault_engine = EvidenceEngine()
        self.scorer = RepScorer(config)
        self.session_state = SessionState()
        self.gemini_client = GeminiClient(config)  # async

    def process_frame(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        """Process one BGR frame, return structured result."""
        # 1. Pose estimation
        # 2. Smoothing + normalization
        # 3. Calibration (if not calibrated)
        # 4. Feature extraction (42-D vector)
        # 5. Sequence buffer → model inference
        # 6. Phase detection → rep segmentation
        # 7. Fault detection + scoring (on rep complete)
        # 8. Return FrameResult dataclass
```

### FrameResult Dataclass

```python
@dataclass
class FrameResult:
    seq: int
    timestamp: float

    # Calibration
    calibration_status: str          # "in_progress" | "complete" | None
    calibration_progress: float      # 0.0 - 1.0
    view_type: str | None            # "side" | "front" (after calibration)

    # Per-frame analysis
    landmarks: list[list[float]] | None   # 33 x [x, y, z, visibility] (image landmarks for overlay)
    phase: str | None                      # "TOP" | "DESCENT" | "BOTTOM" | "ASCENT"
    knee_angle: float | None
    hip_angle: float | None
    torso_angle: float | None
    score: float | None
    confidence: float | None

    # Rep events (only on rep completion)
    rep: RepData | None

    # Coaching (async, may arrive later)
    coaching_text: str | None

@dataclass
class RepData:
    rep_index: int
    scores: dict        # {total, depth, trunk_control, posture_stability, movement_consistency}
    faults: list[str]   # fault type names
    coaching_text: str   # immediate coaching cue from rationale builder
```

### WebSocket Protocol

**Client → Server:** Raw binary JPEG bytes. No JSON wrapper, no headers. Each WebSocket message = one frame.

**Server → Client:** JSON messages, one per processed frame, delta-compressed:

```json
// During calibration
{"type": "calibration", "status": "in_progress", "progress": 0.65}
{"type": "calibration", "status": "complete", "view_type": "side"}

// Per-frame analysis (delta — only changed fields included)
{"type": "frame", "seq": 142, "data": {
  "landmarks": [[0.52, 0.31, -0.1, 0.98], ...],
  "phase": "DESCENT",
  "knee_angle": 95.2,
  "hip_angle": 78.1,
  "torso_angle": 12.5,
  "score": 82,
  "confidence": 0.92
}}

// Rep completed (in addition to frame data)
{"type": "rep", "rep_index": 3, "scores": {
  "total": 85, "depth": 90, "trunk_control": 80,
  "posture_stability": 85, "movement_consistency": 85
}, "faults": ["INSUFFICIENT_DEPTH"],
   "coaching_text": "Almost parallel! Push those hips back another inch."}

// Gemini coaching (async, arrives when ready)
{"type": "coaching", "text": "Nice depth on that one! Keep your chest proud."}

// Session end
{"type": "session_end", "total_reps": 12, "avg_score": 81, "trend": "improving"}
```

### Delta Compression

Delta reduces payload size, NOT frequency. Every frame gets a response.

```python
class DeltaCompressor:
    THRESHOLDS = {
        "knee_angle": 1.0,       # degrees
        "hip_angle": 1.0,
        "torso_angle": 1.0,
        "score": 1,              # points
        "confidence": 0.05,
    }

    def compress(self, current: FrameResult, last_sent: dict) -> dict:
        delta = {"seq": current.seq}

        # Phase: send on change
        if current.phase != last_sent.get("phase"):
            delta["phase"] = current.phase

        # Numeric fields: send if changed beyond threshold
        for field, threshold in self.THRESHOLDS.items():
            val = getattr(current, field)
            if val is not None and abs(val - last_sent.get(field, 0)) > threshold:
                delta[field] = round(val, 1)

        # Landmarks: send if any landmark moved > 0.005 in normalized coords
        if self._landmarks_changed(current.landmarks, last_sent.get("landmarks")):
            delta["landmarks"] = current.landmarks

        return delta
```

### WebSocket Handler

```python
@app.websocket("/ws/session")
async def session_ws(websocket: WebSocket):
    await websocket.accept()
    pipeline = SquatCoachPipeline(config)
    delta = DeltaCompressor()
    last_sent = {}
    seq = 0

    try:
        while True:
            # Receive binary JPEG frame
            jpeg_bytes = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            timestamp = time.time()

            # Process
            result = pipeline.process_frame(frame, timestamp)
            result.seq = seq
            seq += 1

            # Handle calibration messages
            if result.calibration_status:
                await websocket.send_json({
                    "type": "calibration",
                    "status": result.calibration_status,
                    "progress": result.calibration_progress,
                    **({"view_type": result.view_type} if result.view_type else {})
                })
                continue

            # Send frame data (delta-compressed)
            compressed = delta.compress(result, last_sent)
            last_sent.update(compressed)
            await websocket.send_json({"type": "frame", "data": compressed})

            # Send rep event if completed
            if result.rep:
                await websocket.send_json({
                    "type": "rep",
                    "rep_index": result.rep.rep_index,
                    "scores": result.rep.scores,
                    "faults": result.rep.faults,
                    "coaching_text": result.rep.coaching_text
                })

            # Coaching text (from Gemini, async)
            if result.coaching_text:
                await websocket.send_json({
                    "type": "coaching",
                    "text": result.coaching_text
                })
    except WebSocketDisconnect:
        pipeline.cleanup()
```

---

## Frontend-Side Design

### Session Flow (User Journey)

1. Player opens Solo Session → selects Squats → hits "Start"
2. Browser requests camera permission (`getUserMedia`, rear-facing on mobile)
3. Local `<video>` element displays camera at device-native framerate (immediate, no server dependency)
4. WebSocket connects to `NEXT_PUBLIC_ANALYSIS_WS_URL` (`ws://server:8000/ws/session`)
5. **Calibration phase**: overlay shows "Stand still for calibration..." with progress bar
6. **Active session**: canvas draws skeleton + angle arcs over video; React HUD shows phase/score/reps
7. On `rep` message: Web Speech API speaks `coaching_text`
8. Player taps "End Session" → navigate to Results page with accumulated rep data

### Component Architecture

```
SoloSessionPage
├── VideoContainer
│   ├── <video>              (local camera feed, device-native fps, fills container)
│   ├── <canvas>             (overlay: skeleton lines, landmark dots, angle arcs)
│   └── CalibrationOverlay   (progress bar + instruction text, shown during calibration)
├── SessionHUD
│   ├── PhaseIndicator       (TOP / DESCENT / BOTTOM / ASCENT badge)
│   ├── RepCounter           (rep count display)
│   ├── ScoreDisplay         (current score, trend arrow)
│   └── CoachingBanner       (latest coaching text, fades after 3s)
├── FaultAlerts              (toast notifications for detected faults)
└── SessionControls          (end session button)
```

**Canvas vs React split:**
- **Canvas**: skeleton, landmark dots, angle arcs — pixel-positioned over video
- **React components**: phase label, rep count, score, coaching text, faults — standard UI elements

### Core Hook: `useSquatSession`

```typescript
interface SquatSessionState {
  status: 'idle' | 'connecting' | 'calibrating' | 'active' | 'ended';
  calibrationProgress: number;
  landmarks: Landmark[] | null;        // 33 landmarks with x, y, z, visibility
  phase: 'TOP' | 'DESCENT' | 'BOTTOM' | 'ASCENT';
  angles: { knee: number; hip: number; torso: number };
  score: number;
  confidence: number;
  repCount: number;
  reps: RepResult[];
  currentFaults: string[];
  coachingText: string | null;
}

function useSquatSession(): {
  state: SquatSessionState;
  videoRef: RefObject<HTMLVideoElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
  startSession: () => void;
  endSession: () => void;
}
```

Internally manages:
- WebSocket connection lifecycle
- Frame capture loop (24fps `setInterval` at 42ms)
- Message handling (merges deltas into accumulated state)
- Interpolation targets for canvas rendering
- TTS playback on rep/coaching messages

### Frame Capture (Phone → Server)

```typescript
const CAPTURE_INTERVAL = 42; // ms, ~24fps
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

function startCapture(video: HTMLVideoElement, ws: WebSocket) {
  const offscreen = document.createElement('canvas');
  offscreen.width = CAPTURE_WIDTH;
  offscreen.height = CAPTURE_HEIGHT;
  const ctx = offscreen.getContext('2d')!;

  return setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
    offscreen.toBlob(
      (blob) => { if (blob) ws.send(blob); },
      'image/jpeg',
      JPEG_QUALITY
    );
  }, CAPTURE_INTERVAL);
}
```

- Downscales to 640x480 for bandwidth (~30-50KB/frame at q=0.7, ~0.8MB/s upload)
- MediaPipe works well at this resolution
- Uses offscreen canvas (not the overlay canvas)

### Message Handler (Server → Phone)

```typescript
function handleMessage(event: MessageEvent, state: MutableRefObject<AccumulatedState>) {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case 'calibration':
      state.current.status = msg.status === 'complete' ? 'active' : 'calibrating';
      state.current.calibrationProgress = msg.progress;
      break;

    case 'frame':
      // Merge delta into accumulated state
      const data = msg.data;
      if (data.landmarks) state.current.landmarks = data.landmarks;
      if (data.phase) state.current.phase = data.phase;
      if (data.knee_angle !== undefined) state.current.angles.knee = data.knee_angle;
      if (data.hip_angle !== undefined) state.current.angles.hip = data.hip_angle;
      if (data.torso_angle !== undefined) state.current.angles.torso = data.torso_angle;
      if (data.score !== undefined) state.current.score = data.score;
      if (data.confidence !== undefined) state.current.confidence = data.confidence;
      // Update interpolation targets
      state.current.prevLandmarks = state.current.currentLandmarks;
      state.current.currentLandmarks = state.current.landmarks;
      state.current.lastUpdateTime = performance.now();
      break;

    case 'rep':
      state.current.repCount = msg.rep_index;
      state.current.reps.push(msg);
      state.current.currentFaults = msg.faults;
      speakCoaching(msg.coaching_text);
      break;

    case 'coaching':
      state.current.coachingText = msg.text;
      speakCoaching(msg.text);
      break;

    case 'session_end':
      state.current.status = 'ended';
      state.current.sessionSummary = msg;
      break;
  }
}
```

### Canvas Overlay Rendering

```typescript
function renderOverlay(
  ctx: CanvasRenderingContext2D,
  state: AccumulatedState,
  width: number,
  height: number
) {
  ctx.clearRect(0, 0, width, height);
  if (!state.currentLandmarks) return;

  // Interpolate landmarks for smooth 60fps rendering
  const t = Math.min((performance.now() - state.lastUpdateTime) / 42, 1);
  const landmarks = state.prevLandmarks
    ? lerpLandmarks(state.prevLandmarks, state.currentLandmarks, t)
    : state.currentLandmarks;

  // Draw skeleton connections
  for (const [i, j] of SKELETON_CONNECTIONS) {
    const a = landmarks[i], b = landmarks[j];
    if (a.visibility < 0.5 || b.visibility < 0.5) continue;
    ctx.strokeStyle = visibilityColor(Math.min(a.visibility, b.visibility));
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  }

  // Draw landmark dots
  for (const lm of landmarks) {
    if (lm.visibility < 0.5) continue;
    ctx.fillStyle = visibilityColor(lm.visibility);
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw angle arcs at knee/hip joints
  drawAngleArc(ctx, landmarks, 'knee', state.angles.knee, width, height);
  drawAngleArc(ctx, landmarks, 'hip', state.angles.hip, width, height);
}
```

Interpolation between 24fps server data and 60fps display — only a 16ms gap to bridge:

```typescript
function lerpLandmarks(from: Landmark[], to: Landmark[], t: number): Landmark[] {
  return from.map((f, i) => ({
    x: f.x + (to[i].x - f.x) * t,
    y: f.y + (to[i].y - f.y) * t,
    z: f.z + (to[i].z - f.z) * t,
    visibility: to[i].visibility,
  }));
}
```

### TTS Integration

```typescript
function speakCoaching(text: string) {
  if (!('speechSynthesis' in window)) return;
  speechSynthesis.cancel(); // stop any current speech
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.1;
  utterance.pitch = 1.0;
  speechSynthesis.speak(utterance);
}
```

Called on `rep` and `coaching` message types.

### Integration with Existing Frontend

- **Dedicated WebSocket**: `useSquatSession` connects to `NEXT_PUBLIC_ANALYSIS_WS_URL` (the FastAPI server). This is separate from the existing `WebSocketProvider` which remains for future multi-user features.
- **SessionProvider bridge**: on rep completion and session end, `useSquatSession` dispatches into the existing `SessionProvider` reducer (`PARTICIPANT_UPDATE`, `SESSION_END`) so data flows to Results, History, and future Coach pages.
- **Existing pages untouched**: coach dashboard, multi session, history, leaderboard, profile — all preserved for future implementation.
- **Solo session page**: updated to use `useSquatSession` hook + new overlay components.
- **Results page**: receives accumulated rep data from session state after session ends.

### Environment Variables

```
NEXT_PUBLIC_ANALYSIS_WS_URL=ws://localhost:8000/ws/session
```

Falls back to existing mock behavior if not set.

---

## Calibration UX

1. WebSocket connects → server creates pipeline
2. Server waits for frames to start arriving
3. First frame received → server starts 2-second calibration timer
4. Server sends `{"type": "calibration", "status": "in_progress", "progress": 0.0}` ... `0.5` ... `1.0`
5. Calibration complete → server sends `{"type": "calibration", "status": "complete", "view_type": "side"}`
6. Frontend transitions from CalibrationOverlay to active session UI
7. User must stand still during calibration — overlay shows instruction text + progress bar

---

## File Changes Summary

### New Files (Server)
- `squat_coach/server/__init__.py`
- `squat_coach/server/main.py` — FastAPI app
- `squat_coach/server/ws_handler.py` — WebSocket endpoint
- `squat_coach/server/pipeline.py` — `SquatCoachPipeline` headless processor
- `squat_coach/server/delta.py` — Delta compressor
- `squat_coach/server/protocol.py` — Message schemas

### Modified Files (Backend)
- `squat_coach/app.py` — extract shared logic into pipeline (or keep as-is for CLI mode, pipeline is new class)
- `squat_coach/requirements.txt` — add `fastapi`, `uvicorn[standard]`

### New Files (Frontend)
- `frontend/hooks/use-squat-session.ts` — core session hook
- `frontend/components/session/overlay-canvas.tsx` — skeleton + angle rendering
- `frontend/components/session/calibration-overlay.tsx` — calibration UI
- `frontend/components/session/session-hud.tsx` — phase/score/rep display
- `frontend/components/session/fault-alerts.tsx` — fault toast notifications
- `frontend/components/session/coaching-banner.tsx` — coaching text display
- `frontend/lib/skeleton.ts` — skeleton connection definitions, drawing utilities
- `frontend/lib/interpolation.ts` — landmark/angle lerp functions
- `frontend/lib/tts.ts` — Web Speech API wrapper

### Modified Files (Frontend)
- `frontend/app/(player)/session/solo/page.tsx` — integrate `useSquatSession` + new components
- `frontend/lib/types.ts` — add Landmark, FrameData, RepResult types
- `frontend/.env.local` — add `NEXT_PUBLIC_ANALYSIS_WS_URL`

### Untouched
- All coach, multi, history, leaderboard, profile, replay pages
- Existing providers (SessionProvider, WebSocketProvider, FishjamProvider, ThemeProvider)
- Existing components not related to solo session

---

## Running the System

### Server
```bash
cd squat_coach
pip install -r requirements.txt
python -m uvicorn squat_coach.server.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
echo "NEXT_PUBLIC_ANALYSIS_WS_URL=ws://<server-ip>:8000/ws/session" > .env.local
pnpm dev
```

### Phone Access
Phone connects to the Next.js dev server on the local network (e.g., `http://192.168.1.x:3000`). Camera permissions required. Server must be reachable from the phone's network.
