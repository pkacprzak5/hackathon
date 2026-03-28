# GymAI Backend API Specification

**Date:** 2026-03-28
**Status:** Draft
**Purpose:** Define the REST API and WebSocket protocol the frontend expects, based on current frontend types, mock data, and integration points.

---

## Table of Contents

1. [Overview](#1-overview)
2. [REST API — Sessions & Data](#2-rest-api--sessions--data)
3. [WebSocket Protocol — Real-Time Events](#3-websocket-protocol--real-time-events)
4. [Video Streaming — Fishjam/Smelter Integration](#4-video-streaming--fishjamsmelter-integration)
5. [Data Models](#5-data-models)
6. [Environment & Configuration](#6-environment--configuration)

---

## 1. Overview

The GymAI frontend currently operates in **full mock mode** with no backend. It expects two real-time channels:

| Channel | Transport | Purpose |
|---------|-----------|---------|
| Metadata | WebSocket | Session state, participant scores, joint angles, AI insights, rep counts |
| Video | Fishjam (WebRTC) | Pre-composited camera feeds with skeleton overlays from Smelter |

Additionally, a REST API is needed for data persistence (session history, replays, leaderboard, user profiles) that the frontend currently renders from hardcoded mock values.

### Auth Model

No authentication for MVP. Users identify by `username` only. One shared room for the entire app.

---

## 2. REST API — Sessions & Data

Base URL: `NEXT_PUBLIC_API_URL` (e.g., `http://localhost:3001/api`)

All endpoints return JSON. Errors use standard HTTP status codes with `{ "error": string }` body.

---

### 2.1 Users

#### `POST /api/users/join`

Register a user for the current session. Called when a user enters their username on the entry screen.

**Request:**
```json
{
  "username": "Alex",
  "role": "player" | "coach"
}
```

**Response (200):**
```json
{
  "id": "usr_abc123",
  "username": "Alex",
  "role": "player",
  "createdAt": "2026-03-28T14:30:00Z"
}
```

**Errors:**
- `409` — Username already taken in active session

---

### 2.2 Sessions

#### `GET /api/sessions/current`

Get the current active session state (or `null` if idle).

**Response (200):**
```json
{
  "id": "sess_xyz789",
  "status": "active",
  "exercise": "Squat",
  "currentSet": 2,
  "totalSets": 3,
  "currentRep": 8,
  "totalReps": 10,
  "sessionTime": 1475,
  "participants": [ /* Participant[] */ ],
  "startedAt": "2026-03-28T14:00:00Z"
}
```

#### `POST /api/sessions/start`

Start a new exercise session (coach or system triggered).

**Request:**
```json
{
  "exercise": "Squat",
  "totalSets": 3,
  "totalReps": 10
}
```

**Response (201):**
```json
{
  "id": "sess_xyz789",
  "status": "active",
  "exercise": "Squat",
  "currentSet": 1,
  "totalSets": 3,
  "currentRep": 0,
  "totalReps": 10,
  "sessionTime": 0,
  "participants": [],
  "startedAt": "2026-03-28T14:00:00Z"
}
```

#### `POST /api/sessions/:sessionId/end`

End the active session.

**Response (200):**
```json
{
  "id": "sess_xyz789",
  "status": "ended",
  "exercise": "Squat",
  "sessionTime": 1820,
  "endedAt": "2026-03-28T14:30:20Z",
  "summary": {
    "totalParticipants": 4,
    "averageScore": 87.5,
    "topPerformer": "Jordan"
  }
}
```

---

### 2.3 Session History

#### `GET /api/sessions/history`

Get past sessions for the history screen. Supports pagination.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `username` | string | required | Filter by participant |
| `limit` | number | 20 | Max results |
| `offset` | number | 0 | Pagination offset |
| `from` | ISO date | — | Start date filter |
| `to` | ISO date | — | End date filter |

**Response (200):**
```json
{
  "sessions": [
    {
      "id": "sess_abc",
      "exercise": "Squat",
      "score": 94,
      "reps": 30,
      "duration": 1820,
      "date": "2026-03-28T14:00:00Z",
      "insights": {
        "warnings": 2,
        "successes": 5,
        "tips": 1
      }
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

#### `GET /api/sessions/history/weekly`

Get weekly aggregated stats for the history chart.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `username` | string | required | Filter by participant |
| `weeks` | number | 4 | Number of weeks to return |

**Response (200):**
```json
{
  "weeks": [
    {
      "weekStart": "2026-03-23",
      "sessionsCount": 5,
      "averageScore": 88,
      "totalReps": 150,
      "totalDuration": 7200,
      "topExercise": "Squat"
    }
  ]
}
```

---

### 2.4 Replays

#### `GET /api/replays/:sessionId`

Get replay data for a completed session, including time-series joint angle data for the angle chart.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `participantId` | string | — | Filter to specific participant (optional) |

**Response (200):**
```json
{
  "sessionId": "sess_xyz789",
  "exercise": "Squat",
  "duration": 1820,
  "date": "2026-03-28T14:00:00Z",
  "videoUrl": "https://storage.example.com/replays/sess_xyz789.mp4",
  "participants": ["Alex", "Sam", "Dana", "Jordan"],
  "angleTimeSeries": [
    {
      "timestamp": 0,
      "knee": 90,
      "back": 15,
      "hip": 105
    },
    {
      "timestamp": 1,
      "knee": 88,
      "back": 18,
      "hip": 102
    }
  ],
  "scoreTimeSeries": [
    {
      "timestamp": 0,
      "score": 92
    }
  ],
  "insights": [
    {
      "participantId": "1",
      "type": "warning",
      "title": "Back Posture",
      "message": "Your back angle exceeded 35° during reps 5-7",
      "timestamp": 845
    }
  ],
  "summary": {
    "finalScore": 94,
    "totalReps": 30,
    "averageKneeAngle": 91,
    "averageBackAngle": 17,
    "averageHipAngle": 104
  }
}
```

#### `GET /api/replays/:sessionId/video`

Stream replay video. Returns the pre-composited video (with skeleton overlay baked in by Smelter).

**Response:** `200` with `Content-Type: video/mp4`, streamed.

---

### 2.5 Leaderboard

#### `GET /api/leaderboard`

Get ranked participant list.

**Query params:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `exercise` | string | — | Filter by exercise (optional) |
| `period` | string | `all` | `daily`, `weekly`, `monthly`, `all` |
| `limit` | number | 20 | Max results |

**Response (200):**
```json
{
  "rankings": [
    {
      "rank": 1,
      "username": "Jordan",
      "score": 91,
      "sessionsCount": 12,
      "totalReps": 360,
      "streak": 5
    },
    {
      "rank": 2,
      "username": "Alex",
      "score": 94,
      "sessionsCount": 10,
      "totalReps": 300,
      "streak": 3
    }
  ],
  "period": "weekly",
  "total": 15
}
```

---

### 2.6 Profile

#### `GET /api/users/:username/profile`

Get user profile and aggregate stats.

**Response (200):**
```json
{
  "username": "Alex",
  "role": "player",
  "joinedAt": "2026-03-20T10:00:00Z",
  "stats": {
    "totalSessions": 15,
    "averageScore": 91,
    "totalReps": 450,
    "totalDuration": 27300,
    "favoriteExercise": "Squat",
    "currentStreak": 3,
    "bestScore": 98
  },
  "recentSessions": [
    {
      "id": "sess_abc",
      "exercise": "Squat",
      "score": 94,
      "date": "2026-03-28T14:00:00Z"
    }
  ]
}
```

---

### 2.7 Exercises

#### `GET /api/exercises`

List available exercises for the home screen exercise cards.

**Response (200):**
```json
{
  "exercises": [
    {
      "id": "squat",
      "name": "Squat",
      "difficulty": "intermediate",
      "muscleGroups": ["quadriceps", "glutes", "hamstrings"],
      "description": "Standard barbell back squat",
      "thumbnailUrl": "/exercises/squat.png"
    },
    {
      "id": "deadlift",
      "name": "Deadlift",
      "difficulty": "advanced",
      "muscleGroups": ["back", "hamstrings", "glutes"],
      "description": "Conventional deadlift",
      "thumbnailUrl": "/exercises/deadlift.png"
    }
  ]
}
```

---

## 3. WebSocket Protocol — Real-Time Events

**Endpoint:** `NEXT_PUBLIC_WS_URL` (e.g., `ws://localhost:8080`)

The frontend's `WebSocketClient` class connects and exchanges JSON messages.

### 3.1 Connection Lifecycle

```
Client                          Server
  |                               |
  |--- ws connect --------------->|
  |--- { type: "join", ... } ---->|   (sent immediately after open)
  |                               |
  |<-- { type: "session_update" } |   (current state snapshot)
  |<-- { type: "participant_join" }|  (one per existing participant)
  |<-- { type: "insight" } -------|   (recent insights)
  |                               |
  |<-- real-time events --------->|   (continuous)
  |                               |
  |--- disconnect --------------->|
```

### 3.2 Client -> Server Messages

#### `join`
Sent immediately after WebSocket connection opens.

```json
{
  "type": "join",
  "username": "Alex",
  "role": "player"
}
```

The server should respond with the current session snapshot (session_update + participant_join for all existing participants + recent insights).

### 3.3 Server -> Client Messages

#### `session_update`
Broadcast when exercise parameters or timer change.

```json
{
  "type": "session_update",
  "exercise": "Squat",
  "currentSet": 2,
  "totalSets": 3,
  "currentRep": 8,
  "totalReps": 10,
  "sessionTime": 1475
}
```

**Frequency:** `sessionTime` increments every 1 second. Other fields change on rep/set transitions.

#### `participant_update`
Broadcast when a participant's real-time metrics change (from AI analysis pipeline).

```json
{
  "type": "participant_update",
  "participant": {
    "id": "1",
    "username": "Alex",
    "score": 94,
    "jointAngles": {
      "knee": 92,
      "back": 18,
      "hip": 105
    },
    "repCount": 8,
    "status": "excellent"
  }
}
```

**Field details:**
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `score` | number | 0–100 | Form quality score from AI |
| `jointAngles.knee` | number | degrees | Knee bend angle |
| `jointAngles.back` | number | degrees | Back lean angle (lower = better, >35 = warning) |
| `jointAngles.hip` | number | degrees | Hip hinge angle |
| `repCount` | number | 0+ | Completed reps in current set |
| `status` | string | "active" / "warning" / "excellent" | AI-determined form quality tier |

**Frequency:** Updated every ~2 seconds per participant based on video analysis pipeline output. Score varies by ±2, angles by ±3 degrees between updates.

#### `participant_join`
Broadcast when a new participant enters the room.

```json
{
  "type": "participant_join",
  "participant": {
    "id": "2",
    "username": "Sam",
    "score": 0,
    "jointAngles": { "knee": 0, "back": 0, "hip": 0 },
    "repCount": 0,
    "status": "active"
  }
}
```

#### `participant_leave`
Broadcast when a participant disconnects.

```json
{
  "type": "participant_leave",
  "participantId": "2"
}
```

#### `insight`
Broadcast when the AI analysis generates feedback for a participant.

```json
{
  "type": "insight",
  "insight": {
    "participantId": "2",
    "type": "warning",
    "title": "Back Posture",
    "message": "Try to keep your back more upright. Current angle suggests too much forward lean.",
    "timestamp": 1475
  }
}
```

**Insight types and their UI treatment:**
| Type | Color | Icon | When generated |
|------|-------|------|----------------|
| `warning` | Red (#EF4444) | Alert triangle | Back angle > 35°, knee angle < 80°, form degradation |
| `success` | Green (#22C55E) | Check circle | Consistent good form, score > 90 |
| `tip` | Amber (#F59E0B) | Lightbulb | Improvement suggestions, technique reminders |

**Note:** The frontend caps stored insights at 50. Older insights are dropped.

#### `session_end`
Broadcast when the session is terminated.

```json
{
  "type": "session_end"
}
```

The frontend sets `sessionStatus` to `"ended"` and navigates to the results screen.

---

## 4. Video Streaming — Fishjam/Smelter Integration

### 4.1 Architecture

```
Camera (participant device)
  │
  ▼
Fishjam Room (WebRTC SFU)
  │
  ├──▶ Smelter (composites skeleton overlay onto video)
  │       │
  │       ▼
  │    Composited stream (video + skeleton) sent back to Fishjam
  │
  ▼
Frontend receives composited streams via Fishjam JS SDK
```

The frontend does **not** do any pose estimation or skeleton rendering — it receives pre-composited video with overlays already baked in.

### 4.2 Fishjam Connection

**Environment variables:**
```
NEXT_PUBLIC_FISHJAM_URL=ws://localhost:5002
NEXT_PUBLIC_FISHJAM_TOKEN=<room-token>
```

**Expected SDK usage** (to be implemented in `providers/fishjam-provider.tsx`):

```typescript
// Fishjam client connects to room
// Each peer in the room = one participant's composited video stream
// FishjamProvider populates: Map<participantId, MediaStream>
// VideoFeed component renders MediaStream into <video> element
```

### 4.3 Room Token API

The backend needs to provide Fishjam room tokens to the frontend.

#### `POST /api/video/token`

Request a Fishjam room token for the current user.

**Request:**
```json
{
  "username": "Alex",
  "role": "player"
}
```

**Response (200):**
```json
{
  "token": "eyJhbGciOi...",
  "fishjamUrl": "ws://fishjam.example.com:5002",
  "roomId": "gym-main-room"
}
```

### 4.4 Video Feed Behavior

The `VideoFeed` component handles two states:

1. **Stream available** — renders `<video autoPlay playsInline muted>` with the MediaStream
2. **No stream** — renders a dark placeholder (`#1A1A2E`) with an SVG skeleton silhouette, colored per participant:
   - Participant 1: `#8B5CF6` (purple)
   - Participant 2: `#F472B6` (pink)
   - Participant 3: `#06B6D4` (cyan)
   - Participant 4: `#14B8A6` (teal)

Each feed also shows:
- Participant name (bottom-left badge)
- Form score (top-right, when provided)

### 4.5 Replay Video Storage

Completed sessions should have their composited video stored for replay. The backend should:

1. Record the Smelter-composited stream during the session
2. Store the recording (e.g., S3/object storage)
3. Serve it via `GET /api/replays/:sessionId/video` as `video/mp4`

---

## 5. Data Models

### 5.1 Core Types (matching frontend `lib/types.ts`)

```typescript
interface JointAngles {
  knee: number;   // degrees, 0-180
  back: number;   // degrees, 0-90 (lower is better, >35 = warning)
  hip: number;    // degrees, 0-180
}

interface Participant {
  id: string;
  username: string;
  score: number;          // 0-100, AI form quality score
  jointAngles: JointAngles;
  repCount: number;
  status: "active" | "warning" | "excellent";
}

interface Insight {
  participantId: string;
  type: "warning" | "success" | "tip";
  title: string;
  message: string;
  timestamp: number;      // seconds into session
}

interface SessionState {
  username: string;
  role: "player" | "coach";
  sessionStatus: "idle" | "active" | "ended";
  exercise: string;
  currentSet: number;
  totalSets: number;
  currentRep: number;
  totalReps: number;
  sessionTime: number;    // seconds since session start
  participants: Participant[];
  insights: Insight[];    // capped at 50 on frontend
}
```

### 5.2 Backend-Only Models (suggested)

```typescript
interface Session {
  id: string;             // "sess_xyz789"
  status: "active" | "ended";
  exercise: string;
  totalSets: number;
  totalReps: number;
  startedAt: string;      // ISO 8601
  endedAt?: string;       // ISO 8601
  participants: string[]; // participant IDs
}

interface SessionRecording {
  sessionId: string;
  videoUrl: string;
  angleTimeSeries: AngleDataPoint[];
  scoreTimeSeries: ScoreDataPoint[];
  insights: Insight[];
}

interface AngleDataPoint {
  timestamp: number;      // seconds into session
  knee: number;
  back: number;
  hip: number;
}

interface ScoreDataPoint {
  timestamp: number;
  score: number;
}

interface UserProfile {
  username: string;
  role: "player" | "coach";
  joinedAt: string;       // ISO 8601
  stats: UserStats;
}

interface UserStats {
  totalSessions: number;
  averageScore: number;
  totalReps: number;
  totalDuration: number;  // seconds
  favoriteExercise: string;
  currentStreak: number;  // consecutive days
  bestScore: number;
}

interface LeaderboardEntry {
  rank: number;
  username: string;
  score: number;          // average form score
  sessionsCount: number;
  totalReps: number;
  streak: number;
}

interface Exercise {
  id: string;
  name: string;
  difficulty: "beginner" | "intermediate" | "advanced";
  muscleGroups: string[];
  description: string;
  thumbnailUrl: string;
}
```

---

## 6. Environment & Configuration

### Frontend Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | No | — | REST API base URL. If unset, frontend uses mock data. |
| `NEXT_PUBLIC_WS_URL` | No | — | WebSocket server URL. If unset, uses `MockWebSocket`. |
| `NEXT_PUBLIC_FISHJAM_URL` | No | — | Fishjam SFU WebSocket URL. If unset, video shows placeholders. |
| `NEXT_PUBLIC_FISHJAM_TOKEN` | No | — | Fishjam room auth token. |

**All variables are optional.** The frontend is always demoable without any backend — this is a core design principle.

### Backend Services Required

| Service | Purpose | Protocol |
|---------|---------|----------|
| REST API Server | Data persistence, user profiles, history, leaderboard | HTTP |
| WebSocket Server | Real-time session metadata, scores, insights | WS |
| Fishjam SFU | WebRTC video routing | WS/WebRTC |
| Smelter | Pose estimation + skeleton overlay compositing | Internal |
| Object Storage | Replay video files | HTTP (S3-compatible) |
| Database | Users, sessions, scores, insights | Internal |

---

## Appendix A: Mock Data Reference

The frontend mock system (`lib/mock-data.ts`) simulates the following for standalone demo:

**4 mock participants:**
| ID | Username | Initial Score | Status | Knee | Back | Hip | Reps |
|----|----------|--------------|--------|------|------|-----|------|
| 1 | Alex | 94 | excellent | 92° | 18° | 105° | 8 |
| 2 | Sam | 87 | warning | 88° | 42° | 98° | 7 |
| 3 | Dana | 78 | active | 78° | 22° | 100° | 6 |
| 4 | Jordan | 91 | excellent | 95° | 15° | 110° | 9 |

**3 mock insights:**
| Participant | Type | Title |
|------------|------|-------|
| Sam | warning | Back Posture — "Try to keep your back more upright..." |
| Jordan | success | Excellent Form — "Great job maintaining proper form..." |
| Dana | tip | Knee Tracking — "Focus on keeping your knees aligned..." |

**Mock timing:**
- Initial state delivered 500ms after connection
- Participant updates every 2000ms (random participant, score ±2, angles ±3°)
- Session timer increments every 1000ms

**Initial session state:** Squat, Set 2/3, Rep 8/10, sessionTime 1475s
