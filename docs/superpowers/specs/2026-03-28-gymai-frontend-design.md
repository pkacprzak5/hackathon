# GymAI MVP Frontend Design Spec

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Pure frontend implementation based on Pencil designs (`pencil-new.pen`)

## Overview

GymAI is a mobile-first PWA for real-time exercise form analysis. Users enter a username and join a shared room as either a player (mobile) or coach (desktop). Video streams with skeleton overlays arrive pre-composited from Smelter/Fishjam. The frontend displays these streams and renders real-time metadata (scores, joint angles, AI insights) received via WebSocket.

No login/auth system. No invitations. One shared room for the entire app.

## Design Source

All UI must be pixel-matched to the screens in `pencil-new.pen`:

- **Row 1 (y=0):** Light — Home, Live Session, Replay, History
- **Row 2 (y=924):** Light — Profile, Multiplayer, Leaderboard, Results
- **Row 3 (y=1948):** Dark — Home, Live Session, Replay, History
- **Row 4 (y=2872):** Dark — Profile, Multiplayer, Leaderboard, Results
- **Row 5 (y=3846):** Desktop — Coach Dashboard Light, Coach Dashboard Dark

**Note:** The entry screen (`/`) is not in the Pencil file. It should be implemented matching the app's visual language — gradient header, centered card with username input field, two CTA buttons ("Join as Player" primary gradient, "Join as Coach" outline), minimal and clean. Both light and dark mode variants.

## Tech Stack

- Next.js 16.1.6 (App Router)
- React 19.2.3
- TypeScript 5.9.3
- Tailwind CSS 4.2.1
- next-themes 0.4.6
- lucide-react 0.577.0
- class-variance-authority 0.7.1
- clsx + tailwind-merge
- @fishjam-cloud/react-client (video streams)

---

## 1. Routes & Navigation

| Route | View | Role |
|-------|------|------|
| `/` | Entry screen — username input + "Join as Player" / "Join as Coach" | All |
| `/home` | Home dashboard — score ring, exercise cards | Player |
| `/session/solo` | Solo live session — single camera feed, AI feedback | Player |
| `/session/multi` | Multiplayer session — dynamic camera grid | Player |
| `/replay` | Replay view — video playback + angle chart + AI summary | Player |
| `/history` | History — weekly chart + error tracking | Player |
| `/leaderboard` | Leaderboard rankings | Player |
| `/results` | Results & Share screen | Player |
| `/profile` | Profile & Settings (includes theme toggle) | Player |
| `/coach` | Coach desktop dashboard — all participant feeds + insights | Coach |

### Navigation behavior

- Player screens use a bottom tab bar with 4 tabs: Home, Session, History, Profile.
- Session tab navigates to `/session/solo` by default. Multiplayer is accessible from Home or within the session flow.
- Coach has no tab bar — single full-screen dashboard.
- Entry screen (`/`) has no navigation chrome.
- After entering a username, role + username are stored in `SessionContext`. Navigating to any player/coach route without a username redirects to `/`.
- Coach route is only accessible when role is "coach".

---

## 2. State Management & Data Flow

### SessionContext

Single React Context + `useReducer`, wraps the entire app:

```typescript
interface SessionState {
  // User
  username: string;
  role: "player" | "coach";

  // Session
  sessionStatus: "idle" | "active" | "ended";
  exercise: string;
  currentSet: number;
  totalSets: number;
  currentRep: number;
  totalReps: number;
  sessionTime: number; // seconds

  // Participants (including self)
  participants: Participant[];

  // AI Insights
  insights: Insight[];
}

interface Participant {
  id: string;
  username: string;
  score: number;
  jointAngles: { knee: number; back: number; hip: number };
  repCount: number;
  status: "active" | "warning" | "excellent";
}

interface Insight {
  participantId: string;
  type: "warning" | "success" | "tip";
  title: string;
  message: string;
  timestamp: number;
}
```

### Data flow

1. User enters username + role on `/` -> stored in context, WebSocket connects with `{ type: "join", username, role }`.
2. WebSocket receives real-time updates -> context reducer dispatches to participants, scores, insights, rep counts.
3. UI components consume via hooks: `useSession()`, `useParticipants()`, `useInsights()`.
4. For MVP without backend, a `MockWebSocket` class simulates incoming data so frontend works standalone.

### Video streams

Handled separately through Fishjam JS SDK — not part of SessionContext. Video components receive stream objects directly from Fishjam and render in `<video>` elements. Context only holds metadata; actual video pixels flow through Fishjam's peer connection.

---

## 3. Component Architecture

### Layout components

- `MobileLayout` — wraps all player routes, renders bottom tab bar
- `CoachLayout` — wraps `/coach`, no tab bar, full-screen desktop
- `EntryLayout` — wraps `/`, no chrome

### Shared components

| Component | Purpose | Used in |
|-----------|---------|---------|
| `VideoFeed` | Fishjam stream -> styled `<video>` with name/score overlay | Solo, Multi, Coach |
| `ScoreRing` | Circular progress for form score | Home, Results |
| `ParticipantCard` | Avatar + name + score + reps | Coach sidebar |
| `JointAngleRow` | Name dot + knee/back/hip with color coding | Coach panel, Session |
| `InsightCard` | Severity dot + title + message (red/green/amber) | Coach panel, Session |
| `ExerciseCard` | Exercise name + thumbnail + difficulty | Home |
| `StatBlock` | Label + value pair | Stats bars |
| `ThemeToggle` | Dark/light switch via next-themes | Profile |

### Screen-specific components

- `CameraGrid` — dynamic layout: 1=full, 2=side-by-side, 3=2+1, 4=2x2
- `WeeklyChart` — SVG bar chart (no charting library)
- `AngleChart` — SVG line chart for replay
- `ShareCard` — results summary with share action

### File structure

```
app/
  layout.tsx                    # Root — providers, fonts, theme
  page.tsx                      # Entry screen
  (player)/
    layout.tsx                  # MobileLayout with tab bar
    home/page.tsx
    session/solo/page.tsx
    session/multi/page.tsx
    replay/page.tsx
    history/page.tsx
    leaderboard/page.tsx
    results/page.tsx
    profile/page.tsx
  coach/
    layout.tsx                  # CoachLayout
    page.tsx                    # Dashboard
components/
  ui/                           # Shared primitives
  session/                      # Session-specific
  coach/                        # Coach-specific
providers/
  session-provider.tsx
  websocket-provider.tsx
  fishjam-provider.tsx
hooks/
  use-session.ts
  use-participants.ts
  use-insights.ts
lib/
  types.ts
  mock-data.ts
  websocket-client.ts
```

---

## 4. Backend Integration Layer

### WebSocket client (`lib/websocket-client.ts`)

- Connects to configurable URL via `NEXT_PUBLIC_WS_URL`
- Sends on connect: `{ type: "join", username, role }`
- Receives message types:
  - `session_update` — exercise, set, rep, timer changes
  - `participant_update` — score, joint angles, status for one participant
  - `participant_join` / `participant_leave`
  - `insight` — new AI insight
  - `session_end` — session completed
- Auto-reconnect with exponential backoff
- Plain class, not React-coupled — `WebSocketProvider` bridges to context

### Fishjam integration (`providers/fishjam-provider.tsx`)

- Uses `@fishjam-cloud/react-client` SDK
- Connects to Fishjam room with same username
- Exposes streams via `useFishjamPeers()` hook
- `VideoFeed` maps peer stream to `<video>` element
- Without backend: renders dark placeholder with colored skeleton silhouette (matching Pencil designs)

### Mock mode (`lib/mock-data.ts`)

- `MockWebSocket` class implements same interface as real client
- Simulates 4 participants: Alex, Sam, Dana, Jordan (matching designs)
- Emits realistic score updates, angle changes, AI insights on timers
- Activated when `NEXT_PUBLIC_WS_URL` is not set
- Mock video feeds show dark rectangles with colored skeleton outlines

### Environment variables

```
NEXT_PUBLIC_WS_URL=ws://localhost:8080       # WebSocket endpoint (optional)
NEXT_PUBLIC_FISHJAM_URL=ws://localhost:5002   # Fishjam server (optional)
NEXT_PUBLIC_FISHJAM_TOKEN=                    # Room token (optional)
```

All optional — if unset, app runs in full mock mode. Frontend is always demoable without any backend.

---

## 5. Responsive Behavior & Theme

### Layout strategy

- All player screens designed for 390px width (iPhone 14 baseline), fluid up to ~430px
- Player routes are mobile-only — no desktop adaptation needed
- Coach route is desktop-only — designed for 1440px, min-width enforced
- Entry screen works on both mobile and desktop (centered card layout)
- Player on desktop: mobile layout centered with max-width container
- Coach on mobile: message "Coach dashboard requires a desktop browser"

### Theme

- `next-themes` ThemeProvider with `attribute="class"`, `defaultTheme="system"`
- Manual override toggle in Profile screen
- All colors as CSS custom properties in `globals.css`, toggled by `.dark` class

#### Color tokens (from Pencil designs)

**Light mode:**
- `--bg-primary: #FFFFFF`
- `--bg-surface: #F4F4F5`
- `--bg-card: #FFFFFF`
- `--text-primary: #09090B`
- `--text-secondary: #71717A`

**Dark mode:**
- `--bg-primary: #0A0A0F`
- `--bg-surface: #1A1A2E`
- `--bg-card: #1A1A2E`
- `--bg-elevated: #252540`
- `--text-primary: #FFFFFF`
- `--text-secondary: #A1A1AA`

**Shared:**
- `--gradient-start: #8B5CF6`
- `--gradient-end: #F472B6`
- `--gradient2-start: #06B6D4`
- `--gradient2-end: #8B5CF6`
- `--success: #22C55E` / `--success-dark: #14B8A6`
- `--error: #EF4444`
- `--warning: #F59E0B`

---

## 6. PWA

- `manifest.json` — app name "GymAI", theme color `#8B5CF6`, sample SVG icon (purple gradient circle with dumbbell silhouette)
- `apple-touch-icon.png` (180x180) for iOS home screen
- Meta tags in root layout: `apple-mobile-web-app-capable`, `apple-mobile-web-app-status-bar-style`
- Basic service worker: cache-first for static assets (JS/CSS/fonts/icons), network-first for API/WebSocket. Registered in root layout. No offline page — just ensures fast reloads.
