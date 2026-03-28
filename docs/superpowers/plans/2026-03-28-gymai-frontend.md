# GymAI MVP Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mobile-first PWA for real-time exercise form analysis with 8 player screens, 1 coach desktop dashboard, WebSocket integration, and Fishjam video stream support — all pixel-matched to `pencil-new.pen` designs.

**Architecture:** Next.js 16 App Router with React Context + useReducer for state. WebSocket for real-time metadata, Fishjam SDK for video streams. Mock mode when no backend is connected. Route groups `(player)` and `coach` with separate layouts.

**Tech Stack:** Next.js 16.1.6, React 19, TypeScript 5.9, Tailwind CSS 4.2, next-themes, lucide-react, CVA, @fishjam-cloud/react-client

**Design source:** All UI pixel-matched to `pencil-new.pen`. Reference screenshots from Pencil MCP tools during implementation.

---

## File Map

```
lib/
  types.ts                      # All TypeScript interfaces (SessionState, Participant, Insight, WS messages)
  utils.ts                      # cn() helper (clsx + tailwind-merge)
  websocket-client.ts           # WebSocket client class with auto-reconnect
  mock-data.ts                  # MockWebSocket + mock participants/insights

providers/
  session-provider.tsx          # SessionContext + useReducer + hooks
  websocket-provider.tsx        # Bridges WebSocket client to SessionContext
  fishjam-provider.tsx          # Fishjam SDK wrapper (stub for MVP)
  theme-provider.tsx            # next-themes wrapper (client component)

components/
  ui/score-ring.tsx             # Circular SVG score indicator
  ui/stat-block.tsx             # Label + value stat display
  ui/exercise-card.tsx          # Exercise card with icon + difficulty
  ui/insight-card.tsx           # AI insight with severity coloring
  ui/joint-angle-row.tsx        # Joint angle data row with color coding
  ui/participant-card.tsx       # Participant avatar + name + score
  ui/theme-toggle.tsx           # Dark/light mode switch
  session/video-feed.tsx        # Video stream display (Fishjam or placeholder)
  session/camera-grid.tsx       # Dynamic N-person camera layout
  charts/weekly-chart.tsx       # SVG bar chart for history
  charts/angle-chart.tsx        # SVG line chart for replay
  coach/coach-sidebar.tsx       # Coach participant list sidebar
  coach/coach-insights.tsx      # Coach AI insights panel
  coach/coach-joints.tsx        # Coach joint angles table

app/
  globals.css                   # MODIFY: Replace with design tokens
  layout.tsx                    # MODIFY: Add providers, PWA meta, fonts
  page.tsx                      # REPLACE: Entry screen (username + role)
  (player)/
    layout.tsx                  # MobileLayout with bottom tab bar
    home/page.tsx               # Home dashboard
    session/solo/page.tsx       # Solo live session
    session/multi/page.tsx      # Multiplayer session
    replay/page.tsx             # Replay with video + chart
    history/page.tsx            # History with weekly chart
    leaderboard/page.tsx        # Leaderboard rankings
    results/page.tsx            # Results & Share
    profile/page.tsx            # Profile & Settings
  coach/
    layout.tsx                  # CoachLayout (desktop only)
    page.tsx                    # Coach dashboard

public/
  manifest.json                 # PWA manifest
  sw.js                         # Basic service worker
  icon.svg                      # App icon (SVG)
  apple-touch-icon.png          # iOS home screen icon (180x180)
```

---

### Task 1: Foundation — Types, Utilities, CSS Tokens

**Files:**
- Create: `lib/types.ts`
- Create: `lib/utils.ts`
- Modify: `app/globals.css`

- [ ] **Step 1: Create type definitions**

```typescript
// lib/types.ts

export interface JointAngles {
  knee: number;
  back: number;
  hip: number;
}

export interface Participant {
  id: string;
  username: string;
  score: number;
  jointAngles: JointAngles;
  repCount: number;
  status: "active" | "warning" | "excellent";
}

export interface Insight {
  participantId: string;
  type: "warning" | "success" | "tip";
  title: string;
  message: string;
  timestamp: number;
}

export interface SessionState {
  username: string;
  role: "player" | "coach";
  sessionStatus: "idle" | "active" | "ended";
  exercise: string;
  currentSet: number;
  totalSets: number;
  currentRep: number;
  totalReps: number;
  sessionTime: number;
  participants: Participant[];
  insights: Insight[];
}

// WebSocket message types
export type WSIncomingMessage =
  | { type: "session_update"; exercise: string; currentSet: number; totalSets: number; currentRep: number; totalReps: number; sessionTime: number }
  | { type: "participant_update"; participant: Participant }
  | { type: "participant_join"; participant: Participant }
  | { type: "participant_leave"; participantId: string }
  | { type: "insight"; insight: Insight }
  | { type: "session_end" };

export type WSOutgoingMessage =
  | { type: "join"; username: string; role: "player" | "coach" };

export type SessionAction =
  | { type: "SET_USER"; username: string; role: "player" | "coach" }
  | { type: "SESSION_UPDATE"; exercise: string; currentSet: number; totalSets: number; currentRep: number; totalReps: number; sessionTime: number }
  | { type: "PARTICIPANT_UPDATE"; participant: Participant }
  | { type: "PARTICIPANT_JOIN"; participant: Participant }
  | { type: "PARTICIPANT_LEAVE"; participantId: string }
  | { type: "ADD_INSIGHT"; insight: Insight }
  | { type: "SESSION_END" }
  | { type: "RESET" };
```

- [ ] **Step 2: Create cn utility**

```typescript
// lib/utils.ts
import { clsx } from "clsx";
import type { ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 3: Replace globals.css with design tokens**

Replace the entire contents of `app/globals.css` with:

```css
@import "tailwindcss";
@import "tw-animate-css";

:root {
  /* Backgrounds */
  --bg-primary: #FFFFFF;
  --bg-surface: #F4F4F5;
  --bg-card: #FFFFFF;
  --bg-elevated: #FAFAFA;

  /* Text */
  --text-primary: #09090B;
  --text-secondary: #71717A;
  --text-muted: #A1A1AA;

  /* Borders */
  --border: #E4E4E7;
  --border-light: #F4F4F5;

  /* Brand gradients */
  --gradient-start: #8B5CF6;
  --gradient-end: #F472B6;
  --gradient2-start: #06B6D4;
  --gradient2-end: #8B5CF6;

  /* Semantic */
  --success: #22C55E;
  --success-muted: #14B8A6;
  --error: #EF4444;
  --warning: #F59E0B;

  /* Insight backgrounds */
  --insight-error-bg: #FFF5F5;
  --insight-success-bg: #F0FDF4;
  --insight-warning-bg: #FFFBEB;

  /* Camera */
  --camera-bg: #1A1A2E;
  --camera-bg-dark: #0F0F1A;
}

.dark {
  --bg-primary: #0A0A0F;
  --bg-surface: #1A1A2E;
  --bg-card: #1A1A2E;
  --bg-elevated: #252540;

  --text-primary: #FFFFFF;
  --text-secondary: #A1A1AA;
  --text-muted: #71717A;

  --border: #2A2A3E;
  --border-light: #2A2A3E;

  --insight-error-bg: #2A1515;
  --insight-success-bg: #0F2A1A;
  --insight-warning-bg: #2A2010;

  --camera-bg: #151525;
  --camera-bg-dark: #0A0A14;
}

@theme inline {
  --color-bg-primary: var(--bg-primary);
  --color-bg-surface: var(--bg-surface);
  --color-bg-card: var(--bg-card);
  --color-bg-elevated: var(--bg-elevated);
  --color-text-primary: var(--text-primary);
  --color-text-secondary: var(--text-secondary);
  --color-text-muted: var(--text-muted);
  --color-border: var(--border);
  --color-border-light: var(--border-light);
  --color-gradient-start: var(--gradient-start);
  --color-gradient-end: var(--gradient-end);
  --color-gradient2-start: var(--gradient2-start);
  --color-gradient2-end: var(--gradient2-end);
  --color-success: var(--success);
  --color-success-muted: var(--success-muted);
  --color-error: var(--error);
  --color-warning: var(--warning);
  --color-insight-error-bg: var(--insight-error-bg);
  --color-insight-success-bg: var(--insight-success-bg);
  --color-insight-warning-bg: var(--insight-warning-bg);
  --color-camera-bg: var(--camera-bg);
  --color-camera-bg-dark: var(--camera-bg-dark);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}

body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-geist-sans), system-ui, sans-serif;
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`
Expected: No type errors

- [ ] **Step 5: Commit**

```bash
git add lib/types.ts lib/utils.ts app/globals.css
git commit -m "feat: add type definitions, cn utility, and design tokens"
```

---

### Task 2: PWA Setup — Manifest, Service Worker, Icons

**Files:**
- Create: `public/manifest.json`
- Create: `public/sw.js`
- Create: `public/icon.svg`

- [ ] **Step 1: Create PWA manifest**

```json
// public/manifest.json
{
  "name": "GymAI",
  "short_name": "GymAI",
  "description": "Real-time exercise form analysis",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0A0A0F",
  "theme_color": "#8B5CF6",
  "orientation": "portrait",
  "icons": [
    {
      "src": "/icon.svg",
      "sizes": "any",
      "type": "image/svg+xml",
      "purpose": "any"
    }
  ]
}
```

- [ ] **Step 2: Create basic service worker**

```javascript
// public/sw.js
const CACHE_NAME = "gymai-v1";
const STATIC_ASSETS = ["/_next/static/", "/icon.svg", "/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Network-first for API/WebSocket
  if (url.pathname.startsWith("/api") || url.protocol === "ws:" || url.protocol === "wss:") {
    return;
  }

  // Cache-first for static assets
  const isStatic = STATIC_ASSETS.some((path) => url.pathname.includes(path));
  if (isStatic) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          return response;
        });
      })
    );
    return;
  }

  // Network-first for everything else
  event.respondWith(
    fetch(event.request).catch(() => caches.match(event.request))
  );
});
```

- [ ] **Step 3: Create SVG app icon**

```svg
<!-- public/icon.svg -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#8B5CF6"/>
      <stop offset="100%" style="stop-color:#F472B6"/>
    </linearGradient>
  </defs>
  <circle cx="256" cy="256" r="256" fill="url(#bg)"/>
  <!-- Dumbbell -->
  <rect x="176" y="236" width="160" height="40" rx="8" fill="white" opacity="0.9"/>
  <rect x="148" y="212" width="40" height="88" rx="12" fill="white" opacity="0.9"/>
  <rect x="324" y="212" width="40" height="88" rx="12" fill="white" opacity="0.9"/>
  <rect x="136" y="224" width="24" height="64" rx="8" fill="white" opacity="0.7"/>
  <rect x="352" y="224" width="24" height="64" rx="8" fill="white" opacity="0.7"/>
</svg>
```

- [ ] **Step 4: Generate apple-touch-icon placeholder**

Create a simple 180x180 PNG from the SVG. For the hackathon, we can use the SVG directly and also reference it as the apple-touch-icon. Add this to root layout in a later task.

- [ ] **Step 5: Commit**

```bash
git add public/manifest.json public/sw.js public/icon.svg
git commit -m "feat: add PWA manifest, service worker, and app icon"
```

---

### Task 3: Session State — Context, Reducer, Hooks

**Files:**
- Create: `providers/session-provider.tsx`
- Create: `hooks/use-session.ts`
- Create: `hooks/use-participants.ts`
- Create: `hooks/use-insights.ts`

- [ ] **Step 1: Create session provider with reducer**

```typescript
// providers/session-provider.tsx
"use client";

import type { ReactNode } from "react";
import { createContext, useCallback, useContext, useReducer } from "react";

import type { SessionAction, SessionState } from "@/lib/types";

const initialState: SessionState = {
  username: "",
  role: "player",
  sessionStatus: "idle",
  exercise: "",
  currentSet: 0,
  totalSets: 0,
  currentRep: 0,
  totalReps: 0,
  sessionTime: 0,
  participants: [],
  insights: [],
};

function sessionReducer(state: SessionState, action: SessionAction): SessionState {
  switch (action.type) {
    case "SET_USER":
      return { ...state, username: action.username, role: action.role };
    case "SESSION_UPDATE":
      return {
        ...state,
        sessionStatus: "active",
        exercise: action.exercise,
        currentSet: action.currentSet,
        totalSets: action.totalSets,
        currentRep: action.currentRep,
        totalReps: action.totalReps,
        sessionTime: action.sessionTime,
      };
    case "PARTICIPANT_UPDATE":
      return {
        ...state,
        participants: state.participants.map((p) =>
          p.id === action.participant.id ? action.participant : p
        ),
      };
    case "PARTICIPANT_JOIN":
      return {
        ...state,
        participants: [...state.participants.filter((p) => p.id !== action.participant.id), action.participant],
      };
    case "PARTICIPANT_LEAVE":
      return {
        ...state,
        participants: state.participants.filter((p) => p.id !== action.participantId),
      };
    case "ADD_INSIGHT":
      return {
        ...state,
        insights: [action.insight, ...state.insights].slice(0, 50),
      };
    case "SESSION_END":
      return { ...state, sessionStatus: "ended" };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

interface SessionContextValue {
  state: SessionState;
  dispatch: React.Dispatch<SessionAction>;
  setUser: (username: string, role: "player" | "coach") => void;
}

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(sessionReducer, initialState);

  const setUser = useCallback(
    (username: string, role: "player" | "coach") => {
      dispatch({ type: "SET_USER", username, role });
    },
    []
  );

  return (
    <SessionContext.Provider value={{ state, dispatch, setUser }}>
      {children}
    </SessionContext.Provider>
  );
}

export function useSessionContext() {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSessionContext must be used within SessionProvider");
  return ctx;
}
```

- [ ] **Step 2: Create convenience hooks**

```typescript
// hooks/use-session.ts
"use client";

import { useMemo } from "react";

import { useSessionContext } from "@/providers/session-provider";

export function useSession() {
  const { state, setUser } = useSessionContext();

  return useMemo(
    () => ({
      username: state.username,
      role: state.role,
      sessionStatus: state.sessionStatus,
      exercise: state.exercise,
      currentSet: state.currentSet,
      totalSets: state.totalSets,
      currentRep: state.currentRep,
      totalReps: state.totalReps,
      sessionTime: state.sessionTime,
      isAuthenticated: state.username !== "",
      setUser,
    }),
    [state, setUser]
  );
}
```

```typescript
// hooks/use-participants.ts
"use client";

import { useMemo } from "react";

import { useSessionContext } from "@/providers/session-provider";

export function useParticipants() {
  const { state } = useSessionContext();

  return useMemo(
    () => ({
      participants: state.participants,
      count: state.participants.length,
    }),
    [state.participants]
  );
}
```

```typescript
// hooks/use-insights.ts
"use client";

import { useMemo } from "react";

import { useSessionContext } from "@/providers/session-provider";

export function useInsights() {
  const { state } = useSessionContext();

  return useMemo(
    () => ({
      insights: state.insights,
      warnings: state.insights.filter((i) => i.type === "warning"),
      tips: state.insights.filter((i) => i.type === "tip"),
    }),
    [state.insights]
  );
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`
Expected: No type errors

- [ ] **Step 4: Commit**

```bash
git add providers/session-provider.tsx hooks/
git commit -m "feat: add session context, reducer, and hooks"
```

---

### Task 4: WebSocket Client + Mock Data

**Files:**
- Create: `lib/websocket-client.ts`
- Create: `lib/mock-data.ts`
- Create: `providers/websocket-provider.tsx`

- [ ] **Step 1: Create WebSocket client class**

```typescript
// lib/websocket-client.ts
import type { WSIncomingMessage, WSOutgoingMessage } from "@/lib/types";

export type WSMessageHandler = (message: WSIncomingMessage) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private handler: WSMessageHandler;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(url: string, handler: WSMessageHandler) {
    this.url = url;
    this.handler = handler;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WSIncomingMessage;
          this.handler(message);
        } catch {
          console.warn("Failed to parse WebSocket message:", event.data);
        }
      };

      this.ws.onclose = () => {
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.ws?.close();
      };
    } catch {
      this.scheduleReconnect();
    }
  }

  send(message: WSOutgoingMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectAttempts = this.maxReconnectAttempts;
    this.ws?.close();
    this.ws = null;
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
    this.reconnectAttempts++;
    this.reconnectTimer = setTimeout(() => this.connect(), delay);
  }
}
```

- [ ] **Step 2: Create mock data and MockWebSocket**

```typescript
// lib/mock-data.ts
import type { Insight, Participant, WSMessageHandler } from "@/lib/types";
import type { WSIncomingMessage } from "@/lib/types";

export const MOCK_PARTICIPANTS: Participant[] = [
  { id: "1", username: "Alex", score: 94, jointAngles: { knee: 92, back: 18, hip: 105 }, repCount: 8, status: "excellent" },
  { id: "2", username: "Sam", score: 87, jointAngles: { knee: 88, back: 42, hip: 98 }, repCount: 7, status: "warning" },
  { id: "3", username: "Dana", score: 78, jointAngles: { knee: 78, back: 22, hip: 100 }, repCount: 6, status: "active" },
  { id: "4", username: "Jordan", score: 91, jointAngles: { knee: 95, back: 15, hip: 110 }, repCount: 9, status: "excellent" },
];

export const MOCK_INSIGHTS: Insight[] = [
  { participantId: "2", type: "warning", title: "Sam: Back Posture", message: "Spine rounding detected. Cue to brace core now.", timestamp: Date.now() },
  { participantId: "4", type: "success", title: "Jordan: Excellent Form", message: "Consistent depth & great bar path.", timestamp: Date.now() - 5000 },
  { participantId: "3", type: "tip", title: "Dana: Knee Tracking", message: "Slight inward cave at depth.", timestamp: Date.now() - 10000 },
];

export const PARTICIPANT_COLORS: Record<string, string> = {
  "1": "#8B5CF6",
  "2": "#F472B6",
  "3": "#06B6D4",
  "4": "#14B8A6",
};

export class MockWebSocket {
  private handler: WSMessageHandler;
  private intervals: ReturnType<typeof setInterval>[] = [];

  constructor(handler: WSMessageHandler) {
    this.handler = handler;
  }

  connect() {
    // Send initial session state
    setTimeout(() => {
      this.handler({
        type: "session_update",
        exercise: "Squat",
        currentSet: 2,
        totalSets: 3,
        currentRep: 8,
        totalReps: 10,
        sessionTime: 1475,
      });

      // Join all mock participants
      for (const p of MOCK_PARTICIPANTS) {
        this.handler({ type: "participant_join", participant: p });
      }

      // Send initial insights
      for (const insight of MOCK_INSIGHTS) {
        this.handler({ type: "insight", insight });
      }
    }, 500);

    // Simulate periodic updates
    this.intervals.push(
      setInterval(() => {
        const p = MOCK_PARTICIPANTS[Math.floor(Math.random() * MOCK_PARTICIPANTS.length)];
        const updated: Participant = {
          ...p,
          score: Math.max(50, Math.min(100, p.score + Math.floor(Math.random() * 5) - 2)),
          jointAngles: {
            knee: p.jointAngles.knee + Math.floor(Math.random() * 6) - 3,
            back: p.jointAngles.back + Math.floor(Math.random() * 4) - 2,
            hip: p.jointAngles.hip + Math.floor(Math.random() * 6) - 3,
          },
        };
        this.handler({ type: "participant_update", participant: updated });
      }, 2000)
    );

    // Simulate timer
    let time = 1475;
    this.intervals.push(
      setInterval(() => {
        time++;
        this.handler({
          type: "session_update",
          exercise: "Squat",
          currentSet: 2,
          totalSets: 3,
          currentRep: 8,
          totalReps: 10,
          sessionTime: time,
        });
      }, 1000)
    );
  }

  send(_message: unknown) {
    // Mock doesn't process outgoing messages
  }

  disconnect() {
    for (const interval of this.intervals) {
      clearInterval(interval);
    }
    this.intervals = [];
  }
}
```

- [ ] **Step 3: Create WebSocket provider**

```typescript
// providers/websocket-provider.tsx
"use client";

import type { ReactNode } from "react";
import { useCallback, useEffect, useRef } from "react";

import { MockWebSocket } from "@/lib/mock-data";
import type { WSIncomingMessage } from "@/lib/types";
import { WebSocketClient } from "@/lib/websocket-client";
import { useSessionContext } from "@/providers/session-provider";

export function WebSocketProvider({ children }: { children: ReactNode }) {
  const { state, dispatch } = useSessionContext();
  const clientRef = useRef<WebSocketClient | MockWebSocket | null>(null);

  const handleMessage = useCallback(
    (message: WSIncomingMessage) => {
      switch (message.type) {
        case "session_update":
          dispatch({
            type: "SESSION_UPDATE",
            exercise: message.exercise,
            currentSet: message.currentSet,
            totalSets: message.totalSets,
            currentRep: message.currentRep,
            totalReps: message.totalReps,
            sessionTime: message.sessionTime,
          });
          break;
        case "participant_update":
          dispatch({ type: "PARTICIPANT_UPDATE", participant: message.participant });
          break;
        case "participant_join":
          dispatch({ type: "PARTICIPANT_JOIN", participant: message.participant });
          break;
        case "participant_leave":
          dispatch({ type: "PARTICIPANT_LEAVE", participantId: message.participantId });
          break;
        case "insight":
          dispatch({ type: "ADD_INSIGHT", insight: message.insight });
          break;
        case "session_end":
          dispatch({ type: "SESSION_END" });
          break;
      }
    },
    [dispatch]
  );

  useEffect(() => {
    if (!state.username) return;

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL;

    if (wsUrl) {
      const client = new WebSocketClient(wsUrl, handleMessage);
      clientRef.current = client;
      client.connect();
      client.send({ type: "join", username: state.username, role: state.role });
    } else {
      const mock = new MockWebSocket(handleMessage);
      clientRef.current = mock;
      mock.connect();
    }

    return () => {
      clientRef.current?.disconnect();
      clientRef.current = null;
    };
  }, [state.username, state.role, handleMessage]);

  return <>{children}</>;
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`
Expected: No type errors

- [ ] **Step 5: Commit**

```bash
git add lib/websocket-client.ts lib/mock-data.ts providers/websocket-provider.tsx
git commit -m "feat: add WebSocket client, mock data, and WS provider"
```

---

### Task 5: Theme Provider + Root Layout + Entry Screen

**Files:**
- Create: `providers/theme-provider.tsx`
- Create: `providers/fishjam-provider.tsx`
- Modify: `app/layout.tsx`
- Modify: `app/page.tsx`

- [ ] **Step 1: Create theme provider wrapper**

```typescript
// providers/theme-provider.tsx
"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ReactNode } from "react";

export function ThemeProvider({ children }: { children: ReactNode }) {
  return (
    <NextThemesProvider attribute="class" defaultTheme="system" enableSystem>
      {children}
    </NextThemesProvider>
  );
}
```

- [ ] **Step 2: Create Fishjam provider stub**

```typescript
// providers/fishjam-provider.tsx
"use client";

import type { ReactNode } from "react";
import { createContext, useContext } from "react";

// Stub context — will be replaced with real Fishjam SDK integration
interface FishjamContextValue {
  peers: Map<string, MediaStream | null>;
  isConnected: boolean;
}

const FishjamContext = createContext<FishjamContextValue>({
  peers: new Map(),
  isConnected: false,
});

export function FishjamProvider({ children }: { children: ReactNode }) {
  // Stub: no real Fishjam connection yet
  return (
    <FishjamContext.Provider value={{ peers: new Map(), isConnected: false }}>
      {children}
    </FishjamContext.Provider>
  );
}

export function useFishjamPeers() {
  return useContext(FishjamContext);
}
```

- [ ] **Step 3: Update root layout**

Replace `app/layout.tsx` entirely:

```typescript
// app/layout.tsx
import "./globals.css";

import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { FishjamProvider } from "@/providers/fishjam-provider";
import { SessionProvider } from "@/providers/session-provider";
import { ThemeProvider } from "@/providers/theme-provider";
import { WebSocketProvider } from "@/providers/websocket-provider";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GymAI",
  description: "Real-time exercise form analysis",
  manifest: "/manifest.json",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "GymAI",
  },
  icons: {
    icon: "/icon.svg",
    apple: "/icon.svg",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#8B5CF6",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <ThemeProvider>
          <SessionProvider>
            <WebSocketProvider>
              <FishjamProvider>
                {children}
              </FishjamProvider>
            </WebSocketProvider>
          </SessionProvider>
        </ThemeProvider>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                  navigator.serviceWorker.register('/sw.js');
                });
              }
            `,
          }}
        />
      </body>
    </html>
  );
}
```

- [ ] **Step 4: Create entry screen**

Replace `app/page.tsx` entirely. This is the username entry screen — not in Pencil designs, so match the app's visual language (gradient header, centered card):

```typescript
// app/page.tsx
"use client";

import { Dumbbell } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { cn } from "@/lib/utils";
import { useSessionContext } from "@/providers/session-provider";

export default function EntryPage() {
  const [username, setUsername] = useState("");
  const { setUser } = useSessionContext();
  const router = useRouter();

  function handleJoin(role: "player" | "coach") {
    if (!username.trim()) return;
    setUser(username.trim(), role);
    router.push(role === "coach" ? "/coach" : "/home");
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-bg-surface px-4">
      {/* Gradient header accent */}
      <div className="fixed top-0 left-0 h-48 w-full bg-gradient-to-r from-gradient-start to-gradient-end opacity-20 blur-3xl" />

      <div className="relative z-10 flex w-full max-w-sm flex-col items-center gap-8">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-gradient-start to-gradient-end">
            <Dumbbell className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-text-primary">GymAI</h1>
        </div>

        <p className="text-center text-text-secondary">
          Real-time exercise form analysis.
          <br />
          Enter your name to get started.
        </p>

        {/* Card */}
        <div className="w-full rounded-3xl bg-bg-card p-6 shadow-lg ring-1 ring-border-light">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleJoin("player")}
            placeholder="Your name"
            className="mb-6 w-full rounded-2xl border border-border bg-bg-surface px-4 py-3 text-text-primary placeholder:text-text-muted outline-none focus:ring-2 focus:ring-gradient-start"
            autoFocus
          />

          <div className="flex flex-col gap-3">
            <button
              onClick={() => handleJoin("player")}
              disabled={!username.trim()}
              className={cn(
                "w-full rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end py-3 text-sm font-semibold text-white transition-opacity",
                username.trim() ? "hover:opacity-90" : "opacity-40 cursor-not-allowed"
              )}
            >
              Join as Player
            </button>

            <button
              onClick={() => handleJoin("coach")}
              disabled={!username.trim()}
              className={cn(
                "w-full rounded-2xl border border-gradient-start py-3 text-sm font-semibold text-gradient-start transition-colors",
                username.trim() ? "hover:bg-gradient-start/10" : "opacity-40 cursor-not-allowed"
              )}
            >
              Join as Coach
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Verify the app runs**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm dev`
Open http://localhost:3000 — should see the entry screen with gradient accent, username input, and two buttons.

- [ ] **Step 6: Commit**

```bash
git add providers/theme-provider.tsx providers/fishjam-provider.tsx app/layout.tsx app/page.tsx
git commit -m "feat: add providers, root layout with PWA meta, and entry screen"
```

---

### Task 6: Shared UI Components — Batch 1

**Files:**
- Create: `components/ui/score-ring.tsx`
- Create: `components/ui/stat-block.tsx`
- Create: `components/ui/exercise-card.tsx`
- Create: `components/ui/theme-toggle.tsx`

- [ ] **Step 1: Create ScoreRing**

Circular SVG progress ring. Matches the score ring on Home screen in Pencil designs.

```typescript
// components/ui/score-ring.tsx
"use client";

import { cn } from "@/lib/utils";

interface ScoreRingProps {
  score: number; // 0-100
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export function ScoreRing({ score, size = 120, strokeWidth = 8, className }: ScoreRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const center = size / 2;

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg width={size} height={size} className="-rotate-90">
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-border-light"
        />
        {/* Progress circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="url(#scoreGradient)"
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
        />
        <defs>
          <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--gradient-start)" />
            <stop offset="100%" stopColor="var(--gradient-end)" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-3xl font-bold text-text-primary">{score}</span>
        <span className="text-xs text-text-secondary">score</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create StatBlock**

```typescript
// components/ui/stat-block.tsx
import { cn } from "@/lib/utils";

interface StatBlockProps {
  label: string;
  value: string;
  valueClassName?: string;
  className?: string;
}

export function StatBlock({ label, value, valueClassName, className }: StatBlockProps) {
  return (
    <div className={cn("flex flex-col", className)}>
      <span className={cn("text-sm font-bold text-text-primary", valueClassName)}>{value}</span>
      <span className="text-[10px] text-text-secondary">{label}</span>
    </div>
  );
}
```

- [ ] **Step 3: Create ExerciseCard**

```typescript
// components/ui/exercise-card.tsx
import { Dumbbell, Target } from "lucide-react";

import { cn } from "@/lib/utils";

interface ExerciseCardProps {
  name: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  reps?: string;
  className?: string;
  onClick?: () => void;
}

const difficultyColors = {
  Beginner: "bg-success/10 text-success",
  Intermediate: "bg-warning/10 text-warning",
  Advanced: "bg-error/10 text-error",
};

export function ExerciseCard({ name, difficulty, reps, className, onClick }: ExerciseCardProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light transition-colors hover:bg-bg-elevated text-left w-full",
        className
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-gradient-start/20 to-gradient-end/20">
        {name === "Squat" ? (
          <Target className="h-6 w-6 text-gradient-start" />
        ) : (
          <Dumbbell className="h-6 w-6 text-gradient-start" />
        )}
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-text-primary">{name}</p>
        {reps && <p className="text-xs text-text-secondary">{reps}</p>}
      </div>
      <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-semibold", difficultyColors[difficulty])}>
        {difficulty}
      </span>
    </button>
  );
}
```

- [ ] **Step 4: Create ThemeToggle**

```typescript
// components/ui/theme-toggle.tsx
"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

export function ThemeToggle({ className }: { className?: string }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  if (!mounted) return null;

  const isDark = theme === "dark";

  return (
    <button
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className={cn(
        "flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light w-full",
        className
      )}
    >
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-bg-surface">
        {isDark ? <Moon className="h-5 w-5 text-gradient-start" /> : <Sun className="h-5 w-5 text-warning" />}
      </div>
      <div className="flex-1 text-left">
        <p className="text-sm font-semibold text-text-primary">Dark Mode</p>
        <p className="text-xs text-text-secondary">{isDark ? "On" : "Off"}</p>
      </div>
      <div className={cn(
        "h-6 w-11 rounded-full p-0.5 transition-colors",
        isDark ? "bg-gradient-start" : "bg-border"
      )}>
        <div className={cn(
          "h-5 w-5 rounded-full bg-white transition-transform",
          isDark ? "translate-x-5" : "translate-x-0"
        )} />
      </div>
    </button>
  );
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`

- [ ] **Step 6: Commit**

```bash
git add components/ui/
git commit -m "feat: add ScoreRing, StatBlock, ExerciseCard, ThemeToggle components"
```

---

### Task 7: Shared UI Components — Batch 2

**Files:**
- Create: `components/ui/insight-card.tsx`
- Create: `components/ui/joint-angle-row.tsx`
- Create: `components/ui/participant-card.tsx`
- Create: `components/session/video-feed.tsx`
- Create: `components/session/camera-grid.tsx`

- [ ] **Step 1: Create InsightCard**

```typescript
// components/ui/insight-card.tsx
import { cn } from "@/lib/utils";
import type { Insight } from "@/lib/types";

const insightStyles = {
  warning: { bg: "bg-insight-error-bg", dot: "bg-error", title: "text-error" },
  success: { bg: "bg-insight-success-bg", dot: "bg-success", title: "text-success" },
  tip: { bg: "bg-insight-warning-bg", dot: "bg-warning", title: "text-warning" },
};

interface InsightCardProps {
  insight: Insight;
  className?: string;
}

export function InsightCard({ insight, className }: InsightCardProps) {
  const styles = insightStyles[insight.type];

  return (
    <div className={cn("flex gap-3 rounded-xl p-3", styles.bg, className)}>
      <div className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", styles.dot)} />
      <div>
        <p className={cn("text-xs font-semibold", styles.title)}>{insight.title}</p>
        <p className="text-xs text-text-secondary">{insight.message}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create JointAngleRow**

```typescript
// components/ui/joint-angle-row.tsx
import { cn } from "@/lib/utils";
import type { JointAngles } from "@/lib/types";

interface JointAngleRowProps {
  name: string;
  color: string;
  angles: JointAngles;
  className?: string;
}

function angleColor(value: number, joint: "knee" | "back" | "hip"): string {
  // Back: >35 is bad. Knee: <80 is warning. Hip: all fine under 120.
  if (joint === "back" && value > 35) return "text-error";
  if (joint === "back" && value < 20) return "text-success";
  if (joint === "knee" && value < 80) return "text-warning";
  if (joint === "knee" && value > 90) return "text-success";
  return "text-text-primary";
}

export function JointAngleRow({ name, color, angles, className }: JointAngleRowProps) {
  return (
    <div className={cn("flex items-center gap-2 px-4 py-2.5", className)}>
      <div className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
      <span className="w-16 text-xs font-semibold text-text-primary">{name}</span>
      <span className={cn("w-12 text-xs", angleColor(angles.knee, "knee"))}>{angles.knee}°</span>
      <span className={cn("w-12 text-xs", angleColor(angles.back, "back"))}>{angles.back}°</span>
      <span className={cn("w-12 text-xs", angleColor(angles.hip, "hip"))}>{angles.hip}°</span>
    </div>
  );
}
```

- [ ] **Step 3: Create ParticipantCard**

```typescript
// components/ui/participant-card.tsx
import { cn } from "@/lib/utils";
import type { Participant } from "@/lib/types";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";

interface ParticipantCardProps {
  participant: Participant;
  className?: string;
}

export function ParticipantCard({ participant, className }: ParticipantCardProps) {
  const color = PARTICIPANT_COLORS[participant.id] ?? "#8B5CF6";

  return (
    <div className={cn("flex items-center gap-3 px-4 py-4", className)}>
      <div
        className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold text-white"
        style={{ backgroundColor: color }}
      >
        {participant.username[0]}
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-text-primary">{participant.username}</p>
        <p className="text-xs text-text-secondary">Rep {participant.repCount} / 10</p>
      </div>
      <div className="text-right">
        <p className="text-xl font-bold" style={{ color }}>{participant.score}</p>
        <p className="text-[9px] text-text-muted">score</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create VideoFeed**

```typescript
// components/session/video-feed.tsx
"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

interface VideoFeedProps {
  stream?: MediaStream | null;
  name: string;
  score?: number;
  color?: string;
  className?: string;
}

export function VideoFeed({ stream, name, score, color = "#8B5CF6", className }: VideoFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div className={cn("relative overflow-hidden rounded-xl bg-camera-bg", className)}>
      {stream ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />
      ) : (
        /* Placeholder skeleton silhouette */
        <div className="flex h-full w-full items-center justify-center bg-gradient-to-b from-camera-bg to-camera-bg-dark">
          <svg width="80" height="160" viewBox="0 0 80 160" className="opacity-40">
            <circle cx="40" cy="20" r="16" fill={color} opacity="0.4" />
            <rect x="37" y="36" width="6" height="50" rx="3" fill={color} opacity="0.6" />
            <rect x="10" y="50" width="30" height="5" rx="2" fill={color} opacity="0.4" />
            <rect x="40" y="50" width="30" height="5" rx="2" fill={color} opacity="0.4" />
            <rect x="27" y="86" width="5" height="40" rx="2" fill={color} opacity="0.4" />
            <rect x="48" y="86" width="5" height="40" rx="2" fill={color} opacity="0.4" />
            <circle cx="29" cy="82" r="5" fill={color} />
            <circle cx="50" cy="82" r="5" fill={color} />
          </svg>
        </div>
      )}

      {/* Name badge */}
      <div
        className="absolute bottom-2 left-2 rounded-md px-2.5 py-1 text-[11px] font-semibold text-white"
        style={{ backgroundColor: color }}
      >
        {name}
      </div>

      {/* Score badge */}
      {score !== undefined && (
        <div className="absolute top-2 right-2 rounded-full bg-white/20 px-2.5 py-0.5 text-xs font-bold text-white backdrop-blur-sm">
          {score}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Create CameraGrid**

Dynamic layout: 1=full, 2=side-by-side, 3=2+1, 4=2x2.

```typescript
// components/session/camera-grid.tsx
import { cn } from "@/lib/utils";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";
import type { Participant } from "@/lib/types";
import { VideoFeed } from "@/components/session/video-feed";

interface CameraGridProps {
  participants: Participant[];
  className?: string;
}

export function CameraGrid({ participants, className }: CameraGridProps) {
  const count = participants.length;

  if (count === 0) {
    return (
      <div className={cn("flex items-center justify-center rounded-xl bg-camera-bg p-8", className)}>
        <p className="text-sm text-text-secondary">Waiting for participants...</p>
      </div>
    );
  }

  if (count === 1) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <VideoFeed
          name={participants[0].username}
          score={participants[0].score}
          color={PARTICIPANT_COLORS[participants[0].id]}
          className="aspect-[4/3] w-full"
        />
      </div>
    );
  }

  if (count === 2) {
    return (
      <div className={cn("flex gap-2", className)}>
        {participants.map((p) => (
          <VideoFeed
            key={p.id}
            name={p.username}
            score={p.score}
            color={PARTICIPANT_COLORS[p.id]}
            className="aspect-[3/4] flex-1"
          />
        ))}
      </div>
    );
  }

  if (count === 3) {
    return (
      <div className={cn("flex flex-col gap-2", className)}>
        <div className="flex gap-2">
          {participants.slice(0, 2).map((p) => (
            <VideoFeed
              key={p.id}
              name={p.username}
              score={p.score}
              color={PARTICIPANT_COLORS[p.id]}
              className="aspect-[4/3] flex-1"
            />
          ))}
        </div>
        <VideoFeed
          name={participants[2].username}
          score={participants[2].score}
          color={PARTICIPANT_COLORS[participants[2].id]}
          className="aspect-[16/9] w-full"
        />
      </div>
    );
  }

  // 4+ participants: 2x2 grid (show first 4)
  const shown = participants.slice(0, 4);
  return (
    <div className={cn("grid grid-cols-2 gap-2", className)}>
      {shown.map((p) => (
        <VideoFeed
          key={p.id}
          name={p.username}
          score={p.score}
          color={PARTICIPANT_COLORS[p.id]}
          className="aspect-[4/3]"
        />
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`

- [ ] **Step 7: Commit**

```bash
git add components/
git commit -m "feat: add InsightCard, JointAngleRow, ParticipantCard, VideoFeed, CameraGrid"
```

---

### Task 8: Mobile Layout with Bottom Tab Bar

**Files:**
- Create: `app/(player)/layout.tsx`

- [ ] **Step 1: Create mobile layout with bottom tab bar**

This matches the Pencil design: 4 tabs (Home, Session, History, Profile) with pill-style active indicator and gradient accent on active icon.

```typescript
// app/(player)/layout.tsx
"use client";

import { ChartBar, Home, Timer, User } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";
import { useSession } from "@/hooks/use-session";
import { useEffect } from "react";

const tabs = [
  { label: "Home", icon: Home, href: "/home" },
  { label: "Session", icon: Timer, href: "/session/solo" },
  { label: "History", icon: ChartBar, href: "/history" },
  { label: "Profile", icon: User, href: "/profile" },
];

export default function PlayerLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { isAuthenticated } = useSession();

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    }
  }, [isAuthenticated, router]);

  if (!isAuthenticated) return null;

  return (
    <div className="mx-auto flex min-h-dvh max-w-[430px] flex-col bg-bg-surface">
      {/* Page content */}
      <main className="flex-1 overflow-y-auto pb-20">{children}</main>

      {/* Bottom tab bar */}
      <nav className="fixed bottom-0 left-1/2 z-50 w-full max-w-[430px] -translate-x-1/2 border-t border-border-light bg-bg-card/80 backdrop-blur-xl">
        <div className="flex items-center justify-around py-2">
          {tabs.map((tab) => {
            const isActive =
              pathname === tab.href ||
              (tab.href === "/session/solo" && pathname.startsWith("/session")) ||
              (tab.href === "/home" && pathname === "/home");

            return (
              <button
                key={tab.href}
                onClick={() => router.push(tab.href)}
                className="flex flex-col items-center gap-1 px-4 py-1"
              >
                <tab.icon
                  className={cn(
                    "h-5 w-5 transition-colors",
                    isActive ? "text-gradient-start" : "text-text-muted"
                  )}
                />
                <span
                  className={cn(
                    "text-[10px] font-medium transition-colors",
                    isActive ? "text-gradient-start" : "text-text-muted"
                  )}
                >
                  {tab.label}
                </span>
                {isActive && (
                  <div className="h-1 w-4 rounded-full bg-gradient-to-r from-gradient-start to-gradient-end" />
                )}
              </button>
            );
          })}
        </div>
      </nav>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`

- [ ] **Step 3: Commit**

```bash
git add "app/(player)/layout.tsx"
git commit -m "feat: add mobile layout with bottom tab bar"
```

---

### Task 9: Home Screen

**Files:**
- Create: `app/(player)/home/page.tsx`

- [ ] **Step 1: Create home screen**

Pixel-matched to Pencil Row 1 (Home light) and Row 3 (Home dark). Features: gradient header, score ring, exercise cards, quick stats.

```typescript
// app/(player)/home/page.tsx
"use client";

import { ChevronRight, Users } from "lucide-react";
import { useRouter } from "next/navigation";

import { ExerciseCard } from "@/components/ui/exercise-card";
import { ScoreRing } from "@/components/ui/score-ring";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";

export default function HomePage() {
  const router = useRouter();
  const { username } = useSession();
  const { count } = useParticipants();

  return (
    <div className="flex flex-col">
      {/* Gradient header */}
      <div className="bg-gradient-to-r from-gradient-start to-gradient-end px-5 pb-16 pt-12">
        <p className="text-sm text-white/80">Welcome back,</p>
        <h1 className="text-2xl font-bold text-white">{username}</h1>
      </div>

      {/* Score card overlapping header */}
      <div className="-mt-10 px-5">
        <div className="flex items-center gap-6 rounded-3xl bg-bg-card p-5 shadow-lg ring-1 ring-border-light">
          <ScoreRing score={87} size={100} strokeWidth={7} />
          <div className="flex flex-col gap-1">
            <p className="text-sm font-semibold text-text-primary">Form Score</p>
            <p className="text-xs text-text-secondary">Your average across all exercises</p>
            <div className="mt-1 flex items-center gap-1 text-success">
              <ChevronRight className="h-3 w-3 rotate-[-90deg]" />
              <span className="text-xs font-semibold">+4 this week</span>
            </div>
          </div>
        </div>
      </div>

      {/* Multiplayer quick join */}
      <div className="px-5 pt-5">
        <button
          onClick={() => router.push("/session/multi")}
          className="flex w-full items-center gap-3 rounded-2xl bg-gradient-to-r from-gradient2-start to-gradient2-end p-4"
        >
          <Users className="h-5 w-5 text-white" />
          <div className="flex-1 text-left">
            <p className="text-sm font-semibold text-white">Multiplayer Session</p>
            <p className="text-xs text-white/70">{count > 0 ? `${count} active` : "Join the room"}</p>
          </div>
          <ChevronRight className="h-4 w-4 text-white/70" />
        </button>
      </div>

      {/* Exercises */}
      <div className="px-5 pt-5">
        <h2 className="mb-3 text-base font-bold text-text-primary">Exercises</h2>
        <div className="flex flex-col gap-3">
          <ExerciseCard
            name="Squat"
            difficulty="Intermediate"
            reps="3 sets × 10 reps"
            onClick={() => router.push("/session/solo")}
          />
          <ExerciseCard
            name="Deadlift"
            difficulty="Advanced"
            reps="3 sets × 8 reps"
            onClick={() => router.push("/session/solo")}
          />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it renders**

Run dev server and navigate to http://localhost:3000, enter a name, click "Join as Player". Should see the Home screen.

- [ ] **Step 3: Commit**

```bash
git add "app/(player)/home/page.tsx"
git commit -m "feat: add home screen with score ring, exercises, multiplayer CTA"
```

---

### Task 10: Solo Session Screen

**Files:**
- Create: `app/(player)/session/solo/page.tsx`

- [ ] **Step 1: Create solo session screen**

Matches Pencil "2. Live Session" — camera dominant 65/35 split, joint angles, AI feedback below.

```typescript
// app/(player)/session/solo/page.tsx
"use client";

import { ArrowLeft, Users } from "lucide-react";
import { useRouter } from "next/navigation";

import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { VideoFeed } from "@/components/session/video-feed";
import { useInsights } from "@/hooks/use-insights";
import { useSession } from "@/hooks/use-session";

export default function SoloSessionPage() {
  const router = useRouter();
  const { username, exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => router.push("/home")}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <div className="text-center">
          <p className="text-sm font-semibold text-text-primary">{exercise || "Squat"}</p>
          <p className="text-xs text-text-secondary">{timeStr}</p>
        </div>
        <button onClick={() => router.push("/session/multi")}>
          <Users className="h-5 w-5 text-text-secondary" />
        </button>
      </div>

      {/* Camera feed — 65% of remaining height */}
      <div className="px-4">
        <VideoFeed
          name={username}
          score={87}
          color="#8B5CF6"
          className="aspect-[3/4] w-full"
        />
      </div>

      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-3">
        <StatBlock label="Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Rep" value={`${currentRep || 8} / ${totalReps || 10}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Score" value="87" valueClassName="text-gradient-start" />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Time" value={timeStr || "24:35"} />
      </div>

      {/* AI Feedback */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="mb-2 text-sm font-bold text-text-primary">AI Feedback</h3>
        <div className="flex flex-col gap-2">
          {(insights.length > 0 ? insights.slice(0, 3) : [
            { participantId: "self", type: "success" as const, title: "Good depth", message: "Hitting parallel consistently.", timestamp: Date.now() },
            { participantId: "self", type: "tip" as const, title: "Watch your knees", message: "Slight inward cave on rep 6.", timestamp: Date.now() },
          ]).map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add "app/(player)/session/solo/page.tsx"
git commit -m "feat: add solo session screen with camera feed and AI feedback"
```

---

### Task 11: Multiplayer Session Screen

**Files:**
- Create: `app/(player)/session/multi/page.tsx`

- [ ] **Step 1: Create multiplayer session screen**

Matches Pencil "6. Multiplayer Session" — dynamic camera grid, joint comparison bars, AI feedback.

```typescript
// app/(player)/session/multi/page.tsx
"use client";

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";

import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { CameraGrid } from "@/components/session/camera-grid";
import { useInsights } from "@/hooks/use-insights";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";

export default function MultiSessionPage() {
  const router = useRouter();
  const { exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { participants } = useParticipants();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3">
        <button onClick={() => router.push("/home")}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <div className="flex-1">
          <p className="text-sm font-semibold text-text-primary">{exercise || "Squat"} — Multiplayer</p>
          <p className="text-xs text-text-secondary">{participants.length} participants · {timeStr}</p>
        </div>
      </div>

      {/* Camera grid */}
      <div className="px-4">
        <CameraGrid participants={participants} />
      </div>

      {/* Joint comparison */}
      <div className="px-4 pt-3">
        <div className="rounded-2xl bg-bg-card p-3 ring-1 ring-border-light">
          <p className="mb-2 text-xs font-bold text-text-primary">Joint Comparison</p>
          {["Knee", "Back", "Hip"].map((joint) => (
            <div key={joint} className="mb-2">
              <p className="mb-1 text-[10px] text-text-secondary">{joint}</p>
              <div className="flex gap-1">
                {participants.slice(0, 4).map((p) => {
                  const angle = p.jointAngles[joint.toLowerCase() as keyof typeof p.jointAngles];
                  const color = PARTICIPANT_COLORS[p.id] ?? "#8B5CF6";
                  return (
                    <div key={p.id} className="flex-1">
                      <div className="h-2 rounded-full" style={{ backgroundColor: color, width: `${Math.min(100, angle)}%` }} />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
          {/* Legend */}
          <div className="mt-2 flex flex-wrap gap-3">
            {participants.slice(0, 4).map((p) => (
              <div key={p.id} className="flex items-center gap-1">
                <div className="h-2 w-2 rounded-full" style={{ backgroundColor: PARTICIPANT_COLORS[p.id] }} />
                <span className="text-[10px] text-text-secondary">{p.username}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 px-4 py-3">
        <StatBlock label="Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Rep" value={`${currentRep || 8} / ${totalReps || 10}`} />
      </div>

      {/* AI feedback */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="mb-2 text-xs font-bold text-text-primary">AI Feedback</h3>
        <div className="flex flex-col gap-2">
          {insights.slice(0, 3).map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add "app/(player)/session/multi/page.tsx"
git commit -m "feat: add multiplayer session screen with camera grid and joint comparison"
```

---

### Task 12: Replay Screen

**Files:**
- Create: `components/charts/angle-chart.tsx`
- Create: `app/(player)/replay/page.tsx`

- [ ] **Step 1: Create AngleChart**

```typescript
// components/charts/angle-chart.tsx
import { cn } from "@/lib/utils";

interface AngleChartProps {
  data?: number[];
  label?: string;
  color?: string;
  className?: string;
}

export function AngleChart({
  data = [85, 88, 82, 90, 87, 84, 92, 88, 86, 90],
  label = "Knee Angle",
  color = "var(--gradient-start)",
  className,
}: AngleChartProps) {
  const width = 320;
  const height = 100;
  const padding = 8;
  const maxVal = Math.max(...data);
  const minVal = Math.min(...data);
  const range = maxVal - minVal || 1;

  const points = data
    .map((val, i) => {
      const x = padding + (i / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((val - minVal) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className={cn("rounded-xl bg-bg-card p-3 ring-1 ring-border-light", className)}>
      <p className="mb-2 text-xs font-semibold text-text-primary">{label}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {/* Area fill */}
        <polygon
          points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
          fill={color}
          opacity="0.1"
        />
      </svg>
      <div className="mt-1 flex justify-between text-[9px] text-text-muted">
        <span>Rep 1</span>
        <span>Rep {data.length}</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create replay page**

```typescript
// app/(player)/replay/page.tsx
"use client";

import { ArrowLeft, Download, Share2 } from "lucide-react";
import { useRouter } from "next/navigation";

import { AngleChart } from "@/components/charts/angle-chart";
import { InsightCard } from "@/components/ui/insight-card";
import { VideoFeed } from "@/components/session/video-feed";

export default function ReplayPage() {
  const router = useRouter();

  return (
    <div className="flex flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => router.back()}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <p className="text-sm font-semibold text-text-primary">Replay</p>
        <div className="flex gap-2">
          <Download className="h-5 w-5 text-text-secondary" />
          <Share2 className="h-5 w-5 text-text-secondary" />
        </div>
      </div>

      {/* Video replay */}
      <div className="px-4">
        <VideoFeed name="You" score={87} color="#8B5CF6" className="aspect-video w-full" />
      </div>

      {/* Angle charts */}
      <div className="flex flex-col gap-3 px-4 pt-4">
        <AngleChart label="Knee Angle" data={[85, 88, 82, 90, 87, 84, 92, 88, 86, 90]} color="var(--gradient-start)" />
        <AngleChart label="Back Angle" data={[15, 18, 22, 16, 20, 18, 14, 19, 17, 16]} color="var(--gradient2-start)" />
      </div>

      {/* AI Summary */}
      <div className="px-4 pt-4 pb-4">
        <h3 className="mb-2 text-sm font-bold text-text-primary">AI Summary</h3>
        <InsightCard insight={{ participantId: "self", type: "success", title: "Solid Session", message: "Consistent depth across all reps. Knee tracking improved from last session.", timestamp: Date.now() }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add components/charts/angle-chart.tsx "app/(player)/replay/page.tsx"
git commit -m "feat: add replay screen with angle chart and AI summary"
```

---

### Task 13: History Screen

**Files:**
- Create: `components/charts/weekly-chart.tsx`
- Create: `app/(player)/history/page.tsx`

- [ ] **Step 1: Create WeeklyChart**

```typescript
// components/charts/weekly-chart.tsx
import { cn } from "@/lib/utils";

interface WeeklyChartProps {
  data?: { day: string; value: number }[];
  className?: string;
}

const DEFAULT_DATA = [
  { day: "Mon", value: 72 },
  { day: "Tue", value: 85 },
  { day: "Wed", value: 78 },
  { day: "Thu", value: 90 },
  { day: "Fri", value: 88 },
  { day: "Sat", value: 0 },
  { day: "Sun", value: 0 },
];

export function WeeklyChart({ data = DEFAULT_DATA, className }: WeeklyChartProps) {
  const maxVal = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className={cn("rounded-2xl bg-bg-card p-4 ring-1 ring-border-light", className)}>
      <p className="mb-3 text-sm font-bold text-text-primary">This Week</p>
      <div className="flex items-end gap-2">
        {data.map((d, i) => {
          const height = d.value > 0 ? (d.value / maxVal) * 120 : 4;
          const isToday = i === new Date().getDay() - 1;

          return (
            <div key={d.day} className="flex flex-1 flex-col items-center gap-1">
              {d.value > 0 && (
                <span className="text-[10px] font-semibold text-text-secondary">{d.value}</span>
              )}
              <div
                className={cn(
                  "w-full rounded-t-lg transition-all",
                  d.value > 0
                    ? isToday
                      ? "bg-gradient-to-t from-gradient-start to-gradient-end"
                      : "bg-gradient-start/30"
                    : "bg-border-light"
                )}
                style={{ height: `${height}px` }}
              />
              <span className={cn("text-[10px]", isToday ? "font-bold text-gradient-start" : "text-text-muted")}>
                {d.day}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create history page**

```typescript
// app/(player)/history/page.tsx
"use client";

import { WeeklyChart } from "@/components/charts/weekly-chart";
import { StatBlock } from "@/components/ui/stat-block";

const recentSessions = [
  { date: "Today", exercise: "Squat", score: 87, reps: "3×10" },
  { date: "Yesterday", exercise: "Deadlift", score: 82, reps: "3×8" },
  { date: "Mar 26", exercise: "Squat", score: 90, reps: "3×10" },
  { date: "Mar 25", exercise: "Squat", score: 78, reps: "3×10" },
];

export default function HistoryPage() {
  return (
    <div className="flex flex-col bg-bg-surface">
      {/* Header */}
      <div className="px-5 pt-12 pb-4">
        <h1 className="text-2xl font-bold text-text-primary">History</h1>
      </div>

      {/* Weekly chart */}
      <div className="px-5">
        <WeeklyChart />
      </div>

      {/* Quick stats */}
      <div className="flex gap-3 px-5 pt-4">
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Avg Score" value="84" valueClassName="text-lg text-gradient-start" />
        </div>
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Sessions" value="12" valueClassName="text-lg" />
        </div>
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Streak" value="4d" valueClassName="text-lg text-success" />
        </div>
      </div>

      {/* Recent sessions */}
      <div className="px-5 pt-4 pb-4">
        <h2 className="mb-3 text-sm font-bold text-text-primary">Recent Sessions</h2>
        <div className="flex flex-col gap-2">
          {recentSessions.map((session, i) => (
            <div key={i} className="flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
              <div className="flex-1">
                <p className="text-sm font-semibold text-text-primary">{session.exercise}</p>
                <p className="text-xs text-text-secondary">{session.date} · {session.reps}</p>
              </div>
              <span className="text-lg font-bold text-gradient-start">{session.score}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add components/charts/weekly-chart.tsx "app/(player)/history/page.tsx"
git commit -m "feat: add history screen with weekly chart and session list"
```

---

### Task 14: Leaderboard, Results, Profile Screens

**Files:**
- Create: `app/(player)/leaderboard/page.tsx`
- Create: `app/(player)/results/page.tsx`
- Create: `app/(player)/profile/page.tsx`

- [ ] **Step 1: Create leaderboard page**

```typescript
// app/(player)/leaderboard/page.tsx
"use client";

import { Medal, Trophy } from "lucide-react";

import { cn } from "@/lib/utils";
import { useParticipants } from "@/hooks/use-participants";
import { MOCK_PARTICIPANTS, PARTICIPANT_COLORS } from "@/lib/mock-data";

export default function LeaderboardPage() {
  const { participants } = useParticipants();
  const list = participants.length > 0 ? participants : MOCK_PARTICIPANTS;
  const sorted = [...list].sort((a, b) => b.score - a.score);

  return (
    <div className="flex flex-col bg-bg-surface">
      <div className="bg-gradient-to-r from-gradient-start to-gradient-end px-5 pb-8 pt-12">
        <h1 className="text-2xl font-bold text-white">Leaderboard</h1>
        <p className="text-sm text-white/70">Current session rankings</p>
      </div>

      <div className="-mt-4 px-5">
        <div className="flex flex-col gap-2">
          {sorted.map((p, i) => {
            const color = PARTICIPANT_COLORS[p.id] ?? "#8B5CF6";
            return (
              <div
                key={p.id}
                className={cn(
                  "flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light",
                  i === 0 && "ring-2 ring-warning/50"
                )}
              >
                <div className="flex h-8 w-8 items-center justify-center">
                  {i === 0 ? (
                    <Trophy className="h-6 w-6 text-warning" />
                  ) : i < 3 ? (
                    <Medal className="h-5 w-5 text-text-muted" />
                  ) : (
                    <span className="text-sm font-bold text-text-muted">{i + 1}</span>
                  )}
                </div>
                <div
                  className="flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold text-white"
                  style={{ backgroundColor: color }}
                >
                  {p.username[0]}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-text-primary">{p.username}</p>
                  <p className="text-xs text-text-secondary">Rep {p.repCount} / 10</p>
                </div>
                <span className="text-xl font-bold" style={{ color }}>{p.score}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create results page**

```typescript
// app/(player)/results/page.tsx
"use client";

import { Share2 } from "lucide-react";

import { ScoreRing } from "@/components/ui/score-ring";
import { StatBlock } from "@/components/ui/stat-block";

export default function ResultsPage() {
  return (
    <div className="flex flex-col items-center bg-bg-surface">
      {/* Header */}
      <div className="w-full bg-gradient-to-r from-gradient-start to-gradient-end px-5 pb-16 pt-12 text-center">
        <p className="text-sm text-white/70">Session Complete</p>
        <h1 className="text-2xl font-bold text-white">Great Work!</h1>
      </div>

      {/* Results card */}
      <div className="-mt-10 w-full max-w-sm px-5">
        <div className="flex flex-col items-center rounded-3xl bg-bg-card p-6 shadow-lg ring-1 ring-border-light">
          <ScoreRing score={87} size={140} strokeWidth={10} />

          <div className="mt-4 grid w-full grid-cols-3 gap-4">
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Exercise" value="Squat" />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Sets" value="3 × 10" />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Time" value="24:35" />
            </div>
          </div>

          <button className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end py-3 text-sm font-semibold text-white">
            <Share2 className="h-4 w-4" />
            Share Results
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create profile page**

```typescript
// app/(player)/profile/page.tsx
"use client";

import { ChevronRight, HelpCircle, LogOut, Shield, Trophy } from "lucide-react";
import { useRouter } from "next/navigation";

import { ThemeToggle } from "@/components/ui/theme-toggle";
import { useSession } from "@/hooks/use-session";
import { useSessionContext } from "@/providers/session-provider";

export default function ProfilePage() {
  const { username } = useSession();
  const { dispatch } = useSessionContext();
  const router = useRouter();

  function handleLogout() {
    dispatch({ type: "RESET" });
    router.replace("/");
  }

  const menuItems = [
    { icon: Trophy, label: "Leaderboard", href: "/leaderboard" },
    { icon: Shield, label: "Privacy", href: "#" },
    { icon: HelpCircle, label: "Help & Support", href: "#" },
  ];

  return (
    <div className="flex flex-col bg-bg-surface">
      <div className="px-5 pt-12 pb-4">
        <h1 className="text-2xl font-bold text-text-primary">Profile</h1>
      </div>

      {/* User card */}
      <div className="px-5">
        <div className="flex items-center gap-4 rounded-3xl bg-bg-card p-5 ring-1 ring-border-light">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-gradient-start to-gradient-end text-xl font-bold text-white">
            {username[0]?.toUpperCase()}
          </div>
          <div>
            <p className="text-lg font-bold text-text-primary">{username}</p>
            <p className="text-sm text-text-secondary">Player</p>
          </div>
        </div>
      </div>

      {/* Theme toggle */}
      <div className="px-5 pt-4">
        <ThemeToggle />
      </div>

      {/* Menu items */}
      <div className="flex flex-col gap-2 px-5 pt-4">
        {menuItems.map((item) => (
          <button
            key={item.label}
            onClick={() => item.href !== "#" && router.push(item.href)}
            className="flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light"
          >
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-bg-surface">
              <item.icon className="h-5 w-5 text-text-secondary" />
            </div>
            <span className="flex-1 text-left text-sm font-medium text-text-primary">{item.label}</span>
            <ChevronRight className="h-4 w-4 text-text-muted" />
          </button>
        ))}
      </div>

      {/* Logout */}
      <div className="px-5 pt-4 pb-8">
        <button
          onClick={handleLogout}
          className="flex w-full items-center gap-3 rounded-2xl bg-error/10 p-4"
        >
          <LogOut className="h-5 w-5 text-error" />
          <span className="text-sm font-medium text-error">Log Out</span>
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify all pages compile**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`

- [ ] **Step 5: Commit**

```bash
git add "app/(player)/leaderboard/page.tsx" "app/(player)/results/page.tsx" "app/(player)/profile/page.tsx"
git commit -m "feat: add leaderboard, results, and profile screens"
```

---

### Task 15: Coach Layout + Dashboard

**Files:**
- Create: `components/coach/coach-sidebar.tsx`
- Create: `components/coach/coach-insights.tsx`
- Create: `components/coach/coach-joints.tsx`
- Create: `app/coach/layout.tsx`
- Create: `app/coach/page.tsx`

- [ ] **Step 1: Create coach sidebar**

```typescript
// components/coach/coach-sidebar.tsx
"use client";

import { ParticipantCard } from "@/components/ui/participant-card";
import type { Participant } from "@/lib/types";

interface CoachSidebarProps {
  participants: Participant[];
}

export function CoachSidebar({ participants }: CoachSidebarProps) {
  return (
    <aside className="flex h-full w-[220px] flex-col border-r border-border bg-bg-card">
      <div className="border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">Participants ({participants.length})</p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {participants.map((p, i) => (
          <ParticipantCard
            key={p.id}
            participant={p}
            className={i % 2 === 1 ? "bg-bg-elevated" : ""}
          />
        ))}
      </div>
      <div className="border-t border-border-light bg-bg-elevated px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-success" />
          <span className="text-xs font-semibold text-success">Live · Squat</span>
        </div>
        <p className="text-[10px] text-text-muted">Set 2 / 3 · 10 reps</p>
      </div>
    </aside>
  );
}
```

- [ ] **Step 2: Create coach insights panel**

```typescript
// components/coach/coach-insights.tsx
"use client";

import { InsightCard } from "@/components/ui/insight-card";
import type { Insight } from "@/lib/types";

interface CoachInsightsProps {
  insights: Insight[];
}

export function CoachInsights({ insights }: CoachInsightsProps) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">AI Insights</p>
        <span className="rounded-full bg-gradient-to-r from-gradient-start to-gradient-end px-2 py-0.5 text-[9px] font-semibold text-white">
          Live
        </span>
      </div>
      <div className="flex flex-col gap-0">
        {insights.map((insight, i) => (
          <div key={i} className="px-3 py-1">
            <InsightCard insight={insight} />
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create coach joints table**

```typescript
// components/coach/coach-joints.tsx
"use client";

import { JointAngleRow } from "@/components/ui/joint-angle-row";
import type { Participant } from "@/lib/types";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";

interface CoachJointsProps {
  participants: Participant[];
}

export function CoachJoints({ participants }: CoachJointsProps) {
  return (
    <div className="flex flex-col">
      <div className="border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">Joint Angles</p>
      </div>
      {/* Column headers */}
      <div className="flex items-center gap-2 bg-bg-elevated px-4 py-1.5">
        <span className="w-2" />
        <span className="w-16 text-[10px] font-semibold text-text-muted">Name</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Knee</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Back</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Hip</span>
      </div>
      {participants.map((p) => (
        <JointAngleRow
          key={p.id}
          name={p.username}
          color={PARTICIPANT_COLORS[p.id] ?? "#8B5CF6"}
          angles={p.jointAngles}
        />
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Create coach layout**

```typescript
// app/coach/layout.tsx
"use client";

import { useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect } from "react";

import { useSession } from "@/hooks/use-session";

export default function CoachLayout({ children }: { children: ReactNode }) {
  const router = useRouter();
  const { isAuthenticated, role } = useSession();

  useEffect(() => {
    if (!isAuthenticated) {
      router.replace("/");
    } else if (role !== "coach") {
      router.replace("/home");
    }
  }, [isAuthenticated, role, router]);

  if (!isAuthenticated || role !== "coach") return null;

  return (
    <>
      {/* Mobile warning */}
      <div className="flex min-h-dvh items-center justify-center bg-bg-surface p-8 lg:hidden">
        <p className="text-center text-text-secondary">
          Coach dashboard requires a desktop browser.
        </p>
      </div>
      {/* Desktop layout */}
      <div className="hidden min-h-screen bg-bg-surface lg:block">{children}</div>
    </>
  );
}
```

- [ ] **Step 5: Create coach dashboard page**

Pixel-matched to Pencil "9. Coach Dashboard" — gradient top bar, sidebar, 2x2 camera grid, right panel with AI insights + joint angles.

```typescript
// app/coach/page.tsx
"use client";

import { Dumbbell } from "lucide-react";

import { CoachInsights } from "@/components/coach/coach-insights";
import { CoachJoints } from "@/components/coach/coach-joints";
import { CoachSidebar } from "@/components/coach/coach-sidebar";
import { StatBlock } from "@/components/ui/stat-block";
import { CameraGrid } from "@/components/session/camera-grid";
import { useInsights } from "@/hooks/use-insights";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";

export default function CoachDashboard() {
  const { exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { participants } = useParticipants();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  const avgScore = participants.length > 0
    ? Math.round(participants.reduce((sum, p) => sum + p.score, 0) / participants.length)
    : 0;

  return (
    <div className="flex h-screen flex-col">
      {/* Top bar — gradient */}
      <header className="flex h-14 items-center justify-between bg-gradient-to-r from-gradient-start to-gradient-end px-6">
        <div className="flex items-center gap-3">
          <Dumbbell className="h-5 w-5 text-white" />
          <span className="text-lg font-bold text-white">GymAI</span>
          <span className="rounded-md bg-white/20 px-2 py-0.5 text-[10px] font-semibold text-white">Coach</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-white">{exercise || "Morning Squat Session"}</span>
          <div className="flex items-center gap-1.5 rounded-full bg-white/20 px-3 py-1">
            <div className="h-2 w-2 rounded-full bg-green-400" />
            <span className="text-xs font-semibold text-white">{timeStr || "24:35"}</span>
          </div>
        </div>
        <button className="rounded-full bg-error/40 px-4 py-1.5 text-xs font-semibold text-white">
          End Session
        </button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <CoachSidebar participants={participants} />

        {/* Main content */}
        <main className="flex flex-1 flex-col overflow-y-auto p-4">
          <div className="mb-2 flex items-baseline gap-2">
            <h2 className="text-lg font-bold text-text-primary">Live Feeds</h2>
            <span className="text-xs text-text-secondary">{participants.length} active</span>
          </div>

          {/* Camera grid */}
          <CameraGrid participants={participants} className="mb-4" />

          {/* Stats bar */}
          <div className="flex items-center gap-6 rounded-xl bg-bg-card p-4 ring-1 ring-border-light">
            <StatBlock label="Exercise" value={exercise || "Squat"} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Current Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Reps" value={`${currentRep || 8} / ${totalReps || 10}`} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Avg Score" value={String(avgScore || 87)} valueClassName="text-gradient-start" />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Session Time" value={timeStr || "24:35"} />
          </div>
        </main>

        {/* Right panel */}
        <aside className="flex w-[340px] flex-col overflow-y-auto border-l border-border bg-bg-card">
          <CoachInsights insights={insights} />
          <div className="border-t border-border-light">
            <CoachJoints participants={participants} />
          </div>
          <div className="mt-auto border-t border-border-light bg-bg-elevated p-4">
            <p className="text-xs font-bold text-text-primary">Coach Notes</p>
            <p className="mt-1 text-xs text-text-secondary">
              Focus on Sam&apos;s back posture. Dana needs cue for knee tracking at depth.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
```

- [ ] **Step 6: Verify it compiles**

Run: `cd /Users/aleksanderjozwik/study/hackathon && pnpm type-check`

- [ ] **Step 7: Commit**

```bash
git add components/coach/ app/coach/
git commit -m "feat: add coach layout and desktop dashboard with sidebar, camera grid, insights"
```

---

### Task 16: Final Validation & Cleanup

**Files:**
- None new — validation only

- [ ] **Step 1: Run full validation**

```bash
cd /Users/aleksanderjozwik/study/hackathon && pnpm validate
```

Expected: All lint and type checks pass.

- [ ] **Step 2: Run dev server and manually verify all routes**

```bash
cd /Users/aleksanderjozwik/study/hackathon && pnpm dev
```

Check each route in the browser:
1. `/` — Entry screen with username input + two buttons
2. Enter name as player → `/home` — Score ring, exercise cards, multiplayer CTA
3. `/session/solo` — Camera feed + AI feedback
4. `/session/multi` — Camera grid + joint comparison
5. `/replay` — Video + angle charts
6. `/history` — Weekly chart + session list
7. `/leaderboard` — Ranked participant list
8. `/results` — Score ring + share button
9. `/profile` — User card + theme toggle + logout
10. Go back to `/`, join as coach → `/coach` — Desktop dashboard with sidebar, grid, insights

- [ ] **Step 3: Run build to verify production readiness**

```bash
cd /Users/aleksanderjozwik/study/hackathon && pnpm build
```

Expected: Build completes with no errors.

- [ ] **Step 4: Fix any issues found, then commit**

```bash
git add -A
git commit -m "chore: final validation and cleanup"
```
