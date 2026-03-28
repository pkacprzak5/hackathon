import type { Insight, Participant } from "@/lib/types";
import type { WSMessageHandler } from "@/lib/websocket-client";

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

      for (const p of MOCK_PARTICIPANTS) {
        this.handler({ type: "participant_join", participant: p });
      }

      for (const insight of MOCK_INSIGHTS) {
        this.handler({ type: "insight", insight });
      }
    }, 500);

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
