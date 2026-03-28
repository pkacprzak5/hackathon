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
