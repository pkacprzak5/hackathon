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
