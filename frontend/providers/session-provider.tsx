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
