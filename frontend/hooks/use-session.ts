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
