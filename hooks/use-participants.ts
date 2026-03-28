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
