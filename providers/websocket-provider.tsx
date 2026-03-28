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
