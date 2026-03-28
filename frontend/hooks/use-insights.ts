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
