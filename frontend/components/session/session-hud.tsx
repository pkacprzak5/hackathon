"use client";

import type { SquatPhase, SquatRepResult } from "@/lib/squat-types";

interface SessionHudProps {
  phase: SquatPhase;
  repCount: number;
  score: number;
  coachingText: string | null;
}

const PHASE_COLORS: Record<SquatPhase, string> = {
  TOP: "bg-blue-500",
  DESCENT: "bg-amber-500",
  BOTTOM: "bg-red-500",
  ASCENT: "bg-green-500",
};

export function SessionHud({ phase, repCount, score, coachingText }: SessionHudProps) {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 z-20 flex flex-col gap-2 p-3">
      <div className="flex items-center justify-between">
        <div className={`rounded-full px-3 py-1 text-xs font-bold text-white ${PHASE_COLORS[phase]}`}>
          {phase}
        </div>
        <div className="flex items-center gap-3">
          <div className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold text-white backdrop-blur-sm">
            Rep {repCount}
          </div>
          <div className="rounded-full bg-white/20 px-3 py-1 text-xs font-bold text-white backdrop-blur-sm">
            {Math.round(score)}
          </div>
        </div>
      </div>

      {coachingText && (
        <div className="mx-auto max-w-xs rounded-xl bg-gradient-to-r from-gradient-start/80 to-gradient-end/80 px-4 py-2 text-center text-sm font-medium text-white backdrop-blur-sm">
          {coachingText}
        </div>
      )}
    </div>
  );
}
