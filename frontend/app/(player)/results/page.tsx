"use client";

import { Home, Share2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ScoreRing } from "@/components/ui/score-ring";
import { StatBlock } from "@/components/ui/stat-block";
import type { SquatRepResult } from "@/lib/squat-types";

interface SessionResults {
  reps: SquatRepResult[];
  repCount: number;
  avgScore: number;
  sessionTime: number;
}

export default function ResultsPage() {
  const router = useRouter();
  const [results, setResults] = useState<SessionResults | null>(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const raw = sessionStorage.getItem("squat_results");
      if (raw) {
        setResults(JSON.parse(raw));
      }
    }
  }, []);

  const avgScore = results?.avgScore ?? 0;
  const repCount = results?.repCount ?? 0;
  const sessionTime = results?.sessionTime ?? 0;
  const reps = results?.reps ?? [];

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  return (
    <div className="flex flex-col items-center bg-bg-surface">
      {/* Header */}
      <div className="w-full bg-linear-to-r from-gradient-start to-gradient-end px-5 pb-16 pt-12 text-center">
        <p className="text-sm text-white/70">Session Complete</p>
        <h1 className="text-2xl font-bold text-white">
          {avgScore >= 80 ? "Great Work!" : avgScore >= 60 ? "Good Effort!" : "Keep Practicing!"}
        </h1>
      </div>

      {/* Results card */}
      <div className="-mt-10 w-full max-w-sm px-5">
        <div className="flex flex-col items-center rounded-3xl bg-bg-card p-6 shadow-lg ring-1 ring-border-light">
          <ScoreRing score={avgScore} size={140} strokeWidth={10} />

          <div className="mt-4 grid w-full grid-cols-3 gap-4">
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Exercise" value="Squat" />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Reps" value={`${repCount}`} />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Time" value={timeStr} />
            </div>
          </div>

          {/* Per-rep breakdown */}
          {reps.length > 0 && (
            <div className="mt-4 w-full">
              <h3 className="mb-2 text-xs font-semibold text-text-secondary">Rep Breakdown</h3>
              <div className="flex flex-col gap-1.5">
                {reps.map((rep, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg bg-bg-surface px-3 py-2"
                  >
                    <span className="text-xs text-text-secondary">Rep {rep.rep_index}</span>
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] text-text-muted">
                        D:{Math.round(rep.scores.depth)} T:{Math.round(rep.scores.trunk_control)} P:{Math.round(rep.scores.posture_stability)}
                      </span>
                      <span className={`text-xs font-bold ${rep.scores.total >= 80 ? "text-success" : rep.scores.total >= 60 ? "text-warning" : "text-error"}`}>
                        {Math.round(rep.scores.total)}
                      </span>
                    </div>
                    {rep.faults.length > 0 && (
                      <span className="text-[10px] text-error">
                        {rep.faults[0].replace(/_/g, " ")}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4 flex w-full gap-2">
            <button
              onClick={() => router.push("/home")}
              className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-bg-surface py-3 text-sm font-semibold text-text-primary ring-1 ring-border-light"
            >
              <Home className="h-4 w-4" />
              Home
            </button>
            <button className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-linear-to-r from-gradient-start to-gradient-end py-3 text-sm font-semibold text-white">
              <Share2 className="h-4 w-4" />
              Share
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
