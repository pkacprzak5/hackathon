"use client";

import { Home, Link2, Save, Share2, Trophy } from "lucide-react";
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
    <div className="flex flex-col items-center bg-bg-primary px-6">
      {/* Header */}
      <div className="flex w-full flex-col items-center pt-12 pb-6">
        <div className="mb-3 flex h-10 w-10 items-center justify-center">
          <Trophy className="h-10 w-10 text-gradient-start" />
        </div>
        <h1 className="bg-linear-to-r from-gradient-start to-gradient-end bg-clip-text font-heading text-[28px] font-extrabold text-transparent">
          {avgScore >= 80 ? "Great Work!" : avgScore >= 60 ? "Good Effort!" : "Keep Practicing!"}
        </h1>
        <p className="text-sm text-text-secondary">Session Complete</p>
      </div>

      {/* Results card */}
      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center rounded-3xl bg-bg-card p-6">
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

          <div className="mt-4 flex w-full gap-[10px]">
            <button className="flex h-[44px] flex-1 items-center justify-center gap-2 rounded-[22px] bg-linear-to-r from-gradient-start to-gradient-end text-sm font-semibold text-white">
              <Share2 className="h-4 w-4" />
              Share
            </button>
            <button
              onClick={() => router.push("/home")}
              className="flex h-[44px] items-center justify-center gap-2 rounded-[22px] border-[1.5px] border-border px-4 text-[13px] font-semibold text-text-primary"
            >
              <Home className="h-4 w-4" />
              Home
            </button>
            <button className="flex h-[44px] items-center justify-center gap-2 rounded-[22px] border-[1.5px] border-border px-4 text-[13px] font-semibold text-text-primary">
              <Save className="h-4 w-4" />
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
