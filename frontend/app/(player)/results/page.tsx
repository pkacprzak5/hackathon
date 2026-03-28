"use client";

import { Share2 } from "lucide-react";

import { ScoreRing } from "@/components/ui/score-ring";
import { StatBlock } from "@/components/ui/stat-block";

export default function ResultsPage() {
  return (
    <div className="flex flex-col items-center bg-bg-surface">
      {/* Header */}
      <div className="w-full bg-linear-to-r from-gradient-start to-gradient-end px-5 pb-16 pt-12 text-center">
        <p className="text-sm text-white/70">Session Complete</p>
        <h1 className="text-2xl font-bold text-white">Great Work!</h1>
      </div>

      {/* Results card */}
      <div className="-mt-10 w-full max-w-sm px-5">
        <div className="flex flex-col items-center rounded-3xl bg-bg-card p-6 shadow-lg ring-1 ring-border-light">
          <ScoreRing score={87} size={140} strokeWidth={10} />

          <div className="mt-4 grid w-full grid-cols-3 gap-4">
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Exercise" value="Squat" />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Sets" value="3 × 10" />
            </div>
            <div className="flex flex-col items-center rounded-xl bg-bg-surface p-3">
              <StatBlock label="Time" value="24:35" />
            </div>
          </div>

          <button className="mt-4 flex w-full items-center justify-center gap-2 rounded-2xl bg-linear-to-r from-gradient-start to-gradient-end py-3 text-sm font-semibold text-white">
            <Share2 className="h-4 w-4" />
            Share Results
          </button>
        </div>
      </div>
    </div>
  );
}
