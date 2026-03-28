"use client";

import { Dumbbell } from "lucide-react";

import { CoachInsights } from "@/components/coach/coach-insights";
import { CoachJoints } from "@/components/coach/coach-joints";
import { CoachSidebar } from "@/components/coach/coach-sidebar";
import { CameraGrid } from "@/components/session/camera-grid";
import { StatBlock } from "@/components/ui/stat-block";
import { useInsights } from "@/hooks/use-insights";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";

export default function CoachDashboard() {
  const { exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { participants } = useParticipants();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  const avgScore = participants.length > 0
    ? Math.round(participants.reduce((sum, p) => sum + p.score, 0) / participants.length)
    : 0;

  return (
    <div className="flex h-screen flex-col">
      {/* Top bar */}
      <header className="flex h-14 items-center justify-between bg-gradient-to-r from-gradient-start to-gradient-end px-6">
        <div className="flex items-center gap-3">
          <Dumbbell className="h-5 w-5 text-white" />
          <span className="text-lg font-bold text-white">GymAI</span>
          <span className="rounded-md bg-white/20 px-2 py-0.5 text-[10px] font-semibold text-white">Coach</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm font-semibold text-white">{exercise || "Morning Squat Session"}</span>
          <div className="flex items-center gap-1.5 rounded-full bg-white/20 px-3 py-1">
            <div className="h-2 w-2 rounded-full bg-green-400" />
            <span className="text-xs font-semibold text-white">{timeStr || "24:35"}</span>
          </div>
        </div>
        <button className="rounded-full bg-error/40 px-4 py-1.5 text-xs font-semibold text-white">
          End Session
        </button>
      </header>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        <CoachSidebar participants={participants} />

        <main className="flex flex-1 flex-col overflow-y-auto p-4">
          <div className="mb-2 flex items-baseline gap-2">
            <h2 className="text-lg font-bold text-text-primary">Live Feeds</h2>
            <span className="text-xs text-text-secondary">{participants.length} active</span>
          </div>

          <CameraGrid participants={participants} className="mb-4" />

          <div className="flex items-center gap-6 rounded-xl bg-bg-card p-4 ring-1 ring-border-light">
            <StatBlock label="Exercise" value={exercise || "Squat"} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Current Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Reps" value={`${currentRep || 8} / ${totalReps || 10}`} />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Avg Score" value={String(avgScore || 87)} valueClassName="text-gradient-start" />
            <div className="h-8 w-px bg-border-light" />
            <StatBlock label="Session Time" value={timeStr || "24:35"} />
          </div>
        </main>

        <aside className="flex w-[340px] flex-col overflow-y-auto border-l border-border bg-bg-card">
          <CoachInsights insights={insights} />
          <div className="border-t border-border-light">
            <CoachJoints participants={participants} />
          </div>
          <div className="mt-auto border-t border-border-light bg-bg-elevated p-4">
            <p className="text-xs font-bold text-text-primary">Coach Notes</p>
            <p className="mt-1 text-xs text-text-secondary">
              Focus on Sam&apos;s back posture. Dana needs cue for knee tracking at depth.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
