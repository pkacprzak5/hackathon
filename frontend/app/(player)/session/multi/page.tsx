"use client";

import { ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";

import { CameraGrid } from "@/components/session/camera-grid";
import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { useInsights } from "@/hooks/use-insights";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";

export default function MultiSessionPage() {
  const router = useRouter();
  const { exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { participants } = useParticipants();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3">
        <button onClick={() => router.push("/home")}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <div className="flex-1">
          <p className="text-sm font-semibold text-text-primary">{exercise || "Squat"} — Multiplayer</p>
          <p className="text-xs text-text-secondary">{participants.length} participants · {timeStr}</p>
        </div>
      </div>

      {/* Camera grid */}
      <div className="px-4">
        <CameraGrid participants={participants} />
      </div>

      {/* Joint comparison */}
      <div className="px-4 pt-3">
        <div className="rounded-2xl bg-bg-card p-3 ">
          <p className="mb-2 text-xs font-bold text-text-primary">Joint Comparison</p>
          {["Knee", "Back", "Hip"].map((joint) => (
            <div key={joint} className="mb-2">
              <p className="mb-1 text-[10px] text-text-secondary">{joint}</p>
              <div className="flex gap-1">
                {participants.slice(0, 4).map((p) => {
                  const angle = p.jointAngles[joint.toLowerCase() as keyof typeof p.jointAngles];
                  const color = PARTICIPANT_COLORS[p.id] ?? "#8B5CF6";
                  return (
                    <div key={p.id} className="flex-1">
                      <div className="h-2 rounded-full" style={{ backgroundColor: color, width: `${Math.min(100, angle)}%` }} />
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
          {/* Legend */}
          <div className="mt-2 flex flex-wrap gap-3">
            {participants.slice(0, 4).map((p) => (
              <div key={p.id} className="flex items-center gap-1">
                <div className="h-2 w-2 rounded-full" style={{ backgroundColor: PARTICIPANT_COLORS[p.id] }} />
                <span className="text-[10px] text-text-secondary">{p.username}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 px-4 py-3">
        <StatBlock label="Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Rep" value={`${currentRep || 8} / ${totalReps || 10}`} />
      </div>

      {/* AI feedback */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="mb-2 text-xs font-bold text-text-primary">AI Feedback</h3>
        <div className="flex flex-col gap-2">
          {insights.slice(0, 3).map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
}
