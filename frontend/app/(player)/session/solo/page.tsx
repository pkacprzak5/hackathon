"use client";

import { ArrowLeft, Users } from "lucide-react";
import { useRouter } from "next/navigation";

import { VideoFeed } from "@/components/session/video-feed";
import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { useInsights } from "@/hooks/use-insights";
import { useSession } from "@/hooks/use-session";

export default function SoloSessionPage() {
  const router = useRouter();
  const { username, exercise, currentSet, totalSets, currentRep, totalReps, sessionTime } = useSession();
  const { insights } = useInsights();

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => router.push("/home")}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <div className="text-center">
          <p className="text-sm font-semibold text-text-primary">{exercise || "Squat"}</p>
          <p className="text-xs text-text-secondary">{timeStr}</p>
        </div>
        <button onClick={() => router.push("/session/multi")}>
          <Users className="h-5 w-5 text-text-secondary" />
        </button>
      </div>

      {/* Camera feed */}
      <div className="px-4">
        <VideoFeed
          name={username}
          score={87}
          color="#8B5CF6"
          className="aspect-[3/4] w-full"
        />
      </div>

      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-3">
        <StatBlock label="Set" value={`${currentSet || 2} / ${totalSets || 3}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Rep" value={`${currentRep || 8} / ${totalReps || 10}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Score" value="87" valueClassName="text-gradient-start" />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Time" value={timeStr || "24:35"} />
      </div>

      {/* AI Feedback */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="mb-2 text-sm font-bold text-text-primary">AI Feedback</h3>
        <div className="flex flex-col gap-2">
          {(insights.length > 0 ? insights.slice(0, 3) : [
            { participantId: "self", type: "success" as const, title: "Good depth", message: "Hitting parallel consistently.", timestamp: 0 },
            { participantId: "self", type: "tip" as const, title: "Watch your knees", message: "Slight inward cave on rep 6.", timestamp: 0 },
          ]).map((insight, i) => (
            <InsightCard key={i} insight={insight} />
          ))}
        </div>
      </div>
    </div>
  );
}
