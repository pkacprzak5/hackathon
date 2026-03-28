"use client";

import { ArrowLeft, Download, Share2 } from "lucide-react";
import { useRouter } from "next/navigation";

import { AngleChart } from "@/components/charts/angle-chart";
import { VideoFeed } from "@/components/session/video-feed";
import { InsightCard } from "@/components/ui/insight-card";

export default function ReplayPage() {
  const router = useRouter();

  return (
    <div className="flex flex-col bg-bg-primary px-6">
      {/* Header */}
      <div className="flex items-center justify-between py-3">
        <button onClick={() => router.back()}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <p className="text-sm font-semibold text-text-primary">Replay</p>
        <div className="flex gap-2">
          <Download className="h-5 w-5 text-text-secondary" />
          <Share2 className="h-5 w-5 text-text-secondary" />
        </div>
      </div>

      {/* Video replay */}
      <div>
        <VideoFeed name="You" score={87} color="#8B5CF6" className="aspect-video w-full rounded-2xl" />
      </div>

      {/* Stat pills */}
      <div className="flex gap-3 pt-4">
        <div className="rounded-xl bg-bg-card px-4 py-2 text-xs font-semibold text-text-primary">87 avg</div>
        <div className="rounded-xl bg-bg-card px-4 py-2 text-xs font-semibold text-text-primary">10 reps</div>
        <div className="rounded-xl bg-bg-card px-4 py-2 text-xs font-semibold text-success">Good</div>
      </div>

      {/* Angle charts */}
      <div className="flex flex-col gap-3 pt-4">
        <AngleChart label="Knee Angle" data={[85, 88, 82, 90, 87, 84, 92, 88, 86, 90]} color="var(--gradient-start)" />
        <AngleChart label="Back Angle" data={[15, 18, 22, 16, 20, 18, 14, 19, 17, 16]} color="var(--gradient2-start)" />
      </div>

      {/* AI Summary */}
      <div className="pt-4 pb-4">
        <h3 className="mb-2 text-sm font-bold text-text-primary">AI Summary</h3>
        <InsightCard insight={{ participantId: "self", type: "success", title: "Solid Session", message: "Consistent depth across all reps. Knee tracking improved from last session.", timestamp: 0 }} />
      </div>
    </div>
  );
}
