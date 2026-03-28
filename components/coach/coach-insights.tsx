"use client";

import { InsightCard } from "@/components/ui/insight-card";
import type { Insight } from "@/lib/types";

interface CoachInsightsProps {
  insights: Insight[];
}

export function CoachInsights({ insights }: CoachInsightsProps) {
  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-2 border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">AI Insights</p>
        <span className="rounded-full bg-gradient-to-r from-gradient-start to-gradient-end px-2 py-0.5 text-[9px] font-semibold text-white">
          Live
        </span>
      </div>
      <div className="flex flex-col gap-0">
        {insights.map((insight, i) => (
          <div key={i} className="px-3 py-1">
            <InsightCard insight={insight} />
          </div>
        ))}
      </div>
    </div>
  );
}
