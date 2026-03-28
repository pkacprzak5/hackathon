import { Sparkles } from "lucide-react";

import type { Insight } from "@/lib/types";
import { cn } from "@/lib/utils";

const insightStyles = {
  warning: { bg: "bg-insight-error-bg", title: "text-error" },
  success: { bg: "bg-insight-success-bg", title: "text-success" },
  tip: { bg: "bg-bg-card/80 backdrop-blur-md border border-border-light/50", title: "text-text-primary" },
};

interface InsightCardProps {
  insight: Insight;
  className?: string;
}

export function InsightCard({ insight, className }: InsightCardProps) {
  const styles = insightStyles[insight.type];

  return (
    <div className={cn("flex gap-3 rounded-[18px] p-3", styles.bg, className)}>
      {insight.type === "tip" ? (
        <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-gradient-start" />
      ) : (
        <div className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", insight.type === "warning" ? "bg-error" : "bg-success")} />
      )}
      <div>
        <p className={cn("text-xs font-semibold", styles.title)}>{insight.title}</p>
        <p className="text-xs text-text-secondary">{insight.message}</p>
      </div>
    </div>
  );
}
