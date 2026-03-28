import type { Insight } from "@/lib/types";
import { cn } from "@/lib/utils";

const insightStyles = {
  warning: { bg: "bg-insight-error-bg", dot: "bg-error", title: "text-error" },
  success: { bg: "bg-insight-success-bg", dot: "bg-success", title: "text-success" },
  tip: { bg: "bg-insight-warning-bg", dot: "bg-warning", title: "text-warning" },
};

interface InsightCardProps {
  insight: Insight;
  className?: string;
}

export function InsightCard({ insight, className }: InsightCardProps) {
  const styles = insightStyles[insight.type];

  return (
    <div className={cn("flex gap-3 rounded-xl p-3", styles.bg, className)}>
      <div className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", styles.dot)} />
      <div>
        <p className={cn("text-xs font-semibold", styles.title)}>{insight.title}</p>
        <p className="text-xs text-text-secondary">{insight.message}</p>
      </div>
    </div>
  );
}
