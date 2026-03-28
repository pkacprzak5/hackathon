import { cn } from "@/lib/utils";

interface WeeklyChartProps {
  data?: { day: string; value: number }[];
  className?: string;
}

const DEFAULT_DATA = [
  { day: "Mon", value: 72 },
  { day: "Tue", value: 85 },
  { day: "Wed", value: 78 },
  { day: "Thu", value: 90 },
  { day: "Fri", value: 88 },
  { day: "Sat", value: 0 },
  { day: "Sun", value: 0 },
];

export function WeeklyChart({ data = DEFAULT_DATA, className }: WeeklyChartProps) {
  const maxVal = Math.max(...data.map((d) => d.value), 1);

  return (
    <div className={cn("rounded-2xl bg-bg-card p-4", className)}>
      <p className="mb-3 text-sm font-bold text-text-primary">This Week</p>
      <div className="flex items-end gap-2">
        {data.map((d, i) => {
          const height = d.value > 0 ? (d.value / maxVal) * 120 : 4;
          const isToday = i === new Date().getDay() - 1;

          return (
            <div key={d.day} className="flex flex-1 flex-col items-center gap-1">
              {d.value > 0 && (
                <span className="text-[10px] font-semibold text-text-secondary">{d.value}</span>
              )}
              <div
                className={cn(
                  "w-full rounded-t-lg transition-all",
                  d.value > 0
                    ? isToday
                      ? "bg-linear-to-t from-gradient-start to-gradient-end"
                      : "bg-gradient-start/30"
                    : "bg-border-light"
                )}
                style={{ height: `${height}px` }}
              />
              <span className={cn("text-[10px]", isToday ? "font-bold text-gradient-start" : "text-text-muted")}>
                {d.day}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
