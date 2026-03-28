import { cn } from "@/lib/utils";

interface AngleChartProps {
  data?: number[];
  label?: string;
  color?: string;
  className?: string;
}

export function AngleChart({
  data = [85, 88, 82, 90, 87, 84, 92, 88, 86, 90],
  label = "Knee Angle",
  color = "var(--gradient-start)",
  className,
}: AngleChartProps) {
  const width = 320;
  const height = 100;
  const padding = 8;
  const maxVal = Math.max(...data);
  const minVal = Math.min(...data);
  const range = maxVal - minVal || 1;

  const points = data
    .map((val, i) => {
      const x = padding + (i / (data.length - 1)) * (width - padding * 2);
      const y = height - padding - ((val - minVal) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className={cn("rounded-xl bg-bg-card p-3", className)}>
      <p className="mb-2 text-xs font-semibold text-text-primary">{label}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <polygon
          points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
          fill={color}
          opacity="0.1"
        />
      </svg>
      <div className="mt-1 flex justify-between text-[9px] text-text-muted">
        <span>Rep 1</span>
        <span>Rep {data.length}</span>
      </div>
    </div>
  );
}
