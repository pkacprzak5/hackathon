import type { JointAngles } from "@/lib/types";
import { cn } from "@/lib/utils";

interface JointAngleRowProps {
  name: string;
  color: string;
  angles: JointAngles;
  className?: string;
}

function angleColor(value: number, joint: "knee" | "back" | "hip"): string {
  if (joint === "back" && value > 35) return "text-error";
  if (joint === "back" && value < 20) return "text-success";
  if (joint === "knee" && value < 80) return "text-warning";
  if (joint === "knee" && value > 90) return "text-success";
  return "text-text-primary";
}

export function JointAngleRow({ name, color, angles, className }: JointAngleRowProps) {
  return (
    <div className={cn("flex items-center gap-2 px-4 py-2.5", className)}>
      <div className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
      <span className="w-16 text-xs font-semibold text-text-primary">{name}</span>
      <span className={cn("w-12 text-xs", angleColor(angles.knee, "knee"))}>{angles.knee}°</span>
      <span className={cn("w-12 text-xs", angleColor(angles.back, "back"))}>{angles.back}°</span>
      <span className={cn("w-12 text-xs", angleColor(angles.hip, "hip"))}>{angles.hip}°</span>
    </div>
  );
}
