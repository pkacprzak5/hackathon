import { ChevronRight, Dumbbell, Target } from "lucide-react";

import { cn } from "@/lib/utils";

interface ExerciseCardProps {
  name: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  reps?: string;
  className?: string;
  onClick?: () => void;
}

export function ExerciseCard({ name, difficulty, reps, className, onClick }: ExerciseCardProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-4 rounded-3xl bg-bg-card p-5 transition-colors hover:bg-bg-elevated text-left w-full",
        className
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-[14px] bg-linear-to-br from-gradient-start/12 to-gradient-end/12">
        {name === "Squat" ? (
          <Target className="h-6 w-6 text-gradient-start" />
        ) : (
          <Dumbbell className="h-6 w-6 text-gradient-start" />
        )}
      </div>
      <div className="flex-1">
        <p className="text-sm font-bold text-text-primary">{name}</p>
        <p className="text-xs text-text-muted">{difficulty}{reps ? ` · ${reps}` : ""}</p>
      </div>
      <ChevronRight className="h-5 w-5 text-text-muted" />
    </button>
  );
}
