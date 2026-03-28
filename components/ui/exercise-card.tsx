import { Dumbbell, Target } from "lucide-react";

import { cn } from "@/lib/utils";

interface ExerciseCardProps {
  name: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  reps?: string;
  className?: string;
  onClick?: () => void;
}

const difficultyColors = {
  Beginner: "bg-success/10 text-success",
  Intermediate: "bg-warning/10 text-warning",
  Advanced: "bg-error/10 text-error",
};

export function ExerciseCard({ name, difficulty, reps, className, onClick }: ExerciseCardProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light transition-colors hover:bg-bg-elevated text-left w-full",
        className
      )}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-gradient-start/20 to-gradient-end/20">
        {name === "Squat" ? (
          <Target className="h-6 w-6 text-gradient-start" />
        ) : (
          <Dumbbell className="h-6 w-6 text-gradient-start" />
        )}
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-text-primary">{name}</p>
        {reps && <p className="text-xs text-text-secondary">{reps}</p>}
      </div>
      <span className={cn("rounded-full px-2 py-0.5 text-[10px] font-semibold", difficultyColors[difficulty])}>
        {difficulty}
      </span>
    </button>
  );
}
