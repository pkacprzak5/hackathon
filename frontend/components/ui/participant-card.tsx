import { PARTICIPANT_COLORS } from "@/lib/mock-data";
import type { Participant } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ParticipantCardProps {
  participant: Participant;
  className?: string;
}

export function ParticipantCard({ participant, className }: ParticipantCardProps) {
  const color = PARTICIPANT_COLORS[participant.id] ?? "#8B5CF6";

  return (
    <div className={cn("flex items-center gap-3 px-4 py-4", className)}>
      <div
        className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold text-white"
        style={{ backgroundColor: color }}
      >
        {participant.username[0]}
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-text-primary">{participant.username}</p>
        <p className="text-xs text-text-secondary">Rep {participant.repCount} / 10</p>
      </div>
      <div className="text-right">
        <p className="text-xl font-bold" style={{ color }}>{participant.score}</p>
        <p className="text-[9px] text-text-muted">score</p>
      </div>
    </div>
  );
}
