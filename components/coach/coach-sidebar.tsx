"use client";

import { ParticipantCard } from "@/components/ui/participant-card";
import type { Participant } from "@/lib/types";

interface CoachSidebarProps {
  participants: Participant[];
}

export function CoachSidebar({ participants }: CoachSidebarProps) {
  return (
    <aside className="flex h-full w-[220px] flex-col border-r border-border bg-bg-card">
      <div className="border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">Participants ({participants.length})</p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {participants.map((p, i) => (
          <ParticipantCard
            key={p.id}
            participant={p}
            className={i % 2 === 1 ? "bg-bg-elevated" : ""}
          />
        ))}
      </div>
      <div className="border-t border-border-light bg-bg-elevated px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-success" />
          <span className="text-xs font-semibold text-success">Live · Squat</span>
        </div>
        <p className="text-[10px] text-text-muted">Set 2 / 3 · 10 reps</p>
      </div>
    </aside>
  );
}
