"use client";

import { JointAngleRow } from "@/components/ui/joint-angle-row";
import { PARTICIPANT_COLORS } from "@/lib/mock-data";
import type { Participant } from "@/lib/types";

interface CoachJointsProps {
  participants: Participant[];
}

export function CoachJoints({ participants }: CoachJointsProps) {
  return (
    <div className="flex flex-col">
      <div className="border-b border-border-light px-4 py-3">
        <p className="text-sm font-bold text-text-primary">Joint Angles</p>
      </div>
      <div className="flex items-center gap-2 bg-bg-elevated px-4 py-1.5">
        <span className="w-2" />
        <span className="w-16 text-[10px] font-semibold text-text-muted">Name</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Knee</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Back</span>
        <span className="w-12 text-[10px] font-semibold text-text-muted">Hip</span>
      </div>
      {participants.map((p) => (
        <JointAngleRow
          key={p.id}
          name={p.username}
          color={PARTICIPANT_COLORS[p.id] ?? "#8B5CF6"}
          angles={p.jointAngles}
        />
      ))}
    </div>
  );
}
