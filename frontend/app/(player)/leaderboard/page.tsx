"use client";

import { Medal, Trophy } from "lucide-react";

import { useParticipants } from "@/hooks/use-participants";
import { MOCK_PARTICIPANTS, PARTICIPANT_COLORS } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

export default function LeaderboardPage() {
  const { participants } = useParticipants();
  const list = participants.length > 0 ? participants : MOCK_PARTICIPANTS;
  const sorted = [...list].sort((a, b) => b.score - a.score);

  return (
    <div className="flex flex-col bg-bg-surface">
      <div className="bg-linear-to-r from-gradient-start to-gradient-end px-5 pb-8 pt-12">
        <h1 className="text-2xl font-bold text-white">Leaderboard</h1>
        <p className="text-sm text-white/70">Current session rankings</p>
      </div>

      <div className="-mt-4 px-5">
        <div className="flex flex-col gap-2">
          {sorted.map((p, i) => {
            const color = PARTICIPANT_COLORS[p.id] ?? "#8B5CF6";
            return (
              <div
                key={p.id}
                className={cn(
                  "flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light",
                  i === 0 && "ring-2 ring-warning/50"
                )}
              >
                <div className="flex h-8 w-8 items-center justify-center">
                  {i === 0 ? (
                    <Trophy className="h-6 w-6 text-warning" />
                  ) : i < 3 ? (
                    <Medal className="h-5 w-5 text-text-muted" />
                  ) : (
                    <span className="text-sm font-bold text-text-muted">{i + 1}</span>
                  )}
                </div>
                <div
                  className="flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold text-white"
                  style={{ backgroundColor: color }}
                >
                  {p.username[0]}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-text-primary">{p.username}</p>
                  <p className="text-xs text-text-secondary">Rep {p.repCount} / 10</p>
                </div>
                <span className="text-xl font-bold" style={{ color }}>{p.score}</span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
