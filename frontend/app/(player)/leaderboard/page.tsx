"use client";

import { useState } from "react";

import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";
import { MOCK_PARTICIPANTS, PARTICIPANT_COLORS } from "@/lib/mock-data";
import { cn } from "@/lib/utils";

const filters = ["All Time", "This Week", "Today"];

export default function LeaderboardPage() {
  const { participants } = useParticipants();
  const { username } = useSession();
  const [activeFilter, setActiveFilter] = useState("All Time");
  const list = participants.length > 0 ? participants : MOCK_PARTICIPANTS;
  const sorted = [...list].sort((a, b) => b.score - a.score);

  return (
    <div className="flex flex-col bg-bg-primary px-6">
      <div className="pt-12 pb-4">
        <h1 className="font-heading text-[28px] font-extrabold text-text-primary">Leaderboard</h1>
      </div>

      {/* Filter pills */}
      <div className="flex gap-2 pb-4">
        {filters.map((filter) => (
          <button
            key={filter}
            onClick={() => setActiveFilter(filter)}
            className={cn(
              "rounded-full px-4 py-2 text-xs font-semibold transition-colors",
              activeFilter === filter
                ? "bg-linear-to-r from-gradient-start to-gradient-end text-white"
                : "bg-bg-card text-text-secondary"
            )}
          >
            {filter}
          </button>
        ))}
      </div>

      {/* Ranking rows */}
      <div className="flex flex-col gap-2">
        {sorted.map((p, i) => {
          const color = PARTICIPANT_COLORS[p.id] ?? "#8B5CF6";
          const isYou = p.username === username;
          return (
            <div
              key={p.id}
              className={cn(
                "flex items-center gap-3 rounded-2xl py-3 px-4",
                isYou
                  ? "bg-purple-soft border-[1.5px] border-gradient-start/30"
                  : "bg-bg-card"
              )}
            >
              <span
                className={cn(
                  "w-6 text-center font-heading text-base font-bold",
                  i === 0
                    ? "bg-linear-to-r from-gradient-start to-gradient-end bg-clip-text text-transparent"
                    : "text-text-secondary"
                )}
              >
                {i + 1}
              </span>
              <div
                className="flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold text-white"
                style={{ backgroundColor: color }}
              >
                {p.username[0]}
              </div>
              <div className="flex-1">
                <p className="text-sm font-semibold text-text-primary">
                  {p.username} {isYou && <span className="text-xs text-gradient-start">(You)</span>}
                </p>
                <p className="text-xs text-text-secondary">Rep {p.repCount} / 10</p>
              </div>
              <span
                className={cn(
                  "font-heading text-xl font-extrabold",
                  isYou ? "text-gradient-start" : ""
                )}
                style={isYou ? undefined : { color }}
              >
                {p.score}
              </span>
            </div>
          );
        })}
      </div>

      {/* Show top 50 link */}
      <button className="mt-4 text-center text-[13px] font-semibold text-gradient-start">
        Show top 50
      </button>
    </div>
  );
}
