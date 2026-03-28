"use client";

import { ChevronRight, Users } from "lucide-react";
import { useRouter } from "next/navigation";

import { ExerciseCard } from "@/components/ui/exercise-card";
import { ScoreRing } from "@/components/ui/score-ring";
import { useParticipants } from "@/hooks/use-participants";
import { useSession } from "@/hooks/use-session";

export default function HomePage() {
  const router = useRouter();
  const { username } = useSession();
  const { count } = useParticipants();

  return (
    <div className="flex flex-col">
      {/* Gradient header */}
      <div className="bg-linear-to-r from-gradient-start to-gradient-end px-5 pb-16 pt-12">
        <p className="text-sm text-white/80">Welcome back,</p>
        <h1 className="text-2xl font-bold text-white">{username}</h1>
      </div>

      {/* Score card overlapping header */}
      <div className="-mt-10 px-5">
        <div className="flex items-center gap-6 rounded-3xl bg-bg-card p-5 shadow-lg ring-1 ring-border-light">
          <ScoreRing score={87} size={100} strokeWidth={7} />
          <div className="flex flex-col gap-1">
            <p className="text-sm font-semibold text-text-primary">Form Score</p>
            <p className="text-xs text-text-secondary">Your average across all exercises</p>
            <div className="mt-1 flex items-center gap-1 text-success">
              <ChevronRight className="h-3 w-3 rotate-[-90deg]" />
              <span className="text-xs font-semibold">+4 this week</span>
            </div>
          </div>
        </div>
      </div>

      {/* Multiplayer quick join */}
      <div className="px-5 pt-5">
        <button
          onClick={() => router.push("/session/multi")}
          className="flex w-full items-center gap-3 rounded-2xl bg-linear-to-r from-gradient2-start to-gradient2-end p-4"
        >
          <Users className="h-5 w-5 text-white" />
          <div className="flex-1 text-left">
            <p className="text-sm font-semibold text-white">Multiplayer Session</p>
            <p className="text-xs text-white/70">{count > 0 ? `${count} active` : "Join the room"}</p>
          </div>
          <ChevronRight className="h-4 w-4 text-white/70" />
        </button>
      </div>

      {/* Exercises */}
      <div className="px-5 pt-5">
        <h2 className="mb-3 text-base font-bold text-text-primary">Exercises</h2>
        <div className="flex flex-col gap-3">
          <ExerciseCard
            name="Squat"
            difficulty="Intermediate"
            reps="3 sets × 10 reps"
            onClick={() => router.push("/session/solo")}
          />
          <ExerciseCard
            name="Deadlift"
            difficulty="Advanced"
            reps="3 sets × 8 reps"
            onClick={() => router.push("/session/solo")}
          />
        </div>
      </div>
    </div>
  );
}
