"use client";

import { Bell, ChevronRight, Flame, Timer, Users } from "lucide-react";
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
    <div className="flex flex-col px-6">
      {/* Clean header row */}
      <div className="flex items-center justify-between pt-12 pb-6">
        <h1 className="bg-linear-to-r from-gradient-start to-gradient-end bg-clip-text font-heading text-[28px] font-extrabold text-transparent">
          GymAI
        </h1>
        <Bell className="h-6 w-6 text-text-secondary" />
      </div>

      {/* Score section */}
      <div className="flex flex-col items-center pb-6">
        <ScoreRing score={87} size={160} strokeWidth={8} />
        <p className="mt-3 text-sm font-medium text-text-secondary">Average Form Score</p>
      </div>

      {/* Stats row */}
      <div className="flex gap-4 pb-5">
        <div className="flex flex-1 flex-col gap-2 rounded-3xl bg-bg-card p-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-linear-to-br from-gradient-start/12 to-gradient-end/12">
            <Flame className="h-6 w-6 text-gradient-start" />
          </div>
          <span className="font-heading text-[28px] font-extrabold text-text-primary">12</span>
          <span className="text-[13px] font-medium text-text-secondary">Total Sessions</span>
        </div>
        <div className="flex flex-1 flex-col gap-2 rounded-3xl bg-bg-card p-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-linear-to-br from-gradient-start/12 to-gradient-end/12">
            <Timer className="h-6 w-6 text-gradient-start" />
          </div>
          <span className="font-heading text-[28px] font-extrabold text-text-primary">4h</span>
          <span className="text-[13px] font-medium text-text-secondary">Training Time</span>
        </div>
      </div>

      {/* Exercises */}
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

      {/* Multiplayer card */}
      <div className="mt-5">
        <button
          onClick={() => router.push("/session/multi")}
          className="flex w-full items-center gap-3 rounded-3xl bg-linear-to-r from-gradient-start to-gradient-end p-5"
        >
          <Users className="h-5 w-5 text-white" />
          <div className="flex-1 text-left">
            <p className="text-sm font-semibold text-white">Multiplayer Session</p>
            <p className="text-xs text-white/70">{count > 0 ? `${count} active` : "Join the room"}</p>
          </div>
          <ChevronRight className="h-5 w-5 text-white/70" />
        </button>
      </div>
    </div>
  );
}
