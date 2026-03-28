"use client";

import { Dumbbell } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { cn } from "@/lib/utils";
import { useSessionContext } from "@/providers/session-provider";

export default function EntryPage() {
  const [username, setUsername] = useState("");
  const { setUser } = useSessionContext();
  const router = useRouter();

  function handleJoin(role: "player" | "coach") {
    if (!username.trim()) return;
    setUser(username.trim(), role);
    router.push(role === "coach" ? "/coach" : "/home");
  }

  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-bg-surface px-4">
      {/* Gradient header accent */}
      <div className="fixed top-0 left-0 h-48 w-full bg-gradient-to-r from-gradient-start to-gradient-end opacity-20 blur-3xl" />

      <div className="relative z-10 flex w-full max-w-sm flex-col items-center gap-8">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-gradient-start to-gradient-end">
            <Dumbbell className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-3xl font-bold text-text-primary">GymAI</h1>
        </div>

        <p className="text-center text-text-secondary">
          Real-time exercise form analysis.
          <br />
          Enter your name to get started.
        </p>

        {/* Card */}
        <div className="w-full rounded-3xl bg-bg-card p-6 shadow-lg ring-1 ring-border-light">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleJoin("player")}
            placeholder="Your name"
            className="mb-6 w-full rounded-2xl border border-border bg-bg-surface px-4 py-3 text-text-primary placeholder:text-text-muted outline-none focus:ring-2 focus:ring-gradient-start"
            autoFocus
          />

          <div className="flex flex-col gap-3">
            <button
              onClick={() => handleJoin("player")}
              disabled={!username.trim()}
              className={cn(
                "w-full rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end py-3 text-sm font-semibold text-white transition-opacity",
                username.trim() ? "hover:opacity-90" : "opacity-40 cursor-not-allowed"
              )}
            >
              Join as Player
            </button>

            <button
              onClick={() => handleJoin("coach")}
              disabled={!username.trim()}
              className={cn(
                "w-full rounded-2xl border border-gradient-start py-3 text-sm font-semibold text-gradient-start transition-colors",
                username.trim() ? "hover:bg-gradient-start/10" : "opacity-40 cursor-not-allowed"
              )}
            >
              Join as Coach
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
