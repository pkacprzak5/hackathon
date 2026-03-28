"use client";

import { WeeklyChart } from "@/components/charts/weekly-chart";
import { StatBlock } from "@/components/ui/stat-block";

const recentSessions = [
  { date: "Today", exercise: "Squat", score: 87, reps: "3×10" },
  { date: "Yesterday", exercise: "Deadlift", score: 82, reps: "3×8" },
  { date: "Mar 26", exercise: "Squat", score: 90, reps: "3×10" },
  { date: "Mar 25", exercise: "Squat", score: 78, reps: "3×10" },
];

export default function HistoryPage() {
  return (
    <div className="flex flex-col bg-bg-surface">
      {/* Header */}
      <div className="px-5 pt-12 pb-4">
        <h1 className="text-2xl font-bold text-text-primary">History</h1>
      </div>

      {/* Weekly chart */}
      <div className="px-5">
        <WeeklyChart />
      </div>

      {/* Quick stats */}
      <div className="flex gap-3 px-5 pt-4">
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Avg Score" value="84" valueClassName="text-lg text-gradient-start" />
        </div>
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Sessions" value="12" valueClassName="text-lg" />
        </div>
        <div className="flex-1 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
          <StatBlock label="Streak" value="4d" valueClassName="text-lg text-success" />
        </div>
      </div>

      {/* Recent sessions */}
      <div className="px-5 pt-4 pb-4">
        <h2 className="mb-3 text-sm font-bold text-text-primary">Recent Sessions</h2>
        <div className="flex flex-col gap-2">
          {recentSessions.map((session, i) => (
            <div key={i} className="flex items-center gap-3 rounded-2xl bg-bg-card p-4 ring-1 ring-border-light">
              <div className="flex-1">
                <p className="text-sm font-semibold text-text-primary">{session.exercise}</p>
                <p className="text-xs text-text-secondary">{session.date} · {session.reps}</p>
              </div>
              <span className="text-lg font-bold text-gradient-start">{session.score}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
