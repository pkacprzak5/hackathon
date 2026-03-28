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
    <div className="flex flex-col bg-bg-primary px-6">
      {/* Header */}
      <div className="pt-12 pb-4">
        <h1 className="font-heading text-[28px] font-extrabold text-text-primary">History</h1>
      </div>

      {/* Weekly chart */}
      <div>
        <WeeklyChart />
      </div>

      {/* Quick stats */}
      <div className="flex gap-4 pt-4">
        <div className="flex-1 rounded-3xl bg-bg-card p-5">
          <StatBlock label="Avg Score" value="84" valueClassName="text-lg text-gradient-start" />
        </div>
        <div className="flex-1 rounded-3xl bg-bg-card p-5">
          <StatBlock label="Sessions" value="12" valueClassName="text-lg" />
        </div>
        <div className="flex-1 rounded-3xl bg-bg-card p-5">
          <StatBlock label="Streak" value="4d" valueClassName="text-lg text-success" />
        </div>
      </div>

      {/* Recent sessions */}
      <div className="pt-4 pb-4">
        <h2 className="mb-3 text-sm font-bold text-text-primary">Recent Sessions</h2>
        <div className="flex flex-col gap-3">
          {recentSessions.map((session, i) => (
            <div key={i} className="flex items-center gap-3 rounded-3xl bg-bg-card p-5">
              <div className="flex-1">
                <p className="text-sm font-semibold text-text-primary">{session.exercise}</p>
                <p className="text-xs text-text-secondary">{session.date} · {session.reps}</p>
              </div>
              <span className="font-heading text-xl font-extrabold text-gradient-start">{session.score}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
