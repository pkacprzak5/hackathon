"use client";

import { ArrowLeft, Play, Square, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CalibrationOverlay } from "@/components/session/calibration-overlay";
import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { useSquatSession } from "@/hooks/use-squat-session";
import type { Insight } from "@/lib/types";

export default function SoloSessionPage() {
  const router = useRouter();
  const { state, videoRef, streamUrl, startSession, endSession } = useSquatSession();
  const [sessionTime, setSessionTime] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Session timer
  useEffect(() => {
    if (state.status === "active" && !timerRef.current) {
      timerRef.current = setInterval(() => setSessionTime((t) => t + 1), 1000);
    }
    if (state.status === "ended" || state.status === "idle") {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [state.status]);

  const handleEnd = () => {
    endSession();
    if (typeof window !== "undefined") {
      sessionStorage.setItem("squat_results", JSON.stringify({
        reps: state.reps,
        repCount: state.repCount,
        avgScore: state.reps.length > 0
          ? Math.round(state.reps.reduce((sum, r) => sum + r.scores.total, 0) / state.reps.length)
          : 0,
        sessionTime,
      }));
    }
    router.push("/results");
  };

  const mins = Math.floor(sessionTime / 60);
  const secs = sessionTime % 60;
  const timeStr = `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;

  const avgScore = state.reps.length > 0
    ? Math.round(state.reps.reduce((sum, r) => sum + r.scores.total, 0) / state.reps.length)
    : 0;

  // Build insights from faults and coaching
  const insights: Insight[] = [];
  if (state.coachingText) {
    insights.push({
      participantId: "self",
      type: "tip",
      title: "Coach",
      message: state.coachingText,
      timestamp: Date.now(),
    });
  }
  for (const fault of state.currentFaults.slice(0, 2)) {
    insights.push({
      participantId: "self",
      type: "warning",
      title: fault.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      message: `Detected during rep ${state.repCount}`,
      timestamp: Date.now(),
    });
  }
  if (state.reps.length > 0) {
    const lastRep = state.reps[state.reps.length - 1];
    if (lastRep.scores.total >= 80) {
      insights.push({
        participantId: "self",
        type: "success",
        title: "Good Rep!",
        message: `Rep ${lastRep.rep_index} scored ${Math.round(lastRep.scores.total)}`,
        timestamp: Date.now(),
      });
    }
  }

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => { endSession(); router.push("/home"); }}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <div className="text-center">
          <p className="text-sm font-semibold text-text-primary">Squat Coach</p>
          <p className="text-xs text-text-secondary">{timeStr}</p>
        </div>
        <button onClick={() => router.push("/session/multi")}>
          <Users className="h-5 w-5 text-text-secondary" />
        </button>
      </div>

      {/* Video feed */}
      <div className="relative mx-4 aspect-[3/4] overflow-hidden rounded-xl bg-camera-bg">
        {/* Hidden video — captures camera frames to send to server */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={state.status === "active" ? "hidden" : "h-full w-full object-cover"}
        />

        {/* MJPEG stream from server — skeleton rendered on frames, continuous video */}
        {state.status === "active" && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={streamUrl}
            alt="Squat analysis"
            className="h-full w-full object-cover"
          />
        )}

        {/* Calibration overlay */}
        {state.status === "calibrating" && (
          <CalibrationOverlay progress={state.calibrationProgress} />
        )}

        {/* Idle state */}
        {state.status === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40">
            <button
              onClick={startSession}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end px-6 py-3 text-sm font-semibold text-white"
            >
              <Play className="h-4 w-4" />
              Start Session
            </button>
            {state.coachingText && (
              <p className="mx-8 rounded-lg bg-error/20 px-4 py-2 text-center text-xs text-error">
                {state.coachingText}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Stats bar */}
      <div className="flex items-center gap-4 px-4 py-3">
        <StatBlock label="Rep" value={`${state.repCount}`} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Score" value={`${avgScore || "—"}`} valueClassName="text-gradient-start" />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Phase" value={state.status === "active" ? state.phase : "—"} />
        <div className="h-6 w-px bg-border-light" />
        <StatBlock label="Time" value={timeStr} />
      </div>

      {/* Joint Angles */}
      {state.status === "active" && (
        <div className="flex items-center gap-3 px-4 py-2">
          <div className="flex flex-1 items-center justify-between rounded-lg bg-bg-surface px-3 py-2">
            <span className="text-[10px] text-text-muted">Knee</span>
            <span className="text-sm font-bold text-text-primary">
              {state.angles.knee > 0 ? `${Math.round(state.angles.knee)}°` : "—"}
            </span>
          </div>
          <div className="flex flex-1 items-center justify-between rounded-lg bg-bg-surface px-3 py-2">
            <span className="text-[10px] text-text-muted">Hip</span>
            <span className="text-sm font-bold text-text-primary">
              {state.angles.hip > 0 ? `${Math.round(state.angles.hip)}°` : "—"}
            </span>
          </div>
          <div className="flex flex-1 items-center justify-between rounded-lg bg-bg-surface px-3 py-2">
            <span className="text-[10px] text-text-muted">Torso</span>
            <span className="text-sm font-bold text-text-primary">
              {state.angles.torso > 0 ? `${Math.round(state.angles.torso)}°` : "—"}
            </span>
          </div>
          <div className="flex flex-1 items-center justify-between rounded-lg bg-bg-surface px-3 py-2">
            <span className="text-[10px] text-text-muted">Conf</span>
            <span className={`text-sm font-bold ${state.confidence > 0.7 ? "text-success" : state.confidence > 0.4 ? "text-warning" : "text-error"}`}>
              {state.confidence > 0 ? `${Math.round(state.confidence * 100)}%` : "—"}
            </span>
          </div>
        </div>
      )}

      {/* AI Feedback */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h3 className="mb-2 text-sm font-bold text-text-primary">AI Feedback</h3>
        <div className="flex flex-col gap-2">
          {insights.length > 0 ? (
            insights.map((insight, i) => (
              <InsightCard key={i} insight={insight} />
            ))
          ) : (
            <p className="text-xs text-text-muted">
              {state.status === "active"
                ? "Feedback will appear as you exercise..."
                : "Start a session to get AI coaching feedback"}
            </p>
          )}
        </div>
      </div>

      {/* End session button */}
      {state.status === "active" && (
        <div className="flex items-center justify-center px-4 py-4">
          <button
            onClick={handleEnd}
            className="flex items-center gap-2 rounded-2xl bg-error px-6 py-3 text-sm font-semibold text-white"
          >
            <Square className="h-4 w-4" />
            End Session
          </button>
        </div>
      )}
    </div>
  );
}
