"use client";

import { ArrowLeft, Play, Square, Users } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CalibrationOverlay } from "@/components/session/calibration-overlay";
import { OverlayCanvas } from "@/components/session/overlay-canvas";
import { SessionHud } from "@/components/session/session-hud";
import { InsightCard } from "@/components/ui/insight-card";
import { StatBlock } from "@/components/ui/stat-block";
import { useSquatSession } from "@/hooks/use-squat-session";
import type { Insight } from "@/lib/types";

export default function SoloSessionPage() {
  const router = useRouter();
  const { state, videoRef, canvasRef, accRef, startSession, endSession } = useSquatSession();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
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

  // Track container dimensions for canvas sizing
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const handleEnd = () => {
    endSession();
    // Store session data for results page
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

      {/* Video + overlay container */}
      <div ref={containerRef} className="relative mx-4 aspect-[3/4] overflow-hidden rounded-xl bg-camera-bg">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />

        {/* Canvas overlay for skeleton */}
        {state.status === "active" && dimensions.width > 0 && (
          <OverlayCanvas
            canvasRef={canvasRef}
            accRef={accRef}
            angles={state.angles}
            width={dimensions.width}
            height={dimensions.height}
          />
        )}

        {/* Calibration overlay */}
        {state.status === "calibrating" && (
          <CalibrationOverlay progress={state.calibrationProgress} />
        )}

        {/* Session HUD */}
        {state.status === "active" && (
          <SessionHud
            phase={state.phase}
            repCount={state.repCount}
            score={state.score}
            coachingText={state.coachingText}
          />
        )}

        {/* Idle state: start button */}
        {state.status === "idle" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40">
            <button
              onClick={startSession}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end px-6 py-3 text-sm font-semibold text-white"
            >
              <Play className="h-4 w-4" />
              Start Session
            </button>
            {/* Show error message if camera failed */}
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

      {/* Bottom controls */}
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
