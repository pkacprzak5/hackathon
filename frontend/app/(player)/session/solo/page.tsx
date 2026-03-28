"use client";

import { ArrowLeft, Play, Square } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CalibrationOverlay } from "@/components/session/calibration-overlay";
import { OverlayCanvas } from "@/components/session/overlay-canvas";
import { SessionHud } from "@/components/session/session-hud";
import { useSquatSession } from "@/hooks/use-squat-session";

export default function SoloSessionPage() {
  const router = useRouter();
  const { state, videoRef, canvasRef, accRef, startSession, endSession } = useSquatSession();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

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
    router.push("/results");
  };

  return (
    <div className="flex h-full flex-col bg-bg-primary">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <button onClick={() => { endSession(); router.push("/home"); }}>
          <ArrowLeft className="h-5 w-5 text-text-primary" />
        </button>
        <p className="text-sm font-semibold text-text-primary">Squat Coach</p>
        <div className="w-5" />
      </div>

      {/* Video + overlay container */}
      <div ref={containerRef} className="relative mx-4 flex-1 overflow-hidden rounded-xl bg-camera-bg">
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
          <div className="absolute inset-0 flex items-center justify-center bg-black/40">
            <button
              onClick={startSession}
              className="flex items-center gap-2 rounded-2xl bg-gradient-to-r from-gradient-start to-gradient-end px-6 py-3 text-sm font-semibold text-white"
            >
              <Play className="h-4 w-4" />
              Start Session
            </button>
          </div>
        )}
      </div>

      {/* Bottom controls */}
      <div className="flex items-center justify-center px-4 py-4">
        {state.status === "active" && (
          <button
            onClick={handleEnd}
            className="flex items-center gap-2 rounded-2xl bg-error px-6 py-3 text-sm font-semibold text-white"
          >
            <Square className="h-4 w-4" />
            End Session
          </button>
        )}

        {state.status === "idle" && (
          <p className="text-sm text-text-secondary">Tap Start to begin your squat session</p>
        )}

        {(state.status === "calibrating" || state.status === "connecting") && (
          <p className="text-sm text-text-secondary">
            {state.status === "connecting" ? "Connecting..." : "Calibrating..."}
          </p>
        )}
      </div>
    </div>
  );
}
