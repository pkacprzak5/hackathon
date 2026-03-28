"use client";

import { useEffect, useRef, type RefObject } from "react";

import { lerpLandmarks } from "@/lib/interpolation";
import { drawAngleLabel, drawSkeleton } from "@/lib/skeleton";
import type { Landmark, SquatAngles } from "@/lib/squat-types";

interface OverlayCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  accRef: RefObject<{
    landmarks: Landmark[] | null;
    prevLandmarks: Landmark[] | null;
    lastUpdateTime: number;
  }>;
  angles: SquatAngles;
  width: number;
  height: number;
}

export function OverlayCanvas({ canvasRef, accRef, angles, width, height }: OverlayCanvasProps) {
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;

    function render() {
      ctx.clearRect(0, 0, width, height);
      const acc = accRef.current;

      if (acc.landmarks) {
        const t = acc.prevLandmarks
          ? Math.min((performance.now() - acc.lastUpdateTime) / 42, 1)
          : 1;
        const landmarks = acc.prevLandmarks
          ? lerpLandmarks(acc.prevLandmarks, acc.landmarks, t)
          : acc.landmarks;

        drawSkeleton(ctx, landmarks, width, height);

        if (angles.knee > 0) {
          drawAngleLabel(ctx, landmarks, 25, angles.knee, width, height);
        }
        if (angles.hip > 0) {
          drawAngleLabel(ctx, landmarks, 23, angles.hip, width, height);
        }
      }

      rafRef.current = requestAnimationFrame(render);
    }

    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [canvasRef, accRef, angles, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ width, height }}
    />
  );
}
