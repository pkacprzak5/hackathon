"use client";

import { useEffect, type RefObject } from "react";

import { drawAngleLabel, drawSkeleton } from "@/lib/skeleton";
import type { Landmark, SquatAngles } from "@/lib/squat-types";

interface OverlayCanvasProps {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  landmarks: Landmark[] | null;
  angles: SquatAngles;
  width: number;
  height: number;
}

export function OverlayCanvas({ canvasRef, landmarks, angles, width, height }: OverlayCanvasProps) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;

    ctx.clearRect(0, 0, width, height);

    if (!landmarks) return;

    drawSkeleton(ctx, landmarks, width, height);

    if (angles.knee > 0) {
      drawAngleLabel(ctx, landmarks, 25, angles.knee, width, height);
    }
    if (angles.hip > 0) {
      drawAngleLabel(ctx, landmarks, 23, angles.hip, width, height);
    }
  }, [canvasRef, landmarks, angles, width, height]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full"
      style={{ width, height }}
    />
  );
}
