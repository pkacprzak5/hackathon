"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { parseLandmarks } from "@/lib/interpolation";
import type {
  Landmark,
  SquatAngles,
  SquatPhase,
  SquatRepResult,
  SquatSessionState,
  ServerMessage,
} from "@/lib/squat-types";
import { speakCoaching } from "@/lib/tts";

const CAPTURE_INTERVAL = 42; // ms, ~24fps
const CAPTURE_WIDTH = 640;
const CAPTURE_HEIGHT = 480;
const JPEG_QUALITY = 0.7;

interface AccumulatedState {
  landmarks: Landmark[] | null;
  prevLandmarks: Landmark[] | null;
  lastUpdateTime: number;
}

export function useSquatSession() {
  const [state, setState] = useState<SquatSessionState>({
    status: "idle",
    calibrationProgress: 0,
    landmarks: null,
    phase: "TOP",
    angles: { knee: 0, hip: 0, torso: 0 },
    score: 0,
    confidence: 0,
    repCount: 0,
    reps: [],
    currentFaults: [],
    coachingText: null,
  });

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const accRef = useRef<AccumulatedState>({
    landmarks: null,
    prevLandmarks: null,
    lastUpdateTime: 0,
  });

  const handleMessage = useCallback((event: MessageEvent) => {
    const msg: ServerMessage = JSON.parse(event.data);

    switch (msg.type) {
      case "calibration":
        setState((s) => ({
          ...s,
          status: msg.status === "complete" ? "active" : "calibrating",
          calibrationProgress: msg.progress,
        }));
        break;

      case "frame": {
        const data = msg.data;
        setState((s) => {
          const next = { ...s };
          if (data.phase) next.phase = data.phase as SquatPhase;
          if (data.knee_angle !== undefined) next.angles = { ...next.angles, knee: data.knee_angle };
          if (data.hip_angle !== undefined) next.angles = { ...next.angles, hip: data.hip_angle };
          if (data.torso_angle !== undefined) next.angles = { ...next.angles, torso: data.torso_angle };
          if (data.score !== undefined) next.score = data.score;
          if (data.confidence !== undefined) next.confidence = data.confidence;
          if (data.landmarks) {
            const parsed = parseLandmarks(data.landmarks);
            accRef.current.prevLandmarks = accRef.current.landmarks;
            accRef.current.landmarks = parsed;
            accRef.current.lastUpdateTime = performance.now();
            next.landmarks = parsed;
          }
          return next;
        });
        break;
      }

      case "rep":
        setState((s) => ({
          ...s,
          repCount: msg.rep_index,
          reps: [...s.reps, msg as SquatRepResult],
          currentFaults: msg.faults,
          coachingText: msg.coaching_text,
        }));
        speakCoaching(msg.coaching_text);
        break;

      case "coaching":
        setState((s) => ({ ...s, coachingText: msg.text }));
        speakCoaching(msg.text);
        break;

      case "session_end":
        setState((s) => ({ ...s, status: "ended" }));
        break;
    }
  }, []);

  const startCapture = useCallback(() => {
    const video = videoRef.current;
    const ws = wsRef.current;
    if (!video || !ws) return;

    if (!offscreenRef.current) {
      const canvas = document.createElement("canvas");
      canvas.width = CAPTURE_WIDTH;
      canvas.height = CAPTURE_HEIGHT;
      offscreenRef.current = canvas;
    }
    const offscreen = offscreenRef.current;
    const ctx = offscreen.getContext("2d")!;

    captureIntervalRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN) return;
      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      offscreen.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }, CAPTURE_INTERVAL);
  }, []);

  const startSession = useCallback(async () => {
    setState((s) => ({ ...s, status: "connecting" }));

    // Request camera — requires HTTPS on non-localhost (use run.sh which starts HTTPS)
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera API not available. This usually means the page is served over HTTP instead of HTTPS. " +
          "On mobile, camera requires a secure context (HTTPS or localhost)."
        );
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera access failed:", err);
      setState((s) => ({
        ...s,
        status: "idle",
        coachingText: err instanceof Error ? err.message : "Camera access denied",
      }));
      return;
    }

    // Connect WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_ANALYSIS_WS_URL || "ws://localhost:8000/ws/session";
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setState((s) => ({ ...s, status: "calibrating", calibrationProgress: 0 }));
      startCapture();
    };
    ws.onmessage = handleMessage;
    ws.onclose = () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
    };
    ws.onerror = () => {
      ws.close();
      setState((s) => ({ ...s, status: "idle" }));
    };
  }, [handleMessage, startCapture]);

  const endSession = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // Stop camera
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    state,
    videoRef,
    canvasRef,
    accRef,
    startSession,
    endSession,
  };
}
