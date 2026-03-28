"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
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
  const renderedRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  const handleJsonMessage = useCallback((data: string) => {
    const msg: ServerMessage = JSON.parse(data);

    switch (msg.type) {
      case "calibration":
        setState((s) => ({
          ...s,
          status: msg.status === "complete" ? "active" : "calibrating",
          calibrationProgress: msg.progress,
        }));
        break;

      case "frame": {
        const d = msg.data;
        setState((s) => {
          const next = { ...s };
          if (d.phase) next.phase = d.phase as SquatPhase;
          if (d.knee_angle !== undefined) next.angles = { ...next.angles, knee: d.knee_angle };
          if (d.hip_angle !== undefined) next.angles = { ...next.angles, hip: d.hip_angle };
          if (d.torso_angle !== undefined) next.angles = { ...next.angles, torso: d.torso_angle };
          if (d.score !== undefined) next.score = d.score;
          if (d.confidence !== undefined) next.confidence = d.confidence;
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

  const handleBinaryFrame = useCallback((data: Blob) => {
    // Display the server-rendered frame (with skeleton overlay) in the <img>
    const img = renderedRef.current;
    if (!img) return;
    const url = URL.createObjectURL(data);
    const prev = img.src;
    img.src = url;
    // Revoke the previous blob URL to avoid memory leak
    if (prev && prev.startsWith("blob:")) {
      URL.revokeObjectURL(prev);
    }
  }, []);

  const handleMessage = useCallback((event: MessageEvent) => {
    if (event.data instanceof Blob) {
      handleBinaryFrame(event.data);
    } else {
      handleJsonMessage(event.data);
    }
  }, [handleBinaryFrame, handleJsonMessage]);

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

    const wsUrl = process.env.NEXT_PUBLIC_ANALYSIS_WS_URL || "ws://localhost:8000/ws/session";
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "blob"; // receive binary frames as Blob
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
      setState((s) => ({
        ...s,
        status: "idle",
        coachingText: `Cannot connect to analysis server at ${wsUrl}. ` +
          "If using HTTPS, make sure you've accepted the certificate for the backend URL first " +
          "(open it in a browser tab and accept the warning).",
      }));
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
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

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
    renderedRef,
    startSession,
    endSession,
  };
}
