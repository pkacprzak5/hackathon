"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  SquatPhase,
  SquatRepResult,
  SquatSessionState,
  ServerMessage,
} from "@/lib/squat-types";
import { speakCoaching } from "@/lib/tts";

// 3:4 portrait, same resolution everywhere
const CAPTURE_INTERVAL = 40; // ms, 25fps
const CAPTURE_WIDTH = 480;
const CAPTURE_HEIGHT = 640;
const JPEG_QUALITY = 0.5;

// Circular frame buffer
const BUFFER_SIZE = 250; // max frames (10 seconds)
const BUFFER_FILL_COUNT = 8; // initial fill before playback (~0.32s)
const FRAME_DISPLAY_INTERVAL = 40; // ms, 25fps playback

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

  const [frame, setFrame] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const displayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

  // Circular buffer — fixed-size array, write/read indices
  const frameBuffer = useRef<(string | null)[]>(new Array(BUFFER_SIZE).fill(null));
  const writeIndex = useRef(0);
  const readIndex = useRef(0);
  const isReady = useRef(false);
  const framesReceived = useRef(0);

  const handleMessage = useCallback((event: MessageEvent) => {
    if (typeof event.data !== "string") {
      console.log("Non-string message received:", typeof event.data, event.data);
      return;
    }

    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch (e) {
      console.log("Failed to parse message, len:", event.data.length, "first 30:", event.data.substring(0, 30));
      return;
    }

    switch (msg.type) {
      case "calibration":
        console.log("Calibration:", msg.status, msg.progress);
        setState((s) => ({
          ...s,
          status: msg.status === "complete" ? "active" : "calibrating",
          calibrationProgress: msg.progress,
        }));
        break;

      case "new_frame": {
        // Store as data URI in circular buffer
        if (framesReceived.current % 25 === 0) {
          console.log("Received frame", framesReceived.current, "buffer write:", writeIndex.current, "read:", readIndex.current, "ready:", isReady.current);
        }
        frameBuffer.current[writeIndex.current] = `data:image/jpeg;base64,${msg.data}`;
        writeIndex.current = (writeIndex.current + 1) % BUFFER_SIZE;
        framesReceived.current++;

        if (!isReady.current && framesReceived.current >= BUFFER_FILL_COUNT) {
          isReady.current = true;
        }
        break;
      }

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

  // Display frames from buffer at steady 25fps — exactly like TrafficTracking
  const startDisplay = useCallback(() => {
    if (displayIntervalRef.current) return;

    displayIntervalRef.current = setInterval(() => {
      if (!isReady.current) return;

      const currentFrame = frameBuffer.current[readIndex.current];
      if (currentFrame) {
        setFrame(currentFrame);
        readIndex.current = (readIndex.current + 1) % BUFFER_SIZE;
      }
    }, FRAME_DISPLAY_INTERVAL);
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
      if (video.readyState < 2) return;
      ctx.drawImage(video, 0, 0, CAPTURE_WIDTH, CAPTURE_HEIGHT);
      offscreen.toBlob(
        (blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }, CAPTURE_INTERVAL);

    startDisplay();
  }, [startDisplay]);

  const startSession = useCallback(async () => {
    setState((s) => ({ ...s, status: "connecting" }));

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera API not available. On mobile, this requires HTTPS.",
        );
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: CAPTURE_WIDTH },
          height: { ideal: CAPTURE_HEIGHT },
          aspectRatio: { ideal: 3 / 4 },
        },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await new Promise<void>((resolve) => {
          const v = videoRef.current!;
          if (v.readyState >= 2) { resolve(); return; }
          v.onloadeddata = () => resolve();
        });
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
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("Connected to server");
      setState((s) => ({ ...s, status: "calibrating", calibrationProgress: 0 }));
      startCapture();
    };
    ws.onmessage = handleMessage;
    ws.onclose = () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (displayIntervalRef.current) clearInterval(displayIntervalRef.current);
    };
    ws.onerror = () => {
      ws.close();
      setState((s) => ({
        ...s,
        status: "idle",
        coachingText: "Cannot connect to server. Accept the backend certificate first.",
      }));
    };
  }, [handleMessage, startCapture]);

  const endSession = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (displayIntervalRef.current) {
      clearInterval(displayIntervalRef.current);
      displayIntervalRef.current = null;
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
    frameBuffer.current = new Array(BUFFER_SIZE).fill(null);
    writeIndex.current = 0;
    readIndex.current = 0;
    isReady.current = false;
    framesReceived.current = 0;
    setFrame("");
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (displayIntervalRef.current) clearInterval(displayIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  return {
    state,
    frame,
    videoRef,
    startSession,
    endSession,
  };
}
