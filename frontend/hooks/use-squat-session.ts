"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  SquatPhase,
  SquatRepResult,
  SquatSessionState,
  ServerMessage,
} from "@/lib/squat-types";
import { speakCoaching } from "@/lib/tts";

// Portrait orientation, low res for speed
const CAPTURE_INTERVAL = 42; // ms, ~24fps
const CAPTURE_WIDTH = 360;
const CAPTURE_HEIGHT = 640;
const JPEG_QUALITY = 0.5;

// Frame buffer — adds ~2-5 frames of latency for smooth playback
const BUFFER_SIZE = 3;
const PLAYBACK_INTERVAL = 42; // ms, ~24fps playback

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
  const renderedImgRef = useRef<HTMLImageElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const playbackIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const prevBlobUrl = useRef<string | null>(null);

  // Frame buffer: queue of blob URLs waiting to be displayed
  const frameBuffer = useRef<Blob[]>([]);
  const bufferReady = useRef(false);

  const handleJsonMessage = useCallback((text: string) => {
    const msg: ServerMessage = JSON.parse(text);

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

  const handleMessage = useCallback((event: MessageEvent) => {
    if (typeof event.data === "string") {
      handleJsonMessage(event.data);
    } else if (event.data instanceof Blob) {
      // Push frame into buffer
      frameBuffer.current.push(event.data);

      // Drop oldest frames if buffer grows too large (keep max ~8 frames)
      while (frameBuffer.current.length > BUFFER_SIZE * 2) {
        frameBuffer.current.shift();
      }

      // Mark buffer as ready once we have enough frames
      if (!bufferReady.current && frameBuffer.current.length >= BUFFER_SIZE) {
        bufferReady.current = true;
      }
    }
  }, [handleJsonMessage]);

  // Playback loop — pulls frames from buffer at steady rate
  const startPlayback = useCallback(() => {
    if (playbackIntervalRef.current) return;

    playbackIntervalRef.current = setInterval(() => {
      if (!bufferReady.current) return;

      const blob = frameBuffer.current.shift();
      if (!blob) return;

      const img = renderedImgRef.current;
      if (!img) return;

      const url = URL.createObjectURL(blob);
      img.onload = () => {
        if (prevBlobUrl.current) {
          URL.revokeObjectURL(prevBlobUrl.current);
        }
        prevBlobUrl.current = url;
      };
      img.src = url;
    }, PLAYBACK_INTERVAL);
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

    // Start the playback loop too
    startPlayback();
  }, [startPlayback]);

  const startSession = useCallback(async () => {
    setState((s) => ({ ...s, status: "connecting" }));

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error(
          "Camera API not available. On mobile, this requires HTTPS. " +
          "Make sure you're using https:// not http://",
        );
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 640 },
          height: { ideal: 1136 },
          aspectRatio: { ideal: 9 / 16 },
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
    ws.binaryType = "blob";
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected to", wsUrl);
      setState((s) => ({ ...s, status: "calibrating", calibrationProgress: 0 }));
      startCapture();
    };
    ws.onmessage = handleMessage;
    ws.onclose = (e) => {
      console.log("WebSocket closed:", e.code, e.reason);
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (playbackIntervalRef.current) clearInterval(playbackIntervalRef.current);
    };
    ws.onerror = (e) => {
      console.error("WebSocket error:", e);
      ws.close();
      setState((s) => ({
        ...s,
        status: "idle",
        coachingText: `Cannot connect to server at ${wsUrl}. ` +
          "Accept the backend certificate first: open " +
          wsUrl.replace("wss://", "https://").replace("ws://", "http://").replace("/ws/session", "/health") +
          " in a browser tab.",
      }));
    };
  }, [handleMessage, startCapture]);

  const endSession = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (playbackIntervalRef.current) {
      clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
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
    if (prevBlobUrl.current) {
      URL.revokeObjectURL(prevBlobUrl.current);
      prevBlobUrl.current = null;
    }
    frameBuffer.current = [];
    bufferReady.current = false;
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (playbackIntervalRef.current) clearInterval(playbackIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
      if (prevBlobUrl.current) URL.revokeObjectURL(prevBlobUrl.current);
    };
  }, []);

  return {
    state,
    videoRef,
    renderedImgRef,
    startSession,
    endSession,
  };
}
