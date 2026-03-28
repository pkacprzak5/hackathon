"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  SquatPhase,
  SquatRepResult,
  SquatSessionState,
  ServerMessage,
} from "@/lib/squat-types";
import { speakCoaching } from "@/lib/tts";

// 3:4 portrait, matches container aspect ratio
const CAPTURE_INTERVAL = 40; // ms, 25fps
const CAPTURE_WIDTH = 480;
const CAPTURE_HEIGHT = 640;
const JPEG_QUALITY = 0.5;

// Frame buffer — 8 frames initial fill to avoid underruns
const BUFFER_FILL_SIZE = 8;
const PLAYBACK_INTERVAL = 40; // ms, 25fps playback

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
  const renderedCanvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const playbackRafRef = useRef<number>(0);
  const lastDisplayTime = useRef<number>(0);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);

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

      // Drop oldest if buffer grows too large (caps at 16)
      while (frameBuffer.current.length > BUFFER_FILL_SIZE * 2) {
        frameBuffer.current.shift();
      }

      // Start playback once initial buffer is filled
      if (!bufferReady.current && frameBuffer.current.length >= BUFFER_FILL_SIZE) {
        bufferReady.current = true;
      }
    }
  }, [handleJsonMessage]);

  // Draw a frame onto the display canvas, center-cropped to fit.
  // Canvas dimensions are set once (first frame) — never reset mid-stream
  // because setting canvas.width/height clears the pixel buffer.
  const canvasSized = useRef(false);

  const drawFrameToCanvas = useCallback((img: HTMLImageElement) => {
    const canvas = renderedCanvasRef.current;
    if (!canvas) return;

    // Size canvas to its CSS layout size, but only once
    if (!canvasSized.current) {
      const rect = canvas.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvasSized.current = true;
      }
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = canvas.width;
    const ch = canvas.height;
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;

    if (cw === 0 || ch === 0 || iw === 0 || ih === 0) return;

    // Center-crop: scale to cover, then draw the center portion
    const scale = Math.max(cw / iw, ch / ih);
    const sw = cw / scale;
    const sh = ch / scale;
    const sx = (iw - sw) / 2;
    const sy = (ih - sh) / 2;

    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cw, ch);
  }, []);

  // Playback via requestAnimationFrame — smooth, adaptive timing
  const startPlayback = useCallback(() => {
    if (playbackRafRef.current) return;

    const playFrame = (timestamp: number) => {
      if (!bufferReady.current) {
        playbackRafRef.current = requestAnimationFrame(playFrame);
        return;
      }

      const elapsed = timestamp - lastDisplayTime.current;

      if (elapsed >= PLAYBACK_INTERVAL && frameBuffer.current.length > 0) {
        // Peek at next frame but don't remove yet
        const blob = frameBuffer.current[0];
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          // Frame decoded successfully — now draw and remove from buffer
          drawFrameToCanvas(img);
          frameBuffer.current.shift();
          URL.revokeObjectURL(url);
        };
        img.onerror = () => {
          // Bad frame — drop it, keep current display
          frameBuffer.current.shift();
          URL.revokeObjectURL(url);
        };
        img.src = url;
        lastDisplayTime.current = timestamp;
      }

      playbackRafRef.current = requestAnimationFrame(playFrame);
    };

    playbackRafRef.current = requestAnimationFrame(playFrame);
  }, [drawFrameToCanvas]);

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
          width: { ideal: 480 },
          height: { ideal: 640 },
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
      if (playbackRafRef.current) cancelAnimationFrame(playbackRafRef.current);
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
    if (playbackRafRef.current) {
      cancelAnimationFrame(playbackRafRef.current);
      playbackRafRef.current = 0;
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
    frameBuffer.current = [];
    bufferReady.current = false;
    setState((s) => ({ ...s, status: "ended" }));
  }, []);

  useEffect(() => {
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      if (playbackRafRef.current) cancelAnimationFrame(playbackRafRef.current);
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
    renderedCanvasRef,
    startSession,
    endSession,
  };
}
