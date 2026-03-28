"use client";

import { useEffect, useRef } from "react";

import { cn } from "@/lib/utils";

interface VideoFeedProps {
  stream?: MediaStream | null;
  name: string;
  score?: number;
  color?: string;
  className?: string;
}

export function VideoFeed({ stream, name, score, color = "#8B5CF6", className }: VideoFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <div className={cn("relative overflow-hidden rounded-xl bg-camera-bg", className)}>
      {stream ? (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover"
        />
      ) : (
        <div className="flex h-full w-full items-center justify-center bg-linear-to-b from-camera-bg to-camera-bg-dark">
          <svg width="80" height="160" viewBox="0 0 80 160" className="opacity-40">
            <circle cx="40" cy="20" r="16" fill={color} opacity="0.4" />
            <rect x="37" y="36" width="6" height="50" rx="3" fill={color} opacity="0.6" />
            <rect x="10" y="50" width="30" height="5" rx="2" fill={color} opacity="0.4" />
            <rect x="40" y="50" width="30" height="5" rx="2" fill={color} opacity="0.4" />
            <rect x="27" y="86" width="5" height="40" rx="2" fill={color} opacity="0.4" />
            <rect x="48" y="86" width="5" height="40" rx="2" fill={color} opacity="0.4" />
            <circle cx="29" cy="82" r="5" fill={color} />
            <circle cx="50" cy="82" r="5" fill={color} />
          </svg>
        </div>
      )}

      <div
        className="absolute bottom-2 left-2 rounded-md px-2.5 py-1 text-[11px] font-semibold text-white"
        style={{ backgroundColor: color }}
      >
        {name}
      </div>

      {score !== undefined && (
        <div className="absolute top-2 right-2 rounded-full bg-white/20 px-2.5 py-0.5 text-xs font-bold text-white backdrop-blur-sm">
          {score}
        </div>
      )}
    </div>
  );
}
