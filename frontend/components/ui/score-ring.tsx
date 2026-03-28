"use client";

import { cn } from "@/lib/utils";

interface ScoreRingProps {
  score: number;
  size?: number;
  strokeWidth?: number;
  className?: string;
}

export function ScoreRing({ score, size = 120, strokeWidth = 8, className }: ScoreRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const center = size / 2;

  return (
    <div className={cn("relative inline-flex items-center justify-center", className)}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={center} cy={center} r={radius} fill="#F4F4F5" stroke="currentColor" strokeWidth={strokeWidth} className="text-border-light dark:fill-[#1A1A2E]" />
        <circle cx={center} cy={center} r={radius} fill="none" stroke="url(#scoreGradient)" strokeWidth={strokeWidth} strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" />
        <defs>
          <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--gradient-start)" />
            <stop offset="100%" stopColor="var(--gradient-end)" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-heading text-5xl font-extrabold text-text-primary">{score}</span>
        <span className="text-xs text-text-secondary">score</span>
      </div>
    </div>
  );
}
