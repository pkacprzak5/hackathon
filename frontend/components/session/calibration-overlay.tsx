"use client";

interface CalibrationOverlayProps {
  progress: number;
}

export function CalibrationOverlay({ progress }: CalibrationOverlayProps) {
  const percent = Math.round(progress * 100);

  return (
    <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60">
      <div className="flex flex-col items-center gap-4 rounded-2xl bg-bg-card/90 p-8 backdrop-blur-sm">
        <div className="text-lg font-semibold text-text-primary">Stand Still</div>
        <p className="text-sm text-text-secondary">Calibrating your position...</p>

        <div className="h-2 w-48 overflow-hidden rounded-full bg-bg-surface">
          <div
            className="h-full rounded-full bg-gradient-to-r from-gradient-start to-gradient-end transition-all duration-200"
            style={{ width: `${percent}%` }}
          />
        </div>
        <p className="text-xs text-text-muted">{percent}%</p>
      </div>
    </div>
  );
}
