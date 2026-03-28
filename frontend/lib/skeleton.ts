import type { Landmark } from "./squat-types";

// MediaPipe BlazePose skeleton connections (matching squat_coach/pose/landmarks.py)
export const SKELETON_CONNECTIONS: [number, number][] = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  [27, 29], [27, 31], // left foot
  [28, 30], [28, 32], // right foot
];

const MIN_VISIBILITY = 0.5;

function visibilityColor(v: number): string {
  if (v > 0.8) return "#22C55E"; // green
  if (v > 0.6) return "#F59E0B"; // yellow
  return "#EF4444"; // red
}

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  width: number,
  height: number,
): void {
  // Draw connections
  for (const [i, j] of SKELETON_CONNECTIONS) {
    const a = landmarks[i];
    const b = landmarks[j];
    if (a.visibility < MIN_VISIBILITY || b.visibility < MIN_VISIBILITY) continue;

    ctx.strokeStyle = visibilityColor(Math.min(a.visibility, b.visibility));
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x * width, a.y * height);
    ctx.lineTo(b.x * width, b.y * height);
    ctx.stroke();
  }

  // Draw landmark dots
  for (const lm of landmarks) {
    if (lm.visibility < MIN_VISIBILITY) continue;
    ctx.fillStyle = visibilityColor(lm.visibility);
    ctx.beginPath();
    ctx.arc(lm.x * width, lm.y * height, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

export function drawAngleLabel(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  jointIndex: number,
  angle: number,
  width: number,
  height: number,
): void {
  const lm = landmarks[jointIndex];
  if (lm.visibility < MIN_VISIBILITY) return;

  const x = lm.x * width + 12;
  const y = lm.y * height - 8;

  ctx.font = "bold 12px monospace";
  ctx.fillStyle = "#FFFFFF";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 2;
  const text = `${Math.round(angle)}°`;
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}
