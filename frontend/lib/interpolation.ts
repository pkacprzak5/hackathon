import type { Landmark } from "./squat-types";

export function lerpLandmarks(
  from: Landmark[],
  to: Landmark[],
  t: number,
): Landmark[] {
  return from.map((f, i) => ({
    x: f.x + (to[i].x - f.x) * t,
    y: f.y + (to[i].y - f.y) * t,
    z: f.z + (to[i].z - f.z) * t,
    visibility: to[i].visibility,
  }));
}

export function parseLandmarks(raw: number[][]): Landmark[] {
  return raw.map(([x, y, z, visibility]) => ({ x, y, z, visibility }));
}
