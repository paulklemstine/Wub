// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).

export type FieldMode = 'pressure' | 'velocity' | 'air';

export const C = {
  bg: '#0b0e14',
  panel: '#121722',
  grid: '#1f2735',
  grid2: '#2a3446',
  text: '#e7ebf3',
  muted: '#8792a6',
  amber: '#f6a823',
  amberSoft: 'rgba(246,168,35,0.25)',
  teal: '#2ad4bd',
  tealSoft: 'rgba(42,212,189,0.22)',
  violet: '#a98bfa',
  violetSoft: 'rgba(169,139,250,0.25)',
  rose: '#f7708a',
  sky: '#45b8f5',
  bone: '#d8d2c2',
  tooth: '#f4eedf',
  tissue: '#c86b7c',
  tongue: '#df6f87',
  lip: '#e07a8b',
  skin: '#2a2230',
};

/** Colour (0..255) of the sound field inside the tract: teal (low) to amber (high). */
export function fieldRgb(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  const a: [number, number, number] = [16, 60, 80];
  const b: [number, number, number] = [42, 212, 189];
  const c: [number, number, number] = [246, 168, 35];
  const d: [number, number, number] = [255, 236, 190];
  if (x < 0.4) {
    const t = x / 0.4;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  if (x < 0.8) {
    const t = (x - 0.4) / 0.4;
    return [b[0] + (c[0] - b[0]) * t, b[1] + (c[1] - b[1]) * t, b[2] + (c[2] - b[2]) * t];
  }
  const t = (x - 0.8) / 0.2;
  return [c[0] + (d[0] - c[0]) * t, c[1] + (d[1] - c[1]) * t, c[2] + (d[2] - c[2]) * t];
}

/** The field colour as a CSS colour. */
export function fieldColor(v: number, alpha = 1): string {
  const [r, g, bl] = fieldRgb(v);
  return `rgba(${r | 0},${g | 0},${bl | 0},${alpha})`;
}
