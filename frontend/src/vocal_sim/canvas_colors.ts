// Ported from MithrilMan/voice-workshop (Apache-2.0, Copyright 2026 Fabio Angela).

export type FieldMode = 'pressure' | 'velocity' | 'air';

export const C = {
  bg: '#080416',
  panel: '#120a26',
  grid: '#261844',
  grid2: '#392366',
  text: '#f5efff',
  muted: '#a495c9',
  amber: '#ffd23f',
  amberSoft: 'rgba(255,210,63,0.28)',
  teal: '#2ad4bd',
  tealSoft: 'rgba(42,212,189,0.25)',
  violet: '#c084fc',
  violetSoft: 'rgba(192,132,252,0.28)',
  rose: '#ff5cd6',
  sky: '#45e0ff',
  bone: '#e6d8ff',
  tooth: '#ffffff',
  tissue: '#d957a8',
  tongue: '#ff5ca8',
  lip: '#ff66c4',
  skin: '#291547',
};

/** Colour (0..255) of the rave standing-wave sound field: violet-cyan -> neon magenta -> rave gold -> white. */
export function fieldRgb(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  const a: [number, number, number] = [28, 16, 72];
  const b: [number, number, number] = [42, 232, 212];
  const c: [number, number, number] = [255, 76, 216];
  const d: [number, number, number] = [255, 224, 96];
  if (x < 0.38) {
    const t = x / 0.38;
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  if (x < 0.74) {
    const t = (x - 0.38) / 0.36;
    return [b[0] + (c[0] - b[0]) * t, b[1] + (c[1] - b[1]) * t, b[2] + (c[2] - b[2]) * t];
  }
  const t = (x - 0.74) / 0.26;
  return [c[0] + (d[0] - c[0]) * t, c[1] + (d[1] - c[1]) * t, c[2] + (d[2] - c[2]) * t];
}

/** The field colour as a CSS colour. */
export function fieldColor(v: number, alpha = 1): string {
  const [r, g, bl] = fieldRgb(v);
  return `rgba(${r | 0},${g | 0},${bl | 0},${alpha})`;
}
