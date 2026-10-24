// Mirrors wub_core/schema.py — Phase 3 strict types.
export type StemName = "drums" | "bass" | "vocals" | "other";

export interface Segment {
  id: string;
  track_id: string;
  stem: StemName;
  start: number;
  duration: number;
  is_silent: boolean;
  loudness_start: number;
  loudness_max: number;
  loudness_max_time: number;
  pitches: number[]; // 12
  timbre: number[]; // 12
  spectral_centroid: number;
  zero_crossing_start_offset: number;
}

export interface Tatum {
  id: string;
  index: number;
  bar_position: number;
  beat_position: number;
  tatum_position: number;
  start: number;
  duration: number;
  confidence: number;
  stem_segments: Record<StemName, string[]>;
}

export interface TrackAnalysis {
  track_id: string;
  filename: string;
  bpm: number;
  key: string;
  duration: number;
  sample_rate: number;
  stems_available: StemName[];
  tatums: Tatum[];
  segments: Segment[];
}
