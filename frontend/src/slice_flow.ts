/** Final-assembly critic, patient segment memory, and bounded playback traces.
 * All inference is control-rate on block preparation, never in the RAF loop.
 */
import type { SliceSpec } from "./kit_pcm";
import type { PitchAutomationPoint } from "./stretch_voice";
import type { StemName } from "./types";

export const FLOW_DIM = 32;
export const FLOW_INPUT_DIM = 108;
export const MOTIF_INPUT_DIM = FLOW_DIM * 3 + 4 + 3 + 1;
export const VALUE_INPUT_DIM = FLOW_DIM * 2 + 8;
const MOTIF_FORMAT = "wub-motif-return-v1";
const VALUE_FORMAT = "wub-bar-tour-value-v1";
const STEMS: readonly StemName[] = ["drums", "bass", "other", "vocals"];
interface Network { mean: number[]; std: number[]; w1: number[][]; b1: number[]; w2: number[][]; b2: number[] }
interface MotifArtifact { format: string; input_dim: number; net: Network }
interface ValueArtifact { format: string; input_dim: number; net: Network; value_scale?: number }
interface Artifact {
  format: string; feature_dim: number; phrase_ids: string[]; feature_file: string;
  critic: Network; mix: Network; planner: Network; foundation_seconds: number;
}
export interface FlowEntry {
  phrase: { id: string }; kit: { track_id: string }; stem: StemName;
  keyShift: number; rate: number; gain?: number;
}
export interface FlowTask<T extends FlowEntry> {
  stem: StemName; run: T[]; slices?: SliceSpec[]; baseSemitones?: number;
  pitchPoints?: PitchAutomationPoint[];
  voiceRate?: number;
}
export interface FlowCell {
  phraseId: string; trackId: string; stem: StemName; bar: number;
  startFrac: number; endFrac: number; bars: number; gain: number; endGain: number;
  semitones: number; rate: number; mixerGain?: number;
}
export interface FlowContext {
  at: number; bars: number; bpm: number; section: string; targetEnergy: number;
  sectionChanged: boolean; rotation: boolean; bridge: boolean;
}
export interface SliceFlowTrace {
  id: number; stem: StemName; at: number; endAt: number; section: string;
  cells: FlowCell[]; score: number; returnScore: number; seamFeatures: number[] | null;
}
const clamp = (x: number, lo = 0, hi = 1): number => Math.max(lo, Math.min(hi, x));

export function flowTransitionFeatures(a: ArrayLike<number>, b: ArrayLike<number>, stem: StemName, ctx: readonly number[]): Float32Array {
  const out = new Float32Array(FLOW_INPUT_DIM);
  for (let i = 0; i < FLOW_DIM; i++) {
    out[i] = a[i]; out[FLOW_DIM + i] = b[i]; out[2 * FLOW_DIM + i] = Math.abs(a[i] - b[i]);
  }
  out[96 + STEMS.indexOf(stem)] = 1;
  out.set(ctx, 100);
  return out;
}

export function motifFeatures(a: ArrayLike<number>, b: ArrayLike<number>, stem: number, lag: number, age: number): Float32Array {
  const out = new Float32Array(MOTIF_INPUT_DIM);
  for (let i = 0; i < FLOW_DIM; i++) {
    out[i] = a[i]; out[FLOW_DIM + i] = b[i];
    out[2 * FLOW_DIM + i] = Math.abs(a[i] - b[i]);
  }
  out[3 * FLOW_DIM + stem] = 1;
  out[3 * FLOW_DIM + 4 + lag] = 1;
  out[MOTIF_INPUT_DIM - 1] = age;
  return out;
}

export function valueFeatures(current: ArrayLike<number>, recent: ArrayLike<number>, age: number, repetition: number): Float32Array {
  const out = new Float32Array(VALUE_INPUT_DIM);
  for (let i = 0; i < FLOW_DIM; i++) out[i] = current[i];
  for (let i = 0; i < FLOW_DIM; i++) out[FLOW_DIM + i] = recent[i];
  out[2 * FLOW_DIM] = Math.min(2, age / 60);
  out[2 * FLOW_DIM + 1] = Math.min(1, repetition);
  out[2 * FLOW_DIM + 2] = .5;
  out[2 * FLOW_DIM + 3] = .5;
  out[2 * FLOW_DIM + 4] = .7;
  out[2 * FLOW_DIM + 5] = .25;
  return out;
}

export function describeFlowTask<T extends FlowEntry>(task: FlowTask<T>, bars: number, barSec: number): FlowCell[] {
  const out: FlowCell[] = [];
  let position = 0;
  const push = (entry: T, start: number, end: number, duration: number, gain = 1, endGain = gain): void => {
    let shift = task.baseSemitones ?? entry.keyShift;
    for (const point of task.pitchPoints ?? []) if (point.offsetSec <= position * barSec + 1e-6) shift = point.semitones;
    out.push({ phraseId: entry.phrase.id, trackId: entry.kit.track_id, stem: task.stem,
      bar: Math.min(3, Math.floor(start * 4 + 1e-6)), startFrac: start, endFrac: end,
      bars: duration, gain, endGain, semitones: task.stem === "drums" ? 0 : shift, rate: task.voiceRate ?? task.run[0]?.rate ?? entry.rate,
      mixerGain: task.run[0]?.gain ?? 1 });
    position += duration;
  };
  if (task.slices?.length) {
    for (const slice of task.slices) {
      const entry = task.run[slice.itemIndex];
      if (!entry) continue;
      const duration = slice.targetBars ?? (slice.endFrac - slice.startFrac) * 4;
      if (!(duration > 0)) continue;
      // Split longer cuts at source bar boundaries; preserve sub-bar edits.
      const sourceBars = (slice.endFrac - slice.startFrac) * 4;
      let start = slice.startFrac;
      while (start < slice.endFrac - 1e-7) {
        const end = Math.min(slice.endFrac, (Math.floor(start * 4 + 1e-6) + 1) / 4);
        const u = (start - slice.startFrac) / Math.max(1e-8, slice.endFrac - slice.startFrac);
        const v = (end - slice.startFrac) / Math.max(1e-8, slice.endFrac - slice.startFrac);
        const g0 = slice.gain ?? 1, g1 = slice.endGain ?? g0;
        push(entry, start, end, duration * (end - start) * 4 / sourceBars, g0 + (g1 - g0) * u, g0 + (g1 - g0) * v);
        start = end;
      }
    }
  } else {
    for (let b = 0; b < bars && task.run.length; b++) {
      const entry = task.run[Math.floor(b / 4) % task.run.length];
      push(entry, (b % 4) / 4, (b % 4 + 1) / 4, 1);
    }
  }
  return out;
}

function validNetwork(net: Network | undefined, input: number, outputs: number): boolean {
  if (!net || net.mean?.length !== input || net.std?.length !== input || net.w1?.length !== 32 || net.b1?.length !== 32 || net.w2?.length !== outputs || net.b2?.length !== outputs) return false;
  const values = [...net.mean, ...net.std, ...net.b1, ...net.b2, ...net.w1.flat(), ...net.w2.flat()];
  return values.every(Number.isFinite) && net.std.every(x => x > 0) && net.w1.every(r => r.length === input) && net.w2.every(r => r.length === 32);
}

function infer(net: Network, x: ArrayLike<number>): number[] {
  const hidden = new Float64Array(32);
  for (let j = 0; j < 32; j++) {
    let sum = net.b1[j];
    for (let i = 0; i < net.mean.length; i++) sum += net.w1[j][i] * (x[i] - net.mean[i]) / net.std[i];
    hidden[j] = Math.tanh(sum);
  }
  return net.w2.map((row, j) => row.reduce((s, w, i) => s + w * hidden[i], net.b2[j]));
}

export class SliceFlow {
  private artifact: Artifact | null = null;
  private motif: Network | null = null;
  private tourValue: Network | null = null;
  private valueScale: number | null = null;
  private features: Float32Array | null = null;
  private rows = new Map<string, number>();
  private tails: Partial<Record<StemName, FlowCell>> = {};
  private foundations: Partial<Record<StemName, FlowCell>> = {};

  foundation(stem: StemName): FlowCell | undefined { return this.foundations[stem]; }
  private identitySince: Partial<Record<StemName, number>> = {};
  private records: SliceFlowTrace[] = [];
  private nextId = 1;
  private history: Array<{ energy: number; at: number; signature: string }> = [];
  private section = "GROOVE_A";
  private lastIntent = "hold";
  private lastContext: FlowContext | null = null;

  get ready(): boolean { return this.artifact !== null && this.features !== null; }

  async load(base: string): Promise<boolean> {
    this.artifact = null; this.features = null; this.motif = null; this.tourValue = null; this.valueScale = null;
    this.rows.clear();
    try {
      const response = await fetch(`${base}/slice_flow.json`, { signal: AbortSignal.timeout(8000) });
      if (!response.ok) return false;
      const artifact = await response.json() as Artifact;
      if (artifact.format !== "slice_flow_v1" || artifact.feature_dim !== FLOW_DIM || !Array.isArray(artifact.phrase_ids)
        || !validNetwork(artifact.critic, FLOW_INPUT_DIM, 1) || !validNetwork(artifact.mix, 16, 1) || !validNetwork(artifact.planner, 8, 3)
        || !Number.isFinite(artifact.foundation_seconds) || !/^[\w.-]+$/.test(artifact.feature_file)) return false;
      const data = await fetch(`${base}/${artifact.feature_file}`, { signal: AbortSignal.timeout(15000) });
      if (!data.ok) return false;
      const bytes = await data.arrayBuffer();
      if (bytes.byteLength !== artifact.phrase_ids.length * 16 * FLOW_DIM * 4) return false;
      const features = new Float32Array(bytes);
      if (!features.every(Number.isFinite)) return false;
      this.rows.clear(); artifact.phrase_ids.forEach((id, i) => this.rows.set(id, i));
      this.features = features; this.artifact = artifact;
      // Motif-return is optional: the acoustic critic alone still works if
      // this sidecar is missing from an older bank.
      try {
        const response = await fetch(`${base}/motif_return.json`, { signal: AbortSignal.timeout(5000) });
        if (response.ok) {
          const model = await response.json() as MotifArtifact;
          if (model.format === MOTIF_FORMAT && model.input_dim === MOTIF_INPUT_DIM
            && validNetwork(model.net, MOTIF_INPUT_DIM, 1)) this.motif = model.net;
        }
      } catch { /* optional model */ }
      try {
        const response = await fetch(`${base}/bar_tour_value.json`, { signal: AbortSignal.timeout(5000) });
        if (response.ok) {
          const model = await response.json() as ValueArtifact;
          if (model.format === VALUE_FORMAT && model.input_dim === VALUE_INPUT_DIM
            && validNetwork(model.net, VALUE_INPUT_DIM, 1)) {
            this.tourValue = model.net;
            this.valueScale = typeof model.value_scale === "number" && Number.isFinite(model.value_scale) && model.value_scale > 0 ? model.value_scale : null;
          }
        }
      } catch { /* optional model */ }
      console.info(`[SliceFlow] acoustic critic + development planner ready (${artifact.phrase_ids.length} phrases)`);
      return true;
    } catch { return false; }
  }

  reset(): void {
    this.tails = {}; this.foundations = {}; this.identitySince = {}; this.records = []; this.history = [];
    this.nextId = 1; this.lastContext = null;
  }

  private cellFeatures(cell: FlowCell): Float32Array | null {
    const row = this.rows.get(cell.phraseId);
    if (row === undefined || !this.features) return null;
    const offset = ((row * 4 + STEMS.indexOf(cell.stem)) * 4 + cell.bar) * FLOW_DIM;
    const out = this.features.slice(offset, offset + FLOW_DIM);
    const gain = (cell.gain + cell.endGain) / 2 * (cell.mixerGain ?? 1);
    if (gain <= 1e-5) return out.fill(0);
    const delta = 20 * Math.log10(gain) / 60;
    for (let i = 0; i < 3; i++) out[i] = clamp(out[i] + delta);
    const chroma = out.slice(20);
    const shift = ((Math.round(cell.semitones) % 12) + 12) % 12;
    for (let i = 0; i < 12; i++) out[20 + (i + shift) % 12] = chroma[i];
    return out;
  }

  /** Locate a genuinely earlier played cell, not the latest foundation tail. */
  private motifAnchor(stem: StemName, at: number): FlowCell | null {
    for (let i = this.records.length - 1; i >= 0; i--) {
      const record = this.records[i];
      if (record.stem !== stem || record.at > at || record.endAt <= at) continue;
      const bars = record.cells.reduce((sum, cell) => sum + cell.bars, 0);
      const position = (at - record.at) / Math.max(1e-8, record.endAt - record.at) * bars;
      let offset = 0;
      for (const cell of record.cells) {
        offset += cell.bars;
        if (position < offset) return cell;
      }
    }
    return null;
  }

  score(cells: readonly FlowCell[], context: FlowContext): { score: number; seamFeatures: number[] | null; returnScore: number } {
    if (!this.artifact || !cells.length) return { score: .5, seamFeatures: null, returnScore: .5 };
    let previous = this.tails[cells[0].stem];
    let sum = 0, count = 0, minimum = 1;
    let seamFeatures: number[] | null = null;
    let returnSum = 0, returnCount = 0, outputBars = 0;
    for (const cell of cells) {
      const a = previous ? this.cellFeatures(previous) : null;
      const b = this.cellFeatures(cell);
      if (a && b && previous) {
        const age = clamp((context.at - (this.identitySince[cell.stem] ?? context.at)) / 120, 0, 2);
        const input = flowTransitionFeatures(a, b, cell.stem, [Number(context.sectionChanged), Number(previous.trackId !== cell.trackId), Number(previous.phraseId !== cell.phraseId), Number(cell.startFrac === 0), age, 0, context.targetEnergy, .25]);
        const score = 1 / (1 + Math.exp(-infer(this.artifact.critic, input)[0]));
        if (!seamFeatures) seamFeatures = Array.from(input);
        minimum = Math.min(minimum, score); sum += score; count++;
      }
      if (this.motif && b && Math.abs(outputBars - Math.round(outputBars)) < 1e-6) {
        let bestReturn = 0, available = false;
        for (const [lagIndex, lagBars] of [8, 16, 32].entries()) {
          const time = context.at + (outputBars - lagBars) * 240 / context.bpm;
          const remembered = this.motifAnchor(cell.stem, time);
          const anchor = remembered ? this.cellFeatures(remembered) : null;
          if (!anchor) continue;
          const input = motifFeatures(anchor, b, STEMS.indexOf(cell.stem), lagIndex, lagBars / 16);
          bestReturn = Math.max(bestReturn, 1 / (1 + Math.exp(-infer(this.motif, input)[0])));
          available = true;
        }
        if (available) { returnSum += bestReturn; returnCount++; }
      }
      outputBars += cell.bars;
      previous = cell;
    }
    return {
      score: count ? .55 * sum / count + .45 * minimum : .5,
      seamFeatures,
      returnScore: returnCount ? returnSum / returnCount : .5,
    };
  }

  /** How under-represented a source is in the recent listening window. */
  private sourceNovelty(trackId: string): number {
    const recent = this.records.slice(-16);
    if (!recent.length) return 1;
    let seen = 0;
    for (const record of recent) for (const cell of record.cells) {
      if (cell.trackId === trackId) seen++;
    }
    return 1 / (1 + seen);
  }

  /** Long-horizon value estimate for a candidate's first cell (optional). */
  private valueGain(cells: readonly FlowCell[], context: FlowContext): number {
    if (!this.tourValue || !cells.length) return 0;
    const first = this.cellFeatures(cells[0]);
    if (!first) return 0;
    const recentCells = this.records.filter(r=>r.stem===cells[0].stem && r.at<context.at)
      .slice(-4).flatMap(r=>r.cells).slice(-4);
    const vectors = recentCells.map(cell=>this.cellFeatures(cell)).filter((v): v is Float32Array=>v!==null);
    const recent = new Float32Array(FLOW_DIM);
    if (vectors.length) for (const vector of vectors) for (let i=0;i<FLOW_DIM;i++) recent[i]+=vector[i]/vectors.length;
    else recent.set(first);
    const age = Math.max(0, context.at - (this.identitySince[cells[0].stem] ?? context.at));
    const repeated = vectors.length ? vectors.filter(vector=>vector.reduce((sum,v,i)=>sum+Math.abs(v-first[i]),0)/FLOW_DIM < .02).length/vectors.length : 0;
    const x = valueFeatures(first, recent, age, repeated);
    const raw = infer(this.tourValue, x)[0];
    return this.valueScale ? clamp(raw / this.valueScale) : 1 / (1 + Math.exp(-raw));
  }

  mixScore<T extends FlowEntry>(tasks: readonly FlowTask<T>[], context: FlowContext): number {
    if (!this.artifact) return .5;
    const cells = new Map(tasks.map(task => [task.stem, describeFlowTask(task, context.bars, 240/context.bpm)]));
    let sum = 0, minimum = 1;
    for (let bar = 0; bar < context.bars; bar++) {
      const input = new Float32Array(16);
      for (let stem = 0; stem < 4; stem++) {
        let position = 0;
        for (const cell of cells.get(STEMS[stem]) ?? []) {
          const overlap = Math.max(0, Math.min(bar+1,position+cell.bars)-Math.max(bar,position));
          position += cell.bars;
          if (overlap <= 0) continue;
          const features = this.cellFeatures(cell);
          if (features) [0,7,9,11].forEach((index,i)=>{ input[stem*4+i]+=features[index]*overlap; });
        }
      }
      const score = 1/(1+Math.exp(-infer(this.artifact.mix,input)[0]));
      sum += score; minimum = Math.min(minimum,score);
    }
    return .6*sum/Math.max(1,context.bars)+.4*minimum;
  }

  /** Plan first, then gate the FINAL composed cuts, including source changes. */
  select<T extends FlowEntry, Task extends FlowTask<T>>(tasks: Task[], context: FlowContext,
    hold: (stem: StemName) => Task | null, pinned: ReadonlySet<StemName>): Task[] {
    if (!this.ready || !this.artifact) return tasks;
    this.lastContext = context; this.section = context.section;
    const recent = this.history.slice(-16);
    const lastEnergy = recent[recent.length - 1]?.energy ?? context.targetEnergy;
    const age = Math.max(0, context.at - Math.min(...STEMS.map(s => this.identitySince[s] ?? context.at)));
    const repeated = recent.length ? recent.filter(r => r.signature === recent[recent.length-1].signature).length / recent.length : 0;
    const density = STEMS.map(stem=>this.tails[stem]).reduce((sum,cell)=>sum+(cell?this.cellFeatures(cell)?.[11]??0:0),0)/4;
    const logits = infer(this.artifact.planner, [Math.min(2,age/120),lastEnergy,context.targetEnergy-lastEnergy,density,context.targetEnergy-(recent[0]?.energy??context.targetEnergy),repeated,.8,Number(context.sectionChanged)]);
    this.lastIntent = ["hold", "vary", "handoff"][logits.indexOf(Math.max(...logits))];
    const candidates = tasks.map(task => {
      if (pinned.has(task.stem)) return { task, fallback: null, canHold: false, changed: false, priority: 1 };
      const prior = this.foundations[task.stem];
      const candidate = this.score(describeFlowTask(task, context.bars, 240/context.bpm), context);
      const fallback = hold(task.stem);
      const fallbackAssessment = fallback ? this.score(describeFlowTask(fallback,context.bars,240/context.bpm),context) : null;
      const fallbackScore = fallbackAssessment?.score ?? 0;
      const canHold = Boolean(prior && fallback?.run[0]?.kit.track_id === prior.trackId);
      const motifGain = .28 * (candidate.returnScore - (fallbackAssessment?.returnScore ?? .5));
      const candidateCells = describeFlowTask(task, context.bars, 240/context.bpm);
      const valueGain = .10 * (this.valueGain(candidateCells, context)
        - (fallback ? this.valueGain(describeFlowTask(fallback, context.bars, 240/context.bpm), context) : .5));
      const first = task.run[0];
      const changed = Boolean(prior && first && prior.trackId !== first.kit.track_id);
      const stableAge = Math.max(0,context.at-(this.identitySince[task.stem]??context.at));
      const foundation = task.stem === "drums" || task.stem === "bass";
      const protectedFoundation = foundation && stableAge < this.artifact!.foundation_seconds && !context.rotation && !context.bridge;
      if (fallback && ((changed && protectedFoundation && !context.sectionChanged) || (!context.bridge && candidate.score + .18 < fallbackScore))) {
        return { task: fallback, fallback, canHold, changed: !canHold, priority: 0 };
      }
      const noveltyGain = .15 * (this.sourceNovelty(String(first?.kit.track_id ?? ""))
        - (fallback ? this.sourceNovelty(String(fallback.run[0]?.kit.track_id ?? "")) : .5));
      return { task, fallback, canHold, changed, priority: candidate.score-fallbackScore + motifGain + valueGain + noveltyGain + (foundation ? 0 : .02) };
    });
    // Preserve a recognizable anchor even when the old arc requests a reset.
    const maxChanges = context.bars <= 4 || this.lastIntent === "hold" || context.rotation ? 1 : 2;
    const changed = candidates.filter(c => c.changed && c.fallback && c.canHold).sort((a,b)=>b.priority-a.priority);
    for (const item of changed.slice(maxChanges)) if (item.fallback) item.task = item.fallback;
    let selected = candidates.map(c=>c.task);
    let mix = this.mixScore(selected,context);
    // Inspect the complete assembled quartet, not just independent stem joins.
    for (const option of candidates) {
      if (!option.fallback || pinned.has(option.task.stem)) continue;
      const trial = selected.map(task=>task.stem===option.task.stem ? option.fallback! : task);
      const quality = this.mixScore(trial,context);
      if (quality > mix + .08) { selected=trial; mix=quality; }
    }
    return selected;
  }

  record<T extends FlowEntry>(task: FlowTask<T>, at: number, bars: number, bpm: number, section?: string): void {
    const cells = describeFlowTask(task,bars,240/bpm);
    if (!cells.length) return;
    const first = cells[0], previous = this.foundations[task.stem];
    if (!previous || previous.trackId !== first.trackId) this.identitySince[task.stem] = at;
    const context = { at,bars,bpm,section: section??this.section,targetEnergy:this.lastContext?.targetEnergy??.7,sectionChanged:false,rotation:false,bridge:false };
    const assessment = this.score(cells,context);
    this.records.push({id:this.nextId++,stem:task.stem,at,endAt:at+bars*240/bpm,section:context.section,cells,...assessment});
    if (this.records.length>256) this.records.splice(0,this.records.length-256);
    this.tails[task.stem] = cells[cells.length-1];
    this.foundations[task.stem] = cells.filter(cell=>cell.trackId===cells[0].trackId).slice(-1)[0] ?? cells[0];
    if (task.stem === "drums") {
      this.history.push({energy:context.targetEnergy,at,signature:cells.map(c=>`${c.phraseId}:${c.bar}`).join("|")});
      if (this.history.length>32) this.history.shift();
    }
  }

  cancel(stem: StemName, from: number, to = Infinity): void {
    this.records = this.records.filter(r => !(r.stem === stem && r.at >= from && r.at < to));
    for (const record of this.records) if (record.stem === stem && record.at < from && record.endAt > from) record.endAt = from;
    const previous = this.records.filter(r=>r.stem===stem).slice(-1)[0];
    if (previous) {
      this.tails[stem] = previous.cells[previous.cells.length-1];
      this.foundations[stem] = previous.cells.filter(cell=>cell.trackId===previous.cells[0].trackId).slice(-1)[0] ?? previous.cells[0];
    } else { delete this.tails[stem]; delete this.foundations[stem]; }
  }

  retime(after: number, ratio: number): void {
    if (!(ratio > 0) || !Number.isFinite(ratio)) return;
    for (const record of this.records) {
      if (record.endAt <= after) continue;
      if (record.at > after) record.at = after+(record.at-after)*ratio;
      if (record.endAt > after) record.endAt = after+(record.endAt-after)*ratio;
      for (const cell of record.cells) cell.rate /= ratio;
    }
  }

  trace(audioNow = Infinity): { format: string; intent: string; records: SliceFlowTrace[] } {
    return {format:"wub-slice-trace-v1",intent:this.lastIntent,records:this.records.filter(r=>r.at<=audioNow).map(r=>({...r,cells:r.cells.map(c=>({...c})),seamFeatures:r.seamFeatures?.slice()??null}))};
  }
}
