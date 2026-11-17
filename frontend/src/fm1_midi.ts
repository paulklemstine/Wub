// FM-1 / CTL-1 MIDI knob → phrase adapter.
//
// The stock M-VAVE FM-1 firmware is MIDI-IN only: its four upper-right knobs
// change the synth patch and never transmit control changes. Flashing the
// community CTL-1 firmware (https://github.com/charlesvestal/fm1-ctl) turns the
// device into a controller whose KNOB 1-4 send absolute CC values 0..127 in
// four banks. This module is the browser-side receiver for those messages:
// knob N selects a slice for stem N in the same order as the DJ control
// surface (DRUMS, BASS, VOCALS, OTHER).
//
// The module deliberately has no DOM work at import time and imports types
// only, so Node's `--experimental-strip-types` can load it in unit tests.

import type { StemName } from "./types";
import type { NebulaPoint } from "./visualizer";

/** Knob index → stem, matching the DJ control surface order. */
export const FM1_KNOB_STEMS: readonly StemName[] = ["drums", "bass", "vocals", "other"];

export type Fm1BankId = "A" | "B" | "C" | "D";

export const FM1_BANK_ORDER: readonly Fm1BankId[] = ["A", "B", "C", "D"];

/**
 * CTL-1 knob banks. Bank A doubles as mod-wheel (CC 1) and channel volume
 * (CC 7), so bank C is the recommended clean default for phrase control.
 */
export const FM1_CC_BANKS: Record<Fm1BankId, readonly [number, number, number, number]> = {
  A: [74, 71, 1, 7],
  B: [73, 72, 75, 76],
  C: [16, 17, 18, 19],
  D: [20, 21, 22, 23],
};

export const FM1_DEFAULT_BANK: Fm1BankId = "C";

export function isFm1BankId(value: unknown): value is Fm1BankId {
  return value === "A" || value === "B" || value === "C" || value === "D";
}

/** Knob index (0..3) for a controller number in the given bank, or -1. */
export function knobForCc(cc: number, bank: Fm1BankId = FM1_DEFAULT_BANK): number {
  const ccs = FM1_CC_BANKS[bank] ?? FM1_CC_BANKS[FM1_DEFAULT_BANK];
  return ccs.indexOf(cc);
}

/**
 * Linear 0..127 knob value → slice slot. Used to seed the scroll position from
 * the knob's absolute value and by the virtual sliders for coarse jumps. Every
 * detent maps to a fixed slot in the stem's deterministic catalogue order.
 */
export function knobValueToSlot(value: number, count: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(count) || count <= 0) return 0;
  const v = Math.max(0, Math.min(127, Math.round(value)));
  const slot = Math.round((v * (count - 1)) / 127);
  return Math.max(0, Math.min(count - 1, slot));
}

/**
 * Relative scroll step for the CTL-1 endless knobs. The firmware sends absolute
 * 0..127 values with acceleration (1 step per click, up to 6 when spun fast) and
 * wraps 127→0 at the top; unwrap that into a signed delta and accumulate it
 * modulo the slice count so every slice is reachable by turning.
 */
export function scrollSlot(current: number, value: number, previous: number, count: number): number {
  if (!Number.isFinite(count) || count <= 0) return 0;
  const clamp = (n: number): number => Math.max(0, Math.min(count - 1, n));
  if (!Number.isFinite(current)) return clamp(knobValueToSlot(value, count));
  let delta = Math.round(value) - Math.round(previous);
  if (delta > 64) delta -= 128;
  else if (delta < -64) delta += 128;
  if (delta === 0) return clamp(Math.round(current));
  return ((Math.round(current) + delta) % count + count) % count;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function naturalCompare(a: string, b: string): number {
  if (a === b) return 0;
  const ax = a.match(/(\d+|\D+)/g) ?? [a];
  const bx = b.match(/(\d+|\D+)/g) ?? [b];
  const n = Math.min(ax.length, bx.length);
  for (let i = 0; i < n; i++) {
    const av = ax[i];
    const bv = bx[i];
    const an = /^\d/.test(av);
    const bn = /^\d/.test(bv);
    if (an && bn) {
      const diff = Number(av) - Number(bv);
      if (diff !== 0) return diff;
    } else if (av !== bv) {
      return av < bv ? -1 : 1;
    }
  }
  return ax.length - bx.length;
}

/**
 * Deterministic slice order for one stem: catalogue order (track id, then
 * phrase id), with duplicate coordinates collapsed. The order only depends on
 * the point set, never on iteration order of the manifest maps.
 */
export function buildSliceOrder(points: readonly NebulaPoint[]): NebulaPoint[] {
  const seen = new Set<string>();
  const out: NebulaPoint[] = [];
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
    const key = `${point.trackId ?? ""}|${point.phraseId ?? ""}|${point.x.toFixed(4)}|${point.y.toFixed(4)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      x: clamp01(point.x),
      y: clamp01(point.y),
      stem: point.stem,
      trackId: point.trackId,
      phraseId: point.phraseId,
      kitId: point.kitId,
    });
  }
  out.sort((a, b) => {
    const track = naturalCompare(a.trackId ?? "", b.trackId ?? "");
    if (track !== 0) return track;
    const phrase = naturalCompare(a.phraseId ?? "", b.phraseId ?? "");
    if (phrase !== 0) return phrase;
    if (a.x !== b.x) return a.x - b.x;
    return a.y - b.y;
  });
  return out;
}

export interface Fm1MidiDevice {
  id: string;
  name: string;
  manufacturer: string;
  state: MIDIPortDeviceState;
}

export interface Fm1MidiState {
  supported: boolean;
  connected: boolean;
  deviceId: string | null;
  deviceName: string | null;
  bank: Fm1BankId;
  ccs: readonly number[];
  stems: readonly StemName[];
  devices: Fm1MidiDevice[];
  lastMessage: { cc: number; value: number; at: number } | null;
  lastKnob: { stem: StemName; value: number; cc: number } | null;
}

export type Fm1StatusTone = "info" | "ok" | "warn";

export interface Fm1MidiOptions {
  /** Called for each mapped knob turn (value is 0..127). */
  onKnob?: (stem: StemName, value: number, cc: number) => void;
  /** Human-readable connection/status line for the UI strip. */
  onStatus?: (message: string, tone?: Fm1StatusTone) => void;
  /** Called whenever the MIDI input list changes (hot-plug). */
  onDevices?: (devices: Fm1MidiDevice[], selectedId: string | null) => void;
  bank?: Fm1BankId;
}

/**
 * Browser Web MIDI receiver for the CTL-1 firmware. All DOM/navigator access
 * happens inside methods; constructing an instance is safe in Node.
 */
export class Fm1MidiInput {
  private readonly opts: Fm1MidiOptions;
  private access: MIDIAccess | null = null;
  private input: MIDIInput | null = null;
  private selectedId: string | null = null;
  private bank: Fm1BankId;
  private devices: Fm1MidiDevice[] = [];
  private lastMessage: { cc: number; value: number; at: number } | null = null;
  private lastKnob: { stem: StemName; value: number; cc: number } | null = null;
  /** Auto-pick a port until the user makes an explicit selection. */
  private autoPick = true;

  private readonly messageHandler = (event: MIDIMessageEvent): void => {
    const data = event.data;
    if (data !== null && data !== undefined) this.handleRaw(data);
  };

  private readonly stateHandler = (): void => {
    this.refreshDevices(true);
  };

  constructor(opts: Fm1MidiOptions = {}) {
    this.opts = opts;
    this.bank = opts.bank ?? FM1_DEFAULT_BANK;
  }

  /** True when the browser exposes the Web MIDI API. */
  static supported(): boolean {
    return typeof navigator !== "undefined" && typeof navigator.requestMIDIAccess === "function";
  }

  get bankId(): Fm1BankId {
    return this.bank;
  }

  get connected(): boolean {
    return this.input !== null;
  }

  get deviceName(): string | null {
    return this.input?.name ?? null;
  }

  get deviceList(): Fm1MidiDevice[] {
    return this.devices.map((d) => ({ ...d }));
  }

  /** Controller numbers of the current bank, in knob order. */
  get ccs(): readonly [number, number, number, number] {
    return FM1_CC_BANKS[this.bank];
  }

  ccForKnob(index: number): number {
    if (!Number.isInteger(index) || index < 0 || index > 3) return -1;
    return this.ccs[index];
  }

  setBank(bank: Fm1BankId): void {
    if (!isFm1BankId(bank) || bank === this.bank) return;
    this.bank = bank;
    const ccs = this.ccs;
    this.opts.onStatus?.(
      `CTL-1 bank ${bank}: K1=CC ${ccs[0]}, K2=CC ${ccs[1]}, K3=CC ${ccs[2]}, K4=CC ${ccs[3]}`,
      "info",
    );
  }

  /** Request Web MIDI access and start listening. Resolves false if denied. */
  async connect(): Promise<boolean> {
    if (!Fm1MidiInput.supported()) {
      this.opts.onStatus?.(
        "Web MIDI unavailable: use Chrome or Edge on https:// (or localhost).",
        "warn",
      );
      return false;
    }
    try {
      this.access = await navigator.requestMIDIAccess({ sysex: false });
    } catch (err) {
      const name = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      this.opts.onStatus?.(`MIDI permission denied (${name}).`, "warn");
      return false;
    }
    this.access.onstatechange = this.stateHandler;
    this.refreshDevices();
    if (this.input === null && this.autoPick) this.autoSelect();
    return true;
  }

  disconnect(): void {
    if (this.input !== null) this.input.onmidimessage = null;
    this.input = null;
    this.selectedId = null;
    if (this.access !== null) this.access.onstatechange = null;
    this.access = null;
    this.devices = [];
    this.opts.onDevices?.([], null);
    this.opts.onStatus?.("MIDI disconnected.", "info");
  }

  /** Choose a specific input, or null to stop listening. */
  selectInput(id: string | null): void {
    this.autoPick = false;
    if (this.input !== null) this.input.onmidimessage = null;
    this.input = null;
    this.selectedId = null;
    if (id !== null && this.access !== null) {
      for (const port of this.access.inputs.values()) {
        if (port.id === id) {
          this.attach(port);
          break;
        }
      }
    }
    this.opts.onDevices?.(this.deviceList, this.selectedId);
    const attached = this.input as MIDIInput | null;
    if (attached !== null) {
      this.opts.onStatus?.(`Listening to ${attached.name ?? attached.id}.`, "ok");
    } else if (id !== null) {
      this.opts.onStatus?.("Selected MIDI input is no longer available.", "warn");
    }
  }

  /** Pick the first CTL-1/FM-1 input, falling back to the first port. */
  autoSelect(): void {
    if (this.access === null) return;
    const ports = [...this.access.inputs.values()];
    if (ports.length === 0) {
      this.opts.onStatus?.(
        "No MIDI inputs found. Connect the FM-1 over USB, then press Connect again.",
        "warn",
      );
      return;
    }
    const preferred =
      ports.find((port) => /ctl.?1|fm.?1/i.test(port.name ?? "")) ?? ports[0];
    this.attach(preferred);
    this.opts.onDevices?.(this.deviceList, this.selectedId);
    const name = preferred.name ?? preferred.id;
    this.opts.onStatus?.(`Listening to ${name} — bank ${this.bank}.`, "ok");
  }

  /**
   * Consume one raw MIDI message. Returns the mapped knob event, or null when
   * the message is not a mapped control change in the active bank.
   */
  handleRaw(data: ArrayLike<number>): { stem: StemName; value: number; cc: number } | null {
    if (data.length < 3) return null;
    if ((data[0] & 0xf0) !== 0xb0) return null;
    const cc = data[1] & 0x7f;
    const value = data[2] & 0x7f;
    this.lastMessage = { cc, value, at: Date.now() };
    const knob = knobForCc(cc, this.bank);
    if (knob < 0) return null;
    const stem = FM1_KNOB_STEMS[knob];
    this.lastKnob = { stem, value, cc };
    this.opts.onKnob?.(stem, value, cc);
    return { stem, value, cc };
  }

  /** Test/debug entry point: behave as if this CC arrived from the device. */
  injectCC(cc: number, value: number): { stem: StemName; value: number; cc: number } | null {
    return this.handleRaw([0xb0, cc & 0x7f, value & 0x7f]);
  }

  state(): Fm1MidiState {
    return {
      supported: Fm1MidiInput.supported(),
      connected: this.connected,
      deviceId: this.selectedId,
      deviceName: this.deviceName,
      bank: this.bank,
      ccs: [...this.ccs],
      stems: [...FM1_KNOB_STEMS],
      devices: this.deviceList,
      lastMessage: this.lastMessage === null ? null : { ...this.lastMessage },
      lastKnob: this.lastKnob === null ? null : { ...this.lastKnob },
    };
  }

  private attach(port: MIDIInput): void {
    this.input = port;
    this.selectedId = port.id;
    port.onmidimessage = this.messageHandler;
  }

  private refreshDevices(reselect: boolean = false): void {
    if (this.access === null) return;
    const devices: Fm1MidiDevice[] = [];
    for (const port of this.access.inputs.values()) {
      devices.push({
        id: port.id,
        name: port.name ?? port.id,
        manufacturer: port.manufacturer ?? "",
        state: port.state,
      });
    }
    this.devices = devices;
    const stillPresent = this.selectedId !== null && devices.some((d) => d.id === this.selectedId);
    if (!stillPresent) {
      if (this.input !== null) this.input.onmidimessage = null;
      this.input = null;
      this.selectedId = null;
    }
    this.opts.onDevices?.(this.deviceList, this.selectedId);
    if (reselect && this.selectedId === null && this.autoPick) this.autoSelect();
  }
}
