/**
 * One home for every external integration's user storage.
 *
 * Everything lives under a single localStorage key (`wub_integrations_v1`)
 * so broadcaster setup — Twitch channel/bot login, MIDI input/bank —
 * survives reloads in one place. The store is dependency-free and takes an
 * injectable key/value backend so node tests can use a Map instead of DOM
 * storage. Legacy per-key Twitch entries (`wub_tw_channel` …) are migrated
 * forward on first load and then left alone.
 */

export interface TwitchIntegrationState {
  channel: string;
  username: string;
  token: string;
}

export interface MidiIntegrationState {
  /** Web MIDI port id, or "" for none. */
  inputId: string;
  /** CTL-1 bank id (A/B/C/D) or "" for default. */
  bank: string;
}

export interface IntegrationState {
  twitch: TwitchIntegrationState;
  midi: MidiIntegrationState;
}

export const INTEGRATION_STORAGE_KEY = "wub_integrations_v1";

const LEGACY_TWITCH_KEYS = {
  channel: "wub_tw_channel",
  username: "wub_tw_user",
  token: "wub_tw_token",
} as const;

export interface KeyValueStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function browserStorage(): KeyValueStorage | null {
  try {
    if (typeof localStorage !== "undefined") return localStorage;
  } catch {
    // private mode / non-DOM runtimes
  }
  return null;
}

export function emptyIntegrations(): IntegrationState {
  return {
    twitch: { channel: "", username: "", token: "" },
    midi: { inputId: "", bank: "" },
  };
}

function clean(value: unknown, maxLen: number): string {
  if (typeof value !== "string") return "";
  return value.slice(0, maxLen);
}

function normalize(raw: unknown): IntegrationState {
  const out = emptyIntegrations();
  if (typeof raw !== "object" || raw === null) return out;
  const doc = raw as Partial<Record<string, unknown>>;
  const twitch = doc.twitch;
  if (typeof twitch === "object" && twitch !== null) {
    const t = twitch as Partial<Record<string, unknown>>;
    out.twitch.channel = clean(t.channel, 64);
    out.twitch.username = clean(t.username, 64);
    out.twitch.token = clean(t.token, 512);
  }
  const midi = doc.midi;
  if (typeof midi === "object" && midi !== null) {
    const m = midi as Partial<Record<string, unknown>>;
    out.midi.inputId = clean(m.inputId, 256);
    out.midi.bank = clean(m.bank, 8);
  }
  return out;
}

/** Load all integrations, migrating legacy Twitch keys when present. */
export function loadIntegrations(storage?: KeyValueStorage | null): IntegrationState {
  const store = storage ?? browserStorage();
  if (store === null) return emptyIntegrations();
  let raw: unknown = null;
  try {
    const text = store.getItem(INTEGRATION_STORAGE_KEY);
    if (text) raw = JSON.parse(text) as unknown;
  } catch {
    raw = null;
  }
  const state = normalize(raw);
  // Migrate pre-unified Twitch keys once; stored doc wins when both exist.
  if (!state.twitch.channel && !state.twitch.username && !state.twitch.token) {
    try {
      const channel = store.getItem(LEGACY_TWITCH_KEYS.channel) ?? "";
      const username = store.getItem(LEGACY_TWITCH_KEYS.username) ?? "";
      const token = store.getItem(LEGACY_TWITCH_KEYS.token) ?? "";
      if (channel || username || token) {
        state.twitch = {
          channel: clean(channel, 64),
          username: clean(username, 64),
          token: clean(token, 512),
        };
      }
    } catch {
      // ignore
    }
  }
  return state;
}

/** Merge a patch into storage and return the new state. */
export function saveIntegrations(
  patch: Partial<{ twitch: Partial<TwitchIntegrationState>; midi: Partial<MidiIntegrationState> }>,
  storage?: KeyValueStorage | null,
): IntegrationState {
  const store = storage ?? browserStorage();
  const current = loadIntegrations(store);
  if (patch.twitch) {
    if (patch.twitch.channel !== undefined) current.twitch.channel = clean(patch.twitch.channel, 64);
    if (patch.twitch.username !== undefined) current.twitch.username = clean(patch.twitch.username, 64);
    if (patch.twitch.token !== undefined) current.twitch.token = clean(patch.twitch.token, 512);
  }
  if (patch.midi) {
    if (patch.midi.inputId !== undefined) current.midi.inputId = clean(patch.midi.inputId, 256);
    if (patch.midi.bank !== undefined) current.midi.bank = clean(patch.midi.bank, 8);
  }
  if (store !== null) {
    try {
      store.setItem(INTEGRATION_STORAGE_KEY, JSON.stringify(current));
    } catch {
      // quota / private mode — in-memory state still returned
    }
  }
  return current;
}

/** Forget every integration (panel fields reset on next load). */
export function clearIntegrations(storage?: KeyValueStorage | null): IntegrationState {
  const store = storage ?? browserStorage();
  if (store !== null) {
    try {
      store.removeItem(INTEGRATION_STORAGE_KEY);
    } catch {
      // ignore
    }
  }
  return emptyIntegrations();
}

export function selfTestIntegrationStore(): void {
  const check = (cond: boolean, msg: string) => {
    if (!cond) throw new Error(`integration_store: ${msg}`);
  };
  const mem = (): KeyValueStorage => {
    const map = new Map<string, string>();
    return {
      getItem: (k) => (map.has(k) ? (map.get(k) as string) : null),
      setItem: (k, v) => void map.set(k, v),
      removeItem: (k) => void map.delete(k),
    };
  };
  const s1 = mem();
  const fresh = loadIntegrations(s1);
  check(fresh.twitch.channel === "" && fresh.midi.inputId === "", "defaults empty");
  const saved = saveIntegrations(
    { twitch: { channel: "wub", username: "wub_bot", token: "oauth:abc" }, midi: { inputId: "in1", bank: "C" } },
    s1,
  );
  check(saved.twitch.channel === "wub" && saved.midi.bank === "C", "roundtrip saves");
  const reloaded = loadIntegrations(s1);
  check(reloaded.twitch.token === "oauth:abc" && reloaded.midi.inputId === "in1", "roundtrip loads");
  const s2 = mem();
  s2.setItem("wub_tw_channel", "legacy_chan");
  s2.setItem("wub_tw_token", "oauth:legacy");
  const migrated = loadIntegrations(s2);
  check(migrated.twitch.channel === "legacy_chan", "legacy channel migrates");
  check(migrated.twitch.token === "oauth:legacy", "legacy token migrates");
  saveIntegrations({ twitch: { channel: "new_chan" } }, s2);
  check(loadIntegrations(s2).twitch.token === "oauth:legacy", "patch merges, keeps token");
  clearIntegrations(s1);
  check(loadIntegrations(s1).twitch.channel === "", "clear forgets");
  check(loadIntegrations(null).twitch.channel === "", "null storage safe");
}
