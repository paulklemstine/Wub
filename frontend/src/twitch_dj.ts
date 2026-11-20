/**
 * Twitch chat co-DJ: viewers steer the sound with !commands.
 *
 * Read-only and dependency-free: connects anonymously to Twitch IRC
 * (`wss://irc-ws.chat.twitch.tv:443`), JOINs `#channel`, parses PRIVMSGs
 * and routes them to caller-supplied actions. No OAuth, no secrets, no DOM
 * or engine imports here — `main.ts` wires the actions to the real engine
 * and owns the panel UI. All parsing is pure and covered by `selfTest()`.
 *
 * Rate limiting: per-user cooldown (10 s) + global max 3 commands per 4-bar
 * block window. First valid command wins (full co-DJ, not vote).
 */

export type TwitchStem = "drums" | "bass" | "other" | "vocals";

export interface TwitchCommand {
  command: string;
  args: string[];
  user: string;
  raw: string;
}

export interface TwitchActions {
  section?: (name: string, user: string) => void;
  style?: (name: string, user: string) => void;
  bpm?: (bpm: number, user: string) => void;
  bpmHalf?: (user: string) => void;
  bpmDouble?: (user: string) => void;
  key?: (key: string, user: string) => void;
  kill?: (stem: TwitchStem, on: boolean, user: string) => void;
  echo?: (stem: TwitchStem, user: string) => void;
  morph?: (stem: TwitchStem, user: string) => void;
  chaos?: (stem: TwitchStem, value: number, user: string) => void;
  nextKit?: (user: string) => void;
  songMode?: (mode: "song" | "morph", user: string) => void;
  steer?: (stem: TwitchStem, x: number | null, y: number | null, user: string) => void;
  hook?: (user: string) => void;
  status?: (user: string) => string;
  help?: (user: string) => string;
}

const STEMS: readonly TwitchStem[] = ["drums", "bass", "other", "vocals"];
const SECTIONS = ["intro", "verse", "groove", "buildup", "drop", "chorus", "breakdown", "bridge"];
const STYLES = ["breaks", "cyber", "house", "wave"];
const KEYS = ["C", "G", "D", "A", "E", "B", "F#", "F", "Bb", "Eb", "Am", "Em", "Dm"];

export const TWITCH_HELP =
  "!drop !verse !chorus !buildup !breakdown !intro | !style house | !bpm 128 !key Am !half !double | " +
  "!kill drums !unkill drums !echo vocals !morph bass | !chaos drums 0.8 | " +
  "!steer drums [x y] | !hook | !next | !song !morphmode | !status !help";

function cleanToken(token: string): string {
  return token.trim().toLowerCase().replace(/[^a-z0-9#.\-+/]/g, "");
}

/** Parse a raw chat message into a command, or null when not a command. */
export function parseChatCommand(user: string, message: string): TwitchCommand | null {
  const text = (message || "").trim();
  if (!text.startsWith("!")) return null;
  const parts = text.slice(1).split(/\s+/).filter(Boolean);
  if (parts.length === 0) return null;
  const command = cleanToken(parts[0]);
  if (!command) return null;
  const args = parts.slice(1).map((a) => a.trim()).filter((a) => a.length > 0).slice(0, 4);
  return { command, args, user, raw: text.slice(0, 200) };
}

export function isStem(token: string | undefined): token is TwitchStem {
  return !!token && (STEMS as readonly string[]).includes(token.toLowerCase());
}

export function isSection(token: string | undefined): boolean {
  return !!token && SECTIONS.includes(token.toLowerCase());
}

/** Route a parsed command to actions. Returns a reply string or null. */
export function routeCommand(cmd: TwitchCommand, actions: TwitchActions): string | null {
  const c = cmd.command;
  const arg0 = (cmd.args[0] || "").toLowerCase();
  switch (c) {
    case "drop":
    case "chorus":
      actions.section?.("DROP", cmd.user);
      return `@${cmd.user} → DROP 🪩`;
    case "verse":
    case "groove":
      actions.section?.("GROOVE_A", cmd.user);
      return `@${cmd.user} → VERSE 🌀`;
    case "buildup":
    case "build":
      actions.section?.("BUILDUP", cmd.user);
      return `@${cmd.user} → BUILDUP 📈`;
    case "breakdown":
    case "break":
    case "bridge":
      actions.section?.("BREAKDOWN", cmd.user);
      return `@${cmd.user} → BREAKDOWN 🌌`;
    case "intro":
      actions.section?.("INTRO", cmd.user);
      return `@${cmd.user} → INTRO 🌅`;
    case "style": {
      const style = STYLES.find((s) => s === arg0);
      if (!style) return `@${cmd.user} styles: ${STYLES.join(" ")}`;
      actions.style?.(style, cmd.user);
      return `@${cmd.user} → style ${style} 🎨`;
    }
    case "bpm": {
      const bpm = Math.round(Number(arg0));
      if (!Number.isFinite(bpm) || bpm < 80 || bpm > 175) return `@${cmd.user} usage: !bpm 80-175`;
      actions.bpm?.(bpm, cmd.user);
      return `@${cmd.user} → ${bpm} BPM 🥁`;
    }
    case "half":
      actions.bpmHalf?.(cmd.user);
      return `@${cmd.user} → half-time 🐢`;
    case "double":
      actions.bpmDouble?.(cmd.user);
      return `@${cmd.user} → double-time 🐇`;
    case "key": {
      const key = (cmd.args[0] || "").trim();
      const hit = KEYS.find((k) => k.toLowerCase() === key.toLowerCase());
      if (!hit) return `@${cmd.user} keys: ${KEYS.join(" ")}`;
      actions.key?.(hit, cmd.user);
      return `@${cmd.user} → key ${hit} 🎹`;
    }
    case "kill":
      if (!isStem(arg0)) return `@${cmd.user} usage: !kill drums|bass|other|vocals`;
      actions.kill?.(arg0, true, cmd.user);
      return `@${cmd.user} killed ${arg0} 🔇`;
    case "unkill":
    case "unkillall":
    case "revive":
      if (arg0 && !isStem(arg0)) return `@${cmd.user} usage: !unkill drums|bass|other|vocals`;
      if (arg0 && isStem(arg0)) {
        actions.kill?.(arg0, false, cmd.user);
        return `@${cmd.user} revived ${arg0} 🔊`;
      }
      for (const stem of STEMS) actions.kill?.(stem, false, cmd.user);
      return `@${cmd.user} revived all stems 🔊`;
    case "echo":
      if (!isStem(arg0)) return `@${cmd.user} usage: !echo drums|bass|other|vocals`;
      actions.echo?.(arg0, cmd.user);
      return `@${cmd.user} echo throw on ${arg0} ✨`;
    case "morph":
      if (!isStem(arg0)) return `@${cmd.user} usage: !morph drums|bass|other|vocals`;
      actions.morph?.(arg0, cmd.user);
      return `@${cmd.user} morphing ${arg0} ⚡`;
    case "chaos": {
      if (!isStem(arg0)) return `@${cmd.user} usage: !chaos drums 0.8`;
      const value = Number(cmd.args[1]);
      if (!Number.isFinite(value) || value < 0 || value > 1) {
        return `@${cmd.user} usage: !chaos ${arg0} 0-1`;
      }
      actions.chaos?.(arg0, value, cmd.user);
      return `@${cmd.user} chaos ${arg0} → ${value.toFixed(2)} 🎛️`;
    }
    case "steer": {
      if (!isStem(arg0)) return `@${cmd.user} usage: !steer drums [x y]`;
      let x: number | null = null;
      let y: number | null = null;
      if (cmd.args.length >= 3) {
        const nx = Number(cmd.args[1]);
        const ny = Number(cmd.args[2]);
        if (Number.isFinite(nx) && Number.isFinite(ny)) {
          x = Math.max(0, Math.min(1, nx));
          y = Math.max(0, Math.min(1, ny));
        }
      }
      actions.steer?.(arg0, x, y, cmd.user);
      if (x === null || y === null) return `@${cmd.user} steering ${arg0} somewhere new 🧭`;
      return `@${cmd.user} steering ${arg0} → (${x.toFixed(2)}, ${y.toFixed(2)}) 🧭`;
    }
    case "hook":
      actions.hook?.(cmd.user);
      return `@${cmd.user} → vocal hook 🎤`;
    case "next":
    case "nextkit":
    case "skip":
      actions.nextKit?.(cmd.user);
      return `@${cmd.user} → NEXT KIT queued ⏭️`;
    case "song":
      actions.songMode?.("song", cmd.user);
      return `@${cmd.user} → SONG mode 🎼`;
    case "morphmode":
      actions.songMode?.("morph", cmd.user);
      return `@${cmd.user} → MORPH mode ♾️`;
    case "status": {
      const status = actions.status?.(cmd.user) ?? "playing";
      return `@${cmd.user} ${status}`;
    }
    case "help":
    case "commands":
      return actions.help?.(cmd.user) ?? TWITCH_HELP;
    default:
      return null;
  }
}

export interface TwitchRateState {
  lastByUser: Map<string, number>;
  windowStart: number;
  windowCount: number;
}

export const TWITCH_USER_COOLDOWN_MS = 10_000;
export const TWITCH_WINDOW_MS = 16_000;
export const TWITCH_WINDOW_MAX = 3;

/** True when this user/command may run now (updates state). */
export function allowCommand(state: TwitchRateState, user: string, now: number): boolean {
  const key = user.toLowerCase();
  const last = state.lastByUser.get(key) ?? -Infinity;
  if (now - last < TWITCH_USER_COOLDOWN_MS) return false;
  if (now - state.windowStart >= TWITCH_WINDOW_MS) {
    state.windowStart = now;
    state.windowCount = 0;
  }
  if (state.windowCount >= TWITCH_WINDOW_MAX) return false;
  state.lastByUser.set(key, now);
  state.windowCount += 1;
  if (state.lastByUser.size > 500) {
    const oldest = [...state.lastByUser.entries()].sort((a, b) => a[1] - b[1])[0];
    if (oldest) state.lastByUser.delete(oldest[0]);
  }
  return true;
}

export function createRateState(): TwitchRateState {
  return { lastByUser: new Map(), windowStart: 0, windowCount: 0 };
}

/** Minimal anonymous Twitch IRC reader (JOIN + PRIVMSG + PING/PONG). */
export class TwitchChat {
  private socket: WebSocket | null = null;
  private channel = "";
  onMessage: ((user: string, text: string) => void) | null = null;
  onStatus: ((status: string) => void) | null = null;

  get connected(): boolean {
    return this.socket !== null && this.socket.readyState === WebSocket.OPEN;
  }

  get joinedChannel(): string {
    return this.channel;
  }

  connect(channel: string): void {
    const clean = channel.trim().replace(/^#/, "").toLowerCase();
    if (!clean || !/^[a-z0-9_]{1,25}$/.test(clean)) {
      this.onStatus?.("invalid channel (a-z, 0-9, _)");
      return;
    }
    this.disconnect();
    this.channel = clean;
    const nick = `justinfan${Math.floor(1000 + Math.random() * 9000)}`;
    let socket: WebSocket;
    try {
      socket = new WebSocket("wss://irc-ws.chat.twitch.tv:443");
    } catch {
      this.onStatus?.("websocket unavailable (https required)");
      return;
    }
    this.socket = socket;
    this.onStatus?.(`connecting to #${clean} …`);
    socket.onopen = () => {
      socket.send("CAP REQ :twitch.tv/tags twitch.tv/commands");
      socket.send(`PASS SCHMOOPIIE`);
      socket.send(`NICK ${nick}`);
      socket.send(`JOIN #${clean}`);
    };
    socket.onmessage = (event) => {
      const data = typeof event.data === "string" ? event.data : "";
      for (const line of data.split("\r\n")) {
        if (!line) continue;
        if (line.startsWith("PING")) {
          socket.send("PONG :tmi.twitch.tv");
          continue;
        }
        const privmsg = /:([^!]+)![^ ]+ PRIVMSG #[^ ]+ :(.*)$/.exec(line);
        if (privmsg) {
          const user = privmsg[1] ?? "viewer";
          const text = privmsg[2] ?? "";
          this.onMessage?.(user, text);
          continue;
        }
        if (line.includes(`JOIN #${clean}`)) {
          this.onStatus?.(`live in #${clean} — viewers type !help 🎧`);
        }
      }
    };
    socket.onerror = () => this.onStatus?.("chat socket error — retry");
    socket.onclose = () => {
      if (this.socket === socket) {
        this.socket = null;
        this.onStatus?.("disconnected");
      }
    };
  }

  disconnect(): void {
    try {
      this.socket?.close();
    } catch {
      // ignore
    }
    this.socket = null;
    this.channel = "";
  }
}

export function selfTestTwitchDj(): void {
  const check = (cond: boolean, msg: string) => {
    if (!cond) throw new Error(`twitch_dj: ${msg}`);
  };
  check(parseChatCommand("alice", "!drop")?.command === "drop", "parses !drop");
  check(parseChatCommand("bob", "hello") === null, "ignores chat");
  check(parseChatCommand("c", "!") === null, "ignores bare !");
  check(parseChatCommand("d", "!bpm 128")?.args[0] === "128", "parses args");
  check(isStem("drums") && !isStem("guitar"), "stem guard");
  const seen: string[] = [];
  const actions: TwitchActions = {
    section: (name, user) => void seen.push(`section:${name}:${user}`),
    bpm: (bpm, user) => void seen.push(`bpm:${bpm}:${user}`),
    kill: (stem, on, user) => void seen.push(`kill:${stem}:${on}:${user}`),
    help: () => TWITCH_HELP,
  };
  const reply = routeCommand(parseChatCommand("alice", "!kill drums")!, actions);
  check(seen.includes("kill:drums:true:alice"), "routes !kill");
  check(!!reply && reply.includes("alice"), "replies mention user");
  check(routeCommand(parseChatCommand("x", "!nope")!, actions) === null, "unknown ignored");
  check(routeCommand(parseChatCommand("x", "!bpm abc")!, actions)?.includes("usage") === true, "bad bpm usage");
  check(routeCommand(parseChatCommand("x", "!help")!, actions) === TWITCH_HELP, "!help");
  const rate = createRateState();
  check(allowCommand(rate, "alice", 0), "first allowed");
  check(!allowCommand(rate, "alice", 1000), "cooldown blocks");
  check(allowCommand(rate, "bob", 1001), "other user allowed");
  check(allowCommand(rate, "carol", 1002), "third in window allowed");
  check(!allowCommand(rate, "dave", 1003), "fourth in window blocked");
  check(allowCommand(rate, "dave", 20_000), "window resets");
}
