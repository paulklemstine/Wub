# 📺 Twitch Co-DJ — viewers steer the sound with `!commands`

The DJ surface has a **TWITCH CO-DJ** strip. The broadcaster enters their
channel, hits **CONNECT**, and chat takes over: the page joins
`#channel` **read-only and anonymously** (no OAuth, no login, no secrets).
Viewers type `!commands`, first valid command wins (10 s per-user cooldown,
max 3 commands per ~16 s window).

## Commands

| Command | Effect |
| :--- | :--- |
| `!drop` / `!chorus` | Next block leans DROP/chorus (kind + energy nudge, song form intact) |
| `!verse` / `!groove` | Next block leans GROOVE_A/verse |
| `!buildup`, `!breakdown`, `!intro` | Next-block kind request |
| `!style house` | Genre filter to matching bucket (`breaks/cyber/house/wave` or list) |
| `!bpm 128`, `!half`, `!double` | Tempo (clamped 80–175) |
| `!key Am` | Musical key |
| `!kill drums`, `!unkill drums` | Stem kill / revive (all when bare `!unkill`) |
| `!echo vocals` | Echo throw toggle |
| `!morph bass` | Fresh compatible groove for the stem |
| `!chaos drums 0.8` | Chaos slider 0–1 |
| `!steer drums [x y]` | Reticle to `x y` (0–1) or a fresh spot when bare |
| `!hook` | Fresh hook-leaning vocal |
| `!next` | Queue NEXT KIT (bridge on the next boundary) |
| `!song` / `!morphmode` | Song vs infinite-morph mode |
| `!status` | Section • BPM • mode |
| `!help` | Command list |

## How section requests stay song-safe

`!drop` etc. do **not** rewrite the 104-bar blueprint. They call
`ClientGrooveboxConductor.requestSectionKind()` → `KitPlayer.requestKind()`
→ `StemRemixPlayer.requestKind()`, which colors **one block's** selectors
(kind + energy nudge) and expires at the block boundary. Cluster rotation,
pins, the 32-bar vocal holds and the final SliceFlow gate all stay in force.
In legacy (non-kit) modes the same call moves the arranger section directly.
Reply lines (`@user → DROP 🪩`) appear in the strip log (last 8).

## Setup (broadcaster)

1. Open the page (must be `https` or `localhost` for WebSocket chat).
2. Start playback (`▶ PLAY`) — chat only steers a running set.
3. DJ SURFACE → TWITCH CO-DJ → enter channel → CONNECT.
4. Tell chat: `!help`.

To stop: DISCONNECT. The channel name persists in `localStorage`.

## Notes

- Read-only: the page never sends chat (anonymous `justinfan` login can't
  chat). Replies render in the strip log, not in Twitch chat.
- If a command seems ignored: per-user 10 s cooldown, 3-command window cap,
  or an out-of-range arg (`!bpm 80-175`, `!chaos 0-1`, `!steer x y 0-1`).
- `!style` rides the MusicBrainz genre multi-select; without a matching
  bucket it lists valid styles instead of forcing a change.
