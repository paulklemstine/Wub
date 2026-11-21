# 📺 Twitch Co-DJ — viewers steer the sound with `!commands`

The DJ surface has a **TWITCH CO-DJ** strip. The broadcaster enters their
channel, hits **CONNECT**, and chat takes over. Two modes:

- **Read-only (default):** leave bot username + token empty. The page joins
  `#channel` **anonymously** (no OAuth, no login, no secrets) and reacts to
  `!commands`. Replies render in the strip log only.
- **Mirror to chat:** fill bot username + `oauth:…` token and CONNECT.
  Every routed reply (`@user → DROP 🪩`, `!help` output, …) is also posted
  to the channel via `PRIVMSG`, so the whole room sees the output stream.
  Replies are truncated to 450 chars, spaced ≥1.5 s, last-5 queue.

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

To mirror replies into chat: get a token with `chat:read chat:write`
(e.g. open `https://twitchapps.com/tmi/` logged in as your bot account —
a second Twitch account is safest, or your broadcaster account), paste the
bot username + `oauth:…` token, CONNECT. Tokens live only in your browser's
user storage (key `wub_integrations_v1`, shared with the MIDI input/bank —
old `wub_tw_*` keys migrate forward automatically), never in the repo;
revoke them on Twitch settings any time.
Without a token the strip stays read-only (replies in the log only).

To stop: DISCONNECT. The channel name persists in user storage.

## Notes

- Read-only by default: the page never sends chat on an anonymous login.
  With a bot token it posts routed replies (`@user → …`) back to the
  channel.
- If a command seems ignored: per-user 10 s cooldown, 3-command window cap,
  or an out-of-range arg (`!bpm 80-175`, `!chaos 0-1`, `!steer x y 0-1`).
- `!style` rides the MusicBrainz genre multi-select; without a matching
  bucket it lists valid styles instead of forcing a change.
