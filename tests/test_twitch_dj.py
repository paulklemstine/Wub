"""Twitch co-DJ: viewers steer the sound with !commands (read-only chat)."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle(name: str, source: str) -> Path:
    out = Path("/tmp/opencode") / f"{name}.cjs"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_twitch_command_parser_and_router():
    out = bundle("twitch_dj", "twitch_dj.ts")
    script = r"""
const assert = require('node:assert/strict');
const m = require(process.argv[1]);
assert.equal(typeof m.parseChatCommand, 'function', 'parser exported');
assert.equal(typeof m.routeCommand, 'function', 'router exported');
assert.equal(typeof m.allowCommand, 'function', 'rate limit exported');
assert.equal(typeof m.TwitchChat, 'function', 'chat client exported');
assert.equal(typeof m.selfTestTwitchDj, 'function', 'self-test exported');
m.selfTestTwitchDj();
const seen = [];
const actions = {
  section: (n, u) => void seen.push(`section:${n}:${u}`),
  kill: (s, on, u) => void seen.push(`kill:${s}:${on}:${u}`),
};
const cmd = m.parseChatCommand('alice', '!drop');
assert.equal(cmd.command, 'drop', 'parses !drop');
const reply = m.routeCommand(cmd, actions);
assert.ok(seen.includes('section:DROP:alice'), 'routes !drop to section');
assert.ok(reply.includes('alice'), 'reply mentions user');
assert.equal(m.routeCommand(m.parseChatCommand('x', '!nope'), actions), null, 'unknown ignored');
assert.equal(m.parseChatCommand('bob', 'hello'), null, 'ignores chat');
"""
    result = subprocess.run(["node", "-e", script, str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_twitch_panel_elements_exist():
    html = (ROOT / "frontend/index.html").read_text(encoding="utf-8")
    for element_id in ("tw-channel", "tw-username", "tw-token", "tw-connect", "tw-status", "tw-log"):
        assert f'id="{element_id}"' in html, f"missing #{element_id}"
    assert "TWITCH CO-DJ" in html
    assert "twitch_dj" in (ROOT / "frontend/src/main.ts").read_text(encoding="utf-8")
    assert "requestSectionKind" in (ROOT / "frontend/src/client_ml_sequencer.ts").read_text(encoding="utf-8")
    assert "requestKind" in (ROOT / "frontend/src/stem_remix.ts").read_text(encoding="utf-8")


def test_twitch_rate_limit_blocks_spam():
    out = bundle("twitch_rate", "twitch_dj.ts")
    script = r"""
const assert = require('node:assert/strict');
const m = require(process.argv[1]);
const rate = m.createRateState();
assert.ok(m.allowCommand(rate, 'alice', 0), 'first allowed');
assert.ok(!m.allowCommand(rate, 'alice', 1000), 'per-user cooldown');
assert.ok(m.allowCommand(rate, 'bob', 1001), 'other user allowed');
assert.ok(m.allowCommand(rate, 'carol', 1002), 'third in window');
assert.ok(!m.allowCommand(rate, 'dave', 1003), 'window cap');
assert.ok(m.allowCommand(rate, 'dave', 20000), 'window resets');
"""
    result = subprocess.run(["node", "-e", script, str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
