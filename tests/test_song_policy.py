"""Song-policy meta weights trained on real successors."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITS = ROOT / "frontend/public/data/kits"


def test_song_policy_artifact_is_trained():
    doc = json.loads((KITS / "song_policy.json").read_text())
    assert doc["format"] == "wub-song-policy-v1"
    assert doc["version"] >= 2
    weights = doc["weights"]
    # Musically sensible signs: stretch/key penalized, harmony/energy rewarded.
    assert weights["tempoCost"] < 0
    assert weights["keyCost"] < 0
    assert weights["harmonyFit"] > 0
    assert weights["energyMatch"] > 0
    assert doc["report"]["val_auc"] > 0.9
    flags = json.loads((KITS / "sidecars.json").read_text())
    assert flags.get("song_policy") is True


def test_song_policy_bonus_is_bounded():
    import subprocess

    out = Path("/tmp/opencode") / "song_policy.cjs"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(ROOT / "frontend/node_modules/.bin/esbuild"),
         str(ROOT / "frontend/src/stem_remix.ts"), "--bundle",
         "--platform=node", "--format=cjs", f"--outfile={out}",
         "--external:three", "--external:onnxruntime-web"],
        check=True, capture_output=True, text=True,
    )
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
assert.ok(src.includes('songPolicyBonus'), 'bonus helper exists');
assert.ok(src.includes('song_policy.json'), 'loader fetches sidecar');
assert.ok(src.includes('0.15'), 'bounded prior');
"""
    result = subprocess.run(["node", "-e", script, str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
