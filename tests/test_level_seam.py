"""Corpus-measured 20 ms seam gain correction."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITS = ROOT / "frontend/public/data/kits"


def bundle(name: str, source: str) -> Path:
    out = Path("/tmp/opencode") / f"{name}.cjs"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_seam_artifact_is_measured_and_bounded():
    doc = json.loads((KITS / "level_seam.json").read_text())
    assert doc["format"] == "wub-level-seam-v1"
    assert doc["window_sec"] == 0.02
    assert set(doc["correction"]) == {"drums", "bass", "other", "vocals"}
    for stem, value in doc["correction"].items():
        assert 0.88 <= value <= 1.12, stem
        assert doc["report"][stem]["phrases"] >= 3600
    assert json.loads((KITS / "sidecars.json").read_text())["level_seam"] is True
    assert "--kits-dir" in (ROOT / "scripts/build_level_seam.py").read_text()


def test_seam_gain_is_neutral_without_the_sidecar():
    out = bundle("level_seam", "stem_remix.ts")
    doc = json.loads((KITS / "level_seam.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
assert.equal(p.seamGain('vocals',1),1,'no sidecar keeps equal-gain handoff');
p.levelCorrection=model.correction;
assert.ok(Math.abs(p.seamGain('vocals',1)-model.correction.vocals)<1e-9);
assert.ok(Math.abs(p.seamGain('drums',.8)-.8*model.correction.drums)<1e-9);
assert.ok(p.seamGain('bass',0.01)>=0.2,'correction is bounded');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(doc)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
