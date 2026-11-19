"""Implicit preferences from reticule/pin gestures and offline fine-tuning."""
import json
import os
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


def test_reticle_preferences_are_logged_and_exportable():
    stem = (ROOT / "frontend/src/stem_remix.ts").read_text()
    main = (ROOT / "frontend/src/main.ts").read_text()
    assert "__wubPreferenceLog" in stem and "__wubPreferenceExport" in stem
    assert "recordPreference({" in stem
    assert "kind: distance <= 0.06 ? \"hold\" : \"retarget\"" in stem
    assert "kind: \"release\"" in stem
    assert "onReticleDrop" in main and "setReticleSteering(stem, x, y, true)" in main
    assert "onReticleRelease" in main and "clearReticleSteering(stem)" in main
    assert "fine_tune_preferences.py" in " ".join(p.name for p in (ROOT / "scripts").glob("*.py"))


def test_preference_log_is_bounded_and_returns_events():
    out = bundle("pref_log", "stem_remix.ts")
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const p=new StemRemixPlayer({now:()=>0});
const ev=(i)=>({kind:'retarget',at:i,bar:i*4,stem:'bass',from:'a'+i,to:'b'+i,
 fromTrack:'t1',toTrack:'t2',section:'DROP',style:'house',x:.5,y:.5,dropX:.5,dropY:.5});
assert.equal(p.getPreferenceLog().length,0);
p.recordPreference(ev(1));
assert.equal(p.getPreferenceLog().length,1);
assert.equal(p.getPreferenceLog()[0].to,'b1');
for(let i=0;i<500;i++)p.recordPreference(ev(i));
assert.ok(p.getPreferenceLog().length<=400,'log is capped');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_preference_adapter_fine_tunes_without_crashing(tmp_path):
    manifest = json.loads((KITS / "kit_manifest.json").read_text())
    ids = [(k["track_id"], p["id"]) for k in manifest["kits"] for p in k["phrases"]][:12]
    events = [
        {"kind": "retarget", "at": i * 8, "bar": i * 4, "stem": "drums",
         "from": ids[i][1], "to": ids[i + 1][1], "fromTrack": ids[i][0],
         "toTrack": ids[i + 1][0], "section": "DROP", "style": "house",
         "x": .5, "y": .5, "dropX": .5, "dropY": .5, "untilBar": i * 4 + 32}
        for i in range(0, 10, 2)
    ]
    events.append({"kind": "release", "at": 90, "bar": 48, "stem": "bass",
                   "from": ids[0][1], "to": None, "fromTrack": ids[0][0],
                   "toTrack": None, "section": "DROP", "style": "house",
                   "x": .3, "y": .3, "dropX": 0, "dropY": 0})
    prefs = tmp_path / "prefs.json"
    prefs.write_text(json.dumps({"version": 1, "source": "wub-reticle-preferences", "events": events}))
    out = tmp_path / "bank.json"
    result = subprocess.run([
        "python3", str(ROOT / "scripts/fine_tune_preferences.py"),
        "--preferences", str(prefs), "--out", str(out), "--mode", "pg", "--epochs", "2",
    ], cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": f"{ROOT / 'scripts'}:{ROOT}"})
    assert result.returncode == 0, result.stderr
    doc = json.loads(out.read_text())
    assert doc["format"] == "wub-bank-transition-v1"
    assert doc["report"]["events"] == 6 and doc["report"]["pairs"] > 0
    assert doc["report"]["mode"] == "pg"
    assert "topk" in doc and len(doc["topk"]) > 0
