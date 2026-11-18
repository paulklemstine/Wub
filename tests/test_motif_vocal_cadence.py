"""Learned motif return calendar and vocal release cadence."""
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


def test_cadence_artifacts_match_real_song_statistics():
    flags = json.loads((KITS / "sidecars.json").read_text())
    assert flags["motif_calendar"] is True and flags["vocal_cadence"] is True

    motif = json.loads((KITS / "motif_calendar.json").read_text())
    assert motif["format"] == "wub-motif-calendar-v1"
    assert motif["tracks"] >= 1800
    drums = motif["return_rate"]["drums"]
    assert drums["8"] > motif["return_rate"]["bass"]["8"] > motif["return_rate"]["vocals"]["8"]
    assert drums["8"] > drums["16"] > drums["32"]
    assert motif["policy"]["return_bonus"]["drums"] > motif["policy"]["novelty_bonus"]["drums"]
    for stem, bonus in motif["policy"]["novelty_bonus"].items():
        assert 0 < bonus < 0.2, stem

    vocal = json.loads((KITS / "vocal_cadence.json").read_text())
    assert vocal["format"] == "wub-vocal-cadence-v1"
    assert vocal["change_prob"]["section"] > vocal["change_prob"]["inside"] > 0.6
    assert vocal["policy"]["min_hold_bars"] == 8
    assert 0 < vocal["policy"]["section_change_bonus"] <= 0.15
    assert 0.5 < sum(vocal["run_lengths_bars"].values()) <= 1.0
    src = (ROOT / "scripts/build_motif_calendar.py").read_text()
    assert "motif_calendar" in src and "vocal_cadence" in src


def test_vocal_hold_budget_is_optional_only():
    out = bundle("cadence_arc", "arc_planner.ts")
    script = r"""
const assert=require('node:assert/strict');
const {vocalChangeAllowed}=require(process.argv[1]);
assert.equal(vocalChangeAllowed(20,5,false),false,'old 16-bar default');
assert.equal(vocalChangeAllowed(21,5,false),true);
assert.equal(vocalChangeAllowed(13,5,false,8),true,'learned 8-bar budget releases');
assert.equal(vocalChangeAllowed(13,5,false),false,'3-arg callers are unchanged');
assert.equal(vocalChangeAllowed(12,5,true,8),true,'section change always releases');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_motif_prior_rewards_returns_and_fresh_phrases():
    out = bundle("motif_prior", "stem_remix.ts")
    motif = json.loads((KITS / "motif_calendar.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
assert.equal(p.motifPriorFor(['drums'],[{phrase:{id:'a'}}]),0,'neutral without sidecar');
p.motifCalendar=model;
p.stemHistory.drums=['a','b','c'];
const ret=p.motifPriorFor(['drums'],[{phrase:{id:'a'}}]);
assert.ok(ret>0.15 && ret<=0.32,'8-bar return of a drum motif is rewarded: '+ret);
const fresh=p.motifPriorFor(['drums'],[{phrase:{id:'never'}}]);
assert.ok(fresh>0 && fresh<ret,'fresh material gets a smaller novelty bonus');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(motif)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
