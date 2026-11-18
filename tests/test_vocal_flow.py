"""Vocal development priors should stop a one-bar phrase looping verbatim."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vocal_flow_sidecar_mined_from_real_songs():
    base = ROOT / "frontend/public/data/kits"
    prior = json.loads((base / "vocal_flow.json").read_text())
    assert prior["format"] == "wub-vocal-flow-v1"
    assert float(prior["run_lengths"]["3"]) > 0
    # A four-active-bar phrase normally uses about two distinct melodic bars.
    target = prior["distinct_bars_per_active"]["4"]
    assert 1.5 <= float(target) <= 3.0
    assert 0.1 <= float(prior["variation_rate"]) <= 0.9
    assert json.loads((base / "sidecars.json").read_text())["vocal_flow"] is True


def test_bar_composer_develops_vocal_bars_instead_of_copying_one(tmp_path):
    out = tmp_path / "composer.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const c=new BarComposer();
c.vocalFlow={format:'wub-vocal-flow-v1',run_lengths:{},distinct_bars_per_active:{
  '1':1,'2':2,'3':2,'4':2},variation_rate:.35};
const single=[{itemIndex:0,srcBar:0,meta:'y',semitones:0},{itemIndex:1,srcBar:2,meta:'x',semitones:3}];
const schedule=c.developVocalBars(single,8,0);
assert.equal(schedule.length,8);
assert.equal(new Set(schedule.map(s=>s.srcBar)).size,2);
for(let i=1;i<schedule.length;i++)assert.notEqual(schedule[i].srcBar,schedule[i-1].srcBar);
assert.equal(schedule[0].semitones,0);
c.vocalFlow=null;
const fallback=c.developVocalBars([{itemIndex:0,srcBar:1},{itemIndex:0,srcBar:3}],4,0);
assert.equal(new Set(fallback.map(s=>s.srcBar)).size,2);
"""
    subprocess.run(["node", "-e", script, str(out)], check=True, capture_output=True, text=True)
