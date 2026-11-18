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
const four=[0,1,2,3].map(bar=>({itemIndex:0,srcBar:bar}));
const eight=c.developVocalBars(four,8,0);
assert.equal(new Set(eight.map(s=>s.srcBar)).size,4);
const rotated=c.developVocalBars(four,8,1);
assert.notDeepEqual(eight.map(s=>s.srcBar),rotated.map(s=>s.srcBar));
c.vocalFlow=null;
const fallback=c.developVocalBars([{itemIndex:0,srcBar:1},{itemIndex:0,srcBar:3}],4,0);
assert.equal(new Set(fallback.map(s=>s.srcBar)).size,2);
"""
    subprocess.run(["node", "-e", script, str(out)], check=True, capture_output=True, text=True)


def test_vocal_bar_model_is_trained_and_shipped():
    base = ROOT / "frontend/public/data/kits"
    model = json.loads((base / "vocal_bar_model.json").read_text())
    assert model["format"] == "wub-vocal-bar-v1"
    assert model["input_dim"] == 120 and model["next_dim"] == 29
    assert model["report"]["run_mae"] < model["report"]["run_baseline_mae"]
    assert json.loads((base / "sidecars.json").read_text())["vocal_bar_model"] is True


def test_bar_composer_uses_next_vocal_bar_prediction(tmp_path):
    out = tmp_path / "vocal_model.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    model_text = (ROOT / "frontend/public/data/kits/vocal_bar_model.json").read_text()
    model_path = tmp_path / "vocal_bar_model.json"
    model_path.write_text(model_text)
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const model=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
const meta=(root,mask)=>({o:[mask,0,0,0],p:[.8,.7,.6,.5],r:[root,(root+2)%12,(root+7)%12,(root+4)%12]});
const pool=[0,1,2,3].map(bar=>({itemIndex:0,srcBar:bar,meta:meta(bar,0b1010101010101010)}));
const c=new BarComposer();
assert.equal(c.vocalBarModel,null);
const plain=c.developVocalBars(pool,8,0).map(s=>s.srcBar);
c.vocalBarModel=model;
const learned=c.developVocalBars(pool,8,0).map(s=>s.srcBar);
assert.equal(learned.length,8);
assert.equal(new Set(learned).size,4);
assert.ok(learned.some((v,i)=>v!==plain[i]));
"""
    result = subprocess.run(["node", "-e", script, str(out), str(model_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
