"""Learned next-bar patterns and real-song quartet harmony floors."""
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


def test_learned_artifacts_are_complete_and_beat_their_baselines():
    flags = json.loads((KITS / "sidecars.json").read_text())
    for key in ("cell_chroma", "pattern_bar_model", "harmony_continuity"):
        assert flags[key] is True
        assert key in (ROOT / "scripts/build_sidecars.py").read_text()

    pattern = json.loads((KITS / "pattern_bar_model.json").read_text())
    assert pattern["format"] == "wub-pattern-bars-v1"
    assert (pattern["input_dim"], pattern["output_dim"], pattern["hidden"]) == (55, 48, 64)
    assert set(pattern["stems"]) == {"drums", "bass", "other"}
    for net in pattern["stems"].values():
        assert len(net["mean"]) == len(net["std"]) == 55
        assert len(net["w1"]) == 64 and all(len(row) == 55 for row in net["w1"])
        assert len(net["w2"]) == 48 and all(len(row) == 64 for row in net["w2"])
        assert net["val_mse"] < net["baseline_mse"], "model must beat the mean baseline"

    chroma = json.loads((KITS / "cell_chroma.json").read_text())
    assert chroma["format"] == "cell_chroma_v1"
    assert len(chroma["phrases"]) >= 3600
    for stem_map in chroma["phrases"].values():
        assert set(stem_map) == {"bass", "other", "vocals"}
        for bars in stem_map.values():
            assert len(bars) == 4
            assert all(len(bar) == 12 for bar in bars)

    harmony = json.loads((KITS / "harmony_continuity.json").read_text())
    assert harmony["format"] == "wub-harmony-continuity-v1"
    policy = harmony["policy"]
    assert 0.5 < policy["min_bar_pair_cos"] < 0.9
    assert policy["min_boundary_cos"] > 0.85
    assert 0 < policy["max_clash_penalty"] <= 0.3
    assert harmony["root_agreement_rate"] > 0


def test_pattern_vector_and_learned_fit_prefer_the_predicted_bar():
    out = bundle("pattern_vec", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const c=new BarComposer();
const meta={p:[.8,.2,.6,.3],r:[2,5,7,0],q:['minor','major','minor','major'],
 o:[0b1011,0b0101,0b1111,0],chroma:[[.7,.1,.2,0,0,0,0,0,0,0,0,0],
 [0,.8,.2,0,0,0,0,0,0,0,0,0],[.5,.2,.3,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0]]};
const v=c.patternBarVec(meta,0,'GROOVE_A');
assert.equal(v.length,55);
assert.deepEqual(v.slice(0,4),[1,1,0,1]);
assert.equal(v[16],3/16);
assert.equal(v[25],1,'root chroma one-hot');
assert.equal(v[49],1,'GROOVE_A one-hot');
const fit=c.patternBarFit(v,v);
assert.ok(Math.abs(fit-1)<1e-6,'identical pattern fits perfectly');
// A synthetic head whose constant output is exactly the candidate pattern.
const target=v.slice(0,48);
const net={mean:new Array(55).fill(0),std:new Array(55).fill(1),w1:[new Array(55).fill(0)],
 b1:[0],w2:Array.from({length:48},()=>[0]),b2:target};
c.patternBarModel={format:'wub-pattern-bars-v1',version:1,input_dim:55,output_dim:48,hidden:1,
 stems:{bass:net}};
const score=c.patternBarScore('bass',meta,0,meta,0,'GROOVE_A');
assert.ok(score>0.19,'matching prediction gives a positive learned bonus');
c.patternBarModel=null;
assert.equal(c.patternBarScore('bass',meta,0,meta,0,'GROOVE_A'),0,'no model is neutral');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_harmony_floor_penalizes_clashes_and_broken_tails():
    out = bundle("harmony_floor", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const tone=(i)=>{const v=new Array(12).fill(0);v[i]=1;return v;};
const chroma=(...tones)=>{const v=new Array(12).fill(0);for(const t of tones)v[t]=1;return v;};
const doc={format:'cell_chroma_v1',version:1,phrases:{
  good:{bass:[tone(0),tone(0),tone(0),tone(0)],
        other:[tone(0),tone(0),tone(0),tone(0)],
        vocals:[tone(0),tone(0),tone(0),tone(0)]},
  bad:{bass:[tone(0),tone(0),tone(0),tone(0)],
       other:[tone(1),tone(1),tone(1),tone(1)],
       vocals:[tone(0),tone(0),tone(0),tone(0)]},
}};
const policy={format:'wub-harmony-continuity-v1',version:1,
 pair_chroma_cos:{raw:{p10:.65,p50:.85},triad:{p10:0,p50:.25}},
 boundary_chroma_cos:{p10:.898,p50:.99},root_agreement_rate:.3,
 policy:{min_bar_pair_cos:.65,min_boundary_cos:.898,max_clash_penalty:.22}};
const entry=(id)=>({bass:{phrase:{id},keyShift:0},other:{phrase:{id},keyShift:0},
 vocals:{phrase:{id},keyShift:0}});
const c=new BarComposer();
assert.equal(c.harmonyPenaltyFor(entry('good')),0,'neutral without sidecars');
c.loadCellChroma(doc);
c.harmonyContinuity=policy;
assert.equal(c.harmonyPenaltyFor(entry('good')),0,'consonant quartet is free');
const clash=c.harmonyPenaltyFor(entry('bad'));
assert.ok(clash>0.1 && clash<=0.34,'orthogonal pitched stems are penalized and bounded');
const rotated=new BarComposer();rotated.loadCellChroma(doc);rotated.harmonyContinuity=policy;
assert.ok(rotated.harmonyPenaltyFor(entry('bad'))>0.1,'sidecar policy applies the floor');
const tail={meta:{chroma:[tone(1),tone(1),tone(1),tone(1)],q:['mono','mono','mono','mono'],er:['A','A','A','A']},
 srcBar:0,semitones:0,quality:'mono',role:'A',section:'GROOVE_A'};
const withTail=new BarComposer();withTail.loadCellChroma(doc);withTail.harmonyContinuity=policy;
withTail.lastBlockTail.vocals=tail;
const broken=withTail.harmonyPenaltyFor({bass:{phrase:{id:'good'},keyShift:0},
 other:{phrase:{id:'good'},keyShift:0},vocals:{phrase:{id:'bad'},keyShift:0}});
assert.ok(broken>0.08 && broken<=0.34,'incoming vocal that breaks the tail also costs');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
