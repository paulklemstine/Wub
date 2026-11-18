"""Unified short-horizon bar / phrase / sequence ranking bias."""
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


def test_planner_config_is_shipped_and_neutral_by_default():
    doc = json.loads((KITS / "horizon_planner.json").read_text())
    assert doc["format"] == "wub-horizon-planner-v1"
    assert doc["horizon_blocks"] >= 2
    assert abs(sum(doc["weights"].values()) - 1.0) < 1e-6
    assert 0 < doc["max_score"] <= 0.25
    assert json.loads((KITS / "sidecars.json").read_text())["horizon_planner"] is True
    assert "horizon_planner" in (ROOT / "scripts/build_sidecars.py").read_text()


def test_learned_slot_selector_uses_only_the_phrases_own_bars():
    out = bundle("horizon_bars", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const meta={p:[.5,.5,.5,.5],r:[1,4,7,9],q:['minor','major','minor','major'],
 o:[0b1011,0b0101,0b1111,0b0011],es:[.5,.5,.5,.5],er:['A','A','A','A']};
const entry={phrase:{id:'v'},kit:{track_id:'t'},keyShift:0,rate:1,energy:.6,section:'GROOVE_A',rmsDb:-20};
const c=new BarComposer();
const plain=c.buildContiguous4BarCells(entry,meta,'bass',4,0,'GROOVE_A');
assert.deepEqual(plain.map(x=>x.srcBar),[0,1,2,3],'no model keeps contiguous bars');
// A constant head whose output is bar 1's pattern: bar 1 must lead slot 0.
const target=c.patternBarVec(meta,1,'GROOVE_A').slice(0,48);
const net={mean:new Array(55).fill(0),std:new Array(55).fill(1),
 w1:[new Array(55).fill(0)],b1:[0],w2:Array.from({length:48},()=>[0]),b2:target};
c.patternBarModel={format:'wub-pattern-bars-v1',version:1,input_dim:55,output_dim:48,
 hidden:1,stems:{bass:net}};
c.horizonBarSelect=true;
c.lastBlockTail.bass={meta,srcBar:0,semitones:0,quality:'minor',role:'A',section:'GROOVE_A'};
const learned=c.buildContiguous4BarCells(entry,meta,'bass',4,0,'GROOVE_A');
assert.equal(learned[0].srcBar,1,'learned fit leads with the predicted bar');
assert.deepEqual([...learned.map(x=>x.srcBar)].sort(),[0,1,2,3],'each own bar used once');
assert(learned.every(x=>x.entry===entry),'never splices a foreign phrase');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_horizon_score_is_bounded_and_neutral_without_config():
    out = bundle("horizon_score", "stem_remix.ts")
    doc = json.loads((KITS / "horizon_planner.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const cfg=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
const entry=(id)=>({phrase:{id},kit:{track_id:'t'},keyShift:0,rate:1,energy:.7,earQ:.8,
 section:'DROP',rmsDb:-18});
const picks=[entry('a')];
assert.equal(p.horizonScoreFor(['bass'],picks,16,.7,'DROP'),0,'no config is neutral');
p.horizonConfig=cfg;
const s=p.horizonScoreFor(['bass'],picks,16,.7,'DROP');
assert.ok(Number.isFinite(s) && Math.abs(s)<=cfg.max_score+1e-9,'bounded: '+s);
p.horizonConfig=null;
assert.equal(p.horizonScoreFor(['bass'],picks,16,.7,'DROP'),0);
p.horizonConfig=cfg;
p.transitionCohort={p_3plus:{section:0}};
p.stemHistory.bass=['a','a','a'];
const stale=p.horizonScoreFor(['bass'],[entry('a')],16,.7,'DROP');
const fresh=p.horizonScoreFor(['bass'],[entry('brand-new')],16,.7,'DROP');
assert.ok(fresh>stale,'a recently used phrase is downweighted: '+stale+' vs '+fresh);
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(doc)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
