"""Bounded final-gate beam search over next-cut combinations."""
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


def test_beam_config_is_shipped_and_bounded():
    doc = json.loads((KITS / "beam_search.json").read_text())
    assert doc["format"] == "wub-beam-search-v1"
    assert 2 <= doc["depth"] <= 4 and 2 <= doc["width"] <= 8
    assert 0 <= doc["lookahead_weight"] <= 0.6
    assert json.loads((KITS / "sidecars.json").read_text())["beam_search"] is True
    assert "beam_search" in (ROOT / "scripts/build_sidecars.py").read_text()


def test_beam_respects_caps_and_prefers_better_mixes():
    flow = bundle("beam_flow", "slice_flow.ts")
    dim = 32
    net = lambda n, o: dict(mean=[0]*n, std=[1]*n, w1=[[0]*n for _ in range(32)],
                            b1=[0]*32, w2=[[0]*32 for _ in range(o)], b2=[0]*o)
    artifact = dict(format="slice_flow_v1", feature_dim=dim, phrase_ids=["a", "b"],
                    feature_file="features.f32", critic=net(108, 1), mix=net(16, 1),
                    planner=net(8, 3), foundation_seconds=90)
    script = r"""
const assert=require('node:assert/strict');
const {SliceFlow}=require(process.argv[1]);
const artifact=JSON.parse(process.argv[2]);
global.fetch=async()=>({ok:true,json:async()=>artifact,
 arrayBuffer:async()=>new ArrayBuffer(2*4*4*32)});
const stems=['drums','bass','other','vocals'];
const task=(s,id)=>({stem:s,run:[{stem:s,phrase:{id},kit:{track_id:id},keyShift:0,rate:1}]});
(async()=>{
 const flow=new SliceFlow();await flow.load('/kits');
 const ctx={at:0,bars:8,bpm:120,section:'GROOVE_A',targetEnergy:.7,sectionChanged:false,
   rotation:false,bridge:false};
 for(const s of stems)flow.record(task(s,'a'),0,4,120);
 const wanted=stems.map(s=>task(s,'b'));
 const greedy=flow.select(wanted,{...ctx,at:8,beamDepth:1},s=>task(s,'a'),new Set());
 const beam=flow.select(wanted,{...ctx,at:8,beamDepth:3,beamWidth:4,lookaheadWeight:.25},
   s=>task(s,'a'),new Set());
 assert.equal(beam.length,4,'one task per stem');
 assert.deepEqual(stems.map(s=>beam.find(t=>t.stem===s).stem),stems,'all stems present');
 const changes=(arr)=>arr.filter((t,i)=>t.run[0].phrase.id!==greedy[i].run[0].phrase.id).length;
 assert.ok(changes(beam)<=2,'bounded change cap is respected');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(["node", "-e", script, str(flow), json.dumps(artifact)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
