"""Learned section-boundary stem cohorts, mined from real songs."""
import base64
import json
import subprocess
from pathlib import Path

import numpy as np

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


def test_cohort_artifact_is_mined_from_the_real_corpus():
    doc = json.loads((KITS / "transition_cohort.json").read_text())
    assert doc["format"] == "wub-transition-cohort-v1"
    assert doc["tracks"] >= 1800 and doc["boundaries"] >= 50000
    assert doc["p_3plus"]["section"] > doc["p_3plus"]["inside"]
    assert doc["p_3plus"]["section"] > 0.4
    assert doc["policy"]["max_changes_inside"] == 2
    assert doc["policy"]["max_changes_section"] == 3
    pitched = {"bass", "other", "vocals"}
    assert any(
        c["kind"] == "section" and set(c["stems"]) == pitched and c["p"] > 0.1
        for c in doc["cohorts"]
    )
    assert doc["chroma_cos"]["section"]["p10"] > 0.85  # harmony stays intact
    flags = json.loads((KITS / "sidecars.json").read_text())
    assert flags["transition_cohort"] is True
    assert "transition_cohort" in (ROOT / "scripts/build_sidecars.py").read_text()


def test_movement_budget_only_widens_at_learned_section_changes():
    out = bundle("cohort_arc", "arc_planner.ts")
    script = r"""
const assert=require('node:assert/strict');
const {movementBudgetAt}=require(process.argv[1]);
assert.deepEqual(movementBudgetAt(16,4),{minChanges:0,maxChanges:1,maxHoldBars:16});
assert.deepEqual(movementBudgetAt(16,8),{minChanges:1,maxChanges:2,maxHoldBars:16});
assert.deepEqual(movementBudgetAt(16,8,true),{minChanges:1,maxChanges:2,maxHoldBars:16});
const wide=movementBudgetAt(16,8,true,3);
assert.deepEqual(wide,{minChanges:1,maxChanges:3,maxHoldBars:16});
assert.equal(movementBudgetAt(16,8,false,3).maxChanges,2,'no widening inside a section');
assert.equal(movementBudgetAt(16,4,true,3).maxChanges,3);
assert.deepEqual(movementBudgetAt(96,8,true,3),{minChanges:0,maxChanges:0,maxHoldBars:8});
assert.equal(movementBudgetAt(16,8,true,99).maxChanges,4,'cap is bounded');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_final_gate_honours_the_section_cohort_cap():
    out = bundle("cohort_flow", "slice_flow.ts")
    dim = 32
    def net(input_dim: int, outputs: int):
        return dict(mean=[0] * input_dim, std=[1] * input_dim,
                    w1=[[0] * input_dim for _ in range(32)], b1=[0] * 32,
                    w2=[[0] * 32 for _ in range(outputs)], b2=[0] * outputs)

    artifact = dict(
        format="slice_flow_v1", feature_dim=dim, phrase_ids=["a", "b"],
        feature_file="features.f32",
        critic=net(108, 1), mix=net(16, 1), planner=net(8, 3),
        foundation_seconds=90,
    )
    features = np.ones((2, 4, 4, dim), np.float32) * 0.3
    raw = base64.b64encode(features.tobytes()).decode()
    script = r"""
const assert=require('node:assert/strict');
const {SliceFlow}=require(process.argv[1]);
const artifact=JSON.parse(process.argv[2]);
const raw=Buffer.from(process.argv[3],'base64');
global.fetch=async url=>({ok:true,json:async()=>artifact,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)});
const stems=['drums','bass','other','vocals'];
const entry=(stem,id)=>({stem,phrase:{id},kit:{track_id:id},keyShift:2,rate:1});
const task=(stem,id)=>({stem,run:[entry(stem,id)]});
(async()=>{
 const flow=new SliceFlow(); assert.equal(await flow.load('/kits'),true);
 const ctx={at:0,bars:8,bpm:120,section:'GROOVE_A',targetEnergy:.7,sectionChanged:false,rotation:false,bridge:false};
 const initial=stems.map(s=>task(s,'a'));
 for(const t of initial)flow.record(t,0,4,120);
 const wanted=stems.map(s=>task(s,'b'));
 const inside=flow.select(wanted,{...ctx,at:8},s=>task(s,'a'),new Set());
 assert.ok(inside.filter(t=>t.run[0].phrase.id==='b').length<=2,'inside cap stays 2');
 const section=flow.select(wanted,{...ctx,at:16,sectionChanged:true,cohortMaxChanges:3},s=>task(s,'a'),new Set());
 assert.ok(section.filter(t=>t.run[0].phrase.id==='b').length>=3,'section cohort cap widens to 3');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(artifact), raw],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
