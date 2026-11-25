"""Offline-trained end-to-end conductor: artifact, state layout, inference, hooks."""
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


def test_conductor_artifact_is_trained_track_disjoint_and_beats_baselines():
    doc = json.loads((KITS / "conductor.json").read_text())
    assert doc["format"] == "wub-conductor-v2"
    assert (doc["state_dim"], doc["target_dim"], doc["hidden"], doc["history_cells"]) == (1202, 202, 256, 6)
    assert 0 < doc["authority"] < 1
    assert len(doc["layers"]) == 3
    assert len(doc["mean"]) == len(doc["std"]) == 1202
    assert len(doc["layers"][0]["w"]) == 256
    assert len(doc["layers"][2]["w"]) == 202
    r = doc["report"]
    assert r["train_tracks"] >= 1500 and r["val_tracks"] >= 100
    assert r["section_accuracy"] > r["section_majority"] + 0.05
    assert r["section_macro_f1"] > 0.5
    assert r["pattern_mse"] < r["pattern_baseline_mse"] * 0.85
    assert r["energy_mae"] < r["energy_baseline_mae"] * 0.9
    assert all(0.6 < v < 1.0 for v in r["change_auc"].values())
    assert "conductor" in json.loads((KITS / "sidecars.json").read_text())
    assert "wub-conductor-v2" in (ROOT / "scripts/train_conductor_v2.py").read_text()


def test_conductor_state_layout_and_inference_decodes_every_head():
    out = bundle("conductor_infer", "conductor.ts")
    doc = json.loads((KITS / "conductor.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {conductorStateV2,inferConductor,isConductorArtifact}=require(process.argv[1]);
const model=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
assert.ok(isConductorArtifact(model),'artifact passes validation');
const cell=(v)=>Array.from({length:4},()=>new Array(48).fill(v));
const hist=Array.from({length:6},(_,i)=>cell(i===5?1:0));
const st=conductorStateV2(hist,['INTRO','GROOVE_A','GROOVE_A','BUILDUP','GROOVE_A','DROP'],
 [-30,-25,-22,-18,-15,-12],'DROP',-10,2,.5);
assert.equal(st.length,1202);
assert.equal(st[1191],1,'DROP one-hot at 1188+3');
assert.equal(st[192],1,'first stem of second cell starts at 192');
assert.ok(Math.abs(st[1193]-50/60)<1e-6,'context energy scaling');
const d=inferConductor(model,st);
assert.ok(model.sections.includes(d.section));
assert.ok(d.sectionConfidence>=0 && d.sectionConfidence<=1);
assert.ok(d.energy>=0 && d.energy<=1);
assert.equal(d.change.length,4);
assert.ok(d.change.every(v=>v>=0&&v<=1));
assert.equal(d.patterns.length,4);
assert.ok(d.patterns.every(p=>p.length===48&&p.every(v=>v>=0&&v<=1)));
assert.ok(d.authority>0 && d.authority<1,'bias, not arbiter');
"""
    path = Path("/tmp/opencode/conductor_artifact.json")
    path.write_text(json.dumps(doc))
    result = subprocess.run(
        ["node", "-e", script, str(out), str(path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_conductor_patterns_drive_bar_choice_and_change_bias():
    bars = bundle("conductor_bars", "bar_composer.ts")
    remix = bundle("conductor_remix", "stem_remix.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const {StemRemixPlayer}=require(process.argv[2]);
const meta={p:[.5,.5,.5,.5],r:[1,4,7,9],q:['minor','major','minor','major'],
 o:[0b1011,0b0101,0b1111,0b0011],es:[.5,.5,.5,.5],er:['A','A','A','A']};
const entry={phrase:{id:'v'},kit:{track_id:'t'},keyShift:0,rate:1,energy:.6,section:'GROOVE_A',rmsDb:-20};
const c=new BarComposer();
assert.deepEqual(c.buildContiguous4BarCells(entry,meta,'bass',4,0,'DROP').map(x=>x.srcBar),[0,1,2,3]);
const target=c.patternBarVec(meta,2,'DROP').slice(0,48);
c.horizonBarSelect=true;
c.conductorNextPatterns={bass:target};
const learned=c.buildContiguous4BarCells(entry,meta,'bass',4,0,'DROP');
assert.equal(learned[0].srcBar,2,'conductor next-pattern drives slot 0');
assert.deepEqual([...learned.map(x=>x.srcBar)].sort(),[0,1,2,3]);
const p=new StemRemixPlayer({now:()=>0});
assert.equal(p.conductorChangeBonus(['bass']),0,'neutral without decision');
p.conductorDecision={authority:1,change:[.9,.1,.1,.1],section:'DROP',sectionConfidence:1,
 energy:.9,patterns:[]};
assert.ok(p.conductorChangeBonus(['drums'])>0,'drums change is predicted');
assert.ok(p.conductorChangeBonus(['bass'])<0,'bass change is not predicted');
"""
    result = subprocess.run(["node", "-e", script, str(bars), str(remix)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
