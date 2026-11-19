"""Real-song phrase transitions learned in the shipped bank's phrase space."""
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


def test_bank_transition_artifact_is_mined_from_real_chronology():
    doc = json.loads((KITS / "bank_transition.json").read_text())
    assert doc["format"] == "wub-bank-transition-v1"
    assert (doc["input_dim"], doc["hidden"], doc["embedding_dim"]) == (136, 32, 64)
    assert doc["report"]["real_transitions"] >= 2000
    assert doc["report"]["tracks"] >= 400
    assert doc["report"]["val_auc"] > 0.9
    assert len(doc["topk"]) >= 2000
    for nexts in list(doc["topk"].values())[:20]:
        assert nexts and all(0 <= item["p"] <= 1 for item in nexts)
    assert json.loads((KITS / "sidecars.json").read_text())["bank_transition"] is True
    assert "wub-bank-transition-v1" in (ROOT / "scripts/train_bank_transition.py").read_text()
    # The conductor is now a bias, not the arbiter of the whole set.
    conductor = json.loads((KITS / "conductor.json").read_text())
    assert 0 < conductor["authority"] < 1


def test_learned_score_prefers_a_real_successor_phrase():
    out = bundle("bank_transition", "bank_transition.ts")
    script = r"""
const assert=require('node:assert/strict');
const fs=require('fs');
const {bankTransitionScore,inferBank,isBankTransitionArtifact}=require(process.argv[1]);
const model=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const manifest=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
assert.ok(isBankTransitionArtifact(model),'artifact passes validation');
const byId=new Map();
for(const kit of manifest.kits) for(const p of kit.phrases)
  byId.set(p.id,{id:p.id,section:p.section,embedding:p.embedding||[],energy:p.energy||.5,
    bpm:kit.bpm||120,keyRoot:0});
const curId=Object.keys(model.topk)[0];
const cur=byId.get(curId);
const nextId=model.topk[curId][0].id;
const real=byId.get(nextId);
assert.ok(cur&&real,'real transition phrases are in the manifest');
const other={id:'unknown',section:cur.section,embedding:new Array(64).fill(0),
 energy:cur.energy,bpm:cur.bpm,keyRoot:0};
const a=inferBank(model,require(process.argv[1]).bankTransitionFeatures(cur,real,'GROOVE_A'));
const b=bankTransitionScore(model,cur,real,'GROOVE_A');
const c=bankTransitionScore(model,cur,other,'GROOVE_A');
assert.ok(a>=0&&a<=1&&b>=0&&b<=1&&c>=0&&c<=1);
assert.ok(b>c,'the actually following phrase scores higher');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out),
         str(KITS / "bank_transition.json"), str(KITS / "kit_manifest.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
