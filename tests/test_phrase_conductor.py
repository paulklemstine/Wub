"""Phrase-space conductor trained on real song phrase chronology."""
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


def test_phrase_artifact_beats_its_baseline():
    doc = json.loads((KITS / "phrase_conductor.json").read_text())
    assert doc["format"] == "wub-phrase-conductor-v1"
    assert (doc["state_dim"], doc["target_dim"], doc["embed_dim"]) == (264, 70, 64)
    assert len(doc["mean"]) == len(doc["std"]) == 264
    assert len(doc["layers"]) == 3 and len(doc["layers"][2]["w"]) == 70
    r = doc["report"]
    assert r["pairs"] >= 4000 and r["tracks"] >= 400
    assert r["val_embedding_cosine"] > r["baseline_cosine"] + 0.1
    assert r["section_accuracy"] > r["section_majority"] + 0.05
    assert json.loads((KITS / "sidecars.json").read_text())["phrase_conductor"] is True
    assert "wub-phrase-conductor-v1" in (ROOT / "scripts/train_phrase_conductor.py").read_text()


def test_phrase_conductor_scores_real_successors_above_random():
    out = bundle("phrase_conductor", "phrase_conductor.ts")
    script = r"""
const assert=require('node:assert/strict');
const fs=require('fs');
const {phraseConductorState,inferPhraseConductor,phraseConductorScore,
 isPhraseConductorArtifact}=require(process.argv[1]);
const model=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const bank=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const manifest=JSON.parse(fs.readFileSync(process.argv[4],'utf8'));
const cells=JSON.parse(fs.readFileSync(process.argv[5],'utf8')).phrases;
const chromaDoc=JSON.parse(fs.readFileSync(process.argv[6],'utf8')).phrases;
assert.ok(isPhraseConductorArtifact(model),'artifact validates');
const pcs={C:0,'C#':1,D:2,'D#':3,E:4,F:5,'F#':6,G:7,'G#':8,A:9,'A#':10,B:11};
const rootOf=(s)=>pcs[(s||'C').split()[0].replace('m','')]||0;
function pat(meta,row){const v=new Array(48).fill(0);const mask=meta.o?.[0]||0;
 for(let s=0;s<16;s++)v[s]=(mask>>s)&1;const n=v.slice(0,16).reduce((a,b)=>a+b,0);
 v[16]=n/16;v[17]=[0,4,8,12].reduce((a,s)=>a+((mask>>s)&1),0)/4;
 v[18]=[1,3,5,7,9,11,13,15].reduce((a,s)=>a+((mask>>s)&1),0)/8;
 for(let q=0;q<4;q++)v[19+q]=[0,1,2,3].reduce((a,i)=>a+((mask>>(q*4+i))&1),0)/4;
 const r=meta.r?.[0]??-1;if(r>=0)v[23+(r%12)]=1;
 if(row){const sum=row.reduce((a,b)=>a+b,0);if(sum>1e-9)for(let i=0;i<12;i++)v[35+i]=row[i]/sum;}
 v[47]=n>0?1:0;return v;}
const byId=new Map();
for(const kit of manifest.kits) for(const p of kit.phrases)
  byId.set(p.id,{id:p.id,embedding:p.embedding||[],section:p.section,energy:p.energy,kit});
const curId=Object.keys(bank.topk)[0];
const cur=byId.get(curId), real=byId.get(bank.topk[curId][0].id);
assert.ok(cur&&real&&cur.embedding.length===64&&real.embedding.length===64);
const stems=['drums','bass','other','vocals'];
const patterns=stems.map(s=>pat((cells[curId]||{})[s]||{},((chromaDoc[curId]||{})[s]||[])[0]));
const state=phraseConductorState({embedding:cur.embedding,patterns,section:cur.section,
 energy:cur.energy,bpm:cur.kit.bpm,keyRoot:rootOf(cur.kit.key)});
assert.equal(state.length,264);
const predicted=inferPhraseConductor(model,state);
assert.equal(predicted.length,64);
const good=phraseConductorScore(predicted,real.embedding);
const bad=phraseConductorScore(predicted,new Array(64).fill(0));
assert.ok(good>=0&&good<=1&&bad>=0&&bad<=1);
assert.ok(good>bad,'real successor phrase scores higher');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), str(KITS / "phrase_conductor.json"),
         str(KITS / "bank_transition.json"), str(KITS / "kit_manifest.json"),
         str(KITS / "cell_index.json"), str(KITS / "cell_chroma.json")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
