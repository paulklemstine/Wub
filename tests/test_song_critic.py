"""Learned four-stem song-ness critic as a bounded final-gate adjustment."""
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


def test_song_critic_artifact_is_trained_and_bounded():
    doc = json.loads((KITS / "song_critic.json").read_text())
    assert doc["format"] == "wub-song-critic-v1"
    assert doc["input_dim"] == 218 and doc["hidden"] == 64
    assert len(doc["mean"]) == len(doc["std"]) == 218
    assert len(doc["layers"]) == 3 and len(doc["layers"][2]["w"]) == 1
    r = doc["report"]
    assert r.get("songs", r.get("tracks", 0)) >= 150
    assert r["val_tracks"] >= 10
    assert r["val_auc"] > 0.55
    assert r["real_score_p10"] > 0.1
    assert "teacher_val_auc" in r or "songs" in r
    assert json.loads((KITS / "sidecars.json").read_text())["song_critic"] is True
    assert "wub-song-critic-v1" in (ROOT / "scripts/train_song_critic.py").read_text()


def test_critic_inference_and_final_gate_adjustment():
    module = bundle("song_critic", "song_critic.ts")
    doc = json.loads((KITS / "song_critic.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {inferSongCritic,isSongCriticArtifact}=require(process.argv[1]);
const model=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
assert.ok(isSongCriticArtifact(model));
const ones=new Array(218).fill(0);
const p=inferSongCritic(model,ones);
assert.ok(p>=0&&p<=1,'critic output is bounded: '+p);
"""
    path = Path("/tmp/opencode/song_critic_art.json")
    path.write_text(json.dumps(doc))
    result = subprocess.run(["node", "-e", script, str(module), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_final_gate_accepts_critic_adjustment_but_stays_bounded():
    flow = bundle("critic_flow", "slice_flow.ts")
    # Use the same stub artifact pattern as test_slice_flow.py.
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
 const ctx={at:0,bars:4,bpm:120,section:'GROOVE_A',targetEnergy:.7,sectionChanged:false,
   rotation:false,bridge:false};
 for(const s of stems)flow.record(task(s,'a'),0,4,120);
 const wanted=stems.map(s=>task(s,'b'));
 const neutral=flow.select(wanted,{...ctx,at:8,songCriticAdjustment:0},s=>task(s,'a'),new Set());
 const boosted=flow.select(wanted,{...ctx,at:8,songCriticAdjustment:.1},s=>task(s,'a'),new Set());
 const changed=(arr)=>arr.filter(t=>t.run[0].phrase.id==='b').length;
 assert.ok(changed(boosted)>=changed(neutral),'a positive critic adjustment never rejects a change');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(["node", "-e", script, str(flow), json.dumps(artifact)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
