"""History-aware phrase-sequence conductor training/inference contract."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITS = ROOT / "frontend/public/data/kits"


def bundle(name: str, source: str) -> Path:
    out = Path("/tmp/opencode") / f"{name}.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_sequence_conductor_artifact_and_inference():
    doc = json.loads((KITS / "sequence_conductor.json").read_text())
    assert doc["format"] == "wub-sequence-conductor-v1"
    assert (doc["state_dim"], doc["target_dim"], doc["history"]) == (1284, 74, 6)
    assert len(doc["mean"]) == len(doc["std"]) == 1284
    assert doc["report"]["sequences"] >= 500
    assert 0 <= doc["report"]["val_embedding_cosine"] <= 1
    out = bundle("seq_cond", "sequence_conductor.ts")
    script = r"""
const assert=require('node:assert/strict');
const fs=require('fs');
const m=require(process.argv[1]);
const doc=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
assert.ok(m.isSequenceConductorArtifact(doc));
const steps=Array.from({length:6},()=>new Array(214).fill(0.1));
const state=m.sequenceState(steps);
assert.equal(state.length,1284);
const pred=m.inferSequence(doc,state);
assert.equal(pred.embedding.length,64);
assert.equal(pred.change.length,4);
assert.ok(m.sequenceScore(pred.embedding,pred.embedding)>0.99);
"""
    path = Path("/tmp/opencode/seq_art.json")
    path.write_text(json.dumps(doc))
    result = subprocess.run(["node", "-e", script, str(out), str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
