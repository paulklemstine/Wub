"""Learned donor prior for synthetic two-song mixes (Stages 1+2)."""
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


def test_mix_preference_artifact_is_sound():
    assert json.loads((KITS / "sidecars.json").read_text())["mix_preference"] is True
    doc = json.loads((KITS / "mix_preference.json").read_text())
    assert doc["format"] == "wub-mix-preference-v1"
    assert doc["version"] == 1
    assert doc["features"] == ["sim", "tempoCost", "keyCost", "sectionMatch",
                               "energyMatch", "harmonyFit"]
    # weak-label head must have learned the no-trainwreck direction
    w = doc["weights"]
    assert w["sim"] > 0, "similar donors preferred"
    assert w["sectionMatch"] > 0 and w["energyMatch"] > 0 and w["harmonyFit"] > 0
    assert w["tempoCost"] < 0, "near tempo preferred"
    assert all(abs(v) < 100 for v in w.values())
    assert abs(doc["bias"]) < 100
    assert doc["report"]["val_auc"] >= 0.85
    assert doc["report"]["pos_synth"] > 1000 and doc["report"]["neg"] > 1000
    src = (ROOT / "scripts/train_mix_preference.py").read_text()
    assert "gatePass" not in src.split("FEATURES")[1].split("]")[0], \
        "head must learn sub-components, not re-discover the gate"


def test_similarity_and_mix_prior_are_bounded_tiebreakers():
    out = bundle("mix_prior", "stem_remix.ts")
    model = json.loads((KITS / "mix_preference.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
// Neutral without signals: no embeddings, no sidecar.
assert.equal(p.phraseSimilarityBonus('a','b'),0,'no embeddings -> neutral');
const entry=(id,root,bpm,key,section,energy)=>({phrase:{id,bass_root:root},kit:{bpm,key},section,energy});
const cur=entry('cur',0,128,'C major','GROOVE_A',0.7);
assert.equal(p.mixPreferenceBonus('cur',entry('x',0,128,'C major','GROOVE_A',0.7),cur),0,'no sidecar -> neutral');
// With signals: good synthetic donor beats gated-out donor, bounded.
p.mixPreference=model;
p.barComposer.phraseEmbeddings={
  cur:[1,0,0,0], good:[0.9,0.1,0,0], bad:[-0.9,0.1,0,0],
};
const good=entry('good',0,129,'C major','GROOVE_A',0.72);
const bad=entry('bad',6,160,'F# major','INTRO',0.1);
const g=p.mixPreferenceBonus('cur',good,cur);
const b=p.mixPreferenceBonus('cur',bad,cur);
assert.ok(g>b,'good synthetic '+g+' beats gated-out '+b);
assert.ok(g<=0.15 && g>=-0.15 && b<=0.15 && b>=-0.15,'bounded +-0.15');
// Stage 1 similarity term: neighbor wins ties, bounded +-0.15.
const s1=p.phraseSimilarityBonus('cur','good');
const s2=p.phraseSimilarityBonus('cur','bad');
assert.ok(s1>s2,'neighbor beats distant');
assert.ok(Math.abs(s1)<=0.15 && Math.abs(s2)<=0.15,'sim bounded');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(model)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
