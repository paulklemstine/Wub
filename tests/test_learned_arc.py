"""Corpus-mined song arc and rare full-source handoff gate."""
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


def test_arc_artifact_is_mined_and_shipped():
    doc = json.loads((KITS / "song_arc_model.json").read_text())
    assert doc["format"] == "wub-song-arc-v1"
    assert doc["tracks"] >= 1800
    for sec, row in doc["transition"].items():
        assert 0.95 <= sum(row.values()) <= 1.05, sec
    assert {"INTRO", "GROOVE_A", "BUILDUP", "DROP", "BREAKDOWN"} <= set(doc["lengths"])
    for sec in ("INTRO", "BUILDUP", "BREAKDOWN"):
        assert sec in doc["section_slope"]
    assert doc["section_slope"]["BUILDUP"]["mean_db_per_bar"] > 0
    assert doc["section_slope"]["BREAKDOWN"]["mean_db_per_bar"] < 0
    assert "song_arc_model" in json.loads((KITS / "sidecars.json").read_text())

    cohort = json.loads((KITS / "transition_cohort.json").read_text())
    h = cohort["handoff"]
    assert h["enabled"] is True
    assert 0.05 <= h["probability"] <= 0.3
    assert h["max_per_source"] == 4 and h["max_stems"] == 4
    cohort_src = (ROOT / "scripts/build_transition_cohort.py").read_text()
    assert '"handoff"' in cohort_src


def test_sampled_arc_is_contiguous_deterministic_and_builds_and_fades():
    out = bundle("learned_arc", "learned_arc.ts")
    artifact = json.loads((KITS / "song_arc_model.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {sampleSongArc,songArcTargetAt}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const a=sampleSongArc(model,12345);
const b=sampleSongArc(model,12345);
assert.deepEqual(a,b,'same seed gives the same arc');
assert.notDeepEqual(a,sampleSongArc(model,54321),'different seed gives a different arc');
assert.equal(a[0].section,'INTRO');
assert.equal(a[a.length-1].section,'BRIDGE');
assert.equal(a[a.length-1].bars,8);
assert.ok(a.some(s=>s.section==='DROP'),'every set has a drop');
let bar=0;
for(const s of a){assert.equal(s.startBar,bar);bar+=s.bars;}
assert.equal(bar,104,'segments cover the full set');
const t=songArcTargetAt(a,100);
assert.equal(t.section,'BRIDGE');
assert.equal(songArcTargetAt(a,0).section,'INTRO');
const build=sampleSongArc(model,777).find(s=>s.section==='BUILDUP');
if(build) assert.ok(build.slopeDbPerBar>0,'buildups rise like the corpus');
const fade=sampleSongArc(model,777).find(s=>s.section==='BREAKDOWN');
if(fade) assert.ok(fade.slopeDbPerBar<0,'breakdowns fade like the corpus');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(artifact)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_handoff_gate_is_rare_and_can_be_disabled():
    out = bundle("handoff_gate", "stem_remix.ts")
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const p=new StemRemixPlayer({now:()=>0});
p.sampledArcKey='test-set';
p.transitionCohort={handoff:{enabled:true,probability:0.10,max_per_source:4,max_stems:4}};
let hits=0;
for(let bar=0;bar<2000;bar++) if(p.handoffAllowed(bar)) hits++;
assert.ok(hits>120 && hits<280,'roughly 10% of section changes are takeovers: '+hits);
p.transitionCohort.handoff.enabled=false;
assert.equal(p.handoffAllowed(1),false,'disabled handoff is neutral');
assert.equal(p.handoffAllowed(2),p.handoffAllowed(2),'deterministic');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
