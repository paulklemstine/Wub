"""Learned 4/8/16-bar sung sections and their backing cohorts."""
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


def test_vocal_section_artifact_covers_4_8_16_and_backing():
    doc = json.loads((KITS / "vocal_section_model.json").read_text())
    assert doc["format"] == "wub-vocal-section-v1"
    assert doc["tracks"] >= 400
    for cat, row in doc["lengths"].items():
        assert set(row) == {"4", "8", "16"}, cat
        assert abs(sum(row.values()) - 1.0) < 0.02, cat
        assert row["4"] > 0 and row["8"] > 0 and row["16"] > 0, cat
    assert doc["lengths"]["CHORUS"]["16"] > doc["lengths"]["CHORUS"]["8"]
    assert doc["lengths"]["BREAKDOWN"]["8"] > doc["lengths"]["BREAKDOWN"]["16"]
    for cat in ("CHORUS", "VERSE"):
        b = doc["backing_change"][cat]
        assert b["bass"] > b["drums"] and b["other"] > b["drums"]
    assert json.loads((KITS / "sidecars.json").read_text())["vocal_section_model"] is True
    assert "vocal_section_model" in (ROOT / "scripts/build_sidecars.py").read_text()


def test_vocal_length_bonus_follows_the_section():
    out = bundle("vocal_section_bar", "bar_composer.ts")
    doc = json.loads((KITS / "vocal_section_model.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer,vocalSectionCategory,vocalLengthBin}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const c=new BarComposer();
const ctx=(role,section)=>({role,kitSection:section,isBuildup:false});
assert.equal(c.vocalSectionBonus({bars:16},ctx('CHORUS_1','DROP')),0,'neutral without sidecar');
c.vocalSectionModel=model;
const unit=(bars)=>({bars});
const chorus4=c.vocalSectionBonus(unit(4),ctx('CHORUS_1','DROP'));
const chorus8=c.vocalSectionBonus(unit(8),ctx('CHORUS_1','DROP'));
const chorus16=c.vocalSectionBonus(unit(16),ctx('CHORUS_1','DROP'));
assert.ok(chorus4>chorus16 && chorus16>chorus8,'chorus prefers short then long units');
const bd8=c.vocalSectionBonus(unit(8),ctx('BREAKDOWN','BREAKDOWN'));
const bd16=c.vocalSectionBonus(unit(16),ctx('BREAKDOWN','BREAKDOWN'));
assert.ok(bd8>bd16,'breakdown prefers shorter sung sections');
assert.equal(vocalSectionCategory('CHORUS_2','DROP'),'CHORUS');
assert.equal(vocalLengthBin(12),8);
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(doc)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_backing_cohort_follows_the_vocal_section():
    out = bundle("vocal_section_backing", "stem_remix.ts")
    doc = json.loads((KITS / "vocal_section_model.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const model=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
assert.equal(p.vocalBackingBonus('DROP',['vocals','bass']),0,'neutral without sidecar');
p.vocalSectionModel=model;p.targetKind='CHORUS_1';
assert.equal(p.vocalBackingBonus('DROP',['bass','other']),0,'no vocal change, no bonus');
const pitched=p.vocalBackingBonus('DROP',['vocals','bass','other']);
const drums=p.vocalBackingBonus('DROP',['vocals','drums']);
assert.ok(pitched>0,'bass+other change with a new chorus vocal');
assert.ok(drums<0,'drums hold under a chorus vocal like the corpus');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(doc)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
