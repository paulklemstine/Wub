"""Bounded variety prior: less track reuse, fewer repeated vocal hooks."""
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


def test_variety_artifact_is_shipped_and_measured():
    doc = json.loads((KITS / "variety_prior.json").read_text())
    assert doc["format"] == "wub-variety-prior-v1"
    assert 0 < doc["track_reuse_penalty"] <= 0.5
    assert 0 < doc["vocal_freshness_weight"] <= 0.6
    assert doc["joint_reserve"] >= 2
    assert doc["source_novelty_weight"] > 0.15
    assert doc["max_recall_per_set"] == 1
    judge = doc["report"]["findings"]["per_stem_judge_top1_median"]
    assert judge["drums"] > 0.9 and judge["vocals"] > 0.9
    assert json.loads((KITS / "sidecars.json").read_text())["variety_prior"] is True
    assert "variety_prior" in (ROOT / "scripts/build_sidecars.py").read_text()


def test_theme_recall_is_budgeted_when_variety_is_on():
    out = bundle("variety_architect", "song_architect.ts")
    script = r"""
const assert=require('node:assert/strict');
const {SongArchitect}=require(process.argv[1]);
const call=(a,bar,bars)=>a.describeBlock(bar,bars,'house',0,'minor',.9,'DROP');
const plain=new SongArchitect();plain.resetForNewSet('house',0,'minor');
const c1=call(plain,32,8);plain.recordSectionTheme(c1,{vocals:'hookA'});
assert.equal(call(plain,40,8).recalledVocalPhraseId,'hookA','old path repeats hook in sub-blocks');
const v=new SongArchitect();v.resetForNewSet('house',0,'minor');
v.setVarietyConfig({max_recall_per_set:1});
const d1=call(v,32,8);v.recordSectionTheme(d1,{vocals:'hookA'});
assert.equal(call(v,40,8).recalledVocalPhraseId,null,'no sub-block replay with variety on');
assert.equal(call(v,80,16).recalledVocalPhraseId,'hookA','one chorus recall survives');
assert.equal(call(v,88,8).recalledVocalPhraseId,null,'recall budget is respected');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_track_and_vocal_freshness_helpers_prefer_unused_sources():
    out = bundle("variety_helpers", "stem_remix.ts")
    doc = json.loads((KITS / "variety_prior.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const cfg=JSON.parse(process.argv[2]);
const p=new StemRemixPlayer({now:()=>0});
const entry=(id,tid)=>({phrase:{id},kit:{track_id:tid},stem:'bass'});
p.varietyPrior=cfg;
p.entryById.set('a:bass',entry('a','hot'));p.entryById.set('b:bass',entry('b','hot'));
p.entryById.set('c:bass',entry('c','hot'));
p.stemHistory.bass=['a','b','c'];
assert.ok(p.varietyBonusFor('bass','hot')<p.varietyBonusFor('bass','cold'),
 'recently used source is penalized');
p.playCount.set('a:vocals',6);p.stemHistory.vocals=['a','a','a'];
const stale=p.vocalFreshnessOf('a');
const fresh=p.vocalFreshnessOf('never-heard');
assert.ok(fresh>stale,'played vocal phrase is less fresh: '+stale+' vs '+fresh);
p.varietyPrior=null;
assert.equal(p.varietyBonusFor('bass','hot'),0,'neutral without sidecar');
"""
    result = subprocess.run(
        ["node", "-e", script, str(out), json.dumps(doc)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_vocal_chooser_uses_the_freshness_callback():
    out = bundle("variety_vocal", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const meta=(root)=>({p:[.8,.8,.8,.8],r:[root,(root+2)%12,(root+3)%12,(root+5)%12],
 q:['mono','mono','mono','mono'],o:[0b1,0b1,0b1,0b1],es:[.8,.8,.8,.8]});
const entry=(id,track)=>({kit:{track_id:track,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:0,duration_sec:7.5},stem:'vocals',keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const a=entry('va','trackA'),b=entry('vb','trackB');
const ctx={mode:'song',role:'VERSE_A',kitSection:'GROOVE_A',targetEnergy:.7,progressionIndex:0,
 progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),isBuildup:false,isPreDrop:false,
 isDropDownbeat:false,isBreakdownStart:false,recalledVocalPhraseId:null,recalledDonorPhraseIds:{},
 sectionStartBar:0,sectionEndBar:32,
 vocalFreshness:(id)=>id==='va'?.2:1,vocalFreshnessWeight:.3};
const c=new BarComposer();c.loaded=true;c.index={phrases:{va:{vocals:meta(0)},vb:{vocals:meta(0)}}};
assert.equal(c.chooseVocalPrimary(a,[a,b],ctx,false).entry.phrase.id,'vb',
 'fresh phrase wins over an identical overplayed one');
ctx.vocalFreshnessWeight=0;
assert.equal(c.chooseVocalPrimary(a,[a,b],ctx,false).entry.phrase.id,'va',
 'tie falls back to the known-good bias');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
