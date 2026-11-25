"""The vocal AI may rank, but the known-good vocal scheduling must not change."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle(tmp_path, name, source):
    out = tmp_path / f"{name}.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_vocal_ai_artifacts_are_mined_and_shipped():
    base = ROOT / "frontend/public/data/kits"
    bar = json.loads((base / "vocal_bar_model.json").read_text())
    assert bar["format"] == "wub-vocal-bar-v1"
    assert bar["input_dim"] == 120 and bar["next_dim"] == 29
    assert bar["report"]["run_mae"] < bar["report"]["run_baseline_mae"]

    units = json.loads((base / "vocal_phrases.json").read_text())
    assert units["format"] == "wub-vocal-phrases-v1"
    assert len(units["feature_names"]) == len(units["weights"]) == len(units["mean"])
    assert units["report"]["tracks"] >= 300 and units["report"]["units"] >= 1000
    assert units["report"]["val_auc"] > 0.6
    manifest = json.loads((base / "kit_manifest.json").read_text())
    known = {p["id"] for kit in manifest["kits"] for p in kit["phrases"]}
    flat = [unit for track in units["tracks"].values() for unit in track]
    assert any(unit["bars"] >= 16 for unit in flat)
    assert any(unit["repeat"] == 1 for unit in flat)
    for unit in flat:
        assert unit["bars"] == len(unit["ids"]) * 4
        assert all(phrase_id in known for phrase_id in unit["ids"])
    sidecars = json.loads((base / "sidecars.json").read_text())
    assert sidecars["vocal_bar_model"] is True
    assert sidecars["vocal_phrases"] is True


def test_next_bar_model_orders_the_same_active_bars(tmp_path):
    out = bundle(tmp_path, "vocal_order", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
// A net whose output is a constant vector: hidden is tanh(0)=0, output=b2.
const net=(root)=>({mean:new Array(120).fill(0),std:new Array(120).fill(1),
 w1:[new Array(120).fill(0)],b1:[0],w2:Array.from({length:29},()=>[0]),b2:(()=>{const b=new Array(29).fill(0);
 b[root]=1;b[28]=1;return b;})()});
const meta={p:[.8,.8,.8,.8],r:[2,5,2,5],q:['mono','mono','mono','mono'],
 o:[0b1,0b1,0b1,0b1],es:[.5,.5,.5,.5]};
const c=new BarComposer();
assert.deepEqual(c.orderVocalActiveBars(meta,[0,1]),[0,1],'no model keeps old order');
c.vocalBarModel={format:'wub-vocal-bar-v1',version:1,input_dim:120,next_dim:29,
 next:net(5),length:net(0)};
const ordered=c.orderVocalActiveBars(meta,[0,1]);
assert.deepEqual(ordered,[1,0],'model reorders toward its predicted root');
assert.deepEqual([...ordered].sort(),[0,1],'same active-bar set, never new bars');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_known_good_vocal_repair_is_preserved_without_the_model(tmp_path):
    out = bundle(tmp_path, "vocal_repair", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const meta=(root,mask)=>({p:[.8,0,.8,0],r:[root,0,root,0],q:['mono','mono','mono','mono'],
 o:[mask,mask,mask,mask],es:[.8,0,.8,0]});
const entry=(id,track,stem)=>({kit:{track_id:track,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:0,duration_sec:7.5},stem,keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const base={drums:entry('d','dtrk','drums'),bass:entry('b','btrk','bass'),
 other:entry('o','otrk','other'),vocals:entry('v1','trk','vocals')};
const pools={drums:[base.drums],bass:[base.bass],other:[base.other],vocals:[base.vocals]};
const ctx={mode:'song',role:'VERSE_A',kitSection:'GROOVE_A',targetEnergy:.7,progressionIndex:0,
 progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),isBuildup:false,isPreDrop:false,
 isDropDownbeat:false,isBreakdownStart:false,recalledVocalPhraseId:null,recalledDonorPhraseIds:{},
 sectionStartBar:0,sectionEndBar:32};
const c=new BarComposer(); c.loaded=true; c.index={phrases:{}};
c.index.phrases.v1={vocals:meta(5,0b1)};
for(const k of ['drums','bass','other']) c.index.phrases[base[k].phrase.id]={[k]:meta(0,0b1)};
const args={baseEntries:base,pools,sectionCtx:ctx,blockBars:4,blockDurationSec:7.5,blockIndex:0,
 sessionRoot:0,sessionMode:'minor',isBridgeBlock:false,pinnedStems:new Set(),commit:false};
const plan=c.composeBlock(args).stems.vocals;
const bars=plan.slices.filter(s=>(s.targetBars||0)===1).map(s=>Math.round(s.startFrac*4));
assert.deepEqual(bars,[0,2,2,2],'old activeSrcBars[b % length] cadence');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_remeasures_repair_order_but_keeps_the_same_bars(tmp_path):
    out = bundle(tmp_path, "vocal_repair_ai", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const net=(root)=>({mean:new Array(120).fill(0),std:new Array(120).fill(1),
 w1:[new Array(120).fill(0)],b1:[0],w2:Array.from({length:29},()=>[0]),b2:(()=>{const b=new Array(29).fill(0);
 b[root]=1;b[28]=1;return b;})()});
const meta=(root,mask)=>({p:[.8,0,.8,0],r:[2,0,5,0],q:['mono','mono','mono','mono'],
 o:[mask,mask,mask,mask],es:[.5,0,.5,0]});
const entry=(id,track,stem)=>({kit:{track_id:track,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:0,duration_sec:7.5},stem,keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const base={drums:entry('d','dtrk','drums'),bass:entry('b','btrk','bass'),
 other:entry('o','otrk','other'),vocals:entry('v1','trk','vocals')};
const pools={drums:[base.drums],bass:[base.bass],other:[base.other],vocals:[base.vocals]};
const ctx={mode:'song',role:'VERSE_A',kitSection:'GROOVE_A',targetEnergy:.7,progressionIndex:0,
 progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),isBuildup:false,isPreDrop:false,
 isDropDownbeat:false,isBreakdownStart:false,recalledVocalPhraseId:null,recalledDonorPhraseIds:{},
 sectionStartBar:0,sectionEndBar:32};
const compose=()=>{
 const c=new BarComposer(); c.loaded=true; c.index={phrases:{}};
 c.index.phrases.v1={vocals:meta(5,0b1)};
 for(const k of ['drums','bass','other']) c.index.phrases[base[k].phrase.id]={[k]:meta(0,0b1)};
 return c.composeBlock({baseEntries:base,pools,sectionCtx:ctx,blockBars:4,blockDurationSec:7.5,
  blockIndex:0,sessionRoot:0,sessionMode:'minor',isBridgeBlock:false,pinnedStems:new Set(),
  commit:false}).stems.vocals;
};
const plain=compose();
const c=new BarComposer(); // model on, same setup through a second class instance
c.loaded=true; c.index={phrases:{}}; c.index.phrases.v1={vocals:meta(5,0b1)};
for(const k of ['drums','bass','other']) c.index.phrases[base[k].phrase.id]={[k]:meta(0,0b1)};
c.vocalBarModel={format:'wub-vocal-bar-v1',version:1,input_dim:120,next_dim:29,
 next:net(5),length:net(0)};
const learned=c.composeBlock({baseEntries:base,pools,sectionCtx:ctx,blockBars:4,blockDurationSec:7.5,
 blockIndex:0,sessionRoot:0,sessionMode:'minor',isBridgeBlock:false,pinnedStems:new Set(),
 commit:false}).stems.vocals;
const bars=(p)=>p.slices.filter(s=>(s.targetBars||0)===1).map(s=>Math.round(s.startFrac*4));
assert.deepEqual(bars(plain),[0,2,2,2]);
assert.deepEqual(bars(learned),[0,0,2,0],'root-5 prediction leads with bar 2');
assert(bars(learned).every((bar)=>[0,2].includes(bar)),'only the phrase active bars are used');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mined_units_bias_vocal_phrase_selection(tmp_path):
    out = bundle(tmp_path, "vocal_unit_select", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const meta=(root)=>({p:[.8,.8,.8,.8],r:[root,(root+2)%12,(root+3)%12,(root+5)%12],
 q:['mono','mono','mono','mono'],o:[0b1,0b1,0b1,0b1],es:[.8,.8,.8,.8]});
const entry=(id,track)=>({kit:{track_id:track,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:0,duration_sec:7.5},stem:'vocals',keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const a=entry('va','trackA'), b=entry('vb','trackB');
const ctx={mode:'song',role:'VERSE_A',kitSection:'GROOVE_A',targetEnergy:.7,progressionIndex:0,
 progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),isBuildup:false,isPreDrop:false,
 isDropDownbeat:false,isBreakdownStart:false,recalledVocalPhraseId:null,recalledDonorPhraseIds:{},
 sectionStartBar:0,sectionEndBar:32};
const c=new BarComposer(); c.loaded=true; c.index={phrases:{va:{vocals:meta(0)},vb:{vocals:meta(0)}}};
c.vocalPhraseModel={format:'wub-vocal-phrases-v1',version:1,feature_names:[],mean:[],scale:[],
 weights:[],bias:0,tracks:{trackB:[{ids:['vb'],bars:4,start_bar:0,features:[],recurrence:1,
 repeat:1,score:.9}]}};
const chosen=c.chooseVocalPrimary(a,[a,b],ctx,false);
assert.equal(chosen.entry.phrase.id,'vb','mined unit phrase wins the tie');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
