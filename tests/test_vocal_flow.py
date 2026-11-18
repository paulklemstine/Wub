"""Vocal development priors should stop a one-bar phrase looping verbatim."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vocal_flow_sidecar_mined_from_real_songs():
    base = ROOT / "frontend/public/data/kits"
    prior = json.loads((base / "vocal_flow.json").read_text())
    assert prior["format"] == "wub-vocal-flow-v1"
    assert float(prior["run_lengths"]["3"]) > 0
    # A four-active-bar phrase normally uses about two distinct melodic bars.
    target = prior["distinct_bars_per_active"]["4"]
    assert 1.5 <= float(target) <= 3.0
    assert 0.1 <= float(prior["variation_rate"]) <= 0.9
    assert json.loads((base / "sidecars.json").read_text())["vocal_flow"] is True


def test_bar_composer_develops_vocal_bars_instead_of_copying_one(tmp_path):
    out = tmp_path / "composer.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const c=new BarComposer();
c.vocalFlow={format:'wub-vocal-flow-v1',run_lengths:{},distinct_bars_per_active:{
  '1':1,'2':2,'3':2,'4':2},variation_rate:.35};
const single=[{itemIndex:0,srcBar:0,meta:'y',semitones:0},{itemIndex:1,srcBar:2,meta:'x',semitones:3}];
const schedule=c.developVocalBars(single,8,0);
assert.equal(schedule.length,8);
assert.equal(new Set(schedule.map(s=>s.srcBar)).size,2);
for(let i=1;i<schedule.length;i++)assert.notEqual(schedule[i].srcBar,schedule[i-1].srcBar);
assert.equal(schedule[0].semitones,0);
const four=[0,1,2,3].map(bar=>({itemIndex:0,srcBar:bar}));
const eight=c.developVocalBars(four,8,0);
assert.equal(new Set(eight.map(s=>s.srcBar)).size,4);
const rotated=c.developVocalBars(four,8,1);
assert.notDeepEqual(eight.map(s=>s.srcBar),rotated.map(s=>s.srcBar));
c.vocalFlow=null;
const fallback=c.developVocalBars([{itemIndex:0,srcBar:1},{itemIndex:0,srcBar:3}],4,0);
assert.equal(new Set(fallback.map(s=>s.srcBar)).size,2);
"""
    subprocess.run(["node", "-e", script, str(out)], check=True, capture_output=True, text=True)


def test_vocal_bar_model_is_trained_and_shipped():
    base = ROOT / "frontend/public/data/kits"
    model = json.loads((base / "vocal_bar_model.json").read_text())
    assert model["format"] == "wub-vocal-bar-v1"
    assert model["input_dim"] == 120 and model["next_dim"] == 29
    assert model["report"]["run_mae"] < model["report"]["run_baseline_mae"]
    assert json.loads((base / "sidecars.json").read_text())["vocal_bar_model"] is True


def test_bar_composer_uses_next_vocal_bar_prediction(tmp_path):
    out = tmp_path / "vocal_model.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    model_text = (ROOT / "frontend/public/data/kits/vocal_bar_model.json").read_text()
    model_path = tmp_path / "vocal_bar_model.json"
    model_path.write_text(model_text)
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const model=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
const meta=(root,mask)=>({o:[mask,0,0,0],p:[.8,.7,.6,.5],r:[root,(root+2)%12,(root+7)%12,(root+4)%12]});
const pool=[0,1,2,3].map(bar=>({itemIndex:0,srcBar:bar,meta:meta(bar,0b1010101010101010)}));
const c=new BarComposer();
assert.equal(c.vocalBarModel,null);
const plain=c.developVocalBars(pool,8,0).map(s=>s.srcBar);
c.vocalBarModel=model;
const learned=c.developVocalBars(pool,8,0).map(s=>s.srcBar);
assert.equal(learned.length,8);
assert.equal(new Set(learned).size,4);
assert.ok(learned.some((v,i)=>v!==plain[i]));
"""
    result = subprocess.run(["node", "-e", script, str(out), str(model_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_next_bar_prediction_selects_the_slices_that_get_scheduled(tmp_path):
    out = tmp_path / "slices.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    model_path = tmp_path / "vocal_bar_model.json"
    model_path.write_text((ROOT / "frontend/public/data/kits/vocal_bar_model.json").read_text())
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const model=JSON.parse(require('fs').readFileSync(process.argv[2],'utf8'));
const meta=(root,mask)=>({s:'AAAB',p:[.8,.7,.6,.5],r:[root,(root+2)%12,(root+7)%12,(root+4)%12],
 q:['mono','mono','mono','mono'],o:[mask,mask>>1,mask>>2,mask>>3],es:[.7,.6,.5,.4],er:['A','A','A','T']});
const entry=(stem,id)=>({kit:{track_id:id,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:0,duration_sec:7.5},stem,keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const base={drums:entry('drums','d'),bass:entry('bass','b'),other:entry('other','o'),vocals:entry('vocals','v0')};
const pools={drums:[base.drums],bass:[base.bass],other:[base.other],
 vocals:[base.vocals,entry('vocals','v1')]};
const sectionCtx={mode:'song',role:'VERSE_A',kitSection:'GROOVE_A',targetEnergy:.7,
 progressionIndex:0,progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),
 isBuildup:false,isPreDrop:false,isDropDownbeat:false,isBreakdownStart:false,
 recalledVocalPhraseId:null,recalledDonorPhraseIds:{},sectionStartBar:0,sectionEndBar:32};
const args={baseEntries:base,pools,sectionCtx,blockBars:4,blockDurationSec:7.5,blockIndex:0,
 sessionRoot:0,sessionMode:'minor',isBridgeBlock:false,pinnedStems:new Set(),commit:false};
const c=new BarComposer();
c.index={phrases:{v0:{vocals:meta(0,0b1010101010101010)},v1:{vocals:meta(5,0b0101010101010101)}}};
c.loaded=true;
const plain=c.composeBlock(args).stems.vocals.slices.map(s=>s.startFrac+':'+s.targetBars).join(',');
c.vocalBarModel=model;
let calls=0;const orig=c.inferVocal.bind(c);c.inferVocal=(n,x)=>{calls++;return orig(n,x);};
const learned=c.composeBlock(args).stems.vocals.slices.map(s=>s.startFrac+':'+s.targetBars).join(',');
assert.ok(calls>0,'next-bar inference must run during composition');
assert.notEqual(learned,plain,'predictions must change the scheduled slices');
"""
    result = subprocess.run(["node", "-e", script, str(out), str(model_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_vocal_phrase_model_is_mined_from_real_songs():
    base = ROOT / "frontend/public/data/kits"
    model = json.loads((base / "vocal_phrases.json").read_text())
    assert model["format"] == "wub-vocal-phrases-v1"
    assert len(model["feature_names"]) == len(model["weights"]) == len(model["mean"])
    assert model["report"]["tracks"] >= 300
    assert model["report"]["units"] >= 1000
    assert model["report"]["val_auc"] > 0.6
    manifest = json.loads((base / "kit_manifest.json").read_text())
    known = {p["id"] for kit in manifest["kits"] for p in kit["phrases"]}
    units = [unit for track in model["tracks"].values() for unit in track]
    assert any(unit["bars"] >= 16 for unit in units)
    assert any(unit["repeat"] == 1 for unit in units)
    for unit in units:
        assert unit["bars"] == len(unit["ids"]) * 4
        assert all(phrase_id in known for phrase_id in unit["ids"])
    assert json.loads((base / "sidecars.json").read_text())["vocal_phrases"] is True


def test_composer_schedules_whole_mined_phrase_units(tmp_path):
    out = tmp_path / "vocal_units.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/bar_composer.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const meta=(root)=>({s:'ABAB',p:[.8,.8,.8,.8],r:[root,(root+2)%12,(root+3)%12,(root+5)%12],
 q:['mono','mono','mono','mono'],o:[0b1010101010101010,0b0101010101010101,0b1100110011001100,0b0011001100110011],
 es:[.8,.8,.8,.8],er:['A','A','A','A'],fid:.9,vr:'verse'});
const entry=(id,track,bar,stem)=>({kit:{track_id:track,bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'GROOVE_A',start_bar:bar,duration_sec:7.5},stem,keyShift:0,rate:1,
 energy:.7,section:'GROOVE_A',rmsDb:-18});
const ids=['v1','v2','v3','v4'];
const vocals=ids.map((id,i)=>entry(id,'trk',i*4,'vocals'));
const base={drums:entry('d','dtrk',0,'drums'),bass:entry('b','btrk',0,'bass'),
 other:entry('o','otrk',0,'other'),vocals:vocals[0]};
const pools={drums:[base.drums],bass:[base.bass],other:[base.other],vocals:vocals.slice()};
const c=new BarComposer(); c.index={phrases:{}}; c.loaded=true;
ids.forEach((id,i)=>{c.index.phrases[id]={vocals:meta(i)}});
for(const k of ['drums','bass','other']) c.index.phrases[base[k].phrase.id]={[k]:meta(0)};
const ctx={mode:'song',role:'CHORUS_1',kitSection:'DROP',targetEnergy:.9,progressionIndex:1,
 progression:[0,1,2,3].map(i=>({relRoot:i,quality:'minor'})),isBuildup:false,isPreDrop:false,
 isDropDownbeat:false,isBreakdownStart:false,recalledVocalPhraseId:null,recalledDonorPhraseIds:{},
 sectionStartBar:0,sectionEndBar:32};
const args={baseEntries:base,pools,sectionCtx:ctx,blockBars:16,blockDurationSec:30,blockIndex:0,
 sessionRoot:0,sessionMode:'minor',isBridgeBlock:false,pinnedStems:new Set(),commit:false};
c.vocalPhraseModel={format:'wub-vocal-phrases-v1',version:1,feature_names:[],mean:[],scale:[],
 weights:[],bias:0,tracks:{trk:[{ids,bars:16,start_bar:0,features:[],recurrence:1,repeat:1,score:.95}]}};
const plan=c.composeBlock(args).stems.vocals;
assert.deepEqual(plan.donors.map(d=>d.phrase.id),ids);
assert.equal(plan.slices.reduce((sum,s)=>sum+(s.targetBars||0),0),16);
// Each phrase's own four bars must play in order before the next phrase.
const spans=[];
for(const slice of plan.slices){
  const last=spans[spans.length-1];
  if(last && last.item===slice.itemIndex) last.end=slice.endFrac;
  else spans.push({item:slice.itemIndex,start:slice.startFrac,end:slice.endFrac});
}
assert.deepEqual(spans.map(s=>s.item),[0,1,2,3]);
for(let i=0;i<spans.length;i++){
  assert.equal(spans[i].start,0);
  assert.equal(spans[i].end,1);
}
assert.equal(plan.pitchPoints.length,0,'one key shift for the whole performance');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_vocal_unit_wrap_loops_the_whole_chorus_not_the_last_cell(tmp_path):
    out = tmp_path / "stem_remix.cjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src/stem_remix.ts"), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    script = r"""
const assert=require('node:assert/strict');
const {StemRemixPlayer}=require(process.argv[1]);
const player=new StemRemixPlayer({now:()=>0});
const entry=(id,bar)=>({kit:{track_id:'trk',bpm:120,key:'C minor',style:'house'},
 phrase:{id,section:'DROP',start_bar:bar,duration_sec:7.5},stem:'vocals',keyShift:0,rate:1,
 energy:.8,section:'DROP',rmsDb:-18});
player.entryById=new Map([['v1:vocals',entry('v1',0)],['v2:vocals',entry('v2',4)],['v3:vocals',entry('v3',8)]]);
player.barComposer.vocalPhraseModel={format:'wub-vocal-phrases-v1',version:1,feature_names:[],
 mean:[],scale:[],weights:[],bias:0,tracks:{trk:[{ids:['v1','v2','v3'],bars:12,start_bar:0,
 features:[],recurrence:1,repeat:1,score:.9}]}};
const last=player.entryById.get('v3:vocals');
const middle=player.entryById.get('v2:vocals');
player.sessionBar=40; player.lastVocalWrapBar=-1;
assert.equal(player.vocalUnitWrap(middle),null,'only the unit end wraps');
assert.equal(player.vocalUnitWrap(last).phrase.id,'v1','wrap to the unit start, not the last cell');
player.sessionBar=44;
assert.equal(player.vocalUnitWrap(last),null,'one extra pass per 16 bars');
player.sessionBar=60;
assert.equal(player.vocalUnitWrap(last).phrase.id,'v1','a later hold may loop again');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
