"""Acoustic extraction, split hygiene, browser inference and final-plan policy."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pytest
from wub_core.slice_flow import (DIM, INPUT_DIM, acoustic_teacher, describe_audio,
                                mix_features, mix_teacher, transform, transition_features)
from scripts.train_slice_flow import partition

ROOT = Path(__file__).resolve().parents[1]


def test_real_audio_features_are_finite_and_silence_is_distinct():
    sr=16000;t=np.arange(sr)/sr
    bass=.3*np.sin(2*np.pi*110*t)
    low=describe_audio(bass,sr,'bass')
    high=describe_audio(.3*np.sin(2*np.pi*4000*t),sr,'other')
    assert low.shape==(DIM,)
    assert np.isfinite(low).all()
    assert low[7]>high[7]+.5
    assert high[4]>low[4]
    assert not describe_audio(np.zeros((sr,2)),sr,'vocals').any()
    assert not describe_audio(np.empty((0,2)),sr,'vocals').any()
    assert low[16]==0
    stereo=np.column_stack((bass,-bass))
    assert describe_audio(stereo,sr,'bass')[16]>.8


def test_teacher_allows_cross_song_matches_and_rejects_dropouts():
    sr=8000;t=np.arange(sr)/sr
    a=describe_audio(.4*np.sin(2*np.pi*80*t),sr,'bass')
    b=a.copy()
    assert acoustic_teacher(a,b,'bass')>.9
    assert acoustic_teacher(a,np.zeros(DIM),'bass')<.4
    assert acoustic_teacher(a,b,'bass')==acoustic_teacher(a,b,'bass')
    shifted=transform(a,5,.5)
    assert np.allclose(shifted[20:],np.roll(a[20:],5))
    assert shifted[0]<a[0]
    assert not transform(a,0,0).any()
    assert len(transition_features(a,b,'bass'))==INPUT_DIM
    assert len(mix_features(np.stack([a]*4)))==16
    assert 0<=mix_teacher(np.stack([a]*4))<=1


def network(input_dim,outputs=1):
    return dict(mean=[0]*input_dim,std=[1]*input_dim,w1=[[0]*input_dim for _ in range(32)],b1=[0]*32,w2=[[0]*32 for _ in range(outputs)],b2=[1]*outputs)


def test_browser_policy_describes_actual_edits_and_limits_final_changes(tmp_path):
    out=tmp_path/'flow.cjs'
    subprocess.run([str(ROOT/'frontend/node_modules/.bin/esbuild'),str(ROOT/'frontend/src/slice_flow.ts'),'--bundle','--platform=node','--format=cjs',f'--outfile={out}'],check=True,capture_output=True)
    artifact=dict(format='slice_flow_v1',feature_dim=DIM,phrase_ids=['a','b'],feature_file='features.f32',critic=network(INPUT_DIM),mix=network(16),planner=network(8,3),foundation_seconds=90)
    features=np.ones((2,4,4,DIM),np.float32)*.3
    script=r'''
const assert=require('node:assert/strict');
const {SliceFlow,describeFlowTask,flowTransitionFeatures}=require(process.argv[1]);
const artifact=JSON.parse(process.argv[2]);
const raw=Buffer.from(process.argv[3],'base64');
global.fetch=async url=>({ok:true,json:async()=>artifact,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)});
const stems=['drums','bass','other','vocals'];
const entry=(stem,id)=>({stem,phrase:{id},kit:{track_id:id},keyShift:2,rate:1});
const task=(stem,id)=>({stem,run:[entry(stem,id)]});
(async()=>{
 const flow=new SliceFlow();assert.equal(await flow.load('/kits'),true);
 const ctx={at:0,bars:4,bpm:120,section:'GROOVE_A',targetEnergy:.7,sectionChanged:false,rotation:false,bridge:false};
 const initial=stems.map(s=>task(s,'a'));
 for(const t of initial)flow.record(t,0,4,120);
 const wanted=stems.map(s=>task(s,'b'));
 const selected=flow.select(wanted,{...ctx,at:8},s=>task(s,'a'),new Set());
 assert.equal(selected.find(t=>t.stem==='drums').run[0].phrase.id,'a');
 assert.equal(selected.find(t=>t.stem==='bass').run[0].phrase.id,'a');
 assert.ok(selected.filter(t=>t.run[0].phrase.id==='b').length<=1);
 // At a cluster rotation preserve at least three existing sources.
 const rotated=flow.select(wanted,{...ctx,at:104,rotation:true},s=>task(s,'a'),new Set());
 assert.ok(rotated.filter(t=>t.run[0].phrase.id==='b').length<=1);
 const pinned=flow.select(wanted,{...ctx,at:8},s=>task(s,'a'),new Set(['drums']));
 assert.equal(pinned.find(t=>t.stem==='drums').run[0].phrase.id,'b');
 const edited={stem:'other',run:[entry('other','a'),entry('other','b')],slices:[{itemIndex:1,startFrac:.5,endFrac:.75,targetBars:1,gain:.5},{itemIndex:0,startFrac:.75,endFrac:1,targetBars:1}],pitchPoints:[{offsetSec:2,semitones:-3}]};
 const cells=describeFlowTask(edited,2,2);
 assert.equal(cells[0].phraseId,'b');assert.equal(cells[0].bar,2);assert.equal(cells[0].gain,.5);
 assert.equal(cells[1].phraseId,'a');assert.equal(cells[1].semitones,-3);
 assert.equal(cells.reduce((n,c)=>n+c.bars,0),2);
 flow.record(edited,50,2,120);
 assert.equal(flow.trace(49).records.length,4); // prefetched events are not playing
 assert.equal(flow.trace(50).records.at(-1).cells[0].phraseId,'b');
 const copy=flow.trace();copy.records[0].cells[0].phraseId='mutated';
 assert.equal(flow.trace().records[0].cells[0].phraseId,'a');
 for(let i=0;i<300;i++)flow.record(task('drums','a'),100+i*8,4,120);
 assert.equal(flow.trace().records.length,256);
 flow.reset();assert.equal(flow.trace().records.length,0);
 const invalid={...artifact,critic:{...artifact.critic,std:artifact.critic.std.map(()=>0)}};
 global.fetch=async()=>({ok:true,json:async()=>invalid});
 assert.equal(await new SliceFlow().load('/kits'),false);
 const fallback=new SliceFlow();assert.equal(fallback.select(wanted,ctx,()=>null,new Set()),wanted);
 const arr=flowTransitionFeatures(new Float32Array(32).fill(.2),new Float32Array(32).fill(.4),'bass',[0,1,1,1,.5,0,.7,.25]);
 console.log(JSON.stringify(Array.from(arr)));
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    import base64
    result=subprocess.run(['node','-e',script,str(out),json.dumps(artifact),base64.b64encode(features.tobytes()).decode()],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    js=json.loads(result.stdout.splitlines()[-1])
    py=transition_features(np.full(DIM,.2,np.float32),np.full(DIM,.4,np.float32),'bass',[0,1,1,1,.5,0,.7,.25])
    assert np.allclose(js,py,atol=1e-7)


def test_partition_is_track_level_and_deterministic():
    tracks=[f'track_{i}' for i in range(100)]
    assert [partition(t) for t in tracks]==[partition(t) for t in tracks]
    assert 3<sum(partition(t) for t in tracks)<25


def test_feedback_and_playback_traces_cannot_contaminate_holdout(tmp_path):
    from scripts.train_slice_flow import load_feedback,trace_examples
    train=next(f't{i}' for i in range(100) if not partition(f't{i}'))
    held=next(f't{i}' for i in range(100) if partition(f't{i}'))
    p=tmp_path/'feedback.jsonl'
    item={'a':[.1]*INPUT_DIM,'b':[.2]*INPUT_DIM,'winner':'a','source_tracks':[train]}
    p.write_text(json.dumps(item)+'\n'+json.dumps({**item,'winner':'tie'})+'\n')
    assert len(load_feedback(p))==2
    p.write_text(json.dumps({**item,'source_tracks':[held]}))
    with pytest.raises(ValueError,match='training partition'):load_feedback(p)
    p.write_text(json.dumps({**item,'a':[float('nan')]*INPUT_DIM}))
    with pytest.raises(ValueError,match='non-finite'):load_feedback(p)
    def cell(track):return dict(stem='bass',phraseId='a',trackId=track,bar=0,semitones=0,gain=1,endGain=1,startFrac=0)
    trace={'format':'wub-slice-trace-v1','records':[{'at':i,'cells':[cell(t)]} for i,t in enumerate([train,train,held,train])]}
    p=tmp_path/'trace.json';p.write_text(json.dumps(trace))
    rows=trace_examples([p],np.full((1,4,4,DIM),.3,np.float32),['a'])
    assert len(rows)==1  # holdout resets context, so it cannot bridge two training rows


def test_blind_review_exports_parseable_jsonl_and_has_no_teacher_labels(tmp_path):
    import re
    from scripts.review_slice_flow import review_html
    case={'id':'one','a':{'features':[.1]*INPUT_DIM,'source_tracks':['train']},'b':{'features':[.2]*INPUT_DIM,'source_tracks':['train2']}}
    page=review_html([case])
    assert 'teacher' not in page and 'preferable' not in page
    js=re.search(r'<script>(.*?)</script>',page,re.S).group(1)
    result=subprocess.run(['node','--check','-'],input=js,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '.join("\\n")' in js


def test_shipping_artifact_is_complete_finite_and_split_report_is_honest():
    base=ROOT/'frontend/public/data/kits'
    model=json.loads((base/'slice_flow.json').read_text())
    assert model['format']=='slice_flow_v1'
    data=np.fromfile(base/model['feature_file'],dtype='<f4')
    assert data.size==len(model['phrase_ids'])*16*DIM
    assert np.isfinite(data).all()
    report=model['report']
    assert report['cross_song_positive_examples']>0
    assert report['development_accuracy']>=report['development_majority_accuracy']
    assert 'weak supervision' in report['label_provenance']
    for t in report['validation_tracks']:assert partition(t)


def test_unobserved_columns_have_no_random_runtime_effect():
    from scripts.train_slice_flow import train_net
    rng=np.random.default_rng(505)
    x=rng.normal(size=(200,4)).astype(np.float32);x[:,3]=.7
    y=(x[:,0]>0).astype(np.float32);val=np.arange(200)%5==0
    exported,pred,loss=train_net(x,y,val,4,'cpu')
    assert np.isfinite(pred).all() and np.isfinite(loss)
    assert all(row[3]==0 for row in exported['w1'])


def test_motif_return_sidecar_is_complete_and_bounded():
    base=ROOT/'frontend/public/data/kits'
    model=json.loads((base/'motif_return.json').read_text())
    assert model['format']=='wub-motif-return-v1'
    assert model['input_dim']==3*DIM+4+3+1
    net=model['net']
    assert len(net['mean'])==model['input_dim']
    assert len(net['std'])==model['input_dim']
    assert all(std>0 for std in net['std'])
    assert len(net['w1'])==32 and all(len(row)==model['input_dim'] for row in net['w1'])
    assert len(net['w2'])==1 and len(net['w2'][0])==32
    assert model['report']['val_auc']>0.6
    assert model['report']['val_positive_rate']>0.2
    sidecars=json.loads((base/'sidecars.json').read_text())
    assert sidecars['motif_return'] is True


def test_browser_loads_motif_sidecar_as_a_soft_term():
    source=(ROOT/'frontend/src/slice_flow.ts').read_text(encoding='utf-8')
    assert 'motif_return.json' in source
    assert 'returnScore' in source
    assert 'motifGain' in source
    assert 'MOTIF_INPUT_DIM' in source


def test_browser_accepts_valid_motif_sidecar(tmp_path):
    out=tmp_path/'motif.cjs'
    subprocess.run([str(ROOT/'frontend/node_modules/.bin/esbuild'),str(ROOT/'frontend/src/slice_flow.ts'),'--bundle','--platform=node','--format=cjs',f'--outfile={out}'],check=True,capture_output=True)
    base=ROOT/'frontend/public/data/kits'
    motif=(base/'motif_return.json').read_text()
    artifact=dict(format='slice_flow_v1',feature_dim=DIM,phrase_ids=['a','b'],feature_file='features.f32',critic=network(INPUT_DIM),mix=network(16),planner=network(8,3),foundation_seconds=90)
    features=np.ones((2,4,4,DIM),np.float32)*.3
    import base64
    raw=base64.b64encode(features.astype('<f4').tobytes()).decode()
    script=r'''
const assert=require('node:assert/strict');
const {SliceFlow}=require(process.argv[1]);
const artifact=JSON.parse(process.argv[2]);
const motif=JSON.parse(process.argv[3]);
const raw=Buffer.from(process.argv[4],'base64');
global.fetch=async url=>{
 if(String(url).endsWith('motif_return.json'))return {ok:true,json:async()=>motif};
 return {ok:true,json:async()=>artifact,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};
};
(async()=>{
 const flow=new SliceFlow();
 assert.equal(await flow.load('/kits'),true);
 assert.ok(flow.motif && flow.motif.w1.length===32);
 console.log('ok');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result=subprocess.run(['node','-e',script,str(out),json.dumps(artifact),motif,raw],capture_output=True,text=True)
    assert result.returncode==0, result.stderr


def test_auto_review_decides_without_human_input(tmp_path):
    import importlib
    auto = importlib.import_module('scripts.auto_review')
    from scripts.train_slice_flow import load_feedback
    out=tmp_path/'auto.jsonl'
    subprocess.run(['python3','-m','scripts.auto_review','--cases','2','--out',str(out)],
                   cwd=ROOT,check=True,capture_output=True,text=True)
    rows=load_feedback(out)
    assert len(rows)==2
    for row in rows:
        assert row['origin']=='auto-reward-v1'
        assert row['winner'] in ('a','b')
        assert len(row['a'])==INPUT_DIM and len(row['b'])==INPUT_DIM


def test_seam_dip_detects_a_level_hole():
    from scripts.auto_review import seam_dip
    import numpy as np
    a=np.ones(22050,np.float32)*.5
    assert seam_dip(np.concatenate([a,a]),len(a))<1e-6
    assert seam_dip(np.concatenate([a,np.zeros(22050,np.float32)]),len(a))>.9


def test_browser_accepts_value_sidecar(tmp_path):
    out=tmp_path/'value.cjs'
    subprocess.run([str(ROOT/'frontend/node_modules/.bin/esbuild'),str(ROOT/'frontend/src/slice_flow.ts'),'--bundle','--platform=node','--format=cjs',f'--outfile={out}'],check=True,capture_output=True)
    base=ROOT/'frontend/public/data/kits'
    artifact=dict(format='slice_flow_v1',feature_dim=DIM,phrase_ids=['a','b'],feature_file='features.f32',critic=network(INPUT_DIM),mix=network(16),planner=network(8,3),foundation_seconds=90)
    features=np.ones((2,4,4,DIM),np.float32)*.3
    import base64
    raw=base64.b64encode(features.astype('<f4').tobytes()).decode()
    value=(base/'bar_tour_value.json').read_text()
    script=r'''
const assert=require('node:assert/strict');
const {SliceFlow}=require(process.argv[1]);
const artifact=JSON.parse(process.argv[2]);const value=JSON.parse(process.argv[3]);
const raw=Buffer.from(process.argv[4],'base64');
global.fetch=async url=>{
 if(String(url).endsWith('bar_tour_value.json'))return {ok:true,json:async()=>value};
 if(String(url).endsWith('motif_return.json'))return {ok:false};
 return {ok:true,json:async()=>artifact,arrayBuffer:async()=>raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength)};
};
(async()=>{const flow=new SliceFlow();assert.equal(await flow.load('/kits'),true);
 assert.ok(flow.tourValue && flow.tourValue.w1.length===32);console.log('ok')})().catch(e=>{console.error(e);process.exitCode=1});
'''
    result=subprocess.run(['node','-e',script,str(out),json.dumps(artifact),value,raw],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
