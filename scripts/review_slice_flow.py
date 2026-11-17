#!/usr/bin/env python3
"""Render blind A/B handoffs with the shipping PCM assembler + Signalsmith.

Starts isolated local Vite/Chrome processes; never controls the user's browser.
Creates a standalone listening page, WAV clips, JSONL preference export, and a
render report. No original assets are modified. Requires Chrome + websockets.
The exported playback trace can be replayed with --trace.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import json
import os
import signal
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
import soundfile as sf
import websockets

ROOT=Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]


class Browser:
    def __init__(self):self.processes=[];self.temp=None;self.seq=0
    async def __aenter__(self):
        port=free_port();debug=free_port();self.temp=tempfile.TemporaryDirectory(prefix='wub-flow-chrome-')
        self.url=f'http://127.0.0.1:{port}'
        config=Path(self.temp.name)/'vite.config.mjs'
        config.write_text('export default '+json.dumps({'root':str(ROOT/'frontend'),'base':'/','server':{'hmr':False},'optimizeDeps':{'exclude':['signalsmith-stretch']},'resolve':{'conditions':['onnxruntime-web-use-extern-wasm']}}))
        self.processes.append(subprocess.Popen(['npm','run','dev','--','--host','127.0.0.1','--port',str(port),'--strictPort','--config',str(config)],cwd=ROOT/'frontend',stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True))
        chrome=shutil.which('google-chrome') or shutil.which('chromium')
        if not chrome:raise RuntimeError('Chrome or Chromium is required for production-equivalent audio rendering')
        self.processes.append(subprocess.Popen([chrome,'--headless=new','--no-sandbox','--disable-dev-shm-usage','--disable-gpu','--no-first-run','--autoplay-policy=no-user-gesture-required','--mute-audio',f'--user-data-dir={self.temp.name}',f'--remote-debugging-port={debug}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True))
        for _ in range(100):
            try:
                urllib.request.urlopen(self.url+'/src/kit_pcm.ts',timeout=1).read(1)
                targets=json.load(urllib.request.urlopen(f'http://127.0.0.1:{debug}/json',timeout=1))
                target=next(t for t in targets if t['type']=='page');break
            except Exception:await asyncio.sleep(.2)
        else:raise RuntimeError('local rendering server did not start')
        self.ws=await websockets.connect(target['webSocketDebuggerUrl'],max_size=128*1024*1024)
        await self.command('Page.navigate',{'url':self.url+'/src/kit_pcm.ts'})
        await asyncio.sleep(.5)
        return self
    async def command(self,method,params):
        self.seq+=1;ident=self.seq
        await self.ws.send(json.dumps({'id':ident,'method':method,'params':params}))
        while True:
            item=json.loads(await self.ws.recv())
            if item.get('id')==ident:
                if 'error' in item:raise RuntimeError(item['error'])
                return item.get('result',{})
    async def evaluate(self,expression):
        result=await self.command('Runtime.evaluate',{'expression':expression,'awaitPromise':True,'returnByValue':True})
        if 'exceptionDetails' in result:raise RuntimeError(result['exceptionDetails'])
        return result['result'].get('value')
    async def __aexit__(self,*_):
        if hasattr(self,'ws'):await self.ws.close()
        for p in self.processes:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        if self.temp:self.temp.cleanup()


RENDER_JS=r'''(async()=>{
 const job=JOB;
 const {assembleSlicedStereoRun}=await import('/src/kit_pcm.ts');
 const {StretchVoice}=await import('/src/stretch_voice.ts');
 const sr=44100, barSec=240/job.bpm;
 const totalBars=Math.max(...job.parts.map(p=>p.atBar+p.bars));
 const ctx=new OfflineAudioContext(2,Math.ceil((totalBars*barSec+.5)*sr),sr);
 const master=ctx.createDynamicsCompressor();
 master.threshold.value=-12;master.ratio.value=4;master.attack.value=.003;master.release.value=.2;
 const limiter=ctx.createDynamicsCompressor();limiter.threshold.value=-3;limiter.knee.value=0;limiter.ratio.value=20;limiter.attack.value=.001;limiter.release.value=.08;
 const trim=ctx.createGain();trim.gain.value=.62;
 const shaper=ctx.createWaveShaper();const curve=new Float32Array(4096);
 for(let i=0;i<curve.length;i++){const x=i/(curve.length-1)*2-1,a=Math.abs(x);curve[i]=Math.sign(x)*(a<=.8?a:.8+.2*Math.tanh((a-.8)/.2));}
 shaper.curve=curve;shaper.oversample='4x';trim.connect(master);master.connect(limiter);limiter.connect(shaper);shaper.connect(ctx.destination);
 const voices=[];
 for(const part of job.parts){
  const phrases=[];
  for(const file of part.files){
   const response=await fetch('/data/kits/'+file);
   if(!response.ok)throw new Error('Missing asset '+file);
   const buffer=await ctx.decodeAudioData(await response.arrayBuffer());
   phrases.push([buffer.getChannelData(0).slice(),buffer.getChannelData(buffer.numberOfChannels>1?1:0).slice()]);
  }
  const run=assembleSlicedStereoRun(phrases,null,part.slices,sr,28);
  const voice=await StretchVoice.create(ctx,trim);
  voices.push(voice);
  await voice.loadSamples(run.samples,true);
  const rate=part.rate??Math.max(.94,Math.min(1.06,part.sourceDuration/(4*barSec)));
  voice.schedule({when:.15+part.atBar*barSec,rate,semitones:part.semitones,gain:1,fadeInSec:part.stem==='drums'?.006:.08});
  if(part.pitchPoints?.length)voice.schedulePitchAutomation(.15+part.atBar*barSec,rate,part.pitchPoints,part.stem==='vocals');
  voice.fadeOut(.15+(part.atBar+part.bars)*barSec,.08);
 }
 const output=await ctx.startRendering();
 const interleaved=new Float32Array(output.length*2);
 for(let c=0;c<2;c++){const data=output.getChannelData(c);for(let i=0;i<data.length;i++)interleaved[i*2+c]=data[i];}
 const bytes=new Uint8Array(interleaved.buffer);let encoded='';
 for(let i=0;i<bytes.length;i+=16384)encoded+=String.fromCharCode(...bytes.subarray(i,i+16384));
 for(const voice of voices)voice.dispose();
 return {pcm:btoa(encoded),sampleRate:sr,bars:totalBars};
})()'''


def make_reviews(base,count,seed):
    from wub_core.slice_flow import transition_features
    from scripts.train_slice_flow import partition
    model=json.loads((base/'slice_flow.json').read_text());dim=model['feature_dim']
    features=np.fromfile(base/model['feature_file'],dtype='<f4').reshape(-1,4,4,dim)
    rows={p:i for i,p in enumerate(model['phrase_ids'])}
    manifest=json.loads((base/'kit_manifest.json').read_text());rng=np.random.default_rng(seed)
    phrases=[(k,p) for k in manifest['kits'] for p in k['phrases'] if p['id'] in rows and not partition(k['track_id']) and all(s in p['stems'] for s in ('drums','bass','other','vocals'))]
    references=[(k,p) for k,p in phrases if k.get('style')=='wave' and p.get('energy',0)>.4]
    reviews=[]
    for case in range(count):
        kit,phrase=references[int(rng.integers(len(references)))];sr=rows[phrase['id']]
        # Same key and near tempo keeps this test focused on texture/phrasing.
        eligible=[(k,p) for k,p in phrases if k['track_id']!=kit['track_id'] and k['key']==kit['key'] and abs(p['duration_sec']/phrase['duration_sec']-1)<=.06]
        if len(eligible)<2:continue
        stem='other';si=2;a=features[sr,si,3]
        def score_candidate(p):
            x=transition_features(a,features[rows[p['id']],si,0],stem,[0,1,1,1,.5,0,.7,.25])
            net=model['critic'];z=(x-np.array(net['mean']))/np.array(net['std'])
            hidden=np.tanh(np.array(net['w1'])@z+net['b1'])
            return float((np.array(net['w2'])@hidden+net['b2'])[0])
        scored=sorted([(score_candidate(p),k,p) for k,p in eligible],key=lambda item:item[0])
        choices=[scored[-1],scored[max(0,len(scored)//4)]]
        rng.shuffle(choices);variants=[]
        for _,dk,dp in choices:
            parts=[]
            for st in ('drums','bass','other','vocals'):
                if st!=stem:
                    parts.append(dict(stem=st,atBar=0,bars=8,files=[phrase['stems'][st]['file']],slices=[dict(itemIndex=0,startFrac=0,endFrac=1,targetBars=4),dict(itemIndex=0,startFrac=0,endFrac=1,targetBars=4)],sourceDuration=phrase['duration_sec'],semitones=0))
                else:
                    parts.append(dict(stem=st,atBar=0,bars=4,files=[phrase['stems'][st]['file']],slices=[dict(itemIndex=0,startFrac=0,endFrac=1,targetBars=4)],sourceDuration=phrase['duration_sec'],semitones=0))
                    parts.append(dict(stem=st,atBar=4,bars=4,files=[dp['stems'][st]['file']],slices=[dict(itemIndex=0,startFrac=0,endFrac=1,targetBars=4)],sourceDuration=dp['duration_sec'],semitones=0))
            x=transition_features(a,features[rows[dp['id']],si,0],stem,[0,1,1,1,.5,0,.7,.25]).tolist()
            variants.append(dict(bpm=960/phrase['duration_sec'],parts=parts,features=x,source_tracks=[kit['track_id'],dk['track_id']]))
        reviews.append(dict(id=f'flow-{seed}-{case}',a=variants[0],b=variants[1]))
    return reviews


def trace_job(path,base):
    trace=json.loads(Path(path).read_text())
    if trace.get('format')!='wub-slice-trace-v1':raise ValueError('unsupported playback trace')
    manifest=json.loads((base/'kit_manifest.json').read_text())
    phrases={p['id']:p for k in manifest['kits'] for p in k['phrases']}
    records=sorted(trace['records'],key=lambda r:r['at'])
    if not records:raise ValueError('empty playback trace')
    starts=sorted(set(r['at'] for r in records))[:2]
    selected=[r for r in records if r['at'] in starts]
    first=selected[0];bars=sum(c['bars'] for c in first['cells'])
    bar_sec=(first['endAt']-first['at'])/bars;bpm=240/bar_sec
    parts=[]
    for record in selected:
        cells=record['cells'];pids=list(dict.fromkeys(c['phraseId'] for c in cells))
        slices=[];pitch=[];offset=0
        for cell in cells:
            slices.append(dict(itemIndex=pids.index(cell['phraseId']),startFrac=cell['startFrac'],endFrac=cell['endFrac'],targetBars=cell['bars'],gain=cell['gain'],endGain=cell['endGain']))
            pitch.append(dict(offsetSec=offset*bar_sec,semitones=cell['semitones']));offset+=cell['bars']
        parts.append(dict(stem=record['stem'],atBar=(record['at']-starts[0])/bar_sec,bars=offset,files=[phrases[pid]['stems'][record['stem']]['file'] for pid in pids],slices=slices,sourceDuration=phrases[pids[0]]['duration_sec'],semitones=cells[0]['semitones'],rate=cells[0]['rate'],pitchPoints=pitch))
    return dict(bpm=bpm,parts=parts,seamBar=(starts[1]-starts[0])/bar_sec if len(starts)>1 else bars/2)


def review_html(cases):
    public=[dict(id=c['id'],a=c['a']['features'],b=c['b']['features'],source_tracks=sorted(set(c['a']['source_tracks']+c['b']['source_tracks']))) for c in cases]
    cards=''.join(f'<section><h2>Transition {i+1}</h2><p>Which version keeps the music flowing?</p><label>A <audio controls preload="none" src="{c["id"]}-a.wav"></audio></label><label>B <audio controls preload="none" src="{c["id"]}-b.wav"></audio></label><p><button data-id="{c["id"]}" data-winner="a">Prefer A</button> <button data-id="{c["id"]}" data-winner="b">Prefer B</button> <button data-id="{c["id"]}" data-winner="tie">No preference</button></p><output id="{c["id"]}"></output></section>' for i,c in enumerate(cases))
    return '<!doctype html><meta charset="utf-8"><title>Wub listening comparisons</title><style>body{background:#111425;color:#eee;font:18px system-ui;max-width:850px;margin:40px auto;padding:20px}section{padding:22px;border-bottom:1px solid #556}label{display:block;margin:15px 0}audio{vertical-align:middle;width:85%}button{padding:12px;cursor:pointer}output{color:#8fe}</style><h1>Which transition feels earned?</h1><p>Listen at a comfortable volume. A few comparisons are enough; you can save and return later.</p>'+cards+'<p><button id="save">Download preferences</button></p><script>const cases='+json.dumps(public)+';const choices=JSON.parse(localStorage.getItem("wub-flow-v1")||"{}");function show(id){document.getElementById(id).textContent=choices[id]?"Saved: "+choices[id]:""}cases.forEach(c=>show(c.id));document.querySelectorAll("[data-winner]").forEach(b=>b.onclick=()=>{choices[b.dataset.id]=b.dataset.winner;localStorage.setItem("wub-flow-v1",JSON.stringify(choices));show(b.dataset.id)});document.querySelectorAll("audio").forEach(a=>a.onplay=()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause()}));document.getElementById("save").onclick=()=>{const rows=cases.filter(c=>choices[c.id]).map(c=>({...c,winner:choices[c.id]}));const url=URL.createObjectURL(new Blob([rows.map(r=>JSON.stringify(r)).join("\\n")],{type:"application/x-ndjson"}));const a=document.createElement("a");a.href=url;a.download="slice-flow-preferences.jsonl";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};</script>'


async def main_async(args):
    base=Path(args.kits);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cases=make_reviews(base,args.cases,args.seed) if not args.trace else []
    report=[]
    if args.trace:
        job=trace_job(args.trace,base)
        async with Browser() as browser:
            value=await asyncio.wait_for(browser.evaluate(RENDER_JS.replace('JOB',json.dumps(job))),120)
        pcm=np.frombuffer(base64.b64decode(value['pcm']),dtype='<f4').reshape(-1,2)
        if not np.isfinite(pcm).all() or np.max(np.abs(pcm))<1e-6:raise RuntimeError('invalid trace render')
        sf.write(out/'session-excerpt.wav',pcm,value['sampleRate'],subtype='FLOAT')
        print(out/'session-excerpt.wav')
        return
    for case in cases:
        for label in ('a','b'):
            job=case[label]
            # Isolate offline WASM graphs between clips; no accumulating
            # worklet contexts while rendering a large review batch.
            async with Browser() as browser:
                value=await asyncio.wait_for(browser.evaluate(RENDER_JS.replace('JOB',json.dumps(job))),120)
            pcm=np.frombuffer(base64.b64decode(value['pcm']),dtype='<f4').reshape(-1,2)
            sf.write(out/f'{case["id"]}-{label}.wav',pcm,value['sampleRate'],subtype='FLOAT')
            seam=int((.15+job.get('seamBar',4)*240/job['bpm'])*value['sampleRate'])
            pre=pcm[max(0,seam-882):seam];post=pcm[seam:seam+882]
            pre_rms=float(np.sqrt(np.mean(pre**2)));post_rms=float(np.sqrt(np.mean(post**2)))
            report.append(dict(id=case['id'],variant=label,rms=float(np.sqrt(np.mean(pcm**2))),peak=float(np.max(np.abs(pcm))),seam_dip_db=float(20*np.log10(max(post_rms,1e-8)/max(pre_rms,1e-8))),finite=bool(np.isfinite(pcm).all())))
            print('rendered',case['id'],label,flush=True)
    (out/'index.html').write_text(review_html(cases))
    (out/'cases.json').write_text(json.dumps(cases))
    (out/'render_report.json').write_text(json.dumps({'format':'slice_flow_render_report_v1','renderer':'shipping kit_pcm + Signalsmith Stretch + master safety stage in Chrome OfflineAudioContext','cases':report},indent=2))
    if not all(r['finite'] and r['rms']>1e-5 for r in report):raise RuntimeError('render produced nonfinite or silent audio')
    print(out/'index.html')


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace',type=Path,help='replay the first two scheduled blocks of an exported listening session')
    ap.add_argument('--kits',default='frontend/public/data/kits');ap.add_argument('--out',default='/tmp/wub-slice-flow/review');ap.add_argument('--cases',type=int,default=6);ap.add_argument('--seed',type=int,default=505)
    asyncio.run(main_async(ap.parse_args()))
