#!/usr/bin/env python3
"""Exercise real Chrome/Web Audio playback, scheduling and a cluster handoff.

Runs the actual player and fixed voice pool in an isolated muted browser.
No acceleration of AudioContext time or fake scheduling. --seconds 330 normally
crosses a complete wave set. Exports actual scheduled slices for replay.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.review_slice_flow import Browser

START=r'''(async()=>{
 const {AudioEngine}=await import('/src/audio_engine.ts');
 const {StemRemixPlayer}=await import('/src/stem_remix.ts');
 const engine=new AudioEngine();await engine.ensureCtx().resume();
 const player=new StemRemixPlayer(engine);await player.load('/data/kits');
 player.setStyleFilter(new Set([STYLE]));await player.startSession();
 globalThis.flowSmoke={engine,player,levels:[],errors:[],longTasks:[],clusters:[],nodes:new Set()};
 addEventListener('unhandledrejection',event=>flowSmoke.errors.push(String(event.reason)));
 if(PerformanceObserver.supportedEntryTypes.includes('longtask'))new PerformanceObserver(list=>flowSmoke.longTasks.push(...list.getEntries().map(e=>e.duration))).observe({type:'longtask'});
 flowSmoke.timer=setInterval(()=>{try{
  const now=engine.now();player.tick(now);engine.updateGlue(now);player.schedule(now+.15,'GROOVE_A');
  const cluster=player.currentCluster?.id;if(cluster && flowSmoke.clusters.at(-1)!==cluster)flowSmoke.clusters.push(cluster);
  for(const pool of Object.values(player.voicePool))for(const voice of pool){
   if(!flowSmoke.nodes.has(voice.node)){flowSmoke.nodes.add(voice.node);voice.node.addEventListener('processorerror',()=>flowSmoke.errors.push('audio worklet processor error'));}
  }
 }catch(e){flowSmoke.errors.push(String(e));}},20);
 flowSmoke.meter=setInterval(()=>{const analyser=engine.getMasterAnalyser();const data=new Float32Array(analyser.fftSize);analyser.getFloatTimeDomainData(data);flowSmoke.levels.push(Math.sqrt(data.reduce((s,v)=>s+v*v,0)/data.length));},250);
 return {ready:player.sliceFlow.ready,bpm:player.bpm};
})()'''
FINISH=r'''(()=>{
 clearInterval(flowSmoke.timer);clearInterval(flowSmoke.meter);
 const trace=flowSmoke.player.sliceFlow.trace(flowSmoke.engine.now());
 const report={ready:flowSmoke.player.sliceFlow.ready,bpm:flowSmoke.player.bpm,errors:flowSmoke.errors,records:trace.records.length,clusters:flowSmoke.clusters,voiceCounts:Object.fromEntries(Object.entries(flowSmoke.player.voicePool).map(([k,v])=>[k,v.length])),levels:flowSmoke.levels,longTasks:flowSmoke.longTasks,trace};
 flowSmoke.player.stop();return report;
})()'''

async def main(args):
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 async with Browser() as browser:
  print('started',await browser.evaluate(START.replace('STYLE',json.dumps(args.style))),flush=True)
  # Output snapshots keep long checks observable without blocking the caller.
  remaining=args.seconds
  forced=False
  while remaining>0:
   interval=min(30,remaining);await asyncio.sleep(interval);remaining-=interval
   elapsed=args.seconds-remaining
   if args.force_rotation and not forced and elapsed>=30:
    await browser.evaluate("flowSmoke.player.sessionBar=104;flowSmoke.player.nextCluster=flowSmoke.player.chooseNeighbourCluster();true")
    forced=True
   print('playback seconds',elapsed,flush=True)
  result=await browser.evaluate(FINISH)
 (out/'listening_trace.json').write_text(json.dumps(result['trace'],indent=2))
 levels=np.array(result.pop('levels'))
 long_tasks=result.pop('longTasks');result.pop('trace')
 longest=run=0
 for value in levels[4:]:
  run=run+1 if value<1e-5 else 0;longest=max(run,longest)
 result.update(seconds=args.seconds,forced_rotation=forced,max_digital_zero_seconds=longest*.25,
               level_percentiles=np.percentile(levels,[0,5,50,95,100]).tolist(),
               long_task_count=len(long_tasks),max_long_task_ms=max(long_tasks,default=0),
               renderer='Chrome AudioContext / shipping StemRemixPlayer')
 (out/'browser_smoke.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 if result['errors'] or not result['ready'] or len(result['voiceCounts'])!=4 or any(n!=2 for n in result['voiceCounts'].values()) or longest>=4:
  raise RuntimeError('live audio smoke failed; inspect browser_smoke.json')

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--seconds',type=int,default=95);ap.add_argument('--style',default='wave')
 ap.add_argument('--force-rotation',action='store_true',help='force an early set boundary after 30s without accelerating the audio clock')
 ap.add_argument('--out',default='/tmp/wub-slice-flow/live-final')
 args=ap.parse_args()
 if args.seconds<5:ap.error('--seconds must be at least 5')
 asyncio.run(main(args))
