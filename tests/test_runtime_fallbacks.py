"""Runtime coverage for browsers without worker decoding or WebGL."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def bundle(tmp_path, name):
    output = tmp_path / f"{name}.mjs"
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / f"frontend/src/{name}.ts"), "--bundle", "--platform=browser",
        "--format=esm", '--define:import.meta.env={"BASE_URL":"/"}', f"--outfile={output}",
    ], check=True, capture_output=True, text=True)
    return json.dumps(output.as_uri())


def run(script):
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_uses_decode_bridge_without_offline_audio_context(tmp_path):
    module = bundle(tmp_path, "kit_audio_worker")
    run(r"""
import assert from 'node:assert/strict';
const replies = [];
let fetches = 0, decodes = 0;
globalThis.fetch = async () => {
  fetches++;
  return {ok:true, arrayBuffer:async () => new ArrayBuffer(8)};
};
globalThis.self = {postMessage(message, transfer) {
  if (message.type === 'decode') {
    decodes++;
    assert.equal(transfer[0], message.bytes);
    const samples = [new Float32Array([0.2, 0.3]), new Float32Array([0.4, 0.5])];
    setImmediate(() => self.onmessage({data:{type:'decoded', decodeId:message.decodeId, samples}}));
  } else replies.push(message);
}};
""" + f"await import({module});\n" + r"""
assert.equal(typeof OfflineAudioContext, 'undefined');
const send = data => self.onmessage({data});
const flush = () => new Promise(resolve => setImmediate(resolve));
send({type:'init', sampleRate:44100});
const item = {key:'phrase:drums', url:'/phrase.opus', preUrl:null};
send({type:'warm',items:[item]});
send({type:'load', id:1, items:[item]});
send({type:'load', id:2, items:[item]});
for(let i=0;i<10;i++) await flush();
assert.equal(fetches, 1);
assert.equal(decodes, 1);
assert.equal(replies.length, 2);
assert.ok(replies.every(r => r.type === 'run'));
assert.deepEqual(replies[0].samples[0], new Float32Array([0.2,0.3]));
// An unsuccessful bridge decode resolves as a failed run, never a hung load.
self.postMessage = message => {
  if (message.type === 'decode') setImmediate(() => send({type:'decoded',decodeId:message.decodeId,message:'codec failed'}));
  else replies.push(message);
};
send({type:'load',id:3,items:[{...item,key:'bad'}]});
for(let i=0;i<10;i++) await flush();
assert.equal(replies.at(-1).type,'error');
assert.equal(replies.at(-1).message,'codec failed');
""")


def test_decode_bridge_loader_keeps_worker_enabled_and_ignores_stale_replies(tmp_path):
    module = bundle(tmp_path, "stem_remix")
    run(f"import {{ StemRemixPlayer }} from {module};\n" + r"""
import assert from 'node:assert/strict';
let worker;
globalThis.Worker = class {
  sent=[];
  constructor(){worker=this;}
  postMessage(message, transfer){this.sent.push({message,transfer});}
  terminate(){this.terminated=true;}
};
globalThis.scheduler={yield:async()=>{}};
const loader = new StemRemixPlayer({}).audioWorker;
const channel = new Float32Array([0.3,0.7]);
let finishDecode;
loader.start(44100, async () => {
  await new Promise(resolve => {finishDecode=resolve;});
  return {numberOfChannels:1,getChannelData:()=>channel};
});
worker.onmessage({data:{type:'decode',decodeId:5,bytes:new ArrayBuffer(8)}});
finishDecode();
await new Promise(resolve => setImmediate(resolve));
const reply=worker.sent.at(-1);
assert.equal(reply.message.type,'decoded');
assert.equal(reply.transfer.length,2);
assert.deepEqual(reply.message.samples[0],channel);
assert.notEqual(reply.message.samples[0].buffer,channel.buffer);
assert.equal(loader.ok,true);
const count=worker.sent.length;
worker.onmessage({data:{type:'decode',decodeId:6,bytes:new ArrayBuffer(8)}});
loader.dispose();
finishDecode();
await new Promise(resolve => setImmediate(resolve));
assert.equal(worker.sent.length,count);
assert.equal(worker.terminated,true);
""")


def test_graphics_are_lazy_and_blocked_webgl_uses_bounded_2d_fallback(tmp_path):
    lightshow = bundle(tmp_path, "lightshow")
    background = bundle(tmp_path, "psychedelic_bg")
    run(f"import {{ RaveLightshowPanel }} from {lightshow};\n"
        + f"import {{ PsychedelicBackground }} from {background};\n" + r"""
import assert from 'node:assert/strict';
let probes=0, paints=0;
const listeners={};
const ctx={fillRect(){paints++;},createRadialGradient(){return {addColorStop(){}};}};
const canvas={clientWidth:3840,clientHeight:2160,width:100,height:100,style:{},
  addEventListener(name,fn){(listeners[name] ??= []).push(fn);},
  getBoundingClientRect(){return {width:this.clientWidth,height:this.clientHeight};},
  getContext(kind){if(kind==='2d')return ctx;probes++;return null;}};
globalThis.window={devicePixelRatio:3,innerWidth:3840,innerHeight:2160,addEventListener(){}};
globalThis.document={hidden:false,getElementById:()=>canvas};
globalThis.fetch=async()=>({ok:true,json:async()=>({presets:[]})});
const panel=new RaveLightshowPanel(canvas);
const bg=new PsychedelicBackground();
assert.equal(probes,0); // no hidden/default-disabled contexts on startup
const frame={timeSec:1,bpm:120,section:'DROP',mix:0.8,drums:0.7,bass:0.6,other:0.4,vocals:0.3,drumPunch:1};
panel.render(frame);
assert.equal(probes,1);
assert.ok(paints>0);
assert.ok(canvas.width*canvas.height<=602000);
panel.render({...frame,timeSec:2});
assert.equal(probes,1); // no repeated blocked-context creation
panel.setState({blackout:true});
const before=paints;
panel.render(frame);
assert.equal(paints,before+1);
bg.setState({enabled:true});
bg.render();
assert.equal(bg.getState().enabled,false);
const attempts=probes;
bg.setState({enabled:true});
bg.render();
assert.equal(probes,attempts);
// The GPU path shares the same bounded resolution across its four targets.
let outputSize;
panel.renderer={setSize(w,h){outputSize=[w,h];}};
assert.equal(panel.ensureTargets(),true);
assert.ok(outputSize[0]*outputSize[1]<=602000);
for(const target of [panel.bgTarget,panel.fgTarget,...panel.trailTargets]) {
  assert.equal(target.width,outputSize[0]);
  assert.equal(target.height,outputSize[1]);
  assert.equal(target.depthBuffer,false);
}
const beforeLoss=paints;
for(const lost of listeners.webglcontextlost) lost();
panel.render(frame);
assert.equal(paints,beforeLoss); // no rendering into a lost context
assert.equal(panel.contextLost,true);
document.hidden=true;
const last=paints;
panel.render(frame);
assert.equal(paints,last);
""")
