#!/usr/bin/env python3
"""Extract real slice acoustics, mine full-song development, train/export critics.

Track partitions are assigned BEFORE any donor sampling. Compatible cross-song
pairs get graded positives; source identity is never the critic's target.
Feedback JSONL accepts {a:[108 floats], b:[108 floats], winner:'a'|'b'|'tie'}.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wub_core.slice_flow import DIM, FEATURES, INPUT_DIM, STEMS, acoustic_teacher, describe_audio, transition_features, mix_features, mix_teacher, transform


def partition(track):
    return int(hashlib.sha256(track.encode()).hexdigest()[:8], 16) % 10 == 0


def extract_job(job):
    pid, stem, filename = job
    audio, sr = sf.read(filename, dtype='float32', always_2d=True)
    return pid, stem, np.stack([describe_audio(bar, sr, stem) for bar in np.array_split(audio, 4)])


def mine_development(path):
    doc = json.loads(path.read_text())
    stems = doc.get('stems_available', [])
    # Do not invent four independent stems out of a master-only analysis.
    if len(stems) < 4:
        return None
    segments = {s['id']: s for s in doc.get('segments', [])}
    tatums = doc.get('tatums', [])
    n = len(tatums) // 16
    if n < 16:
        return None
    values = np.zeros((n, 4, 3), np.float32)
    previous = [None] * 4
    for i, t in enumerate(tatums[:n * 16]):
        for s, stem in enumerate(STEMS):
            ids = t.get('stem_segments', {}).get(stem, [])
            seg = segments.get(ids[-1]) if ids else None
            if seg and not seg.get('is_silent', False):
                values[i // 16, s, 0] += np.clip((seg.get('loudness_max', -60) + 60) / 60, 0, 1) / 16
                values[i // 16, s, 1] += float(ids[-1] != previous[s]) / 16
                pitches = seg.get('pitches', [])
                values[i // 16, s, 2] += (int(np.argmax(pitches)) / 11 if pitches else 0) / 16
                previous[s] = ids[-1]
    windows = np.array([values[i:i+4].mean(axis=0) for i in range(0, n-3, 4)])
    bpm = max(40, float(doc.get('bpm', 120)))
    rows, labels, durations = [], [], []
    age, repeats = 0., 0
    for i in range(2, len(windows)-1):
        cur, prev, nxt = windows[i], windows[i-1], windows[i+1]
        d = np.abs(nxt-cur).mean(axis=1)
        active = np.maximum(nxt[:,0],cur[:,0]) > .15
        change = (d > .075) & active
        count = int(change.sum())
        label = 0 if count == 0 else (1 if count <= 2 else 2)
        e, pe = float(cur[:,0].mean()), float(prev[:,0].mean())
        trend = float(cur[:,0].mean() - windows[max(0,i-4)][:,0].mean())
        rows.append([min(2, age/120), e, e-pe, float(cur[:,1].mean()), trend, min(1,repeats/8), 1-float(np.abs(cur-prev).mean()), float(np.abs(cur-prev).mean()>.1)])
        labels.append(label)
        age += 960/bpm
        repeats = repeats+1 if np.abs(cur-prev).mean()<.025 else 0
        if label == 2:
            if age>8: durations.append(age)
            age = 0
    return str(doc.get('track_id',path.stem)), rows, labels, durations


def train_net(x, y, val, epochs, device, classes=1, feedback=()):
    torch.manual_seed(505)
    torch.set_num_threads(4)
    # Float64 accumulation matters for constant columns in a large bank:
    # float32 cancellation can invent a nonzero standard deviation.
    mean = x[~val].mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_std = x[~val].astype(np.float64).std(axis=0)
    std = np.maximum(raw_std, .03).astype(np.float32)
    z = (x-mean)/std
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,classes)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=.002)
    xt = torch.tensor(z[~val],device=device)
    yt = torch.tensor(y[~val],device=device, dtype=torch.long if classes>1 else torch.float32)
    best, best_loss = None, float('inf')
    xv = torch.tensor(z[val],device=device)
    yv = torch.tensor(y[val],device=device,dtype=torch.long if classes>1 else torch.float32)
    for epoch in range(epochs):
        net.train()
        for idx in torch.randperm(len(xt),device=device).split(1024):
            pred=net(xt[idx])
            loss=torch.nn.functional.cross_entropy(pred,yt[idx]) if classes>1 else torch.nn.functional.binary_cross_entropy_with_logits(pred[:,0],yt[idx])
            if feedback:
                pairs=feedback[:256]
                a=torch.tensor((np.array([p['a'] for p in pairs],np.float32)-mean)/std,device=device)
                b=torch.tensor((np.array([p['b'] for p in pairs],np.float32)-mean)/std,device=device)
                target=torch.tensor([1. if p['winner']=='a' else 0. if p['winner']=='b' else .5 for p in pairs],device=device)
                weights=torch.tensor([.1 if str(p.get('origin','human')).startswith('auto-reward-') else .3 for p in pairs],device=device)
                pair_loss=torch.nn.functional.binary_cross_entropy_with_logits(net(a)[:,0]-net(b)[:,0],target,reduction='none')
                loss=loss+(weights*pair_loss).mean()
            opt.zero_grad();loss.backward();opt.step()
        net.eval()
        with torch.no_grad():
            pv=net(xv)
            vloss=(torch.nn.functional.cross_entropy(pv,yv) if classes>1 else torch.nn.functional.binary_cross_entropy_with_logits(pv[:,0],yv)).item()
        if vloss<best_loss:
            best_loss=vloss;best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    # Constant training fields carry no learned information. Zero their
    # projection so live age/energy/stereo changes cannot activate random
    # initialization on an unobserved dimension.
    best['0.weight'][:, torch.tensor(raw_std < 1e-5)] = 0
    net.load_state_dict(best)
    with torch.no_grad():
        raw=net(xv).cpu().numpy()
    probs=1/(1+np.exp(-raw[:,0])) if classes==1 else np.exp(raw-raw.max(axis=1,keepdims=True))/np.exp(raw-raw.max(axis=1,keepdims=True)).sum(axis=1,keepdims=True)
    state=net.state_dict()
    exported={"mean":mean.tolist(),"std":std.tolist(),"w1":state['0.weight'].cpu().tolist(),"b1":state['0.bias'].cpu().tolist(),"w2":state['2.weight'].cpu().tolist(),"b2":state['2.bias'].cpu().tolist()}
    return exported,probs,best_loss


def load_feedback(path):
    feedback=[]
    if not path:return feedback
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        item=json.loads(line)
        if item.get('winner') not in ('a','b','tie') or len(item.get('a',[]))!=INPUT_DIM or len(item.get('b',[]))!=INPUT_DIM:
            raise ValueError('invalid preference row')
        if not np.isfinite(item['a']+item['b']).all():raise ValueError('non-finite preference')
        if not item.get('source_tracks') or any(partition(track) for track in item['source_tracks']):
            raise ValueError('feedback must include source_tracks from the training partition only')
        feedback.append(item)
    return feedback


def trace_examples(paths,features,phrase_ids):
    rows={pid:i for i,pid in enumerate(phrase_ids)}
    out=[]
    for path in paths:
        trace=json.loads(path.read_text())
        if trace.get('format')!='wub-slice-trace-v1':raise ValueError('unsupported playback trace')
        previous={}
        for record in sorted(trace.get('records',[]),key=lambda r:r['at']):
            for cell in record.get('cells',[]):
                stem=cell['stem'];pid=cell['phraseId']
                if pid not in rows or partition(cell['trackId']):
                    previous.pop(stem,None);continue
                bar=int(cell['bar'])
                if stem not in STEMS or bar not in range(4):raise ValueError('invalid trace cell')
                current=transform(features[rows[pid],STEMS.index(stem),bar],cell['semitones'],(cell['gain']+cell['endGain'])/2*cell.get('mixerGain',1))
                if stem in previous:
                    old,old_track=previous[stem]
                    context=[0,old_track!=cell['trackId'],1,cell['startFrac']==0,.5,0,.7,.25]
                    out.append((transition_features(old,current,stem,context),acoustic_teacher(old,current,stem)))
                previous[stem]=(current,cell['trackId'])
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kits',default='frontend/public/data/kits')
    ap.add_argument('--processed',default='data/processed')
    ap.add_argument('--work',default='/tmp/wub-slice-flow')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 512*1024*1024 else 'cpu')
    ap.add_argument('--feedback',type=Path)
    ap.add_argument('--traces',type=Path,help='directory of exported playback trace JSON files')
    args=ap.parse_args();start=time.time()
    base=Path(args.kits);work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'kit_manifest.json').read_text())
    phrase_ids=[];tracks=[];jobs=[];features=[]
    for kit in manifest['kits']:
        for phrase in kit['phrases']:
            phrase_ids.append(phrase['id']);tracks.append(kit['track_id'])
            for stem in STEMS:
                asset=phrase.get('stems',{}).get(stem)
                if asset:jobs.append((phrase['id'],stem,str(base/asset['file'])))
    cache=work/'acoustics.npz'
    fingerprint=hashlib.sha256((base/'kit_manifest.json').read_bytes()).hexdigest()
    if cache.exists():
        saved=np.load(cache)
        if str(saved['fingerprint'])==fingerprint:
            features=saved['features']
    if not len(features):
        features=np.zeros((len(phrase_ids),4,4,DIM),np.float32)
        row={pid:i for i,pid in enumerate(phrase_ids)}
        with ThreadPoolExecutor(args.workers) as pool:
            for i,(pid,stem,desc) in enumerate(pool.map(extract_job,jobs)):
                features[row[pid],STEMS.index(stem)]=desc
                if i%2000==0:print('audio cells',i,'/',len(jobs),flush=True)
        np.savez_compressed(cache,features=features,fingerprint=fingerprint)
    rng=np.random.default_rng(505);x=[];y=[];split=[];kinds=[]
    pools={flag:np.array([i for i,t in enumerate(tracks) if partition(t)==flag]) for flag in (False,True)}
    for p,track in enumerate(tracks):
        flag=partition(track);pool=pools[flag]
        for stem in range(4):
            for bar in range(3):
                a,b=features[p,stem,bar],features[p,stem,bar+1]
                natural=acoustic_teacher(a,b,STEMS[stem])
                ctx=[0,0,0,bar==0,.5,0,.7,.25]
                x.append(transition_features(a,b,STEMS[stem],ctx));y.append(max(.72,natural));split.append(flag);kinds.append('natural')
                for _ in range(2):
                    donor=int(rng.choice(pool));db=int(rng.integers(4));b=features[donor,stem,db]
                    prepared=bool(rng.integers(2));ctx=[prepared,tracks[donor]!=track,1,1,.5,0,.7,.25]
                    target=acoustic_teacher(a,b,STEMS[stem],prepared)
                    x.append(transition_features(a,b,STEMS[stem],ctx));y.append(target);split.append(flag);kinds.append('cross')
                # Same-source hard corruptions avoid teaching source identity.
                b=features[p,stem,bar+1].copy();b[:3]*=.1;b[7:11]=b[7:11][::-1];b[20:]=np.roll(b[20:],6)
                x.append(transition_features(a,b,STEMS[stem],[0,0,1,1,.5,0,.7,.5]));y.append(.06);split.append(flag);kinds.append('corrupt')
    captured=trace_examples(sorted(args.traces.glob('*.json')) if args.traces else [],features,phrase_ids)
    for vector,target in captured:
        x.append(vector);y.append(target);split.append(False);kinds.append('playback')
    x=np.stack(x);y=np.array(y,np.float32);val=np.array(split,bool)
    feedback=load_feedback(args.feedback)
    model,pred,loss=train_net(x,y,val,args.epochs,args.device,feedback=feedback)
    kinds=np.array(kinds);binary=y[val]>.65
    report={'format':'slice_flow_report_v1','seed':505,'phrases':len(phrase_ids),'audio_cells':int(features.shape[0]*16),'train_tracks':len(set(np.array(tracks)[~np.array([partition(t) for t in tracks])])),'validation_tracks':sorted(set(np.array(tracks)[np.array([partition(t) for t in tracks])])),'critic_examples':len(x),'validation_mae':float(mean_absolute_error(y[val],pred)),'validation_auc_weak_labels':float(roc_auc_score(binary,pred)),'cross_song_positive_examples':int(np.sum((kinds=='cross')&(y>.65))),'feedback_pairs':len(feedback),'human_feedback_pairs':sum(not str(p.get('origin','human')).startswith('auto-reward-') for p in feedback),'automatic_feedback_pairs':sum(str(p.get('origin','human')).startswith('auto-reward-') for p in feedback),'playback_examples':len(captured),'label_provenance':'natural chronology + graded acoustic weak supervision; not human preference','device':args.device}
    mx=[];my=[];mv=[]
    for p,track in enumerate(tracks):
        flag=partition(track)
        for bar in range(4):
            real=features[p,:,bar]
            mx.append(mix_features(real));my.append(mix_teacher(real));mv.append(flag)
            donors=rng.choice(pools[flag],size=4)
            mixed=np.stack([features[int(donors[stem]),stem,bar] for stem in range(4)])
            mx.append(mix_features(mixed));my.append(mix_teacher(mixed));mv.append(flag)
    mix_model,mp,ml=train_net(np.array(mx,np.float32),np.array(my,np.float32),np.array(mv),args.epochs,args.device)
    report['mix_validation_mae']=float(mean_absolute_error(np.array(my)[mv],mp))
    # Full original timelines teach development without requiring full-song playback.
    devcache=work/'development.json'
    if devcache.exists():development=json.loads(devcache.read_text())
    else:
        development=[];paths=sorted(Path(args.processed).glob('*.json'))
        with ThreadPoolExecutor(min(args.workers,4)) as pool:
            for i,result in enumerate(pool.map(mine_development,paths)):
                if result:development.append(result)
                if i%200==0:print('original timelines',i,'/',len(paths),flush=True)
        devcache.write_text(json.dumps(development))
    dx=[];dy=[];dv=[];durations=[]
    for tid,rows,labels,durs in development:
        dx.extend(rows);dy.extend(labels);dv.extend([partition(tid)]*len(rows));durations.extend(durs)
    if not dx:raise ValueError('no four-stem original timelines available')
    planner,dp,dl=train_net(np.array(dx,np.float32),np.array(dy),np.array(dv),args.epochs,args.device,classes=3)
    report.update({'development_tracks':len(development),'development_examples':len(dx),'development_accuracy':float(np.mean(dp.argmax(axis=1)==np.array(dy)[dv])),'development_majority_accuracy':float(np.bincount(np.array(dy)[dv],minlength=3).max()/len(dp)),'foundation_seconds':float(np.clip(np.percentile(durations,75) if durations else 90,60,120)),'elapsed_sec':round(time.time()-start,2)})
    if report['validation_mae'] > .08 or report['mix_validation_mae'] > .10:
        raise RuntimeError('acoustic validation failed; existing shipping model was not replaced')
    if report['development_accuracy'] < report['development_majority_accuracy']:
        raise RuntimeError('development planner failed to beat the held-out majority baseline')
    artifact={'format':'slice_flow_v1','version':1,'feature_dim':DIM,'feature_names':FEATURES,'phrase_ids':phrase_ids,'feature_file':'slice_flow_features.f32','critic':model,'mix':mix_model,'planner':planner,'foundation_seconds':report['foundation_seconds'],'report':report}
    # Only publish complete validated artifacts; model is renamed last.
    np.asarray(features,dtype='<f4').tofile(base/'slice_flow_features.f32.tmp')
    (base/'slice_flow_features.f32.tmp').replace(base/'slice_flow_features.f32')
    (base/'slice_flow.json.tmp').write_text(json.dumps(artifact,separators=(',',':'),allow_nan=False))
    (base/'slice_flow.json.tmp').replace(base/'slice_flow.json')
    (work/'report.json').write_text(json.dumps(report,indent=2))
    flags_path=base/'sidecars.json'
    flags=json.loads(flags_path.read_text()) if flags_path.exists() else {'version':1}
    flags['slice_flow']=True
    flags_path.write_text(json.dumps(flags,indent=2)+'\n')
    np.savez_compressed(work/'validation.npz',x=x[val],y=y[val],prediction=pred,kinds=kinds[val])
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
