#!/usr/bin/env python3
"""Train/validate the slice-flow heads as one staged, reproducible pipeline.

Optional automatic preferences remain explicitly labelled model judgements.
Original audio stays in place and read-only; only complete validated model
artifacts are promoted. This does not rebuild the retired slice/VQ pipeline.
"""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True)


def auto_refinement_allowed(baseline, candidate):
    before=float(baseline['validation_mae']);after=float(candidate['validation_mae'])
    return math.isfinite(after) and after <= before + max(.001, before*.05)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kits',type=Path,default=ROOT/'frontend/public/data/kits')
    ap.add_argument('--processed',type=Path,default=ROOT/'data/processed')
    ap.add_argument('--work',type=Path,default=Path('/tmp/wub-slice-flow'))
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--feedback',type=Path)
    ap.add_argument('--traces',type=Path)
    ap.add_argument('--auto-cases',type=int,default=0)
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--motif-epochs',type=int,default=25)
    ap.add_argument('--value-epochs',type=int,default=25)
    args=ap.parse_args();base=args.kits.resolve();args.work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='staged-',dir=args.work) as folder:
        stage=Path(folder)
        shutil.copy2(base/'kit_manifest.json',stage/'kit_manifest.json')
        (stage/'kits').symlink_to((base/'kits').resolve(),target_is_directory=True)
        if (base/'sidecars.json').exists():shutil.copy2(base/'sidecars.json',stage/'sidecars.json')
        core=['--kits',stage,'--processed',args.processed,'--work',args.work,'--device',args.device,'--epochs',args.epochs]
        if args.traces:core+=['--traces',args.traces]
        if args.feedback:core+=['--feedback',args.feedback]
        run('train_slice_flow.py',*core)
        run('train_motif_return.py','--kits',stage,'--out',stage/'motif_return.json','--epochs',args.motif_epochs,'--device',args.device)
        run('train_bar_tour_value.py','--kits',stage,'--out',stage/'bar_tour_value.json','--epochs',args.value_epochs,'--device',args.device)
        automatic_refinement={"attempted":False}
        if args.auto_cases:
            baseline_bytes=(stage/'slice_flow.json').read_bytes()
            baseline=json.loads(baseline_bytes)['report']
            auto=args.work/'automatic-preferences.jsonl'
            run('auto_review.py','--kits',stage,'--cases',args.auto_cases,'--out',auto,'--report',args.work/'automatic-review-report.json')
            feedback=args.work/'combined-preferences.jsonl'
            human=args.feedback.read_text().rstrip()+'\n' if args.feedback else ''
            feedback.write_text(human+auto.read_text())
            # Keep acoustic extraction cached; fine-tune only the main critic.
            refreshed=[v for v in core]
            if '--feedback' in refreshed:
                idx=refreshed.index('--feedback');refreshed[idx+1]=feedback
            else:refreshed+=['--feedback',feedback]
            run('train_slice_flow.py',*refreshed)
            candidate=json.loads((stage/'slice_flow.json').read_text())['report']
            accepted=auto_refinement_allowed(baseline,candidate)
            automatic_refinement={"attempted":True,"accepted":accepted,
                "baseline_validation_mae":baseline['validation_mae'],
                "candidate_validation_mae":candidate['validation_mae'],
                "criterion":"no automatic update beyond max(0.001, 5%) held-out MAE regression"}
            if not accepted:
                (stage/'slice_flow.json').write_bytes(baseline_bytes)
                print('Automatic refinement rejected: retaining the stronger held-out baseline',flush=True)
        models={name:json.loads((stage/f'{name}.json').read_text()) for name in ('slice_flow','motif_return','bar_tour_value')}
        report={name:model['report'] for name,model in models.items()}
        report['automatic_cases_requested']=args.auto_cases
        report['automatic_refinement']=automatic_refinement
        flags=json.loads((stage/'sidecars.json').read_text())
        flags.update(slice_flow=True,motif_return=True,bar_tour_value=True)
        # Stage on the destination filesystem; publish model JSON after PCM
        # features and availability flags last. Git/hosting deploy one snapshot.
        names=['slice_flow_features.f32','slice_flow.json','motif_return.json','bar_tour_value.json']
        for name in names:shutil.copy2(stage/name,base/(name+'.pending'))
        for name in names:(base/(name+'.pending')).replace(base/name)
        (base/'sidecars.json.pending').write_text(json.dumps(flags,indent=2)+'\n')
        (base/'sidecars.json.pending').replace(base/'sidecars.json')
        (args.work/'pipeline_report.json').write_text(json.dumps(report,indent=2))
        print(f'Published validated slice pipeline to {base}',flush=True)


if __name__=='__main__':main()
