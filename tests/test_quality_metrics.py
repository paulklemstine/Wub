"""Five-dimension quality telemetry for the infinite set."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bundle(name: str, source: str) -> Path:
    out = Path("/tmp/opencode") / f"{name}.cjs"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ROOT / "frontend/node_modules/.bin/esbuild"),
        str(ROOT / "frontend/src" / source), "--bundle", "--platform=node",
        "--format=cjs", f"--outfile={out}",
    ], check=True, capture_output=True, text=True)
    return out


def test_quality_hook_is_wired_in_main():
    main = (ROOT / "frontend/src/main.ts").read_text()
    assert "__wubQualitySummary" in main
    assert "buildQualitySummary(remix, probe, gains)" in main
    assert "qualitySnapshot" in (ROOT / "frontend/src/stem_remix.ts").read_text()
    assert "measureHarmony" in (ROOT / "frontend/src/bar_composer.ts").read_text()


def test_per_bar_quartet_harmony_objective():
    out = bundle("quartet_harmony", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const c=new BarComposer();
const major=(root)=>{const v=new Array(12).fill(0);for(const p of [root,(root+4)%12,(root+7)%12])v[p]=1;return v;};
const bars=(root)=>[major(root),major(root),major(root),major(root)];
c.cellChroma=new Map([
 ['good',{bass:bars(0),other:bars(0),vocals:bars(0)}],
 ['bad',{bass:bars(0),other:bars(6),vocals:bars(3)}],
]);
const prog=[{relRoot:0,quality:'major'},{relRoot:5,quality:'major'},
            {relRoot:7,quality:'major'},{relRoot:5,quality:'major'}];
const entry=(id)=>({bass:{phrase:{id},keyShift:0},other:{phrase:{id},keyShift:0},
 vocals:{phrase:{id},keyShift:0}});
const good=c.quartetHarmonyFit(entry('good'),0,prog);
const bad=c.quartetHarmonyFit(entry('bad'),0,prog);
assert.ok(good>bad+0.15,'aligned quartet scores higher than clashing: '+good+' vs '+bad);
assert.ok(good>=0&&good<=1&&bad>=0&&bad<=1);
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_phrase_key_shift_fits_all_four_bars():
    out = bundle("harmony_shift", "bar_composer.ts")
    script = r"""
const assert=require('node:assert/strict');
const {BarComposer}=require(process.argv[1]);
const c=new BarComposer();
c.cellChroma=new Map();
const meta={p:[.8,.8,.8,.8],r:[0,0,0,0],q:['minor','minor','minor','minor'],
 o:[0b1,0b1,0b1,0b1],es:[.8,.8,.8,.8]};
const entry={phrase:{id:'p'},keyShift:0};
const chord={relRoot:2,quality:'minor'};
const oldShift=c.solvePhraseKeyShift('bass',entry,meta,chord,0);
const prog=[2,5,7,5].map(relRoot=>({relRoot,quality:'minor'}));
const newShift=c.solvePhraseKeyShift('bass',entry,meta,chord,0,prog,'p');
assert.equal(oldShift,2,'old method only fits the first chord');
assert.equal(newShift,5,'new method fits all four bars');
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_quality_score_penalizes_each_failure_mode():
    out = bundle("quality_metrics", "quality_metrics.ts")
    script = r"""
const assert=require('node:assert/strict');
const q=require(process.argv[1]);
const goodHarmony={p10:.9,p50:.95,clashRate:0,meanRootSpread:.1};
const badHarmony={p10:.3,p50:.5,clashRate:.6,meanRootSpread:.8};
const goodArc={sectionChanges:4,uniqueSections:5,arcDeviation:.03,meanEnergySlope:.01};
const badArc={sectionChanges:20,uniqueSections:2,arcDeviation:.3,meanEnergySlope:0};
const goodChurn={uniqueTracks:12,uniquePhrases:40,changes:30,repeatRate:.45};
const badChurn={uniqueTracks:2,uniquePhrases:6,changes:2,repeatRate:.95};
const goodSeams={dipEvents:0,seamEvents:20,p95Jump:.05};
const badSeams={dipEvents:18,seamEvents:20,p95Jump:.8};
const goodMix={gainSpread:.05,duckSpread:.05,dominantStem:'bass'};
const badMix={gainSpread:.9,duckSpread:.9,dominantStem:'drums'};
const base=q.qualityScore(goodHarmony,goodArc,goodChurn,goodSeams,goodMix);
assert.ok(base.score>.8 && base.weakest!=='harmony',JSON.stringify(base));
for(const [name,args] of Object.entries({
 harmony:[badHarmony,goodArc,goodChurn,goodSeams,goodMix],
 arrangement:[goodHarmony,badArc,goodChurn,goodSeams,goodMix],
 churn:[goodHarmony,goodArc,badChurn,goodSeams,goodMix],
 seams:[goodHarmony,goodArc,goodChurn,badSeams,goodMix],
 mix:[goodHarmony,goodArc,goodChurn,goodSeams,badMix]})){
 const s=q.qualityScore(...args);
 assert.ok(s.score<base.score,name+' must lower the score');
 assert.equal(s.weakest,name,'weakest must identify '+name+': '+JSON.stringify(s));
}
const bars=Array.from({length:10},()=>({minCos:.8,rootSpread:.2}));
const h=q.harmonicQuality(bars,.65);
assert.equal(h.clashRate,0);
assert.ok(h.p10>.7 && h.p50>.7);
const entries=[{bar:0,section:'INTRO',energy:.3},{bar:4,section:'GROOVE_A',energy:.7}];
const a=q.arrangementQuality(entries,[.3,.7,.8]);
assert.equal(a.sectionChanges,1);
assert.ok(a.arcDeviation<.1);
const summary=q.buildQualitySummary({harmony:h,arrangement:a,churn:goodChurn},goodSeams,goodMix);
assert.ok(summary && summary.score>=0 && summary.score<=1);
"""
    result = subprocess.run(["node", "-e", script, str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
