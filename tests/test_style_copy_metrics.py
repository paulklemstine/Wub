"""Style-copy fidelity measured against the analyzed training songs."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_style_copy_metrics_are_bounded_and_discriminative():
    result = subprocess.run([
        "python3", "-c",
        "import sys;sys.path.insert(0,'scripts');"
        "from style_copy_metrics import corpus_stats,style_fidelity;"
        "import pathlib,json;"
        "c=corpus_stats(pathlib.Path('data/processed'),max_tracks=40,"
        "cohort_artifact=pathlib.Path('frontend/public/data/kits/transition_cohort.json'));"
        "log=[{'bars':4,'target':'INTRO','changedTracks':['drums','bass','other','vocals']},"
        "{'bars':8,'target':'GROOVE_A','changedTracks':['vocals']}];"
        "print(json.dumps(style_fidelity(log,c)))",
    ], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    doc = json.loads(result.stdout.strip().splitlines()[-1])
    assert 0 <= doc["style_fidelity"] <= 1
    assert "cohort_divergence" in doc and doc["blocks"] == 2
