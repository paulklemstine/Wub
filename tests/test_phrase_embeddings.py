"""Contrastively learned phrase embeddings used by the shipped ranker."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KITS = ROOT / "frontend/public/data/kits"


def test_phrase_embeddings_are_learned_from_real_transitions():
    doc = json.loads((KITS / "phrase_embeddings.json").read_text())
    assert doc["format"] == "wub-phrase-embeddings-v1"
    assert doc["dim"] == 64 and doc["pairs"] >= 2000
    assert len(doc["phrases"]) >= 3600
    for vec in list(doc["phrases"].values())[:20]:
        assert len(vec) == 64 and all(-1 <= v <= 1 for v in vec)
    ranker = json.loads((KITS / "bank_transition.json").read_text())
    assert ranker["report"]["embeddings"] == "learned-contrastive"
    assert "train_phrase_embeddings.py" in " ".join(p.name for p in (ROOT / "scripts").glob("*.py"))
