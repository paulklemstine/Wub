"""Tests for the bar-arranger GPT (dataset layout + model shapes)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_bar_arranger import BarGPT, batch_iter, windows


def test_token_layout_disjoint():
    from scripts.build_arranger_dataset import CHORD_OFF, OFF, SEC_OFF, UNK, VOCAB
    assert VOCAB == 1222
    seen = set()
    for stem, o in OFF.items():
        for r in range(300):
            assert o + r not in seen
            seen.add(o + r)
        assert UNK[stem] not in seen
        seen.add(UNK[stem])
    for i in range(5):
        assert SEC_OFF + i not in seen
        seen.add(SEC_OFF + i)
    for i in range(13):
        assert CHORD_OFF + i not in seen
        seen.add(CHORD_OFF + i)


def test_windows_deterministic_shapes():
    seqs = [{"tokens": list(range(6 * 40))}]  # 40 bars
    w1 = windows(seqs, ctx=192, stride=48)
    w2 = windows(seqs, ctx=192, stride=48)
    assert w1 == w2 and len(w1) > 0
    for w, ln in w1:
        assert len(w) == 192 and 0 < ln <= 192


def test_model_forward_loss_finite():
    torch.manual_seed(0)
    model = BarGPT(vocab=64, d=64, layers=2, heads=2, ctx=192)
    model.eval()
    x = torch.randint(0, 64, (4, 191))
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, 191, 64)
    assert torch.isfinite(logits).all()
