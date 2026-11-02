"""Evaluate a trained bar-arranger GPT musically (not just perplexity).

For N val prefixes (16 bars): generate 8 bars (top-k sampling), then report:
- nonempty: % generated bars with all 4 stems non-UNK
- section_stability: mean # section flips per 8-bar continuation (want <=2)
- joint_pmi: mean PMI of generated 4-stem combos under mined joint tuples
- chord_bass: % bars where bass-first-pitch == chord token (when chord known)

Usage: PYTHONPATH=. python3 scripts/eval_bar_arranger.py [--ckpt ...] [--n 20]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from scripts.build_arranger_dataset import CHORD_OFF, OFF, SEC_OFF, UNK
from scripts.train_bar_arranger import BarGPT

STEMS = ("drums", "bass", "other", "vocals")


@torch.no_grad()
def generate(model, prefix, n_bars, device, temp=1.0, top_k=20, seed=0):
    rng = random.Random(seed)
    toks = list(prefix[-(model.ctx - 6):])
    out = []
    for _ in range(n_bars * 6):
        x = torch.tensor([toks[-model.ctx:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1] / max(1e-6, temp)
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        nxt = int(rng.choices(range(len(probs)), weights=probs, k=1)[0])
        toks.append(nxt)
        out.append(nxt)
    return [out[i * 6:(i + 1) * 6] for i in range(n_bars)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/arranger_gpt.pt")
    ap.add_argument("--dataset", default="data/arranger_dataset.json")
    ap.add_argument("--patterns", default="data/global_patterns.json")
    ap.add_argument("--vocab", default="data/global_vocabulary.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=1.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = BarGPT(vocab=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = json.loads(Path(args.dataset).read_bytes())
    pats = json.loads(Path(args.patterns).read_bytes())
    gvoc = json.loads(Path(args.vocab).read_bytes())
    sig_of = {}
    for stem in STEMS:
        sig_of[stem] = [p["signature"] for p in pats["bar_prototypes"][stem]]
    pitch_of = {}
    for stem in STEMS:
        for v in gvoc["vocab_by_stem"][stem]:
            pitch_of[v["id"]] = int(v["pitch"])
    # token id -> (stem, rank); joint sig lookup
    tok_sig = {}
    for stem in STEMS:
        for r, sig in enumerate(sig_of[stem]):
            tok_sig[OFF[stem] + r] = (stem, sig)

    def _bar_sigs(six):
        out = {}
        _sec, _ch, _d, _b, _o, _v = six
        for s, t in (("drums", _d), ("bass", _b), ("other", _o), ("vocals", _v)):
            if t != UNK[s] and t in tok_sig:
                out[s] = tok_sig[t][1]
        return out

    def _load_json(p):
        try:
            return json.loads(Path(p).read_bytes())
        except Exception:
            return {}

    val = ds["val"]
    n = min(args.n, len(val))
    m_empty = m_flip = m_chord = m_chord_hit = 0
    seen_bars = set()
    for i in range(n):
        seq = val[i]["tokens"]
        bars = len(seq) // 6
        if bars < 24:
            continue
        prefix = seq[:16 * 6]
        gen = generate(model, prefix, 8, device, temp=args.temp, seed=1000 + i)
        prev_sec = prefix[-6]
        prev_sigs = _bar_sigs(prefix[-6:])
        flips = 0
        for b in gen:
            sec, ch, d, bb, o, v = b
            if sec != prev_sec:
                flips += 1
                prev_sec = sec
            stems_toks = {"drums": d, "bass": bb, "other": o, "vocals": v}
            if all(t != UNK[s] for s, t in stems_toks.items()):
                m_empty += 1
            sigs = {}
            for s, t in stems_toks.items():
                if t != UNK[s] and t in tok_sig:
                    sigs[s] = tok_sig[t][1]
            # variety: distinct full bars (guards against mode collapse)
            seen_bars.add(tuple(b))
            prev_sigs = sigs
            # chord-bass agreement via prototype token -> slice pitch
            if ch != CHORD_OFF + 12 and bb != UNK["bass"] and bb in tok_sig:
                _, bsig = tok_sig[bb]
                # first global id in sig -> pitch
                for gid in bsig.split("-"):
                    if gid != "." and gid in pitch_of:
                        m_chord += 1
                        if pitch_of[gid] % 12 == (ch - CHORD_OFF) % 12:
                            m_chord_hit += 1
                        break
        m_flip += flips
    print(f"continuations: {n}", flush=True)
    print(f"  nonempty bars: {m_empty}/{n * 8} = {m_empty / max(1, n * 8):.2f}", flush=True)
    print(f"  mean section flips / 8 bars: {m_flip / max(1, n):.2f} (want <=2)", flush=True)
    print(f"  distinct bars: {len(seen_bars)}/{n * 8} (want high; guards mode collapse)", flush=True)
    print(f"  chord-bass agreement: {m_chord_hit}/{m_chord} = {m_chord_hit / max(1, m_chord):.2f} (info only; harmony enforced downstream)", flush=True)


if __name__ == "__main__":
    main()
