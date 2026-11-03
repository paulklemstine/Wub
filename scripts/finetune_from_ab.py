"""Preference fine-tune the bar arranger on blind A/B votes (filtered behavior cloning).

Each non-skipped vote with played bars becomes training signal: the winning
variant's bars train at full weight, the losing variant's bars are excluded.
This is supervised fine-tuning on human-approved continuations -- NOT full
RLHF/DPO (live play has no shared prompt prefix to pair over). Small data
regime: few epochs, tiny LR, early stop on winner-val perplexity.

A parallel --dry-run mode trains 2 steps on synthetic pairs for CI.

Usage:
  PYTHONPATH=. python3 scripts/finetune_from_ab.py --ab data/ab_results.json
  PYTHONPATH=. python3 scripts/finetune_from_ab.py --dry-run  # CI smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_bar_arranger import BarGPT, batch_iter, evaluate


def load_pairs(ab_path: Path):
    try:
        data = json.loads(ab_path.read_bytes())
    except Exception:
        return [], []
    winners, losers = [], []
    for r in data.get("results", []):
        if r.get("skipped") or not r.get("winner") or not isinstance(r.get("bars"), dict):
            continue
        mapping = r.get("mapping") or {}
        win_variant = mapping.get(r["winner"], r["winner"])
        lose_variant = "old" if win_variant == "new" else "new"
        wb = (r["bars"].get(win_variant) or [])[-16:]
        lb = (r["bars"].get(lose_variant) or [])[-16:]
        wb = [t for b in wb if isinstance(b, list) and len(b) == 6 for t in b]
        lb = [t for b in lb if isinstance(b, list) and len(b) == 6 for t in b]
        if len(wb) >= 12:
            winners.append(wb)
        if len(lb) >= 12:
            losers.append(lb)
    return winners, losers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", default="data/ab_results.json")
    ap.add_argument("--base", default="data/arranger_gpt.pt")
    ap.add_argument("--out", default="data/arranger_gpt_dpo.pt")
    ap.add_argument("--log", default="data/finetune_log.json")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dry_run:
        vocab = 64
        model = BarGPT(vocab=vocab, d=32, layers=1, heads=2, ctx=64).to(device)
        seq = [{"tokens": [random.randrange(vocab) for _ in range(96)], "stems": ["drums"]}]
        from scripts.train_bar_arranger import windows
        tr_w = windows(seq, ctx=64, stride=32)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        for x, y, ln, mk in batch_iter(tr_w, 4):
            x, y, ln, mk = (t.to(device) for t in (x, y, ln, mk))
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
            loss = (loss.reshape_as(y) * (torch.arange(y.size(1), device=device)[None, :] < ln[:, None]).float() * mk).sum()
            loss.backward()
            opt.step()
            assert math.isfinite(float(loss))
        Path("/tmp/finetune_dryrun_ok").write_text("ok")
        print("dry-run ok", flush=True)
        return

    ckpt = torch.load(args.base, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = BarGPT(vocab=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"]).to(device)
    model.load_state_dict(ckpt["model"])

    winners, losers = load_pairs(Path(args.ab))
    print(f"pairs: {len(winners)} winner seqs, {len(losers)} loser seqs", flush=True)
    if len(winners) < 4:
        raise SystemExit(f"need >=4 voted winner seqs, have {len(winners)} -- collect A/B votes first")
    from scripts.train_bar_arranger import windows
    cut = max(1, int(len(winners) * 0.8))
    tr_w = windows([{"tokens": t, "stems": ["drums", "bass", "other", "vocals"]} for t in winners[:cut]],
                   ctx=args.ctx, stride=48)
    va_w = windows([{"tokens": t, "stems": ["drums", "bass", "other", "vocals"]} for t in winners[cut:] or winners[:1]],
                   ctx=args.ctx, stride=48)
    lo_w = windows([{"tokens": t, "stems": ["drums", "bass", "other", "vocals"]} for t in losers],
                   ctx=args.ctx, stride=48)
    base_va = evaluate(model, va_w, args.batch, device)
    base_lo = evaluate(model, lo_w, args.batch, device) if lo_w else float("nan")
    print(f"base: winner_val_ppl={base_va:.2f} loser_val_ppl={base_lo:.2f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, hist = float("inf"), []
    for ep in range(args.epochs):
        model.train()
        for x, y, ln, mk in batch_iter(tr_w, args.batch):
            x, y, ln, mk = (t.to(device) for t in (x, y, ln, mk))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                loss = loss.reshape_as(y)
                mask = (torch.arange(y.size(1), device=device)[None, :] < ln[:, None]).float() * mk
                loss = (loss * mask).sum() / mask.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vp = evaluate(model, va_w, args.batch, device)
        lp = evaluate(model, lo_w, args.batch, device) if lo_w else float("nan")
        hist.append({"epoch": ep, "winner_val_ppl": round(vp, 3), "loser_val_ppl": round(lp, 3) if lp == lp else None})
        print(f"epoch {ep}: winner_val_ppl={vp:.2f} loser_val_ppl={lp:.2f}", flush=True)
        if vp < best:
            best = vp
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "val_ppl": round(vp, 3), "pairs": len(winners)}, args.out)
    Path(args.log).write_text(json.dumps({"best_winner_val_ppl": round(best, 3),
                                          "base_winner_val_ppl": round(base_va, 3),
                                          "base_loser_val_ppl": round(base_lo, 3) if base_lo == base_lo else None,
                                          "history": hist}, separators=(",", ":")))
    print(f"saved {args.out} best_winner_val_ppl={best:.2f} (base {base_va:.2f})", flush=True)


if __name__ == "__main__":
    main()
