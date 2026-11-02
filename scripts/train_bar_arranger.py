"""Train a bar-level GPT arranger over global-vocab prototype sequences.

Sequence: flat tokens, 6 per bar [SEC, CHORD, D, B, O, V] (see
scripts/build_arranger_dataset.py). Standard causal LM: predict next token.
At inference the browser feeds the last <=32 bars and samples the next bar's
6 tokens (SEC first -> drives section; then chord + 4 stem prototype ranks).

Model (~3.5M params, d=256/L=4/H=4) exports to ONNX for onnxruntime-web.
Outputs data/arranger_gpt.pt (best val) + data/arranger_train_log.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BarGPT(nn.Module):
    def __init__(self, vocab=1222, d=256, layers=4, heads=4, ctx=192, dropout=0.1):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu")
        self.tr = nn.TransformerEncoder(layer, layers)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        causal = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))[None, :, :]
        # padding-free: all windows full length by construction
        h = self.tr(h, mask=causal)
        return self.head(self.ln(h))


STEMS = ("drums", "bass", "other", "vocals")
# token positions within a bar: 0 SEC, 1 CHORD, 2 D, 3 B, 4 O, 5 V
STEM_POS = {"drums": 2, "bass": 3, "other": 4, "vocals": 5}
POS_UNK = {2: 300, 3: 601, 4: 902, 5: 1203}  # per-position UNK token ids


def loss_mask_for(seq_tokens, stems):
    """1/0 per token: 0 where a stem is missing from the recording (don't
    teach the model to predict silence for instruments that were never
    separated). Real silence inside stemmed tracks keeps mask 1."""
    present = set(stems or [])
    m = []
    for i, t in enumerate(seq_tokens):
        pos = i % 6
        if pos in (0, 1):
            m.append(1)
            continue
        stem = STEMS[pos - 2]
        if stem not in present and t == POS_UNK[pos]:
            m.append(0)
        else:
            m.append(1)
    return m


def windows(seqs, ctx=192, stride=48):
    out = []
    for s in seqs:
        t = s["tokens"]
        mask = loss_mask_for(t, s.get("stems", STEMS))
        for i in range(0, max(1, len(t) - ctx), stride):
            w = t[i:i + ctx]
            mw = mask[i:i + ctx]
            if len(w) < 16:
                continue
            if len(w) < ctx:
                pad = ctx - len(w)
                w = w + [0] * pad
                mw = mw + [0] * pad  # pad (loss masked)
            out.append((w, min(len(t[i:i + ctx]), ctx), mw))
    return out


def batch_iter(data, bs, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        x = torch.tensor([c[0][:-1] for c in chunk], dtype=torch.long)
        y = torch.tensor([c[0][1:] for c in chunk], dtype=torch.long)
        ln = torch.tensor([c[1] - 1 for c in chunk], dtype=torch.long)
        mk = torch.tensor([c[2][1:] for c in chunk], dtype=torch.float)
        yield x, y, ln, mk


@torch.no_grad()
def evaluate(model, data, bs, device):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for x, y, ln, mk in batch_iter(data, bs, shuffle=False):
        x, y, ln, mk = x.to(device), y.to(device), ln.to(device), mk.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
        loss = loss.reshape_as(y)
        mask = (torch.arange(y.size(1), device=device)[None, :] < ln[:, None]).float() * mk
        tot_loss += float((loss * mask).sum())
        tot_tok += int(mask.sum())
    return math.exp(tot_loss / max(1, tot_tok))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/arranger_dataset.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ctx", type=int, default=192)
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic run for CI")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        vocab = 64
        tr = [{"tokens": [random.randrange(vocab) for _ in range(192)], "stems": list(STEMS)} for _ in range(32)]
        va = [{"tokens": [random.randrange(vocab) for _ in range(192)], "stems": list(STEMS)} for _ in range(8)]
        model = BarGPT(vocab=vocab, d=64, layers=2, heads=2, ctx=192).to(device)
        tr_w = windows(tr, ctx=192, stride=48)
        va_w = windows(va, ctx=192, stride=48)
    else:
        ds = json.loads(Path(args.data).read_bytes())
        vocab = int(ds["vocab"])
        model = BarGPT(vocab=vocab, d=args.d, layers=args.layers, heads=args.heads, ctx=args.ctx).to(device)
        tr_w = windows(ds["train"], ctx=args.ctx, stride=args.stride)
        va_w = windows(ds["val"], ctx=args.ctx, stride=args.stride)
        print(f"windows: train {len(tr_w)}, val {len(va_w)}", flush=True)
        if not tr_w or not va_w:
            raise SystemExit("empty windows; build dataset first")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best, hist = float("inf"), []
    for ep in range(args.epochs if not args.smoke else 2):
        model.train()
        tot, ntok = 0.0, 0
        for x, y, ln, mk in batch_iter(tr_w, args.batch if not args.smoke else 8):
            x, y, ln, mk = x.to(device), y.to(device), ln.to(device), mk.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
                loss = loss.reshape_as(y)
                mask = (torch.arange(y.size(1), device=device)[None, :] < ln[:, None]).float() * mk
                loss = (loss * mask).sum() / mask.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            nt = int(mask.sum())
            tot += float(loss) * nt
            ntok += nt
        sched.step()
        vppl = evaluate(model, va_w, args.batch if not args.smoke else 8, device)
        tppl = math.exp(tot / max(1, ntok))
        hist.append({"epoch": ep, "train_ppl": round(tppl, 3), "val_ppl": round(vppl, 3), "lr": round(sched.get_last_lr()[0], 6)})
        print(f"epoch {ep}: train_ppl={tppl:.2f} val_ppl={vppl:.2f}", flush=True)
        if vppl < best:
            best = vppl
            torch.save({"model": model.state_dict(),
                        "cfg": {"vocab": model.tok.num_embeddings, "d": model.tok.embedding_dim,
                                "layers": len(model.tr.layers), "heads": model.tr.layers[0].self_attn.num_heads,
                                "ctx": model.ctx}}, "data/arranger_gpt.pt" if not args.smoke else "/tmp/arranger_gpt_smoke.pt")
    if not args.smoke:
        Path("data/arranger_train_log.json").write_text(json.dumps({"best_val_ppl": round(best, 3), "history": hist}))
        print(f"best val_ppl={best:.2f} -> data/arranger_gpt.pt", flush=True)


if __name__ == "__main__":
    main()
