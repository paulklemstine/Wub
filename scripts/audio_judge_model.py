"""Two-tower CNN audio transition judge (shared weights).

Each 4-bar bar is a 64x128 log-mel patch.  The encoder maps one bar to a
128-d embedding; the head consumes the concatenation of the two embeddings
and predicts whether the ordered pair is a real consecutive transition.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

MEL_NORM_DIV = 20.0
MEL_FLOOR_DB = -60.0
MEL_CEIL_DB = 40.0


def normalize_bars(x: torch.Tensor) -> torch.Tensor:
    """Per-bar loudness normalisation: subtract each bar's mean dB.

    Removes the absolute loudness/mastering level so the judge cannot use a
    per-bar loudness shortcut.  Input is dB (power_to_db(ref=1.0)); a -60 dB
    floor keeps the source-stem and Opus-codec noise floors comparable.
    """
    x = torch.clamp(x.float(), min=MEL_FLOOR_DB, max=MEL_CEIL_DB)
    mean = x.mean(dim=(2, 3), keepdim=True)
    return (x - mean) / MEL_NORM_DIV


class BarEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128, in_ch: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 32 x 64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16 x 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 8 x 16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 4)),
            nn.Flatten(),
            nn.Linear(128 * 8, 256), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(normalize_bars(x))


class AudioJudge(nn.Module):
    def __init__(self, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoder = BarEncoder(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ea = self.encoder(a)
        eb = self.encoder(b)
        return self.head(torch.cat([ea, eb], dim=1)).squeeze(1)

    @torch.no_grad()
    def score_pairs(self, a: torch.Tensor, b: torch.Tensor,
                    batch: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, embeddings_concat) for two aligned tensors of bars."""
        logits = []
        embs = []
        n = a.shape[0]
        for i in range(0, n, batch):
            xa = a[i:i + batch].float()
            xb = b[i:i + batch].float()
            if xa.dim() == 3:
                xa = xa.unsqueeze(1)
                xb = xb.unsqueeze(1)
            ea = self.encoder(xa)
            eb = self.encoder(xb)
            logits.append(self.head(torch.cat([ea, eb], 1)).squeeze(1))
            embs.append(torch.cat([ea, eb], 1))
        return torch.cat(logits), torch.cat(embs)
