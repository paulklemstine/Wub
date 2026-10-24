"""Phase 4: neural latent stem-slice synthesizer ("WubSynth").

Decodes brand-new drum hits / bass wubs / synth stabs from continuous latent
vectors instead of only retrieving recorded slices. Two backends, both lazily
loaded so `import wub_core.synth_decoder` never fails on a bare install:

- "encodec": Meta EnCodec (pip install encodec). Uses the continuous encoder
  output (pre-quantizer) as latent z; blending happens in that space.
- "rave": RAVE Realtime Audio VAE (requires a stem-appropriate .ts export +
  `rave` package). Cleanest on isolated bass/drums stems.

Blend semantics per stem (the UI "Synthetic / Real" slider):
  alpha=0.0 -> snap z to nearest real embedded slice, decode that slice's
               latent (identical to retrieval, but through the vocoder).
  0<alpha<1 -> interpolate between two real slice latents and decode.
  alpha=1.0 -> decode the raw predicted z (full hallucination).

Without a Transformer latent head, predicted z is sampled from a Gaussian
prior fitted on embedded corpus slices (honest in-distribution hallucination).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

BackendName = Literal["encodec", "rave"]


@dataclass
class SliceLatent:
    track_id: str
    segment_id: str
    stem: str
    z: np.ndarray


class LatentBackend:
    """Abstract continuous-latent audio codec."""

    sr: int = 24000

    def encode(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, z: np.ndarray, duration_sec: Optional[float] = None) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class EnCodecBackend(LatentBackend):
    """Continuous-latent wrapper around the `encodec` package."""

    def __init__(self, model_name: str = "facebook/encodec_24khz", device: Optional[str] = None):
        try:
            import torch
            from encodec import EncodecModel
        except ImportError as e:
            raise RuntimeError("pip install encodec torch torchaudio") from e
        import torch as _t

        self._torch = _t
        self.device = device or ("cuda" if _t.cuda.is_available() else "cpu")
        self.model = EncodecModel.encodec_model_24khz().to(self.device).eval()
        self.sr = 24000

    def _to_in(self, y: np.ndarray) -> "object":
        import librosa

        if y.ndim > 1:
            y = y.mean(axis=-1)
        if getattr(y, "dtype", None) != np.float32:
            y = np.asarray(y, dtype=np.float32)
        if self.sr != 44100:
            y = librosa.resample(y, orig_sr=44100, target_sr=self.sr).astype(np.float32)
        t = self._torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(self.device)
        return t

    def encode(self, y: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            z = self.model.encoder(self._to_in(y))
        return z.squeeze(0).cpu().numpy()

    def decode(self, z: np.ndarray, duration_sec: Optional[float] = None) -> np.ndarray:
        import torch

        with torch.no_grad():
            t = torch.from_numpy(np.asarray(z, dtype=np.float32)).unsqueeze(0).to(self.device)
            y = self.model.decoder(t).squeeze().cpu().numpy()
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if duration_sec:
            n = int(duration_sec * self.sr)
            y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        peak = np.max(np.abs(y)) + 1e-9
        return (y / peak * 0.89).astype(np.float32)


class StemLatentSynth:
    """Per-stem latent space + prior + blend/decode operations."""

    def __init__(self, backend: Optional[LatentBackend] = None):
        self.backend = backend
        self.embedded: Dict[str, List[SliceLatent]] = {}
        self.prior: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # stem -> (mean, std)

    @classmethod
    def with_backend(cls, name: BackendName = "encodec", **kw) -> "StemLatentSynth":
        if name == "encodec":
            return cls(EnCodecBackend(**kw))
        raise ValueError(f"RAVE backend needs an exported model; got {name!r} "
                         f"(pass a LatentBackend instance explicitly)")

    def _need_backend(self) -> LatentBackend:
        if self.backend is None:
            raise RuntimeError("no latent backend; use StemLatentSynth.with_backend('encodec')")
        return self.backend

    def embed_slice(self, y: np.ndarray) -> np.ndarray:
        return self._need_backend().encode(y)

    def embed_corpus(self, slices_by_stem: Dict[str, List[Tuple[str, str, np.ndarray]]],
                     cache_path: Optional[Path] = None) -> None:
        """slices_by_stem[stem] = [(track_id, segment_id, waveform44100), ...]."""
        if cache_path and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            for row in data["rows"]:
                self.embedded.setdefault(row["stem"], []).append(
                    SliceLatent(row["track_id"], row["segment_id"], row["stem"], row["z"]))
        else:
            rows = []
            for stem, items in slices_by_stem.items():
                for tid, sid, y in items:
                    z = self.embed_slice(y)
                    self.embedded.setdefault(stem, []).append(SliceLatent(tid, sid, stem, z))
                    rows.append({"track_id": tid, "segment_id": sid, "stem": stem, "z": z})
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, rows=np.asarray(rows, dtype=object))
        for stem, latents in self.embedded.items():
            Z = np.stack([s.z.mean(axis=-1) if s.z.ndim > 1 else s.z for s in latents])
            self.prior[stem] = (Z.mean(axis=0), Z.std(axis=0) + 1e-6)

    def sample_prior(self, stem: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if stem not in self.prior:
            raise KeyError(f"no prior for {stem}; embed a corpus first")
        mu, sd = self.prior[stem]
        shape = self.embedded[stem][0].z.shape
        eps = rng.normal(size=shape).astype(np.float32)
        return (mu.reshape([-1] + [1] * (len(shape) - 1)) if eps.ndim > 1 else mu) + \
               (sd.reshape([-1] + [1] * (len(shape) - 1)) if eps.ndim > 1 else sd) * eps * 0.5

    @staticmethod
    def _flat(z: np.ndarray) -> np.ndarray:
        if z.ndim > 1:
            return z.mean(axis=-1).astype(np.float64)
        return z.astype(np.float64)

    def snap(self, stem: str, z: np.ndarray) -> SliceLatent:
        cands = self.embedded.get(stem, [])
        if not cands:
            raise KeyError(f"nothing embedded for {stem}")
        q = self._flat(z)
        best = min(cands, key=lambda s: np.linalg.norm(self._flat(s.z) - q))
        return best

    def blend(self, stem: str, z_a: np.ndarray, z_b: np.ndarray, alpha: float) -> np.ndarray:
        a = np.asarray(z_a, dtype=np.float64)
        b = np.asarray(z_b if z_b is not None else z_a, dtype=np.float64)
        if a.shape != b.shape and a.ndim == 2 and b.ndim == 2:
            x_old = np.linspace(0, 1, b.shape[1])
            x_new = np.linspace(0, 1, a.shape[1])
            b = np.stack([np.interp(x_new, x_old, b[c]) for c in range(b.shape[0])])
        # slerp-ish: lerp then rescale to mean norm (keeps vocoder in range)
        z = (1 - alpha) * a + alpha * b
        target = (1 - alpha) * np.linalg.norm(a) + alpha * np.linalg.norm(b) + 1e-9
        return (z / (np.linalg.norm(z) + 1e-9) * target).astype(np.float32)

    def render(self, stem: str, z: np.ndarray,
               duration_sec: Optional[float] = None) -> np.ndarray:
        return self._need_backend().decode(z, duration_sec=duration_sec)

    def render_blend(self, stem: str, z_a: np.ndarray, z_b: Optional[np.ndarray],
                     alpha: float, duration_sec: Optional[float] = None) -> np.ndarray:
        """Full slider path: 0=nearest-real, 1=raw hallucination."""
        if alpha <= 0.0:
            z = self.snap(stem, z_a).z
        elif alpha >= 1.0:
            z = np.asarray(z_a, dtype=np.float32)
        else:
            z = self.blend(stem, z_a, z_b if z_b is not None else z_a, alpha)
        return self.render(stem, z, duration_sec)
