"""Pretrained "ears" embeddings (MERT + OpenL3/PANNs) for Wub phrase rating.

Goal: replace/augment the hand-made 48-d taste features with music-understanding
embeddings so phrase/transition rating can pick smoother DJ-grade transitions
with more variety and fewer clashes. CLAP is intentionally NOT used.

Embedders
---------
- MERT (microsoft/mert-v1-95M, via `transformers`): mean-pooled last hidden
  state per phrase x stem, L2-normalized, PCA-reduced (matrix saved to the
  output JSON). This is the primary embedding.
- OpenL3 (music) or PANNs CNN14, *if importable*: secondary backbone for
  FAD-backbone comparison, stored separately under ``embeddings_alt``.
- FAD (Frechet Audio Distance) is implemented here with numpy/scipy
  (regularized). If ``torchmetrics`` is importable its FAD is also computed
  (best-effort) and the two values are compared in the output.
- ``museval`` (if importable) sanity-checks SDR/SIR/SAR on a synthetic stem
  pair for stem-QC reference values.

Usage (laptop, RTX 4050, corpus at /mnt/e/Wub/data/stems):
    PYTHONPATH=. python3 scripts/ears_embed.py \
      --manifest frontend/public/data/kits/kit_manifest.json \
      --kits-dir frontend/public/data/kits \
      --out /tmp/ears_embed_full.json --device cuda --batch-size 4 --fp16

This checkout only has the shipped bank (frontend/public/data/kits, ~2565
phrases x 4 stems). The corpus on the laptop is at /mnt/e/Wub/data/stems.

First real run downloads MERT weights (~360 MB FP32) from Hugging Face
(needs network once, HF_TOKEN if the repo is gated). Until then use:

    python3 scripts/ears_embed.py --smoke --limit 8 \
      --out /tmp/opencode/ears_embed_smoke.json --device cpu

--smoke uses synthetic sine/noise clips (no audio files, no weights) and a
deterministic spectral feature in place of MERT, but runs the identical
PCA/FAD/diversity/JSON math so the logic is fully testable offline.

Memory guard (6 GB GPU): batch size (--batch-size, default 4) times clip
length bounds transformer activations; model runs in eval()/no_grad in
fp32 by default, --fp16 halves activation memory on the RTX 4050.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

STEMS = ("drums", "bass", "other", "vocals")
MERT_SR = 24000  # MERT native sample rate
MERT_FRAME = 320  # conv-stack total stride (75 Hz frame rate)
DEFAULT_MERT_ID = "microsoft/mert-v1-95M"
VERSION = "ears-0.1.0"
TOP_DIR = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Frechet distance (numpy/scipy, regularized)
# --------------------------------------------------------------------------

def frechet_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet distance between two embedding distributions (FAD formula).

    d^2 = ||m1 - m2||^2 + tr(C1 + C2 - 2 * sqrt(C1 @ C2))
    Covariances are regularized with eps * I. Implemented with
    numpy (+ scipy.linalg.sqrtm when scipy is available, eigenvalue
    fallback otherwise) so no torchmetrics install is required.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got {a.shape} and {b.shape}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"dim mismatch {a.shape[1]} vs {b.shape[1]}")
    d = a.shape[1]
    m1 = a.mean(axis=0)
    m2 = b.mean(axis=0)
    c1 = np.cov(a, rowvar=False) if a.shape[0] > 1 else np.zeros((d, d), np.float64)
    c2 = np.cov(b, rowvar=False) if b.shape[0] > 1 else np.zeros((d, d), np.float64)
    c1 = np.atleast_2d(c1) + eps * np.eye(d)
    c2 = np.atleast_2d(c2) + eps * np.eye(d)
    diff = m1 - m2
    tr_covmean = None
    try:  # preferred: matrix square root
        from scipy import linalg as _linalg

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covmean, _ = _linalg.sqrtm(c1 @ c2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = float(np.trace(covmean))
    except Exception:
        tr_covmean = None
    if tr_covmean is None or not math.isfinite(tr_covmean):
        # fallback: eigenvalues of C1 @ C2 are real & >= 0 in theory
        w = np.linalg.eigvals(c1 @ c2).real
        tr_covmean = float(np.sum(np.sqrt(np.clip(w, 0.0, None))))
    value = float(diff @ diff + np.trace(c1) + np.trace(c2) - 2.0 * tr_covmean)
    return max(value, 0.0)


def try_torchmetrics_fad(real: np.ndarray, synth: np.ndarray, sr: int) -> tuple[float | None, str]:
    """Best-effort torchmetrics FAD on the same embeddings, for comparison.

    Returns (value_or_None, note). Any failure is captured in the note;
    the numpy/scipy FAD from frechet_distance() remains authoritative.
    """
    try:
        from torchmetrics.audio.fad import FrechetAudioDistance
    except Exception as exc:  # not installed or moved in this version
        return None, f"torchmetrics not importable ({exc}); numpy/scipy FAD used"
    try:
        import torch

        fad = FrechetAudioDistance(use_correction=False, sample_rate=sr)
        r = torch.as_tensor(np.asarray(real, dtype=np.float32))
        s = torch.as_tensor(np.asarray(synth, dtype=np.float32))
        try:
            fad.update(r, real=True)
            fad.update(s, real=False)
        except TypeError:
            fad.update(r)
            fad.update(s)
        val = float(fad.compute())
        return val, "torchmetrics FrechetAudioDistance computed on same embeddings"
    except Exception as exc:
        return None, f"torchmetrics FAD unavailable on embeddings ({exc}); numpy/scipy FAD used"


# --------------------------------------------------------------------------
# Synthetic audio (smoke mode + FAD probe baseline)
# --------------------------------------------------------------------------

def synth_wave(stem: str, rng: np.random.Generator, sr: int = MERT_SR, dur: float = 4.0) -> np.ndarray:
    """Deterministic synthetic mono clip for one stem.

    Used in --smoke mode (as the 'real' set) and in real mode as a fixed
    synth probe distribution so FAD always has a defined n_synth.
    """
    n = int(sr * dur)
    t = np.arange(n) / sr
    if stem == "drums":
        x = rng.standard_normal(n) * 0.08
        every = int(sr * 60.0 / 125.0)  # 125 BPM eighth positions
        for start in range(0, n, every):
            k = min(int(0.09 * sr), n - start)
            x[start:start + k] += rng.standard_normal(k) * 0.9 * np.exp(-np.arange(k) / (0.03 * sr))
        return np.clip(x, -1.0, 1.0).astype(np.float32)
    if stem == "bass":
        f0 = 50.0 * rng.uniform(0.9, 1.4)
        env = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (t / 0.96)))  # half-note gates
        x = (0.75 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)) * env
        return np.clip(x * 0.8, -1.0, 1.0).astype(np.float32)
    if stem == "other":
        base = 220.0 * rng.uniform(0.9, 1.2)
        ratios = np.array([1.0, 1.26, 1.5])
        trem = 0.75 + 0.25 * np.sin(2 * np.pi * 0.8 * t)
        x = np.sum([0.4 * np.sin(2 * np.pi * base * r * t) for r in ratios], axis=0) * trem
        return np.clip(x * 0.5, -1.0, 1.0).astype(np.float32)
    # vocals: harmonic series with vibrato + syllable envelopes
    f0 = 210.0 * rng.uniform(0.9, 1.25)
    vib = 1.0 + 0.03 * np.sin(2 * np.pi * 5.0 * t)
    harm = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * vib * t) for k in range(1, 7))
    syll = np.abs(np.sin(2 * np.pi * 2.0 * t)) ** 1.5
    return np.clip(harm * 0.45 * syll, -1.0, 1.0).astype(np.float32)


def smoke_feature(x: np.ndarray, sr: int = MERT_SR, bands: int = 256) -> np.ndarray:
    """Deterministic spectral stand-in for a MERT vector (smoke mode only).

    Log-spaced magnitude spectrum + early/late energy, standardized and
    L2-normalized. Deterministic given the waveform, no weights needed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mag = np.abs(np.fft.rfft(x * np.hanning(n), n=2 * n))
    freqs = np.fft.rfftfreq(2 * n, 1.0 / sr)
    targets = np.geomspace(40.0, sr / 2.0, bands)
    idx = np.clip(np.searchsorted(freqs, targets), 0, len(mag) - 1)
    v = np.log1p(mag[idx] + 1e-9)
    early = float(np.mean(x[: n // 2] ** 2))
    late = float(np.mean(x[n // 2:] ** 2))
    v = np.concatenate([v, [math.log(early + 1e-12), math.log(late + 1e-12)]])
    v = (v - v.mean()) / (v.std() + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(np.float32)


# --------------------------------------------------------------------------
# Audio loading (opus via soundfile OGG, librosa fallback, ffmpeg last)
# --------------------------------------------------------------------------

def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import librosa

        return librosa.resample(np.asarray(x, dtype=np.float32), orig_sr=sr_in, target_sr=sr_out)
    except Exception:
        from scipy.signal import resample_poly

        g = math.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_wave(path: Path, target_sr: int = MERT_SR) -> tuple[np.ndarray, str]:
    """Load one clip as mono float32 at target_sr. Returns (wave, method)."""
    methods_tried: list[str] = []
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        methods_tried.append(f"soundfile:{exc.__class__.__name__}")
        data, sr = None, None
    if data is None:
        try:
            import librosa

            data, sr = librosa.load(str(path), sr=target_sr, mono=True)
            return np.asarray(data, dtype=np.float32), "librosa"
        except Exception as exc:
            methods_tried.append(f"librosa:{exc.__class__.__name__}")
    if data is not None:
        if data.ndim > 1:
            data = data.mean(axis=1)
        return _resample(data.astype(np.float32), int(sr), target_sr), "soundfile"
    # ffmpeg last resort (needed on builds whose libsndfile lacks Opus)
    try:
        import subprocess

        proc = subprocess.run(
            ["ffmpeg", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-"],
            capture_output=True, check=True, timeout=60,
        )
        return np.frombuffer(proc.stdout, dtype=np.float32).copy(), "ffmpeg"
    except Exception as exc:
        methods_tried.append(f"ffmpeg:{exc.__class__.__name__}")
    raise RuntimeError(f"could not decode {path} ({'; '.join(methods_tried)}). "
                       "Install ffmpeg (apt install ffmpeg) or a libsndfile with Opus support.")


# --------------------------------------------------------------------------
# MERT encoder (lazy)
# --------------------------------------------------------------------------

class MertEncoder:
    """MERT encoder with lazy weight loading (weights only load on first embed())."""

    def __init__(self, model_id: str = DEFAULT_MERT_ID, device: str = "cpu",
                 fp16: bool = False, local_files_only: bool = False):
        self.model_id = model_id
        self.device_name = device
        self.fp16 = fp16
        self.local_files_only = local_files_only
        self._fe = None
        self._model = None
        self._torch = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)
        try:
            import torch
        except Exception as exc:
            self._load_error = f"torch is not importable ({exc}); cannot run MERT."
            raise RuntimeError(self._load_error) from exc
        try:
            from transformers import AutoFeatureExtractor, AutoModel  # type: ignore
        except Exception as exc:
            self._load_error = f"transformers is not importable ({exc}); cannot run MERT."
            raise RuntimeError(self._load_error) from exc
        self._torch = torch
        device = torch.device(self.device_name)
        kwargs = {"local_files_only": self.local_files_only}
        fe = None
        for loader in ("AutoFeatureExtractor", "Wav2Vec2FeatureExtractor"):
            try:
                mod = __import__("transformers", fromlist=[loader])
                fe = getattr(mod, loader).from_pretrained(self.model_id, **kwargs)
                break
            except Exception as exc:
                last_fe_exc = exc
                fe = None
                if loader == "AutoFeatureExtractor":
                    continue
                last_fe_exc = exc
        model = None
        last_exc: Exception | None = None
        if fe is not None:
            for loader in ("AutoModel", "Wav2Vec2Model"):
                try:
                    mod = __import__("transformers", fromlist=[loader])
                    model = getattr(mod, loader).from_pretrained(self.model_id, **kwargs)
                    break
                except Exception as exc:
                    last_exc = exc
                    model = None
        if fe is None or model is None:
            detail = repr(last_exc) if last_exc is not None else repr(last_fe_exc)
            self._load_error = (
                f"could not load MERT weights '{self.model_id}' ({detail}). "
                "First run on the laptop downloads ~360 MB (FP32) from Hugging Face "
                "(set HF_TOKEN if needed); subsequent runs use the local cache. "
                "Use --smoke to test the embedding/FAD logic without weights."
            )
            raise RuntimeError(self._load_error)
        model.eval()
        if device.type == "cuda" and self.fp16:
            try:
                model = model.half()
            except Exception as exc:
                log(f"WARN: fp16 was requested but model.half() failed ({exc}); staying fp32.")
        model.to(device)
        self._fe = fe
        self._model = model

    def embed(self, waves: list[np.ndarray], batch_size: int) -> np.ndarray:
        self._load()
        torch = self._torch
        device = torch.device(self.device_name)
        out: list[np.ndarray] = []
        for start in range(0, len(waves), batch_size):
            chunk = waves[start:start + batch_size]
            batch = self._fe(
                [np.asarray(w, dtype=np.float32) for w in chunk],
                sampling_rate=MERT_SR, return_tensors="pt", padding=True,
            )
            input_values = batch["input_values"].to(device)
            if device.type == "cuda" and self.fp16:
                input_values = input_values.half()
            lengths = [len(w) for w in chunk]
            max_len = max(lengths)
            with torch.no_grad():
                hidden = self._model(input_values).last_hidden_state
            hidden = hidden.float().cpu().numpy()
            T = hidden.shape[1]
            rows = []
            for i, n in enumerate(lengths):
                steps = max(1, int(round(T * n / max(1, max_len))))
                vec = hidden[i, :steps].mean(axis=0)
                rows.append(vec)
            out.append(np.stack(rows))
        E = np.concatenate(out, axis=0).astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E


# --------------------------------------------------------------------------
# Optional secondary backbone (OpenL3 / PANNs) — best effort only
# --------------------------------------------------------------------------

def build_alt_encoder(sr: int = MERT_SR):
    """Return (name, fn(wave)->vec) or (None, reason). Never raises."""
    try:
        import openl3  # type: ignore

        def _openl3_embed(wave: np.ndarray) -> np.ndarray:
            emb, _meta = openl3.get_audio_embedding(np.asarray(wave, dtype=np.float32), sr, content_type="music")
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 2:
                emb = emb.mean(axis=0)
            return emb / (np.linalg.norm(emb) + 1e-9)

        return "openl3-music", _openl3_embed
    except Exception as exc:
        openl3_note = f"openl3 unavailable ({exc})"
    try:
        from panns_inference import AudioTagging  # type: ignore
        import torch

        tagger = AudioTagging(checkpoint_path=None, device="cpu")
        tagger.eval()

        @torch.no_grad()
        def _panns_embed(wave: np.ndarray) -> np.ndarray:
            x = torch.as_tensor(np.asarray(wave, dtype=np.float32)).unsqueeze(0)
            out, _ = tagger(x)
            vec = np.asarray(out, dtype=np.float32).reshape(-1)
            return vec / (np.linalg.norm(vec) + 1e-9)

        return "panns-cnn14", _panns_embed
    except Exception as exc:
        return None, f"{openl3_note}; panns-inference unavailable ({exc})"


# --------------------------------------------------------------------------
# PCA (sklearn preferred, numpy SVD fallback)
# --------------------------------------------------------------------------

def fit_pca(X: np.ndarray, n_comp: int) -> tuple[dict, object]:
    """Fit PCA on rows of X. Returns (serializable_info, transform_fn)."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    n_comp = int(min(n_comp, n - 1 if n > 1 else 1, d))
    n_comp = max(n_comp, 1)
    mean = X.mean(axis=0)
    method = "numpy-svd"
    components = None
    evr = None
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_comp)
        pca.fit(X)
        components = pca.components_
        evr = pca.explained_variance_ratio_
        method = "sklearn"
    except Exception:
        Xc = X - mean
        _u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        components = vt[:n_comp]
        var = s ** 2 / max(n - 1, 1)
        evr = var[:n_comp] / (var.sum() + 1e-12)
    components = np.asarray(components, dtype=np.float64)

    def transform(Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        Z = (Y - mean) @ components.T
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z.astype(np.float32)

    info = {
        "n_components_in": int(d),
        "n_components_out": int(n_comp),
        "mean": np.round(mean, 6).tolist(),
        "components": np.round(components, 6).tolist(),
        "explained_variance_ratio": np.round(evr, 6).tolist(),
        "fitted_on": f"real rows (n={n})",
        "method": method,
    }
    return info, transform


# --------------------------------------------------------------------------
# Diversity / museval / manifest helpers
# --------------------------------------------------------------------------

def diversity_stats(E: np.ndarray) -> dict:
    """Mean pairwise cosine similarity & L2 over normalized rows (n, d)."""
    E = np.asarray(E, dtype=np.float32)
    n = len(E)
    if n < 2:
        return {"n": n, "mean_cosine_sim": None, "mean_pairwise_l2": None}
    G = np.clip(E @ E.T, -1.0, 1.0)
    iu = np.triu_indices(n, 1)
    cos = G[iu]
    l2 = np.sqrt(np.clip(2.0 - 2.0 * cos, 0.0, None))
    return {
        "n": int(n),
        "mean_cosine_sim": round(float(cos.mean()), 6),
        "mean_pairwise_l2": round(float(l2.mean()), 6),
    }


def museval_note() -> str:
    """SDR/SIR/SAR on a synthetic stem pair via museval/bsseval, if present."""
    bss_eval = None
    import_name = None
    try:
        from museval.metrics import bss_eval as museval_bss_eval

        bss_eval, import_name = museval_bss_eval, "museval"
    except Exception:
        try:
            from bsseval import bss_eval as bsseval_bss_eval

            bss_eval, import_name = bsseval_bss_eval, "bsseval"
        except Exception:
            bss_eval = None
    if bss_eval is None:
        return "museval/bsseval not installed; stem-QC SDR/SIR/SAR skipped (pip install museval on the laptop)"
    try:
        t = np.arange(MERT_SR * 4) / MERT_SR
        ref = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)]).astype(np.float32)
        rng = np.random.default_rng(0)
        est = ref.copy()
        est[0] += (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
        est[1] = np.roll(est[1], 32)
        sdr, sir, sar = bss_eval(ref, est, MERT_SR)
        sdr = np.asarray(sdr, dtype=float)
        return (f"{import_name} synthetic stem pair: SDR mean={float(np.nanmean(sdr)):.2f} dB "
                f"(per-source {np.round(sdr.reshape(-1), 2).tolist()}); "
                f"SIR mean={float(np.nanmean(np.asarray(sir, dtype=float))):.2f} dB; "
                f"SAR mean={float(np.nanmean(np.asarray(sar, dtype=float))):.2f} dB")
    except Exception as exc:
        return f"{import_name} importable but bss_eval failed ({exc}); skipped"


def manifest_rows(manifest_path: Path | None, kits_dir: Path, limit: int | None) -> tuple[list[dict], list[str]]:
    """Flatten the manifest to capped (phrase_id, stem, file) rows."""
    if manifest_path is None or not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path} (pass --manifest, or run with --smoke)")
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict] = []
    notes: list[str] = []
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            pid = phrase.get("id")
            stems = phrase.get("stems", {})
            if not pid:
                continue
            for stem in STEMS:
                info = stems.get(stem) or {}
                rel = info.get("file")
                if not rel:
                    notes.append(f"no file listed for {pid}/{stem}; skipped")
                    continue
                rows.append({"phrase_id": pid, "stem": stem, "path": kits_dir / rel})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"no embeddable rows in {manifest_path}")
    return rows, notes


def resolve_device(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def rlist(v: np.ndarray) -> list:
    return np.round(np.asarray(v, dtype=np.float64), 6).tolist()


def embed_and_score(args, rows: list[dict], embed_fn, synth_by_stem: dict[str, np.ndarray],
                    notes: list[str], alt_name: str | None = None,
                    alt_embs: dict[str, np.ndarray] | None = None) -> dict:
    """Shared PCA -> FAD -> diversity -> JSON assembly for real & smoke runs.

    rows carry 'phrase_id'/'stem'/'vec' (full-dim, pre-PCA). synth_by_stem
    holds full-dim synth probe embeddings per stem.
    """
    ids = [r["phrase_id"] for r in rows]
    stems = [r["stem"] for r in rows]
    X = np.stack([r["vec"] for r in rows]).astype(np.float32)
    if not np.isfinite(X).all():
        bad = int((~np.isfinite(X).all(axis=1)).sum())
        raise SystemExit(f"{bad} non-finite embedding rows; refusing to write output")
    n_fit = min(len(X), int(args.pca_sample))
    step = max(1, len(X) // n_fit)
    fit_rows = X[::step][:n_fit]
    pca_info, pca_transform = fit_pca(fit_rows, int(args.pca_dim))
    Z = pca_transform(X)
    if not np.isfinite(Z).all():
        raise SystemExit("non-finite PCA-reduced embeddings; refusing to write output")
    synth_Z = {s: pca_transform(np.asarray(synth_by_stem[s], dtype=np.float32))
               for s in STEMS if s in synth_by_stem and len(synth_by_stem[s])}

    embeddings: dict[str, dict[str, list]] = {}
    for i, r in enumerate(rows):
        embeddings.setdefault(ids[i], {})[stems[i]] = rlist(Z[i])

    per_stem: dict[str, dict] = {}
    for s in STEMS:
        idx = [i for i, st in enumerate(stems) if st == s]
        real = Z[idx] if idx else np.zeros((0, Z.shape[1]), np.float32)
        synth = synth_Z.get(s, np.zeros((0, Z.shape[1]), np.float32))
        entry: dict = {"n_real": int(len(real)), "n_synth": int(len(synth)),
                       "fad": None, "self_fad": None, "torchmetrics_fad": None}
        if len(real) and len(synth):
            entry["fad"] = round(frechet_distance(real, synth), 6)
        if len(real):
            entry["self_fad"] = round(frechet_distance(real, real), 6)  # sanity: ~0
        if len(real) and len(synth):
            tm_val, tm_note = try_torchmetrics_fad(real, synth, MERT_SR)
            entry["torchmetrics_fad"] = round(tm_val, 6) if tm_val is not None else None
            if s == STEMS[0]:
                notes.append(tm_note)
        per_stem[s] = entry
    fads = [v["fad"] for v in per_stem.values() if v["fad"] is not None]
    fad_block = {
        "per_stem": per_stem,
        "mean_fad": round(float(np.mean(fads)), 6) if fads else None,
        "self_fad_max": round(max([v["self_fad"] for v in per_stem.values() if v["self_fad"] is not None], default=0.0), 6),
        "note": "FAD computed on PCA-reduced embeddings; self_fad (real vs real) is the sanity check and must be ~0",
    }
    diversity = {"per_stem": {}}
    for s in STEMS:
        idx = [i for i, st in enumerate(stems) if st == s]
        diversity["per_stem"][s] = diversity_stats(Z[idx]) if idx else {"n": 0, "mean_cosine_sim": None, "mean_pairwise_l2": None}

    out: dict = {
        "version": VERSION,
        "model_ids": {"mert": (None if args.smoke else args.mert_id),
                      "openl3": alt_name if alt_name == "openl3-music" else None,
                      "panns": alt_name if alt_name == "panns-cnn14" else None,
                      "smoke": bool(args.smoke)},
        "n_embedded": int(len(rows)),
        "embeddings": embeddings,
        "pca": {"pca_dim": int(args.pca_dim), **pca_info},
        "fad": fad_block,
        "diversity": diversity,
        "notes": notes,
    }
    if alt_embs:
        out["embeddings_alt"] = {
            ids[i]: {stems[i]: rlist(alt_embs[ids[i]][stems[i]])} for i in range(len(ids))
            if ids[i] in alt_embs and stems[i] in alt_embs.get(ids[i], {})
        }
        out["embeddings_alt"]["_note"] = f"{alt_name} raw embeddings (not PCA-reduced)"
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrained 'ears' embeddings (MERT) for Wub phrase rating.")
    p.add_argument("--manifest", default="frontend/public/data/kits/kit_manifest.json",
                   help="kit_manifest.json (not required with --smoke)")
    p.add_argument("--kits-dir", default="frontend/public/data/kits",
                   help="directory that manifest 'file' paths are relative to")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--limit", type=int, default=None, help="cap the number of clips (phrase x stem)")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--batch-size", type=int, default=4,
                   help="MERT batch size (6 GB GPU guard; keep 4-8 with --fp16)")
    p.add_argument("--pca-dim", type=int, default=128, help="PCA-reduced embedding dim")
    p.add_argument("--pca-sample", type=int, default=4096, help="max rows used to fit PCA")
    p.add_argument("--synth-n", type=int, default=32, help="synth probe clips per stem (FAD baseline)")
    p.add_argument("--max-seconds", type=float, default=10.0, help="truncate clips to this many seconds")
    p.add_argument("--mert-id", default=DEFAULT_MERT_ID, help="HF repo for MERT (mirror: m-a-p/MERT-v1-95M)")
    p.add_argument("--fp16", action="store_true",
                   help="fp16 model on CUDA (RTX 4050 laptop); default fp32")
    p.add_argument("--local-files-only", action="store_true", help="use HF cache only (no downloads)")
    p.add_argument("--smoke", action="store_true",
                   help="synthetic audio, no weights, verifies embedding + FAD math")
    p.add_argument("--seed", type=int, default=0, help="synthetic/synth baseline seed")
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    if args.smoke and args.limit is None:
        args.limit = 8
    args.device = resolve_device(args.device)
    return args


def run_smoke(args) -> dict:
    notes = ["--smoke: synthetic sine/noise clips, no audio files, no model weights loaded.",
             f"seed={args.seed}; synth baseline n_synth={args.synth_n} per stem."]
    n_phrases = math.ceil(args.limit / len(STEMS))
    rows = []
    for pi in range(n_phrases):
        pid = f"smoke_p{pi + 1:02d}"
        for si, stem in enumerate(STEMS):
            if len(rows) >= args.limit:
                break
            rng = np.random.default_rng(args.seed * 10000 + pi * 101 + si)
            wave = synth_wave(stem, rng)
            rows.append({"phrase_id": pid, "stem": stem, "vec": smoke_feature(wave)})
    synth_by_stem: dict[str, np.ndarray] = {}
    for si, stem in enumerate(STEMS):
        vecs = []
        for k in range(args.synth_n):
            rng = np.random.default_rng(args.seed * 99991 + si * 1009 + k)
            vecs.append(smoke_feature(synth_wave(stem, rng)))
        synth_by_stem[stem] = np.stack(vecs).astype(np.float32)
    return embed_and_score(args, rows, None, synth_by_stem, notes)


def run_real(args) -> dict:
    notes = ["MERT mean-pooled last hidden state, L2-normalized, then shared PCA; FAD on reduced embeddings.",
             f"device={args.device}, batch_size={args.batch_size}, fp16={bool(args.fp16)}, max_seconds={args.max_seconds}."]
    rows_spec, manifest_notes = manifest_rows(Path(args.manifest), Path(args.kits_dir), args.limit)
    notes.extend(manifest_notes)
    if args.device == "cuda" and not args.fp16:
        notes.append("running fp32 on CUDA (default); pass --fp16 on the RTX 4050 to halve activation memory")
    encoder = MertEncoder(args.mert_id, args.device, args.fp16, args.local_files_only)
    alt_name, alt_fn = build_alt_encoder()
    if alt_name is None:
        notes.append(alt_fn)
        alt_fn = None
    else:
        notes.append(f"secondary FAD backbone active: {alt_name} (stored in embeddings_alt)")

    t0 = time.time()
    rows: list[dict] = []
    alt_embs: dict[str, dict[str, np.ndarray]] = {}
    decode_methods: dict[str, int] = {}
    missing = 0
    # stream: load + embed in batches so 10k clips never sit in RAM as audio
    batch: list[dict] = []

    alt_fn_holder: list = [alt_fn]  # mutable so flush() can disable it on failure

    def flush(batch_items: list[dict]) -> None:
        if not batch_items:
            return
        waves = [it["wave"] for it in batch_items]
        E = encoder.embed(waves, batch_size=args.batch_size)
        for it, vec in zip(batch_items, E):
            rows.append({"phrase_id": it["phrase_id"], "stem": it["stem"], "vec": vec})
            if alt_fn_holder[0] is not None:
                try:
                    alt_embs.setdefault(it["phrase_id"], {})[it["stem"]] = alt_fn_holder[0](it["wave"])
                except Exception as exc:
                    notes.append(f"alt backbone failed on {it['phrase_id']}/{it['stem']} ({exc}); alt disabled for that clip")
                    alt_fn_holder[0] = None

    max_samples = int(args.max_seconds * MERT_SR)
    for spec in rows_spec:
        try:
            wave, method = load_wave(spec["path"])
        except Exception as exc:
            log(f"WARN: skipping {spec['phrase_id']}/{spec['stem']}: {exc}")
            missing += 1
            continue
        decode_methods[method] = decode_methods.get(method, 0) + 1
        if len(wave) > max_samples:
            wave = wave[:max_samples]
        if len(wave) < 1600:
            log(f"WARN: skipping {spec['phrase_id']}/{spec['stem']}: clip too short ({len(wave)} samples)")
            missing += 1
            continue
        batch.append({"phrase_id": spec["phrase_id"], "stem": spec["stem"], "wave": wave})
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            done = len(rows)
            if done % 500 == 0:
                log(f"embedded {done}/{len(rows_spec)} clips in {time.time() - t0:.1f}s")
    flush(batch)
    alt_enabled = alt_fn_holder[0] is not None
    if rows_spec and missing:
        notes.append(f"{missing} clips skipped (missing/undecodable/short); n_embedded counts only embedded clips")
    notes.append(f"audio decode methods: {decode_methods}")

    # synth probe baseline through the same MERT encoder (n_synth per stem)
    synth_by_stem: dict[str, np.ndarray] = {}
    for si, stem in enumerate(STEMS):
        waves = []
        for k in range(args.synth_n):
            rng = np.random.default_rng(args.seed * 99991 + si * 1009 + k)
            waves.append(synth_wave(stem, rng))
        synth_by_stem[stem] = encoder.embed(waves, batch_size=args.batch_size)

    log(f"embedded {len(rows)} clips in {time.time() - t0:.1f}s")
    return embed_and_score(args, rows, encoder, synth_by_stem, notes,
                           alt_name=alt_name if alt_enabled else None,
                           alt_embs=alt_embs if alt_enabled else None)


def main() -> int:
    args = parse_args()
    notes_extra = museval_note()
    try:
        out = run_smoke(args) if args.smoke else run_real(args)
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        return 2
    if out["fad"]["self_fad_max"] is None or out["fad"]["self_fad_max"] > 1e-3:
        log(f"WARN: self-FAD sanity check above 1e-3 (self_fad_max={out['fad']['self_fad_max']})")
    out["notes"] = [*out["notes"], notes_extra]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(out, allow_nan=False))
    except (ValueError, TypeError) as exc:
        log(f"ERROR: output contains non-JSON value ({exc}); refusing to write")
        return 1
    per = out["fad"]["per_stem"]
    print(f"n_embedded={out['n_embedded']} mean_fad={out['fad']['mean_fad']} "
          f"self_fad_max={out['fad']['self_fad_max']} "
          f"pca={out['pca']['n_components_in']}->{out['pca']['n_components_out']} "
          f"mode={'smoke' if args.smoke else 'real'}")
    for s in STEMS:
        e = per[s]
        print(f"  {s}: fad={e['fad']} self_fad={e['self_fad']} n_real={e['n_real']} n_synth={e['n_synth']}")
    print(f"finite_check=pass (allow_nan=False single pass) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
