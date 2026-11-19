"""Train an audio-level song-ness judge from real stems, then distill to browser.

Teacher sees rendered four-stem audio: log-mel of the real mix plus per-stem
RMS/centroid envelopes. Positives are real song windows; negatives replace two
stems with stems from a similar-tempo song. The teacher output then labels a
compact analysis-feature student (song_critic.json) that the browser can run.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from wub_core.section_arcs import bar_features, segment_sections
from train_song_critic import song_data, window, features as analysis_features

SR = 16000
MELS = 64
FRAMES = 16
AUDIO_DIM = FRAMES * MELS + 4 * 32 + 12  # 1164
ANALYSIS_DIM = 218
HIDDEN = 128


def resample(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=SR)


def mel_frames(y: np.ndarray) -> np.ndarray:
    import librosa
    s = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    mel = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=MELS)
    m = np.log1p(1000 * mel @ s)
    if m.shape[1] < FRAMES:
        m = np.pad(m, ((0, 0), (0, FRAMES - m.shape[1])))
    idx = np.linspace(0, m.shape[1] - 1, FRAMES).astype(int)
    return m[:, idx].T.astype(np.float32)  # 16 x 64


def stem_envelope(y: np.ndarray) -> np.ndarray:
    hop = max(1, len(y) // FRAMES)
    rms = np.array([float(np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2) + 1e-10))
                    for i in range(FRAMES)], dtype=np.float32)
    centroid = np.zeros(FRAMES, dtype=np.float32)
    spec = np.abs(np.fft.rfft(y[:min(len(y), 8192)]))
    freqs = np.fft.rfftfreq(min(len(y), 8192), 1 / SR)
    c = float((spec * freqs).sum() / (spec.sum() + 1e-8)) / (SR / 2)
    centroid[:] = c
    return np.concatenate([np.clip(rms / (rms.max() + 1e-8), 0, 1), centroid])


def audio_features(stems: List[np.ndarray]) -> np.ndarray:
    mix = np.sum(stems, axis=0)
    v = list(mel_frames(mix).reshape(-1))
    for stem in stems:
        v.extend(stem_envelope(stem).tolist())
    import librosa
    chroma = librosa.feature.chroma_stft(y=mix, sr=SR, n_fft=1024, hop_length=512)
    v.extend(chroma.mean(axis=1).astype(np.float32).tolist())
    assert len(v) == AUDIO_DIM, len(v)
    return np.asarray(v, dtype=np.float32)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(AUDIO_DIM, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ANALYSIS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


WINDOW_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def windows_for(tid: str, pproc: Path, bpm: float, key: str, count: int):
    if tid not in WINDOW_CACHE:
        bars = song_data(json.loads(pproc.read_text()))
        if not bars:
            WINDOW_CACHE[tid] = []
        else:
            WINDOW_CACHE[tid] = [
                window(bars[c * 4:c * 4 + 4], bpm, key)
                for c in range(min(count, len(bars) // 4))
            ]
    return WINDOW_CACHE[tid]


def load_tracks(stems_dir: Path, processed: Path, max_tracks: int):
    paths = []
    for p in sorted(processed.glob("*.json")):
        d = stems_dir / p.stem
        if all((d / f"{s}.wav").exists() for s in ("drums", "bass", "other", "vocals")):
            paths.append((p.stem, p, d))
    if max_tracks > 0:
        paths = paths[:max_tracks]
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", type=Path, required=True)
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--student-out", type=Path, required=True)
    ap.add_argument("--max-tracks", type=int, default=300)
    ap.add_argument("--windows-per-track", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    torch.manual_seed(413)
    np.random.seed(413)
    random.seed(413)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracks = load_tracks(args.stems, args.processed, args.max_tracks)

    ax, ax_an, ay, at = [], [], [], []
    for tid, pproc, d in tracks:
        try:
            doc = json.loads(pproc.read_text())
            bpm = float(doc.get("bpm") or 120)
            bar4 = 960.0 / bpm
            secs = min(120.0, args.windows_per_track * bar4)
            waves = []
            for stem in ("drums", "bass", "other", "vocals"):
                path = d / f"{stem}.wav"
                native_sr = sf.info(str(path)).samplerate
                y, sr = sf.read(str(path), frames=int(secs * native_sr),
                                dtype="float32", always_2d=True)
                waves.append(resample(y.mean(axis=1), sr))
            bars = song_data(doc)
            if not bars:
                continue
            wins = [window(bars[c * 4:c * 4 + 4], bpm, str(doc.get("key") or ""))
                    for c in range(min(args.windows_per_track, len(bars) // 4))]
        except Exception as exc:
            import traceback
            print("skip", tid, repr(exc), flush=True)
            traceback.print_exc()
            continue
        for i, w in enumerate(wins):
            start = int(i * bar4 * SR)
            end = start + int(bar4 * SR)
            if end > min(len(y) for y in waves):
                continue
            chunk = [y[start:end] for y in waves]
            ax.append(audio_features(chunk))
            ax_an.append(analysis_features(w))
            ay.append(1.0)
            at.append(tid)
            # Cross-song negative: two replaced stems from a similar song.
            donor_tid, donor_proc, donor_dir = random.choice(tracks)
            try:
                donor_sr = sf.info(str(donor_dir / "other.wav")).samplerate
                dy, sr = sf.read(str(donor_dir / "other.wav"),
                                 frames=int((i + 1) * bar4 * donor_sr) + 1,
                                 dtype="float32", always_2d=True)
                dy = resample(dy.mean(axis=1), sr)
                dstart = int(i * bar4 * SR)
                dend = dstart + int(bar4 * SR)
                if dend <= len(dy):
                    swapped = random.sample([0, 1, 2, 3], 2)
                    mixed = list(chunk)
                    donor_wins = windows_for(donor_tid, donor_proc,
                                             float(json.loads(donor_proc.read_text()).get("bpm") or 120),
                                             "", args.windows_per_track)
                    dw = donor_wins[i % len(donor_wins)] if donor_wins else None
                    w_neg = dict(w)
                    w_neg["pattern"] = dict(w["pattern"])
                    w_neg["chroma"] = dict(w["chroma"])
                    w_neg["mask"] = dict(w["mask"])
                    for s in swapped:
                        mixed[s] = dy[dstart:dend]
                        if dw is not None:
                            stem = ("drums", "bass", "other", "vocals")[s]
                            w_neg["pattern"][stem] = dw["pattern"][stem]
                            w_neg["chroma"][stem] = dw["chroma"][stem]
                            w_neg["mask"][stem] = dw["mask"][stem]
                    ax.append(audio_features(mixed))
                    ax_an.append(analysis_features(w_neg))
                    ay.append(0.0)
                    at.append(tid)
            except Exception:
                pass

    if len(ax) < 500:
        raise SystemExit(f"not enough audio windows: {len(ax)}")
    X = np.stack(ax).astype(np.float32)
    A = np.stack(ax_an).astype(np.float32)
    y = np.asarray(ay, dtype=np.float32)
    cache = Path("/tmp/audio_judge_cache.npz")
    if cache.exists():
        with np.load(cache) as z:
            X, A, y, at = z["X"], z["A"], z["y"], list(z["at"])
    else:
        np.savez_compressed(cache, X=X, A=A, y=y, at=np.asarray(at))
    unique = sorted(set(at))
    random.shuffle(unique)
    val_tracks = set(unique[:max(1, len(unique) // 10)])
    val = np.asarray([t in val_tracks for t in at], dtype=bool)
    tr = ~val
    print(f"windows={len(y)} real={int((y>0.5).sum())} neg={int((y<0.5).sum())} "
          f"tracks={len(unique)} val_tracks={len(val_tracks)}", flush=True)

    teacher = Teacher().to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=0.0015)
    mean = X[tr].mean(axis=0); std = X[tr].std(axis=0); std[std < 1e-4] = 1
    xt = torch.tensor((X[tr] - mean) / std, dtype=torch.float32, device=device)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=device)
    for _ in range(args.epochs):
        teacher.train()
        perm = torch.randperm(xt.shape[0], device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                teacher(xt[idx]).squeeze(1), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    teacher.eval()
    with torch.no_grad():
        xv = torch.tensor((X[val] - mean) / std, dtype=torch.float32, device=device)
        val_logit = teacher(xv).squeeze(1).cpu().numpy()
        all_logit = teacher(torch.tensor((X - mean) / std, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
    auc = float(roc_auc_score(y[val], val_logit))

    # Distill teacher soft labels into the 218-d analysis student.
    amean = A[tr].mean(axis=0); astd = A[tr].std(axis=0); astd[astd < 1e-4] = 1
    student = Student().to(device)
    opt2 = torch.optim.Adam(student.parameters(), lr=0.001)
    labels = torch.tensor(1 / (1 + np.exp(-np.clip(all_logit, -30, 30))), dtype=torch.float32, device=device)
    ast = torch.tensor((A[tr] - amean) / astd, dtype=torch.float32, device=device)
    ytr = torch.tensor(y[tr], dtype=torch.float32, device=device)
    for _ in range(max(30, args.epochs * 2)):
        student.train()
        perm = torch.randperm(ast.shape[0], device=device)
        for s in range(0, perm.numel(), 256):
            idx = perm[s:s + 256]
            # Hard real/cross-song labels: teacher soft scores were compressed
            # near 0.5 and a plain MSE collapsed the student to the mean.
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                student(ast[idx]).squeeze(1), ytr[idx])
            opt2.zero_grad(); loss.backward(); opt2.step()
    student.eval()
    with torch.no_grad():
        av = torch.tensor((A[val] - amean) / astd, dtype=torch.float32, device=device)
        sv = student(av).squeeze(1).cpu().numpy()
    student_auc = float(roc_auc_score(y[val], sv))

    def layer(mod):
        return {"w": [[round(float(v), 5) for v in row] for row in mod.weight.detach().cpu().numpy()],
                "b": [round(float(v), 5) for v in mod.bias.detach().cpu().numpy()]}

    teacher_doc = {
        "format": "wub-audio-judge-v1", "version": 1,
        "input_dim": AUDIO_DIM, "hidden": HIDDEN,
        "mean": [round(float(v), 5) for v in mean],
        "std": [round(float(v), 5) for v in std],
        "layers": [layer(teacher.net[0]), layer(teacher.net[2]), layer(teacher.net[4])],
        "report": {"windows": len(y), "tracks": len(unique), "val_auc": round(auc, 4),
                   "device": str(device), "epochs": args.epochs},
    }
    student_doc = {
        "format": "wub-song-critic-v1", "version": 2,
        "input_dim": ANALYSIS_DIM, "hidden": 64,
        "mean": [round(float(v), 5) for v in amean],
        "std": [round(float(v), 5) for v in astd],
        "layers": [layer(student.net[0]), layer(student.net[2]), layer(student.net[4])],
        "report": {
            "source": "distilled from audio judge trained on real stems",
            "windows": len(y), "tracks": len(unique), "train_tracks": len(unique) - len(val_tracks),
            "val_tracks": len(val_tracks), "teacher_val_auc": round(auc, 4),
            "val_auc": round(student_auc, 4),
            "real_score_p10": round(float(np.percentile(
                labels.detach().cpu().numpy()[y > 0.5], 10)), 4),
            "epochs": args.epochs, "device": str(device),
            "use": "bounded final-gate adjustment; never schedules or renders audio",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.student_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(teacher_doc, separators=(",", ":")) + "\n")
    args.student_out.write_text(json.dumps(student_doc, separators=(",", ":")) + "\n")
    print(json.dumps({"teacher_auc": round(auc, 4), "student_auc": round(student_auc, 4)}, indent=2))


if __name__ == "__main__":
    main()
