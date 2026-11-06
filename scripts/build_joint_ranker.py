"""Build the joint four-stem transition ranker.

The player can already mix stems, but the one-stem morph scored each stem in
isolation.  This script learns the **joint** transition: it scores a candidate
four-stem mix against the current mix and trains a small student that can run
in the browser from per-phrase features.

The shipped player follows a 4 -> 8 -> 16-bar arc.  A block is a run of
consecutive 4-bar phrases, and held stems continue to their real next phrase
(or loop when the source run ends).  Training therefore includes:

* 4-, 8- and 16-bar current / next block lengths
* real-run boundaries (last phrase -> first phrase of the next run)
* held-stem continuation (last phrase -> real next phrase, else run start)
* 1- and 2-stem donor swaps against that continuation

Teacher
-------
The mix-level audio judge (`data/audio_judge.pt`, val AUC 0.951) scores
`(current mix, candidate mix)` for real consecutive 4-bar states and for
candidate 1-/2-stem swaps.  Its logit is the target.

Student
-------
For every phrase we build a 424-d joint state:

    4 stems x (90 per-stem features) + 64-d mix embedding

    90 = 16 onset bits, 12 chroma, 12 root, energy, 5 section one-hot,
         stem RMS, run position, run length, has-next, key shift, tempo fit,
         onset density, offbeat ratio, backbeat ratio, syncopation,
         vocal presence, vocal crest, 32-d per-stem timbre

The four stem chunks are **interleaved**: drums 0:90, bass 90:180,
other 180:270, vocals 270:360, then the 64-d mix embedding at 360:424.

The transition input is:

    prev 424, current 424, candidate 424, candidate-current delta,
    current-prev delta, 4-bit change mask + count, 6-d target section/energy,
    3-d current block one-hot (4/8/16), 3-d next block one-hot (4/8/16)

(or 2,137 numbers).  A 3-layer MLP predicts the teacher logit.  A held-out
track split reports real-vs-swap AUC for both teacher and student, overall and
per block length.  Hold thresholds are calibrated per block length.

Run on the laptop:
    PYTHONPATH=. python3 scripts/build_joint_ranker.py \
      --manifest kits_flow/kit_manifest.json \
      --dataset /tmp/wubtrain/taste_dataset_flow.json \
      --bars data/stem_bars_flow --mix-judge data/audio_judge.pt \
      --out data/joint_ranker --device cuda --epochs 30

Add ``--dump-val /tmp/joint_val.npz`` to save the raw held-out features,
labels and teacher targets for comparing multiple runs on a common split.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_judge_model import AudioJudge  # noqa: E402

STEMS = ("drums", "bass", "other", "vocals")
SEC_IDX = {"INTRO": 0, "GROOVE_A": 1, "BUILDUP": 2, "DROP": 3, "BREAKDOWN": 4}
SEC_ENERGY = {"INTRO": 0.25, "GROOVE_A": 0.83, "BUILDUP": 0.65, "DROP": 0.90, "BREAKDOWN": 0.20}
STEM_DIM = 90
STATE_DIM = 4 * STEM_DIM + 64
MIX_OFFSET = 4 * STEM_DIM
BLOCK_BARS = (4, 8, 16)
BLOCK_INDEX = {bars: index for index, bars in enumerate(BLOCK_BARS)}
INPUT_DIM = STATE_DIM * 5 + 5 + 6 + 6
SUBSETS = [(s,) for s in STEMS] + [
    (a, b) for i, a in enumerate(STEMS) for b in STEMS[i + 1:]
]
# Current/next block pairs seen in the arc.  4->4 and 8->8 dominate; the
# one-off 4->8, 8->16 and 16->4 boundaries are included so the ranker sees
# every hand-off in the set.
ARC_PAIRS = ((1, 1), (2, 2), (1, 2), (2, 4), (4, 1))

PITCH = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
         "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
CAMELOT_MAJOR = [8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1]
CAMELOT_MINOR = [5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10]


def parse_key(name: str):
    import re
    m = re.match(r"^([A-Ga-g])([#b♯♭]?)\s*(major|minor|maj|min)$", name.strip())
    if not m:
        return None
    acc = "#" if m.group(2) == "♯" else ("B" if m.group(2) == "♭" else m.group(2).upper())
    root = PITCH.get(f"{m.group(1).upper()}{acc}")
    if root is None:
        return None
    return root, m.group(3).lower().startswith("min")


def camelot(name: str):
    p = parse_key(name)
    if p is None:
        return None
    root, minor = p
    return (CAMELOT_MINOR if minor else CAMELOT_MAJOR)[root], "A" if minor else "B"


def key_shift(session_key: str, phrase_key: str) -> float:
    a = parse_key(session_key)
    b = parse_key(phrase_key)
    ca = camelot(session_key)
    cb = camelot(phrase_key)
    if not a or not b or not ca or not cb:
        return 0.0
    if ca[0] == cb[0]:
        return 0.0
    if ca[1] != cb[1]:
        return 0.0
    dist = min(abs(ca[0] - cb[0]), 12 - abs(ca[0] - cb[0]))
    if dist != 1:
        return 0.0
    shift = (b[0] - a[0] + 12) % 12
    return shift - 12 if shift > 6 else shift


def rot12(vec, shift: int):
    out = [0.0] * 12
    for i, v in enumerate(vec):
        out[(i + shift) % 12] = float(v)
    return out


class Student(nn.Module):
    def __init__(self, d: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Dropout(0.10),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, teacher: np.ndarray):
        self.x = x
        self.y = y
        self.teacher = teacher

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.x[i]),
            torch.tensor(float(self.y[i])),
            torch.tensor(float(self.teacher[i])),
        )


def load_data(manifest_path: Path, dataset_path: Path, bars_dir: Path):
    manifest = json.loads(manifest_path.read_text())
    dataset = json.loads(dataset_path.read_text())
    rows = {row["phrase_id"]: row for row in dataset.get("kit_phrases", [])}
    index = json.loads((bars_dir / "drums" / "bar_index.json").read_text())
    # All stem datasets share the same phrase order.
    phrase_ids = [str(p) for p in index["phrase_id"]]
    row_of = {pid: i for i, pid in enumerate(phrase_ids)}
    mels = {}
    for stem in STEMS:
        mels[stem] = np.load(bars_dir / stem / "mel.npy", mmap_mode="r")
    return manifest, rows, row_of, mels


def phrase_duration(phrase: dict) -> float:
    return float(phrase.get("duration_sec", 0.0) or 0.0)


def duration_fits(a: dict, b: dict, tolerance: float = 0.01) -> bool:
    before = phrase_duration(a)
    after = phrase_duration(b)
    if before <= 0 or after <= 0:
        return True
    return abs(after - before) / before <= tolerance


def run_info(kit: dict, start_bar: int, tolerance: float = 0.01):
    """Run position/length for the 1%-duration run, matching the browser."""
    by_start = {int(p["start_bar"]): p for p in kit["phrases"]}
    if start_bar not in by_start:
        return 0.0, 1 / 6, 0.0
    first = by_start[start_bar]
    pos = 0
    b = start_bar
    while b - 4 in by_start and duration_fits(by_start[b - 4], first, tolerance):
        b -= 4
        pos += 1
    ln = 1
    e = start_bar
    while e + 4 in by_start and duration_fits(first, by_start[e + 4], tolerance):
        e += 4
        ln += 1
    has_next = 1.0 if e > start_bar else 0.0
    return pos / max(1, ln - 1), ln / 6.0, has_next


def build_runs(kit: dict, tolerance: float = 0.01) -> list[list[dict]]:
    """Maximal consecutive 4-bar runs with the browser's 1% tempo tolerance."""
    by_start = {int(p["start_bar"]): p for p in kit.get("phrases", [])}
    runs: list[list[dict]] = []
    for start in sorted(by_start):
        if start - 4 in by_start and duration_fits(by_start[start - 4], by_start[start], tolerance):
            continue
        run = [by_start[start]]
        bar = start
        while bar + 4 in by_start and duration_fits(run[0], by_start[bar + 4], tolerance):
            bar += 4
            run.append(by_start[bar])
        runs.append(run)
    return runs


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def stem_state_vector(
    *,
    stem: str,
    onset_bits,
    chroma,
    root,
    energy: float,
    section: int,
    rms_db: float,
    peak: float,
    rpos: float,
    rlen: float,
    has_next: float,
    key_shift_norm: float,
    tempo_rate: float,
    timbre,
) -> np.ndarray:
    """90-d per-stem phrase state, interleaved so each chunk is self-contained.

    0:16 onset bits, 16:28 chroma, 28:40 root, 40 energy,
    41:46 section one-hot, 46 stem RMS (normalised), 47 run position,
    48 run length, 49 has-next, 50 key shift / 5, 51 tempo rate * 10,
    52 onset density, 53 offbeat ratio, 54 backbeat ratio, 55 syncopation,
    56 vocal presence, 57 vocal crest, 58:90 per-stem timbre embedding.
    """
    onset = np.asarray(onset_bits, dtype=np.float32).reshape(-1)
    v = np.zeros(STEM_DIM, dtype=np.float32)
    v[0:16] = onset[:16]
    v[16:28] = chroma
    v[28:40] = root
    v[40] = energy
    v[41 + section] = 1.0
    v[46] = clamp01((rms_db + 40.0) / 30.0)
    v[47] = rpos
    v[48] = rlen
    v[49] = has_next
    v[50] = key_shift_norm
    v[51] = tempo_rate * 10.0
    count = float(onset[:16].sum())
    if count > 0:
        denom = count
        v[52] = count / 16.0
        v[53] = float(onset[[2, 6, 10, 14]].sum()) / denom
        v[54] = float(onset[[4, 12]].sum()) / denom
        v[55] = 1.0 - float(onset[[0, 4, 8, 12]].sum()) / denom
    # Silence/absent stem: density and all ratios stay 0, not 1.0 syncopation.
    if stem == "vocals":
        v[56] = clamp01((rms_db + 40.0) / 25.0)
        peak_db = 20.0 * math.log10(max(float(peak), 1e-6))
        v[57] = clamp01((peak_db - rms_db) / 24.0)
    timbre_vec = np.asarray(timbre, dtype=np.float32).reshape(-1)
    v[58:58 + min(32, timbre_vec.size)] = timbre_vec[:32]
    return v


def build_phrase_states(manifest, rows, row_of, mels, emb32):
    """Return phrase_id -> 424-d joint state, and the kit/cluster maps."""
    clusters = manifest.get("clusters") or []
    kit_cluster = {}
    for cluster in clusters:
        for tid in cluster.get("kits", []):
            kit_cluster.setdefault(tid, cluster)
    states = {}
    phrase_meta = {}
    for kit in manifest.get("kits", []):
        cluster = kit_cluster.get(kit["track_id"], {"key": kit["key"], "bpm": kit["bpm"]})
        block_sec = 960.0 / max(1e-6, float(cluster["bpm"]))
        shift = key_shift(cluster["key"], kit["key"])
        for phrase in kit.get("phrases", []):
            pid = phrase["id"]
            row = rows.get(pid)
            if row is None or pid not in row_of:
                continue
            feats = row["features"]
            onset = feats[48:112]
            bass_chroma = feats[112:124]
            other_chroma = feats[124:136]
            bass_root = feats[136:148]
            chord_root = feats[148:160]
            rpos, rlen, has_next = run_info(kit, int(phrase["start_bar"]))
            energy = float(phrase.get("energy", 0.5))
            section = SEC_IDX.get(str(phrase.get("section", "GROOVE_A")).upper(), 1)
            tempo_rate = float(phrase.get("duration_sec", block_sec)) / block_sec - 1.0
            stem_stats = phrase.get("stems") or {}
            vec = []
            # Interleave the four 90-d stem chunks; never group feature blocks.
            for si, stem in enumerate(STEMS):
                if stem == "bass":
                    chroma, root = bass_chroma, bass_root
                elif stem == "other":
                    chroma, root = other_chroma, chord_root
                elif stem == "vocals":
                    chroma, root = chord_root, chord_root
                else:
                    chroma, root = [0.0] * 12, [0.0] * 12
                stats = stem_stats.get(stem) or {}
                vec.append(stem_state_vector(
                    stem=stem,
                    onset_bits=onset[si * 16:(si + 1) * 16],
                    chroma=chroma,
                    root=root,
                    energy=energy,
                    section=section,
                    rms_db=float(stats.get("rms_db", -20.0) or -20.0),
                    peak=float(stats.get("peak", 0.0) or 0.0),
                    rpos=rpos,
                    rlen=rlen,
                    has_next=has_next,
                    key_shift_norm=shift / 5.0,
                    tempo_rate=tempo_rate,
                    timbre=emb32[stem].get(pid, [0.0] * 32),
                ))
            mix = np.asarray(
                phrase.get("embedding") or [0.0] * 64, dtype=np.float32
            ).reshape(-1)
            if mix.size < 64:
                mix = np.pad(mix, (0, 64 - mix.size))
            vec.append(mix[:64])
            state = np.concatenate(vec).astype(np.float32)
            if state.size != STATE_DIM:
                raise ValueError(f"{pid}: state has {state.size} dims, expected {STATE_DIM}")
            states[pid] = state
            phrase_meta[pid] = {
                "track_id": kit["track_id"], "kit": kit, "phrase": phrase,
                "cluster": cluster, "shift": shift, "rpos": rpos, "rlen": rlen,
                "has_next": has_next, "tempo_rate": tempo_rate,
            }
    return states, phrase_meta, kit_cluster


def mix_rows(rows: dict, mels: dict):
    power = None
    for stem in STEMS:
        db = np.asarray(mels[stem][rows[stem]], dtype=np.float32)
        p = np.power(10.0, db / 10.0, dtype=np.float32)
        power = p if power is None else power + p
    return (10.0 * np.log10(power + 1e-12)).astype(np.float32)


def state_for(states, phrase_for_stem):
    """4 x 90 interleaved stem chunks + mean of the four 64-d mix embeddings."""
    chunks = []
    emb = np.zeros(64, dtype=np.float32)
    for stem in STEMS:
        s = states[phrase_for_stem[stem]]
        si = STEMS.index(stem)
        chunks.append(s[si * STEM_DIM:(si + 1) * STEM_DIM])
        emb += s[MIX_OFFSET:MIX_OFFSET + 64]
    return np.concatenate(chunks + [emb / 4.0])


def block_one_hot(bars: int) -> np.ndarray:
    out = np.zeros(3, dtype=np.float32)
    out[BLOCK_INDEX.get(int(bars), 0)] = 1.0
    return out


def transition_vector(
    prev,
    cur,
    cand,
    changed,
    target_section,
    target_energy,
    current_bars: int = 4,
    next_bars: int = 4,
):
    mask = np.zeros(5, dtype=np.float32)
    for stem in changed:
        mask[STEMS.index(stem)] = 1.0
    mask[4] = len(changed) / 2.0
    target = np.zeros(6, dtype=np.float32)
    target[SEC_IDX.get(target_section.upper(), 1)] = 1.0
    target[5] = target_energy
    return np.concatenate([
        prev, cur, cand, cand - cur, cur - prev, mask, target,
        block_one_hot(current_bars), block_one_hot(next_bars),
    ]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="kits_flow/kit_manifest.json")
    ap.add_argument("--dataset", default="/tmp/wubtrain/taste_dataset_flow.json")
    ap.add_argument("--bars", default="data/stem_bars_flow")
    ap.add_argument("--mix-judge", default="data/audio_judge.pt")
    ap.add_argument("--encoders", default="data/stem_judge")
    ap.add_argument("--out", default="data/joint_ranker")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=505)
    ap.add_argument("--max-kits", type=int, default=0)
    ap.add_argument("--max-neg", type=int, default=2)
    ap.add_argument("--max-examples", type=int, default=120000)
    ap.add_argument("--pos-subsets", type=int, default=3)
    ap.add_argument("--neg-subsets", type=int, default=6)
    ap.add_argument(
        "--dump-val",
        default="",
        help="optional .npz path for the raw held-out features/labels/teacher targets",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest, rows, row_of, mels = load_data(
        Path(args.manifest), Path(args.dataset), Path(args.bars)
    )
    # Per-stem timbre embeddings, PCA 32, from the four trained stem judges.
    emb32 = {stem: {} for stem in STEMS}
    for stem in STEMS:
        enc_path = Path(f"{args.encoders}_{stem}.pt")
        if not enc_path.is_file():
            print(f"[joint] missing stem encoder {enc_path}; using zeros", flush=True)
            continue
        ckpt = torch.load(enc_path, map_location="cpu")
        enc = AudioJudge(emb_dim=int(ckpt.get("cfg", {}).get("emb_dim", 128)))
        enc.load_state_dict(ckpt["state_dict"])
        enc.to(device).eval()
        arr = np.load(Path(args.bars) / stem / "mel.npy", mmap_mode="r")
        out = []
        with torch.no_grad():
            for start in range(0, len(arr), 256):
                batch = np.asarray(arr[start:start + 256]).astype(np.float32)
                xb = torch.from_numpy(batch).unsqueeze(1).to(device)
                out.append(enc.encode(xb).float().cpu().numpy())
        emb = np.concatenate(out)
        pca = PCA(n_components=32, random_state=505, svd_solver="full")
        reduced = pca.fit_transform(emb)
        pids = [str(p) for p in json.loads((Path(args.bars) / stem / "bar_index.json").read_text())["phrase_id"]]
        for pid, vec in zip(pids, reduced):
            emb32[stem][pid] = vec.astype(np.float32)
        print(f"[joint] {stem} emb {emb.shape} -> 32 "
              f"(PC1 {pca.explained_variance_ratio_[0]:.3f})", flush=True)

    states, meta, kit_cluster = build_phrase_states(manifest, rows, row_of, mels, emb32)
    print(f"[joint] phrases with state: {len(states)} / {sum(len(k['phrases']) for k in manifest['kits'])}", flush=True)

    # The mix judge is the teacher.
    ckpt = torch.load(args.mix_judge, map_location="cpu")
    mix_judge = AudioJudge(emb_dim=int(ckpt.get("cfg", {}).get("emb_dim", 128)))
    mix_judge.load_state_dict(ckpt["state_dict"])
    mix_judge.to(device).eval()
    print(f"[joint] teacher mix judge val_auc={ckpt.get('val_auc')}", flush=True)

    # Run length remaining and donor pools are built inside the example
    # generator, using the same 1% duration rule as the browser.

    kits = list(manifest.get("kits", []))
    if args.max_kits:
        kits = kits[:args.max_kits]
    random.shuffle(kits)

    # Run length remaining from each phrase, matching the browser's 1% run.
    run_remaining = {}
    for kit in manifest.get("kits", []):
        for run in build_runs(kit):
            for index, phrase in enumerate(run):
                run_remaining[phrase["id"]] = len(run) - index

    # Cluster -> stem -> donor run starts.  A phrase is a legal start for any
    # stem that actually has audio on it.
    donor_pool = defaultdict(lambda: defaultdict(list))
    for pid, info in meta.items():
        phrase = info["phrase"]
        for stem in STEMS:
            if ((phrase.get("stems") or {}).get(stem) or {}).get("file"):
                donor_pool[info["cluster"]["id"]][stem].append(
                    (pid, info["track_id"], run_remaining.get(pid, 1))
                )

    def sample_donor(cluster_id, stem, exclude_tracks, min_len):
        pool = [item for item in donor_pool[cluster_id][stem]
                if item[1] not in exclude_tracks]
        good = [item for item in pool if item[2] >= min_len]
        if not good:
            good = pool
        return random.choice(good)[0] if good else None

    x_rows, y_rows, teacher_rows, track_rows, bars_rows = [], [], [], [], []
    teacher_batch_a, teacher_batch_b, teacher_meta = [], [], []
    example_target = args.max_examples if args.max_examples > 0 else 10 ** 12
    stop = False

    def example_count():
        return len(y_rows) + len(teacher_batch_a)

    def flush_teacher():
        if not teacher_batch_a:
            return
        a = torch.from_numpy(np.stack(teacher_batch_a)).float().unsqueeze(1).to(device)
        b = torch.from_numpy(np.stack(teacher_batch_b)).float().unsqueeze(1).to(device)
        with torch.no_grad():
            logits, _ = mix_judge.score_pairs(a, b, batch=256)
        logits = logits.float().cpu().numpy()
        for logit, (x, y, track, bars) in zip(logits, teacher_meta):
            x_rows.append(x)
            y_rows.append(y)
            teacher_rows.append(float(logit))
            track_rows.append(track)
            bars_rows.append(bars)
        teacher_batch_a.clear()
        teacher_batch_b.clear()
        teacher_meta.clear()

    def add_example(x, y, cur_rows, cand_rows, track, next_bars):
        teacher_batch_a.append(mix_rows(cur_rows, mels))
        teacher_batch_b.append(mix_rows(cand_rows, mels))
        teacher_meta.append((x, float(y), track, int(next_bars)))
        if len(teacher_batch_a) >= 384:
            flush_teacher()

    # Arc-aware boundary examples.  For every phrase position we walk the
    # current/next block pairs in the set (4->4, 8->8, 4->8, 8->16, 16->4)
    # and compare a real next phrase with donor swaps.
    for kit in kits:
        if stop:
            break
        first_pid = next((p["id"] for p in kit["phrases"] if p["id"] in meta), None)
        if first_pid is None:
            continue
        cluster_id = meta[first_pid]["cluster"]["id"]
        by_start = {int(p["start_bar"]): p for p in kit.get("phrases", [])}
        runs = build_runs(kit)
        for run in runs:
            for i in range(len(run)):
                if stop:
                    break
                for current_phrases, next_phrases in ARC_PAIRS:
                    if i + current_phrases > len(run):
                        continue
                    cur = run[i + current_phrases - 1]
                    prev = run[i - 1] if i > 0 else by_start.get(
                        int(run[0]["start_bar"]) - 4, cur
                    )
                    next_phrase = run[i + current_phrases] if i + current_phrases < len(run) else run[i]
                    if (cur["id"] not in states or prev["id"] not in states
                            or next_phrase["id"] not in states):
                        continue
                    cur_rows = {stem: row_of[cur["id"]] for stem in STEMS}
                    next_rows = {stem: row_of[next_phrase["id"]] for stem in STEMS}
                    cur_state = state_for(states, {stem: cur["id"] for stem in STEMS})
                    prev_state = state_for(states, {stem: prev["id"] for stem in STEMS})
                    target_section = str(next_phrase.get("section", "GROOVE_A")).upper()
                    target_energy = SEC_ENERGY.get(target_section, 0.65)
                    cur_bars = current_phrases * 4
                    next_bars = next_phrases * 4

                    # Positive: the real next run start for every stem.  The
                    # change mask says which stems are being "swapped" into it.
                    pos_state = state_for(states, {stem: next_phrase["id"] for stem in STEMS})
                    for subset in random.sample(SUBSETS, min(args.pos_subsets, len(SUBSETS))):
                        x = transition_vector(
                            prev_state, cur_state, pos_state, subset,
                            target_section, target_energy, cur_bars, next_bars,
                        )
                        add_example(x, 1.0, cur_rows, next_rows, kit["track_id"], next_bars)
                        if example_count() >= example_target:
                            stop = True
                            break
                    if stop:
                        break

                    # Negatives: held stems continue to the real next phrase,
                    # the subset moves to donor run starts from other tracks.
                    for subset in random.sample(SUBSETS, min(args.neg_subsets, len(SUBSETS))):
                        exclude = {kit["track_id"], meta[next_phrase["id"]]["track_id"]}
                        donors = {}
                        for stem in subset:
                            donor = sample_donor(cluster_id, stem, exclude, next_phrases)
                            if donor is None:
                                donors = {}
                                break
                            donors[stem] = donor
                            exclude.add(meta[donor]["track_id"])
                        if not donors:
                            continue
                        cand_map = {stem: next_phrase["id"] for stem in STEMS}
                        cand_rows = dict(next_rows)
                        for stem, donor in donors.items():
                            cand_map[stem] = donor
                            cand_rows[stem] = row_of[donor]
                        cand_state = state_for(states, cand_map)
                        x = transition_vector(
                            prev_state, cur_state, cand_state, subset,
                            target_section, target_energy, cur_bars, next_bars,
                        )
                        add_example(x, 0.0, cur_rows, cand_rows, kit["track_id"], next_bars)
                        if example_count() >= example_target:
                            stop = True
                            break
                    if stop:
                        break
                if stop:
                    break
        if example_count() >= 10000 and example_count() % 10000 < 40:
            print(f"[joint] {example_count()} examples", flush=True)

    flush_teacher()

    flush_teacher()
    x = np.stack(x_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float32)
    t = np.asarray(teacher_rows, dtype=np.float32)
    tracks = np.asarray(track_rows)
    bars = np.asarray(bars_rows, dtype=np.int32)
    print(f"[joint] examples={len(y)} pos={int(y.sum())} neg={int((y == 0).sum())} input={x.shape[1]}", flush=True)
    print("[joint] by next block: " + " ".join(
        f"{b} bars={int((bars == b).sum())}" for b in BLOCK_BARS
    ), flush=True)

    # Track-disjoint split.
    unique_tracks = sorted(set(tracks.tolist()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_tracks)
    n_val = max(2, int(round(0.10 * len(unique_tracks))))
    val_tracks = set(unique_tracks[:n_val])
    train_mask = np.array([tr not in val_tracks for tr in tracks])
    val_mask = ~train_mask
    if val_mask.sum() == 0 or len(np.unique(y[val_mask])) < 2:
        print("[joint] validation split too small; using random split", flush=True)
        perm = rng.permutation(len(y))
        cut = max(1, int(0.9 * len(y)))
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[perm[:cut]] = True
        val_mask = ~train_mask

    if args.dump_val:
        np.savez_compressed(
            args.dump_val,
            x=x[val_mask].astype(np.float32),
            y=y[val_mask].astype(np.float32),
            t=t[val_mask].astype(np.float32),
        )
        print(
            f"[joint] dumped validation set {args.dump_val} "
            f"({int(val_mask.sum())} rows)",
            flush=True,
        )

    mean = x[train_mask].mean(axis=0)
    std = x[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    model = Student(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    train_ds = FeatureDataset(x[train_mask], y[train_mask], t[train_mask])
    val_ds = FeatureDataset(x[val_mask], y[val_mask], t[val_mask])
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False)

    best_auc = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb, tb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            tb = tb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb) + 0.5 * mse(logits, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * len(yb)
        sched.step()
        model.eval()
        vals, ts, ys = [], [], []
        with torch.no_grad():
            for xb, yb, tb in val_dl:
                vals.append(torch.sigmoid(model(xb.to(device))).float().cpu().numpy())
                ts.append(tb.numpy())
                ys.append(yb.numpy())
        v = np.concatenate(vals)
        tv = np.concatenate(ts)
        yv = np.concatenate(ys)
        auc = float(roc_auc_score(yv, v))
        tauc = float(roc_auc_score(yv, tv))
        history.append({"epoch": epoch, "auc": auc, "teacher_auc": tauc,
                        "loss": total / max(1, len(train_ds))})
        print(f"[joint] ep {epoch} auc={auc:.4f} teacher={tauc:.4f} "
              f"loss={total / max(1, len(train_ds)):.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Calibrate hold thresholds from the positive validation scores.  The
    # arc's 4/8/16-bar blocks have different score distributions, so keep one
    # threshold per block length (plus a global default).
    vals = []
    with torch.no_grad():
        for xb, yb, tb in val_dl:
            vals.append(torch.sigmoid(model(xb.to(device))).float().cpu().numpy())
    v = np.concatenate(vals)
    val_y = y[val_mask]
    val_t = t[val_mask]
    val_bars = bars[val_mask]
    pos = v[val_y > 0.5]
    threshold = float(np.percentile(pos, 20)) if len(pos) else 0.5
    thresholds = {"default": threshold}
    block_auc = {}
    for block_len in BLOCK_BARS:
        mask = val_bars == block_len
        if mask.sum() < 10 or len(np.unique(val_y[mask])) < 2:
            continue
        positives = v[mask & (val_y > 0.5)]
        block_threshold = (float(np.percentile(positives, 20))
                           if len(positives) else threshold)
        thresholds[str(block_len)] = block_threshold
        block_auc[str(block_len)] = {
            "auc": float(roc_auc_score(val_y[mask], v[mask])),
            "teacher_auc": float(roc_auc_score(val_y[mask], val_t[mask])),
            "threshold": block_threshold,
            "n": int(mask.sum()),
        }
        print(f"[joint] {block_len}-bar val auc={block_auc[str(block_len)]['auc']:.4f} "
              f"teacher={block_auc[str(block_len)]['teacher_auc']:.4f} "
              f"threshold={block_threshold:.4f} n={int(mask.sum())}", flush=True)
    print(f"[joint] threshold={threshold:.4f} best_auc={best_auc:.4f}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "thresholds": thresholds,
        "block_auc": block_auc,
        "val_auc": best_auc,
        "input_dim": x.shape[1],
        "history": history,
    }, out_dir / "joint_ranker.pt")

    # Export the phrase states and the student weights for the browser.
    phrase_ids = []
    feature = []
    for kit in manifest.get("kits", []):
        for phrase in kit.get("phrases", []):
            pid = phrase["id"]
            if pid in states:
                phrase_ids.append(pid)
                feature.append(states[pid])
    feature = np.stack(feature).astype(np.float32)
    (out_dir / "joint_features.f32").write_bytes(feature.tobytes())

    sd = model.state_dict()
    def flat(name):
        return sd[name].detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
    export = {
        "version": 3,
        "model": "joint_transition_arc_v3",
        "val_auc": best_auc,
        "threshold": threshold,
        "thresholds": thresholds,
        "block_auc": block_auc,
        "input_dim": int(x.shape[1]),
        "state_dim": STATE_DIM,
        "stem_dim": STEM_DIM,
        "feature_dim": int(feature.shape[1]),
        "phrase_ids": phrase_ids,
        "feature_file": "joint_features.f32",
        "scaler": {"mean": mean.tolist(), "std": std.tolist()},
        "weights": {
            "w1": flat("net.0.weight"), "b1": flat("net.0.bias"),
            "w2": flat("net.3.weight"), "b2": flat("net.3.bias"),
            "w3": flat("net.5.weight"), "b3": flat("net.5.bias"),
        },
        "epochs": args.epochs,
        "history": history,
    }
    (out_dir / "joint_ranker.json").write_text(json.dumps(export, separators=(",", ":")))
    print(f"[joint] wrote {out_dir}  features={feature.shape}  "
          f"weights={sum(len(v) for v in export['weights'].values())} floats", flush=True)


if __name__ == "__main__":
    main()
