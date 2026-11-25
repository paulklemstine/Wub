"""Unit tests for scripts/build_sidecars.py schema, shapes, and numerical invariants."""

import json
from pathlib import Path

import numpy as np
import torch

from scripts.build_sidecars import (
    RHYTHM_FEAT_NAMES,
    RHYTHM_PAIRS,
    SECTIONS,
    STEMS,
    ArcTransformer,
    build_arc_sidecars,
    build_progression_sidecars,
    build_rhythm_and_vocal_sidecars,
    build_syntax_successors,
    parse_key,
    rhythm_pair_features,
)
from scripts.train_taste_model import PairwiseRanker


def _synthetic_fixture(tmp_path: Path):
    rng = np.random.default_rng(505)
    kits = []
    kit_phrases = []
    tracks = []
    onset_index = {}
    onset_rows = []
    idx_counter = 0

    keys = ["C minor", "F# minor", "A major", "D minor", "G minor", "E minor"]
    styles = ["breaks", "cyber", "house", "wave"]

    for t_i in range(12):
        tid = f"trk{t_i:04d}"
        key = keys[t_i % len(keys)]
        style = styles[t_i % len(styles)]
        windows = []
        phrases = []
        for w_i in range(6):
            feat = np.zeros(160, dtype=np.float32)
            feat[0:12] = rng.dirichlet(np.ones(12)).astype(np.float32)
            feat[12:24] = rng.normal(0.0, 1.0, size=12).astype(np.float32)
            feat[24:28] = rng.uniform(0.1, 0.9, size=4).astype(np.float32)
            feat[28:32] = rng.uniform(0.3, 0.9, size=4).astype(np.float32)
            energy = float(0.2 + 0.12 * w_i)
            feat[32] = energy
            feat[33] = 0.05
            sec = SECTIONS[w_i % len(SECTIONS)]
            feat[34 + SECTIONS.index(sec)] = 1.0
            feat[40] = 0.7
            feat[48:112] = (rng.random(64) > 0.6).astype(np.float32)
            feat[112:124] = rng.dirichlet(np.ones(12)).astype(np.float32)
            feat[124:136] = rng.dirichlet(np.ones(12)).astype(np.float32)
            br = (t_i + w_i) % 12
            cr = (t_i + w_i * 5) % 12
            feat[136 + br] = 1.0
            feat[148 + cr] = 1.0

            flist = feat.tolist()
            windows.append({
                "start_bar": w_i * 4,
                "section": sec,
                "energy": energy,
                "aligned": True,
                "chroma": feat[0:12].tolist(),
                "bass_root": br,
                "is_intro": w_i == 0,
                "is_outro": w_i == 5,
                "once": w_i in (0, 5),
                "features": flist,
            })
            if w_i < 4:
                pid = f"kit_{tid}_p{w_i:02d}"
                phrases.append({
                    "id": pid,
                    "section": sec,
                    "start_bar": w_i * 4,
                    "energy": energy,
                    "chroma": feat[0:12].tolist(),
                    "bass_root": br,
                })
                kit_phrases.append({
                    "phrase_id": pid,
                    "track_id": tid,
                    "start_bar": w_i * 4,
                    "features": flist,
                })
                onset_index[pid] = idx_counter
                idx_counter += 1
                flux = rng.uniform(0.0, 2.0, size=(4, 64)).astype(np.float32)
                if w_i % 2 == 0:
                    flux[3, :2] = 0.0
                    flux[3, -2:] = 0.0
                onset_rows.append(flux)

        tracks.append({"track_id": tid, "style": style, "bpm": 128.0, "key": key, "windows": windows})
        kits.append({"track_id": tid, "style": style, "bpm": 128.0, "key": key, "phrases": phrases})

    manifest = {"version": 2, "kits": kits}
    dataset = {"tracks": tracks, "kit_phrases": kit_phrases}

    onsets_dir = tmp_path / "onsets"
    onsets_dir.mkdir(parents=True, exist_ok=True)
    flux_arr = np.stack(onset_rows).astype(np.float32)
    (onsets_dir / "mix_compat_onsets.f32").write_bytes(flux_arr.tobytes())
    (onsets_dir / "mix_compat_onsets.json").write_text(json.dumps({"index": onset_index}))

    ranker = PairwiseRanker(feature_dim=160, d_model=32, n_layers=1, n_heads=2, max_len=6,
                            mean=[0.0] * 160, std=[1.0] * 160)
    ranker_path = tmp_path / "taste_ranker.pt"
    torch.save({
        "model": ranker.state_dict(),
        "cfg": {"feature_dim": 160, "d_model": 32, "n_layers": 1, "n_heads": 2,
                "max_len": 6, "dropout": 0.1, "init_scale": 10.0,
                "mean": [0.0] * 160, "std": [1.0] * 160},
    }, ranker_path)

    return manifest, dataset, onsets_dir, ranker_path


def test_parse_key_and_rhythm_features():
    assert parse_key("F# minor") == (6, "minor")
    assert parse_key("Db major") == (1, "major")
    assert parse_key("C major") == (0, "major")

    ga = np.zeros(64, dtype=np.uint8)
    ga[::4] = 1
    gb = np.zeros(64, dtype=np.uint8)
    gb[2::4] = 1
    fa = ga.astype(np.float32) * 1.5
    fb = gb.astype(np.float32) * 1.2
    feats = rhythm_pair_features(ga, fa, gb, fb)
    assert feats.shape == (19,)
    assert np.all(np.isfinite(feats))


def test_build_all_sidecars_contracts(tmp_path: Path):
    manifest, dataset, onsets_dir, ranker_path = _synthetic_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    arc_res = build_arc_sidecars(dataset, out, device, epochs=2)
    assert arc_res["phrases"] == 48
    arc_model = json.loads((out / "arc_model.json").read_text())
    assert arc_model["format"] == "wub-arc-model-v1"
    assert len(arc_model["feat_mean"]) == 48
    assert len(arc_model["feat_std"]) == 48
    arc_ctx = json.loads((out / "arc_context_features.json").read_text())
    assert len(arc_ctx["byPhraseId"]) == 48

    # Verify ArcTransformer PyTorch forward produces finite outputs with same weights
    m = ArcTransformer(feature_dim=48, d_model=96, n_layers=2, n_heads=4, max_len=6, n_sections=5)
    m.load_state_dict({k: torch.tensor(v) for k, v in arc_model["weights"].items()})
    m.eval()
    x = torch.zeros(1, 6, 48)
    mask = torch.tensor([[False, False, False, True, True, True]])
    s_log, e_log, d_pred = m(x, mask)
    assert s_log.shape == (1, 5)
    assert torch.isfinite(s_log).all() and torch.isfinite(e_log).all() and torch.isfinite(d_pred).all()

    prog_res = build_progression_sidecars(manifest, dataset, out, device, epochs=2)
    assert prog_res["phrases"] == 48
    prog_model = json.loads((out / "progression_model.json").read_text())
    assert prog_model["format"] == "wub-progression-v1"
    assert len(prog_model["mlp"]["fc1_weight"]) == 64
    assert len(prog_model["mlp"]["fc1_weight"][0]) == 56
    assert len(prog_model["mlp"]["fc2_weight"]) == 12
    prog_ctx = json.loads((out / "progression_context.json").read_text())
    assert prog_ctx["format"] == "wub-progression-context-v1"
    assert len(prog_ctx["byPhraseId"]) == 48

    rhy_res = build_rhythm_and_vocal_sidecars(manifest, onsets_dir, out)
    assert rhy_res["phrases"] == 48
    rhy_scores = json.loads((out / "rhythm_scores.json").read_text())
    assert rhy_scores["format"] == "wub-rhythm-scores-v1"
    assert len(rhy_scores["phrases"]) == 48
    assert all(len(p[s]) == 64 for p in rhy_scores["phrases"] for s in STEMS)
    flux_bytes = (out / "rhythm_flux.f32").read_bytes()
    assert len(flux_bytes) == 48 * 4 * 64 * 4
    mesh = json.loads((out / "rhythm_mesh_model.json").read_text())
    assert mesh["feature_names"] == RHYTHM_FEAT_NAMES
    for sa, sb in RHYTHM_PAIRS:
        pm = mesh["pairs"][f"{sa}-{sb}"]
        assert len(pm["coef"]) == 19
        assert len(pm["feature_mean"]) == 19
        assert len(pm["feature_std"]) == 19

    vm = json.loads((out / "vocal_mix.json").read_text())
    assert len(vm) == 48
    for pid, entry in vm.items():
        assert len(entry["vocal_edge_strength"]) == 2
        assert len(entry["vocal_activity16"]) == 64

    syn_res = build_syntax_successors(manifest, dataset, ranker_path, out, device, top_k=6)
    assert syn_res["anchors"] == 48
    syn = json.loads((out / "syntax_successors.json").read_text())["successors"]
    assert len(syn) == 48
    assert all(len(v["overall_top"]) > 0 for v in syn.values())
