"""Unit & integration tests for the music theory engine, motif miner,
cell_index / cell_compat shipped artifacts, CellCompatNet scoring, and hybrid server routes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scripts.train_cell_compat import (
    CellCompatNet,
    _build_arr_ctx,
    build_cell_feature_vector,
)
from server.main import app
from wub_core.essential_slices import (
    ESSENTIAL_ROLES,
    bar_reconstitution_similarity,
    mine_processed_track_essential_bars,
    pitch_invariant_chroma_sim,
    score_bar_seam_continuity,
    select_essential_slices_for_song,
)
from wub_core.motif_miner import (
    bar_similarity_matrix,
    classify_motif_schema,
    classify_vocal_role,
    classify_vowel_from_centroid,
    detect_drum_turnaround_fill,
)
from wub_core.theory_engine import (
    classify_triad_quality,
    get_progression,
    parse_key,
    score_cell_purity,
    score_kick_bass_pocket,
    score_synth_vocal_unmasking,
    set_to_onset_bits,
    solve_cell_pitch_shift,
    third_clash_penalty,
)

KITS_DIR = Path("frontend/public/data/kits")


def test_parse_key_and_progressions() -> None:
    assert parse_key("F# minor") == (6, "minor")
    assert parse_key("Bb major") == (10, "major")
    assert parse_key("Am") == (9, "minor")
    assert parse_key("invalid") == (0, "minor")

    for style in ("wave", "cyber", "house", "breaks"):
        prog = get_progression(style, "minor", bars=8, index=0)
        assert len(prog) == 8
        assert all(0 <= r < 12 and q in ("minor", "major", "sus") for r, q in prog)


def test_triad_quality_and_clash_prevention() -> None:
    c_minor = [1.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0]
    root, qual, clarity = classify_triad_quality(c_minor)
    assert root == 0 and qual == "minor" and clarity > 0.3

    c_major = [1.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.0]
    root_m, qual_m, _ = classify_triad_quality(c_major)
    assert root_m == 0 and qual_m == "major"

    bass_mono = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    root_b, qual_b, _ = classify_triad_quality(bass_mono)
    assert root_b == 2 and qual_b == "mono"

    assert third_clash_penalty("minor", "major") == pytest.approx(0.65)
    assert third_clash_penalty("mono", "major") == 0.0

    # Relative minor of C major is A minor (shift -3 from C=0 if cell is A minor=9 -> (0+9-9)=0!)
    sh, pen = solve_cell_pitch_shift("other", 9, "minor", 0, 0, "major", max_shift=6, allow_relative=True)
    assert abs(sh) <= 6
    assert pen <= 0.16

    # Without relative pivot enabled, minor over major triggers clash penalty 0.65
    _, clash_pen = solve_cell_pitch_shift("other", 0, "minor", 0, 0, "major", max_shift=6, allow_relative=False)
    assert clash_pen >= 0.65

    # Vocals stay within ±6 and lock onto chord triad tones
    v_sh, v_pen = solve_cell_pitch_shift("vocals", 2, "mono", 0, 0, "minor", max_shift=6)
    assert abs(v_sh) <= 6
    assert (2 + v_sh) % 12 in (0, 3, 7)
    assert v_pen <= 0.05


def test_counterpoint_and_purity() -> None:
    kicks = [0, 4, 8, 12]
    tight_bass = [0, 2, 6, 8, 10, 14]
    rub_bass = [1, 5, 9, 13]
    assert score_kick_bass_pocket(kicks, tight_bass) > score_kick_bass_pocket(kicks, rub_bass)

    vocal_steps = [0, 1, 2, 4, 5, 8, 9]
    call_other = [3, 6, 11, 13, 14]
    clash_other = list(range(16))
    assert score_synth_vocal_unmasking(call_other, vocal_steps, 5 / 16, 7 / 16) > score_synth_vocal_unmasking(
        clash_other, vocal_steps, 1.0, 7 / 16
    )

    assert score_cell_purity("bass", -65.0, 0.0001, [0.0] * 12, 0) == 0.0
    assert score_cell_purity("drums", -14.0, 0.85, [0.0] * 12, 6) > 0.6


def test_motif_miner_schemas_fills_and_vocals() -> None:
    a = [1.0, 0.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0, 0.0]
    c = [0.0, 0.0, 1.0, 0.0]
    d = [0.0, 0.0, 0.0, 1.0]

    assert len(bar_similarity_matrix([a, b, c, d])) == 4
    assert classify_motif_schema([a, a, a, b]) == "AAAB"
    assert classify_motif_schema([a, b, a, b]) == "ABAB"
    assert classify_motif_schema([a, a, b, b]) == "AABB"
    assert classify_motif_schema([a, b, a, c]) == "ABAC"
    assert classify_motif_schema([a, b, c, d]) == "ABCD"

    is_fill, is_dense = detect_drum_turnaround_fill([4, 4, 4, 11])
    assert is_fill and is_dense

    hook = classify_vocal_role("DROP", -16.0, 0.72, [5, 5, 5, 2], "AAAB", 0.30)
    assert hook["is_chorus_hook"] is True
    assert hook["vocal_role"] == "hook_1bar"

    airy = classify_vocal_role("BREAKDOWN", -20.0, 0.65, [2, 2, 2, 2], "AAAB", 0.80)
    assert airy["vocal_role"] == "airy"

    assert classify_vowel_from_centroid(850.0) == "OO"
    assert classify_vowel_from_centroid(2300.0) == "AH"
    assert classify_vowel_from_centroid(4200.0) == "EE"


def test_shipped_cell_assets_and_model_behavior() -> None:
    ci_path = KITS_DIR / "cell_index.json"
    cc_path = KITS_DIR / "cell_compat.json"
    sc_path = KITS_DIR / "sidecars.json"
    assert ci_path.exists()
    assert cc_path.exists()
    assert sc_path.exists()

    sidecars = json.loads(sc_path.read_text())
    assert sidecars.get("cell_index") is True
    assert sidecars.get("cell_compat") is True

    ci = json.loads(ci_path.read_text())
    assert ci["format"] in ("cell_index_v1", "cell_index_v2")
    assert ci["count"] >= 2500
    assert len(ci["phrases"]) == ci["count"]
    assert "DROP" in ci["section_motif_priors"]
    for style in ("wave", "cyber", "house", "breaks"):
        assert style in ci["style_progressions"]

    first_phrase = next(iter(ci["phrases"].values()))
    for stem in ("drums", "bass", "other", "vocals"):
        assert len(first_phrase[stem]["p"]) == 4
        assert len(first_phrase[stem]["o"]) == 4
        assert all(0 <= m <= 0xFFFF for m in first_phrase[stem]["o"])

    if ci["format"] == "cell_index_v2":
        assert float(ci["mean_reconstitution_fidelity"]) >= 0.80
        assert float(ci.get("mean_seam_continuity", 0.0)) >= 0.70
        rt = ci["essential_role_transitions"]
        assert set(rt.keys()) == set(ESSENTIAL_ROLES)
        for r1 in ESSENTIAL_ROLES:
            assert sum(rt[r1].values()) == pytest.approx(1.0, abs=0.02)
        for stem in ("drums", "bass", "other", "vocals"):
            assert len(first_phrase[stem]["es"]) == 4
            assert len(first_phrase[stem]["er"]) == 4
            assert len(first_phrase[stem]["hr"]) == 4
            assert len(first_phrase[stem]["tr"]) == 4
            assert len(first_phrase[stem]["hc"]) == 4
            assert len(first_phrase[stem]["tc"]) == 4
            assert len(first_phrase[stem]["sm"]) == 4
            assert all(0.0 <= float(x) <= 1.0 for x in first_phrase[stem]["es"])
            assert all(r in ESSENTIAL_ROLES for r in first_phrase[stem]["er"])
            assert 0.0 <= float(first_phrase[stem]["fid"]) <= 1.0

    cc = json.loads(cc_path.read_text())
    assert cc["format"] in ("cell_compat_v1", "cell_compat_v2")
    assert cc["input_dim"] == 24
    assert cc["embed_dim"] == 32
    assert cc["val_auc"] >= 0.90

    # Rebuild PyTorch weights and check good quartet > clashing/muddy quartet
    enc_hidden = len(cc["enc_b1"])
    mix_hidden = len(cc["mix_b1"])
    arr_hidden = len(cc["arr_b1"]) if cc["format"] == "cell_compat_v2" else 48
    net = CellCompatNet(
        cc["input_dim"],
        cc["embed_dim"],
        enc_hidden=enc_hidden,
        mix_hidden=mix_hidden,
        arr_hidden=arr_hidden,
    )
    with torch.no_grad():
        net.enc_fc1.weight.copy_(torch.tensor(cc["enc_w1"], dtype=torch.float32))
        net.enc_fc1.bias.copy_(torch.tensor(cc["enc_b1"], dtype=torch.float32))
        net.enc_fc2.weight.copy_(torch.tensor(cc["enc_w2"], dtype=torch.float32))
        net.enc_fc2.bias.copy_(torch.tensor(cc["enc_b2"], dtype=torch.float32))
        net.mix_fc1.weight.copy_(torch.tensor(cc["mix_w1"], dtype=torch.float32))
        net.mix_fc1.bias.copy_(torch.tensor(cc["mix_b1"], dtype=torch.float32))
        net.mix_fc2.weight.copy_(torch.tensor(cc["mix_w2"], dtype=torch.float32))
        net.mix_fc2.bias.copy_(torch.tensor(cc["mix_b2"], dtype=torch.float32))
        if cc["format"] == "cell_compat_v2":
            net.arr_fc1.weight.copy_(torch.tensor(cc["arr_w1"], dtype=torch.float32))
            net.arr_fc1.bias.copy_(torch.tensor(cc["arr_b1"], dtype=torch.float32))
            net.arr_fc2.weight.copy_(torch.tensor(cc["arr_w2"], dtype=torch.float32))
            net.arr_fc2.bias.copy_(torch.tensor(cc["arr_b2"], dtype=torch.float32))
    net.eval()

    drum_m = set_to_onset_bits([0, 4, 8, 12])
    bass_good_m = set_to_onset_bits([0, 2, 6, 8, 10, 14])
    bass_rub_m = set_to_onset_bits([1, 5, 9, 13])
    other_good_m = set_to_onset_bits([3, 7, 11, 15])
    voc_m = set_to_onset_bits([0, 4, 8, 12])

    good_q = np.array([
        build_cell_feature_vector("drums", 0.85, drum_m, -1, "perc", 0, "minor", False, True, "AAAB"),
        build_cell_feature_vector("bass", 0.82, bass_good_m, 0, "mono", 0, "minor", False, False, "AAAB"),
        build_cell_feature_vector("other", 0.80, other_good_m, 0, "minor", 0, "minor", False, False, "AAAB"),
        build_cell_feature_vector("vocals", 0.78, voc_m, 0, "minor", 0, "minor", True, False, "AAAB"),
    ], dtype=np.float32)

    bad_q = np.array([
        build_cell_feature_vector("drums", 0.85, drum_m, -1, "perc", 0, "minor", False, False, "ABCD"),
        build_cell_feature_vector("bass", 0.40, bass_rub_m, 6, "major", 6, "minor", False, False, "ABCD"),
        build_cell_feature_vector("other", 0.35, 0xFFFF, 6, "major", 6, "minor", False, False, "ABCD"),
        build_cell_feature_vector("vocals", 0.30, 0xFFFF, 6, "major", 6, "minor", False, False, "ABCD"),
    ], dtype=np.float32)

    with torch.no_grad():
        logits, embs = net(torch.from_numpy(np.stack([good_q, bad_q], axis=0)))
        probs = torch.sigmoid(logits).tolist()
    assert embs.shape == (2, 4, 32)
    assert probs[0] > probs[1]
    assert probs[0] > 0.65 and probs[1] < 0.35

    if cc["format"] == "cell_compat_v2":
        assert cc["arr_val_auc"] >= 0.85
        assert cc["tracks_mined"] >= 1500
        assert cc["essential_slices_mined"] > 10000
        assert len(cc["arr_w1"]) == arr_hidden and arr_hidden >= 48 and len(cc["arr_w1"][0]) == 72

        prev_v = torch.tensor(
            build_cell_feature_vector("other", 0.82, other_good_m, 0, "minor", 0, "minor", False, False, "AAAB"),
            dtype=torch.float32,
        ).unsqueeze(0)
        good_next = torch.tensor(
            build_cell_feature_vector("other", 0.80, other_good_m, 7, "minor", 0, "minor", False, False, "AAAB"),
            dtype=torch.float32,
        ).unsqueeze(0)
        dead_next = torch.tensor(
            build_cell_feature_vector("other", 0.04, 0, -1, "major", 6, "minor", False, False, "ABCD"),
            dtype=torch.float32,
        ).unsqueeze(0)
        ctx_good = torch.from_numpy(
            _build_arr_ctx(1, False, 0.85, 0.82, 0.82, 0.80, seam_score=0.90, rms_seam_delta=0.04)
        ).unsqueeze(0)
        ctx_bad = torch.from_numpy(
            _build_arr_ctx(1, False, 0.85, 0.05, 0.82, 0.04, seam_score=0.14, rms_seam_delta=0.58)
        ).unsqueeze(0)
        with torch.no_grad():
            p_good = float(torch.sigmoid(net.forward_transition(prev_v, good_next, ctx_good)).item())
            p_bad = float(torch.sigmoid(net.forward_transition(prev_v, dead_next, ctx_bad)).item())
        assert p_good > 0.65 and p_bad < 0.35
        assert p_good > p_bad + 0.35


def test_server_cells_routes() -> None:
    client = TestClient(app)
    resp = client.get("/api/cells/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["cell_index_available"] is True
    assert body["cell_compat_available"] is True

    bad = client.get("/api/cells/slice/nonexistent_track/invalidstem.wav")
    assert bad.status_code == 400


def test_essential_slices_reconstitution_and_mining() -> None:
    c_min = [1.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
    g_min = [0.0] * 12
    g_min[7] = 1.0
    g_min[10] = 0.85
    g_min[2] = 0.8
    assert pitch_invariant_chroma_sim(c_min, g_min) > 0.80

    beat = set_to_onset_bits([0, 4, 8, 12])
    off = set_to_onset_bits([1, 5, 9, 13])
    b_good = {"purity": 0.85, "mask": beat, "root": 0, "qual": "minor", "chroma": c_min}
    b_copy = {"purity": 0.80, "mask": beat, "root": 7, "qual": "minor", "chroma": g_min}
    b_mud = {"purity": 0.04, "mask": 0, "root": -1, "qual": "mono", "chroma": [0.0] * 12}
    assert bar_reconstitution_similarity(b_good, b_copy, "other") > bar_reconstitution_similarity(b_mud, b_good, "other")

    # Build a 12-bar synthetic song with recurring groove (bars 0..2,4..6,8..10) + turnaround fills (3,7,11)
    drums = []
    other = []
    bass = []
    vocals = []
    fill_m = set_to_onset_bits([0, 2, 4, 6, 8, 10, 12, 13, 14, 15])
    for b in range(12):
        is_t = (b % 4) == 3
        drums.append({
            "bar_in_phrase": b % 4,
            "section": "DROP" if b >= 4 else "GROOVE_A",
            "purity": 0.85,
            "mask": fill_m if is_t else beat,
            "root": -1,
            "qual": "perc",
            "is_fill": is_t,
            "is_hook": False,
        })
        other.append({
            "bar_in_phrase": b % 4,
            "section": "DROP" if b >= 4 else "GROOVE_A",
            "purity": 0.82 if not is_t else 0.72,
            "mask": beat,
            "root": 0 if b < 8 else 7,
            "qual": "minor",
            "chroma": c_min if b < 8 else g_min,
            "is_fill": False,
            "is_hook": False,
        })
        bass.append({
            "bar_in_phrase": b % 4,
            "section": "DROP" if b >= 4 else "GROOVE_A",
            "purity": 0.80,
            "mask": beat if (b % 2 == 0) else off,
            "root": 0,
            "qual": "mono",
            "is_fill": False,
            "is_hook": False,
        })
        vocals.append({
            "bar_in_phrase": b % 4,
            "section": "DROP" if b >= 4 else "GROOVE_A",
            "purity": 0.78 if b >= 4 else 0.05,
            "mask": beat if b >= 4 else 0,
            "root": 0 if b >= 4 else -1,
            "qual": "mono",
            "is_fill": False,
            "is_hook": b >= 4 and (b % 2 == 0),
        })

    sel = select_essential_slices_for_song(
        {"drums": drums, "bass": bass, "other": other, "vocals": vocals},
        max_codebook_per_stem=5,
    )
    assert sel["mean_fidelity"] >= 0.85
    assert sel["stems"]["drums"]["roles"][3] == "T"
    assert any(r == "H" for r in sel["stems"]["vocals"]["roles"][4:])
    # Silent vocal intro scores far below active hook bars
    assert max(sel["stems"]["vocals"]["essential_scores"][4:]) > max(sel["stems"]["vocals"]["essential_scores"][:4]) + 0.3

    # Test single-stem & 4-stem processed JSON mining
    tatums = []
    segs = []
    for t in range(128):  # 8 bars at 120 BPM (bar=2.0s, tatum=0.125s)
        start = t * 0.125
        sid_d = f"sd_{t}"
        sid_o = f"so_{t}"
        segs.append({
            "id": sid_d,
            "stem": "drums",
            "start": start,
            "duration": 0.12,
            "is_silent": False,
            "loudness_start": -40.0,
            "loudness_max": -12.0 if t % 4 == 0 else -22.0,
            "pitches": [0.5] * 12,
            "timbre": [50.0, 0.0, 0.0, 20.0],
            "spectral_centroid": 4800.0,
        })
        segs.append({
            "id": sid_o,
            "stem": "other",
            "start": start,
            "duration": 0.35,
            "is_silent": False,
            "loudness_start": -36.0,
            "loudness_max": -14.0,
            "pitches": c_min,
            "timbre": [50.0, 0.0, 0.0, 0.0],
            "spectral_centroid": 1200.0,
        })
        tatums.append({
            "start": start,
            "duration": 0.125,
            "stem_segments": {"drums": [sid_d], "bass": [sid_o], "other": [sid_o], "vocals": [sid_o]},
        })

    doc_4 = {
        "track_id": "test_4s",
        "bpm": 120.0,
        "key": "C minor",
        "duration": 16.0,
        "stems_available": ["drums", "bass", "other", "vocals"],
        "tatums": tatums,
        "segments": segs,
    }
    mined_4 = mine_processed_track_essential_bars(doc_4, max_bars=16)
    assert mined_4["is_4stem"] is True
    assert len(mined_4["bar_quartets"]) == 8
    assert mined_4["essential"]["mean_fidelity"] >= 0.8

    doc_1 = dict(doc_4, track_id="test_1s", stems_available=["other"])
    mined_1 = mine_processed_track_essential_bars(doc_1, max_bars=16)
    assert mined_1["is_4stem"] is False
    assert len(mined_1["bar_quartets"]) >= 1
    first_q = mined_4["bar_quartets"][0]["cells"]["other"]
    for k in ("hr", "tr", "hc", "tc", "sm"):
        assert k in first_q and 0.0 <= float(first_q[k]) <= 1.0


def test_bar_seam_continuity_and_frontend_seamless_flow(tmp_path: Path) -> None:
    import subprocess
    import os

    beat = set_to_onset_bits([0, 4, 8, 12])
    b_tail = {"purity": 0.84, "mask": beat, "root": 0, "qual": "minor", "hr": 0.78, "tr": 0.76, "hc": 0.42, "tc": 0.44, "sm": 0.90}
    b_smooth = {"purity": 0.82, "mask": beat, "root": 7, "qual": "minor", "hr": 0.74, "tr": 0.75, "hc": 0.45, "tc": 0.46}
    b_jumpy = {"purity": 0.35, "mask": 0xAAAA, "root": 6, "qual": "major", "hr": 0.14, "tr": 0.22, "hc": 0.88, "tc": 0.85}

    s_cont = score_bar_seam_continuity(b_tail, b_smooth, "other", is_contiguous=True)
    s_smooth = score_bar_seam_continuity(b_tail, b_smooth, "other", is_contiguous=False)
    s_jumpy = score_bar_seam_continuity(b_tail, b_jumpy, "other", is_contiguous=False)
    assert s_cont >= 0.85
    assert s_smooth > s_jumpy + 0.35

    root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "js"
    out_dir.mkdir(parents=True, exist_ok=True)
    esbuild = root / "frontend" / "node_modules" / ".bin" / "esbuild"
    for mod_name in ("kit_pcm.ts", "theory_engine.ts", "bar_composer.ts"):
        subprocess.run(
            [
                str(esbuild),
                str(root / "frontend" / "src" / mod_name),
                "--bundle",
                "--platform=node",
                "--format=cjs",
                f"--outfile={out_dir / (mod_name[:-3] + '.js')}",
            ],
            check=True,
        )

    node_script = """
    const pcm = require(process.env.OUT + '/kit_pcm.js');
    const bc = require(process.env.OUT + '/bar_composer.js');
    const theory = require(process.env.OUT + '/theory_engine.js');

    // 1. Coalesce + isContiguousFullPhrase checks
    const full4 = [0, 1, 2, 3].map((b) => ({
      itemIndex: 0, startFrac: b * 0.25, endFrac: (b + 1) * 0.25, targetBars: 1, gain: 1.0,
    }));
    if (!pcm.isContiguousFullPhrase(full4)) throw new Error('expected contiguous full phrase true');
    const spliced4 = full4.map((s, i) => (i === 3 ? { ...s, itemIndex: 1 } : s));
    if (pcm.isContiguousFullPhrase(spliced4)) throw new Error('expected spliced phrase false');
    const merged = pcm.coalesceContiguousSlices([
      { itemIndex: 0, startFrac: 0, endFrac: 0.25, targetBars: 1, gain: 1 },
      { itemIndex: 0, startFrac: 0.25, endFrac: 0.5, targetBars: 1, gain: 1 },
      { itemIndex: 0, startFrac: 0.5, endFrac: 0.75, targetBars: 1, gain: 1 },
      { itemIndex: 1, startFrac: 0.75, endFrac: 1.0, targetBars: 1, gain: 1 },
    ]);
    if (merged.length !== 2 || Math.abs(merged[0].endFrac - 0.75) > 1e-5) {
      throw new Error('coalesce failed: ' + JSON.stringify(merged));
    }

    // 2. Exact bar resampling eliminates dead silence gap when donor2 is faster (shorter phrase)
    const sr = 48000;
    const len1 = 38400; // 0.8s (0.2s per bar)
    const len2 = 32000; // faster donor: shorter buffer!
    const p1 = [new Float32Array(len1), new Float32Array(len1)];
    const p2 = [new Float32Array(len2), new Float32Array(len2)];
    for (let i = 0; i < len1; i++) {
      const v = 0.35 * Math.sin(2 * Math.PI * 110 * i / sr);
      p1[0][i] = v; p1[1][i] = v;
    }
    for (let i = 0; i < len2; i++) {
      const v = 0.08 * Math.sin(2 * Math.PI * 220 * i / sr);
      p2[0][i] = v; p2[1][i] = v;
    }
    const res = pcm.assembleSlicedStereoRun([p1, p2], null, spliced4, sr, 28);
    if (res.samples[0].length !== len1) {
      throw new Error('expected exact target length ' + len1 + ', got ' + res.samples[0].length);
    }
    const bar4Tail = res.samples[0].subarray(len1 - 4800, len1 - 1200);
    let tailRms = 0;
    for (let i = 0; i < bar4Tail.length; i++) tailRms += bar4Tail[i] * bar4Tail[i];
    tailRms = Math.sqrt(tailRms / bar4Tail.length);
    if (tailRms < 0.05) {
      throw new Error('bar4 tail still has silence gap / low RMS: ' + tailRms);
    }

    // 3. BarComposer contiguous default & single-key-shift lock
    const composer = new bc.BarComposer();
    composer.index = {
      version: 2,
      format: 'cell_index_v2',
      count: 1,
      stem_names: ['drums', 'bass', 'other', 'vocals'],
      style_progressions: {},
      section_motif_priors: {},
      phrases: {
        ph_main: {
          drums: { s: 'ABCD', p: [0.85, 0.85, 0.85, 0.85], r: [-1, -1, -1, -1], q: ['perc','perc','perc','perc'], o: [0x1111,0x1111,0x1111,0x1111], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.7] },
          bass:  { s: 'ABCD', p: [0.82, 0.82, 0.82, 0.82], r: [0, 5, 7, 10], q: ['minor','minor','minor','minor'], o: [0x1111,0x1111,0x1111,0x1111], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.7] },
          other: { s: 'ABCD', p: [0.82, 0.82, 0.82, 0.82], r: [0, 8, 3, 10], q: ['minor','major','major','major'], o: [0x2222,0x2222,0x2222,0x2222], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.7] },
          vocals:{ s: 'ABCD', p: [0.75, 0.75, 0.75, 0.75], r: [0, 0, 3, 7], q: ['mono','mono','mono','mono'], o: [0x1111,0x1111,0x1111,0x1111], hr: [0.7,0.7,0.7,0.7], tr: [0.7,0.7,0.7,0.7], sm: [0.9,0.9,0.9,0.7] },
        },
      },
    };
    composer.loaded = true;
    const mkEntry = (stem) => ({
      kit: { track_id: 'trk_1', bpm: 126, key: 'C minor', style: 'house' },
      phrase: { id: 'ph_main', section: 'GROOVE_A', start_bar: 0, duration_sec: 7.6 },
      stem, keyShift: 0, rate: 1.0, energy: 0.8, section: 'GROOVE_A', rmsDb: -14,
    });
    const plan = composer.composeBlock({
      baseEntries: { drums: mkEntry('drums'), bass: mkEntry('bass'), other: mkEntry('other'), vocals: mkEntry('vocals') },
      pools: { drums: [], bass: [], other: [], vocals: [] },
      sectionCtx: {
        role: 'GROOVE_A',
        kitSection: 'GROOVE_A',
        progression: [
          { relRoot: 0, quality: 'minor' },
          { relRoot: 8, quality: 'major' },
          { relRoot: 3, quality: 'major' },
          { relRoot: 10, quality: 'major' },
        ],
        isBuildup: false,
        isPreDrop: false,
        isDropDownbeat: false,
        isBreakdownStart: false,
      },
      blockBars: 4,
      blockDurationSec: 7.6,
      blockIndex: 2,
      sessionRoot: 0,
      sessionMode: 'minor',
      isBridgeBlock: false,
      pinnedStems: new Set(),
    });
    for (const st of ['drums', 'bass', 'other', 'vocals']) {
      if (!plan.stems[st].isContiguousUncut) throw new Error(st + ' should default to isContiguousUncut');
      if (plan.stems[st].pitchPoints.length !== 0) throw new Error(st + ' should hold single keyshift with zero pitchPoints');
    }
    console.log('OK');
    """
    proc = subprocess.run(
        ["node", "-e", node_script],
        env={**os.environ, "OUT": str(out_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"


def test_vocal_activity_filtering_and_empty_bar_repair(tmp_path: Path) -> None:
    import os
    import subprocess

    root = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "js_vocal"
    out_dir.mkdir(parents=True, exist_ok=True)
    esbuild = root / "frontend" / "node_modules" / ".bin" / "esbuild"
    subprocess.run(
        [
            str(esbuild),
            str(root / "frontend" / "src" / "bar_composer.ts"),
            "--bundle",
            "--platform=node",
            "--format=cjs",
            f"--outfile={out_dir / 'bar_composer.js'}",
        ],
        check=True,
    )
    node_script = """
    const bc = require(process.env.OUT + '/bar_composer.js');
    const composer = new bc.BarComposer();
    composer.index = {
      version: 2,
      format: 'cell_index_v2',
      count: 3,
      stem_names: ['drums', 'bass', 'other', 'vocals'],
      style_progressions: {},
      section_motif_priors: {},
      phrases: {
        ph_sparse: {
          drums: { s: 'ABCD', p: [0.85, 0.85, 0.85, 0.85], r: [-1, -1, -1, -1], q: ['perc','perc','perc','perc'], o: [0x1111,0x1111,0x1111,0x1111], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.9] },
          bass:  { s: 'ABCD', p: [0.82, 0.82, 0.82, 0.82], r: [0, 0, 0, 0], q: ['minor','minor','minor','minor'], o: [0x1111,0x1111,0x1111,0x1111], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.9] },
          other: { s: 'ABCD', p: [0.82, 0.82, 0.82, 0.82], r: [0, 0, 0, 0], q: ['minor','minor','minor','minor'], o: [0x2222,0x2222,0x2222,0x2222], hr: [0.8,0.8,0.8,0.8], tr: [0.8,0.8,0.8,0.8], sm: [0.9,0.9,0.9,0.9] },
          vocals:{ s: 'ABCD', p: [0.75, 0.75, 0.75, 0.75], r: [0, 0, 0, 0], q: ['mono','mono','mono','mono'], o: [0, 0, 0, 0], hr: [0.02,0.02,0.02,0.02], tr: [0.02,0.02,0.02,0.02], sm: [0.9,0.9,0.9,0.9] },
        },
        ph_active: {
          vocals:{ s: 'ABCD', p: [0.78, 0.78, 0.78, 0.78], r: [0, 0, 0, 0], q: ['mono','mono','mono','mono'], o: [0x3333,0x3333,0x3333,0x3333], hr: [0.7,0.7,0.7,0.7], tr: [0.7,0.7,0.7,0.7], sm: [0.9,0.9,0.9,0.9] },
        },
        ph_half_empty: {
          vocals:{ s: 'ABCD', p: [0.70, 0.75, 0.75, 0.70], r: [0, 0, 0, 0], q: ['mono','mono','mono','mono'], o: [0, 0x3333, 0x3333, 0], hr: [0.01,0.7,0.7,0.01], tr: [0.01,0.7,0.7,0.01], sm: [0.9,0.9,0.9,0.9] },
        },
      },
    };
    composer.loaded = true;
    const actMap = new Map([
      ['ph_sparse', { frac: 0.05, barFracs: [0.05, 0, 0, 0], activeBars: 0 }],
      ['ph_active', { frac: 0.75, barFracs: [0.75, 0.75, 0.75, 0.75], activeBars: 4 }],
      ['ph_half_empty', { frac: 0.40, barFracs: [0.0, 0.80, 0.80, 0.0], activeBars: 2 }],
    ]);
    composer.setVocalActivity(actMap);

    const mkEntry = (phraseId, stem) => ({
      kit: { track_id: 'trk_' + phraseId, bpm: 126, key: 'C minor', style: 'house' },
      phrase: { id: phraseId, section: 'GROOVE_A', start_bar: 0, duration_sec: 7.6 },
      stem, keyShift: 0, rate: 1.0, energy: 0.8, section: 'GROOVE_A', rmsDb: -18,
    });
    const secCtx = {
      role: 'VERSE_1',
      kitSection: 'GROOVE_A',
      progression: [
        { relRoot: 0, quality: 'minor' },
        { relRoot: 0, quality: 'minor' },
        { relRoot: 0, quality: 'minor' },
        { relRoot: 0, quality: 'minor' },
      ],
      isBuildup: false, isPreDrop: false, isDropDownbeat: false, isBreakdownStart: false,
    };

    // 1. Sparse primary in baseEntries should be replaced by active pool vocal
    const planSwap = composer.composeBlock({
      baseEntries: {
        drums: mkEntry('ph_sparse', 'drums'),
        bass: mkEntry('ph_sparse', 'bass'),
        other: mkEntry('ph_sparse', 'other'),
        vocals: mkEntry('ph_sparse', 'vocals'),
      },
      pools: {
        drums: [],
        bass: [],
        other: [],
        vocals: [mkEntry('ph_sparse', 'vocals'), mkEntry('ph_active', 'vocals')],
      },
      sectionCtx: secCtx,
      blockBars: 4, blockDurationSec: 7.6, blockIndex: 2,
      sessionRoot: 0, sessionMode: 'minor', isBridgeBlock: false,
      pinnedStems: new Set(),
    });
    if (planSwap.stems.vocals.donors[0].phrase.id !== 'ph_active') {
      throw new Error('expected ph_active vocal replacement, got ' + planSwap.stems.vocals.donors[0].phrase.id);
    }

    // 2. Half-empty vocal phrase (pinned or selected) must repair bars 0 and 3 using active bars 1 and 2
    const planRepair = composer.composeBlock({
      baseEntries: {
        drums: mkEntry('ph_sparse', 'drums'),
        bass: mkEntry('ph_sparse', 'bass'),
        other: mkEntry('ph_sparse', 'other'),
        vocals: mkEntry('ph_half_empty', 'vocals'),
      },
      pools: { drums: [], bass: [], other: [], vocals: [] },
      sectionCtx: secCtx,
      blockBars: 4, blockDurationSec: 7.6, blockIndex: 2,
      sessionRoot: 0, sessionMode: 'minor', isBridgeBlock: false,
      pinnedStems: new Set(),
    });
    const vocSlices = planRepair.stems.vocals.slices;
    for (const s of vocSlices) {
      if (s.startFrac < 0.24 || s.startFrac >= 0.74) {
        throw new Error('silent bar sliced into vocal plan: ' + JSON.stringify(vocSlices));
      }
    }
    console.log('OK');
    """
    proc = subprocess.run(
        ["node", "-e", node_script],
        env={**os.environ, "OUT": str(out_dir)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "OK"

    # Also verify shipped bank retains abundant singing vocal phrases across all 16 clusters
    kits_dir = root / "frontend" / "public" / "data" / "kits"
    manifest_path = kits_dir / "kit_manifest.json"
    vm_path = kits_dir / "vocal_mix.json"
    if manifest_path.exists() and vm_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        vm = json.loads(vm_path.read_text())
        by_pid = {}
        for kit in manifest.get("kits", []):
            for phr in kit.get("phrases", []):
                asset = phr.get("stems", {}).get("vocals")
                if not asset:
                    continue
                act = vm.get(phr["id"], {}).get("vocal_activity16", "")
                frac = act.count("1") / 64.0 if len(act) == 64 else 1.0
                if float(asset.get("rms_db", -99)) >= -34.0 and frac >= 0.20:
                    by_pid[phr["id"]] = float(phr.get("duration_sec", 0.0))
        assert len(by_pid) >= 1500
        for cl in manifest.get("clusters", []):
            block_sec = 4 * 60.0 * 4 / float(cl["bpm"])
            count = sum(1 for dur in by_pid.values() if dur > 0 and 0.94 <= (dur / block_sec) <= 1.06)
            assert count >= 40, f"Cluster {cl['id']} too few active vocals: {count}"
