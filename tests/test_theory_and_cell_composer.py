"""Unit & integration tests for the music theory engine, motif miner,
cell_index / cell_compat shipped artifacts, CellCompatNet scoring, and hybrid server routes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scripts.train_cell_compat import CellCompatNet, build_cell_feature_vector
from server.main import app
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
    assert ci["format"] == "cell_index_v1"
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

    cc = json.loads(cc_path.read_text())
    assert cc["format"] == "cell_compat_v1"
    assert cc["input_dim"] == 24
    assert cc["embed_dim"] == 32
    assert cc["val_auc"] >= 0.90

    # Rebuild PyTorch weights and check good quartet > clashing/muddy quartet
    net = CellCompatNet(cc["input_dim"], cc["embed_dim"])
    with torch.no_grad():
        net.enc_fc1.weight.copy_(torch.tensor(cc["enc_w1"], dtype=torch.float32))
        net.enc_fc1.bias.copy_(torch.tensor(cc["enc_b1"], dtype=torch.float32))
        net.enc_fc2.weight.copy_(torch.tensor(cc["enc_w2"], dtype=torch.float32))
        net.enc_fc2.bias.copy_(torch.tensor(cc["enc_b2"], dtype=torch.float32))
        net.mix_fc1.weight.copy_(torch.tensor(cc["mix_w1"], dtype=torch.float32))
        net.mix_fc1.bias.copy_(torch.tensor(cc["mix_b1"], dtype=torch.float32))
        net.mix_fc2.weight.copy_(torch.tensor(cc["mix_w2"], dtype=torch.float32))
        net.mix_fc2.bias.copy_(torch.tensor(cc["mix_b2"], dtype=torch.float32))
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
