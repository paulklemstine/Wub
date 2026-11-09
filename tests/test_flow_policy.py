"""Cover the DJ flow policy mirror (signatures, timbre budget, techniques).

The browser engine is the live path (frontend/src/arc_planner.ts is pure and
dependency-free; there is no JS test harness in the repo).  This test file
covers the Python mirror in wub_core.flow_policy so the math stays green in
CI, and asserts the shipped signals the TS engine derives from
(kit_manifest.json quality/energy/style + arc energy targets) actually exist.
"""

import json
import math
from pathlib import Path

from wub_core import flow_policy as fp

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frontend" / "public" / "data" / "kits" / "kit_manifest.json"
TASTE = ROOT / "frontend" / "public" / "data" / "kits" / "taste_scores.json"
ARC_TS = ROOT / "frontend" / "src" / "arc_planner.ts"


def _sample_candidates():
    return [
        {"phraseId": "a", "trackId": "t1", "quality": 0.9, "earQ": 0.9, "energy": 0.95, "section": "DROP"},
        {"phraseId": "b", "trackId": "t2", "quality": 0.8, "earQ": 0.5, "energy": 0.90, "section": "DROP"},
        {"phraseId": "c", "trackId": "t1", "quality": 0.6, "earQ": 0.9, "energy": 0.95, "section": "DROP"},
        {"phraseId": "intro", "trackId": "t3", "quality": 1.0, "earQ": 1.0, "energy": 0.95, "section": "DROP", "isIntro": True},
    ]


def test_constants_sane():
    assert fp.FLOW_SIGNATURES_PER_STEM == 2
    assert 0.85 <= fp.FLOW_SIGNATURE_PEAK_ENERGY <= 0.95
    assert fp.FLOW_SIGNATURE_MIN_GAP_BARS == 16
    assert abs(fp.FLOW_SIGNATURE_BONUS - 0.14) < 1e-9
    assert fp.FLOW_TIMBRE_WINDOW_BARS == 32
    assert abs(fp.FLOW_TIMBRE_BONUS - 0.08) < 1e-9
    assert fp.FLOW_EQ_OUT_FLOOR_HZ == 300
    assert fp.FLOW_VOCAL_FADE_LOUD_SEC >= 0.10 and fp.FLOW_VOCAL_FADE_LOUD_SEC <= 0.20


def test_ts_constants_in_sync():
    text = ARC_TS.read_text(encoding="utf-8")
    for const in [
        "FLOW_SIGNATURES_PER_STEM = 2",
        "FLOW_SIGNATURE_PEAK_ENERGY = 0.90",
        "FLOW_SIGNATURE_MIN_GAP_BARS = 16",
        "FLOW_TIMBRE_WINDOW_BARS = 32",
        "FLOW_EQ_OUT_FLOOR_HZ = 300",
    ]:
        assert const in text, f"TS missing constant {const!r}"


def test_pick_signatures_prefers_quality_and_distinct_tracks():
    picked = fp.pick_signatures(_sample_candidates())
    assert picked == {"a", "b"}
    assert "c" not in picked  # same track as "a" wins only on track diversity
    assert "intro" not in picked  # intro moments excluded


def test_pick_signatures_caps_and_empty_and_missing_quality():
    assert len(fp.pick_signatures(_sample_candidates(), 1)) == 1
    assert fp.pick_signatures([]) == set()
    no_quality = [dict(c, quality=None) for c in _sample_candidates()[:3]]
    assert len(fp.pick_signatures(no_quality)) == 2


def test_pick_signatures_deterministic():
    first = fp.pick_signatures(_sample_candidates())
    second = fp.pick_signatures(_sample_candidates())
    assert first == second


def test_signature_energy_scale_monotonic_and_gated():
    assert fp.signature_energy_scale(0.95) > fp.signature_energy_scale(0.72)
    assert fp.signature_energy_scale(0.72) > fp.signature_energy_scale(0.30)
    assert fp.signature_energy_scale(0.20) == 0.0
    assert fp.signature_energy_scale(0.50) == 0.0
    assert fp.signature_energy_scale(0.95) == 1.0


def test_signature_bonus_gating():
    assert fp.signature_bonus(True, 0.95, 20) > 0
    assert fp.signature_bonus(True, 0.20, 20) == 0.0  # low energy backs off
    assert fp.signature_bonus(True, 0.95, 4) == 0.0  # inside 16-bar rest
    assert fp.signature_bonus(False, 0.95, 20) == 0.0
    assert fp.signature_bonus(True, 0.95, 20, pinned=True) == 0.0
    assert fp.signature_bonus(True, 0.95, -1) > 0  # never played = rested
    assert fp.signature_bonus(True, float("nan"), 20) == 0.0 or math.isfinite(
        fp.signature_bonus(True, float("nan"), 20)
    )


def test_timbre_bonus_monotonic_and_bounded():
    assert fp.timbre_bonus(0, 8) == fp.FLOW_TIMBRE_BONUS
    assert fp.timbre_bonus(1, 8) < fp.timbre_bonus(0, 8)
    assert fp.timbre_bonus(2, 8) < fp.timbre_bonus(1, 8)
    assert fp.timbre_bonus(3, 8) < 0
    assert fp.timbre_bonus(0, 0) == 0.0
    for count in range(10):
        bonus = fp.timbre_bonus(count, 8)
        assert -0.08 - 1e-9 <= bonus <= fp.FLOW_TIMBRE_BONUS + 1e-9


def test_transition_eq_out_gate():
    hot = fp.select_transition_technique(
        "other", 0.40, 0.80, -20.0, 0.95, False
    )
    assert hot["eq_out"] is True and hot["eq_out_floor_hz"] == 300
    mild = fp.select_transition_technique(
        "other", 0.70, 0.75, -20.0, 0.95, False
    )
    assert mild["eq_out"] is False
    drums = fp.select_transition_technique(
        "drums", 0.40, 0.90, -20.0, 0.95, False
    )
    assert drums["eq_out"] is False  # never EQ a kick to 300 Hz


def test_transition_vocal_treatment():
    loud = fp.select_transition_technique(
        "vocals", 0.60, 0.70, -12.0, 0.95, False, True
    )
    assert loud["vocal_reverb_tail"] is True
    fade = loud["vocal_fade_in_sec"]
    assert fade is not None and 0.10 <= float(fade) <= 0.20
    quiet = fp.select_transition_technique(
        "vocals", 0.60, 0.70, -30.0, 0.95, False, True
    )
    assert quiet["vocal_fade_in_sec"] is None
    bridge = fp.select_transition_technique(
        "vocals", 0.40, 0.80, -12.0, 0.20, True, True
    )
    assert bridge["eq_out"] is False and bridge["vocal_reverb_tail"] is False


def test_repetition_discount_factor():
    assert fp.repetition_discount_factor(False) == 1.0
    assert abs(fp.repetition_discount_factor(True) - 0.4) < 1e-9


def test_shipped_signals_exist_for_signatures_and_timbre():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    kits = manifest["kits"]
    assert len(kits) > 0
    styles = {k["style"] for k in kits}
    assert len(styles) >= 2  # timbre budget needs >1 timbral family
    clusters = manifest.get("clusters", [])
    assert len(clusters) > 0
    energies = [
        p["energy"]
        for k in kits[:20]
        for p in k["phrases"][:5]
    ]
    assert all(0.0 <= e <= 1.0 for e in energies if e is not None)
    # DROP peaks near the 0.95 arc target (signature home).
    drop_energies = [
        p["energy"]
        for k in kits
        for p in k["phrases"]
        if str(p.get("section", "")).upper() == "DROP"
    ]
    assert len(drop_energies) > 0
    assert max(drop_energies) >= 0.85


def test_taste_quality_rows_align_with_phrase_order():
    taste = json.loads(TASTE.read_text(encoding="utf-8"))
    kit_scores = taste.get("kits", {})
    assert kit_scores, "taste_scores.json has no kit rows"
    checked = 0
    for track_id, scores in list(kit_scores.items())[:5]:
        order = scores.get("phrase_order", [])
        quality = scores.get("quality", [])
        assert order and quality, f"missing order/quality for {track_id}"
        assert len(order) == len(quality), f"order/quality mismatch for {track_id}"
        assert all(0.0 <= float(q) <= 1.0 for q in quality)
        checked += 1
    assert checked > 0


def test_arc_energy_targets_match_engine_table():
    # arc_planner.ts ARC_SEGMENTS are the contract for signature gating.
    assert (0, 0.30) in [(0, 0.30)]  # placeholder shape guard (kept explicit below)
    text = ARC_TS.read_text(encoding="utf-8")
    for token in [
        'startBar: 0, bars: 16, section: "INTRO", energy: 0.30',
        'startBar: 64, bars: 32, section: "DROP", energy: 0.95',
        'startBar: 96, bars: 8, section: "BREAKDOWN", energy: 0.20',
    ]:
        assert token in text, f"arc table drifted: {token!r}"
