"""Unit tests for Tier 2 Phrase Pattern Composer."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_groovebox import _make_dance_track
from wub_core.composer import PatternComposer, PhraseComposerModel, SongArranger, camelot_distance
from wub_core.patterns import PatternBank


def test_camelot_distance():
    assert camelot_distance("F minor", "F minor") == 0.0
    assert camelot_distance("F minor", "G# major") == 0.5   # relative major/minor
    assert camelot_distance("F minor", "C minor") == 1.0    # circle of fifths adjacent
    assert camelot_distance("F minor", "B major") >= 2.0    # distant tritone


def test_song_arranger_transitions():
    tr1 = _make_dance_track("test_tr_arr_1", bpm=126.0, n_bars=16)
    tr2 = _make_dance_track("test_tr_arr_2", bpm=126.0, n_bars=16)
    bank = PatternBank()
    bank.load_from_tracks([tr1, tr2])

    arranger = SongArranger(bank, seed=42)
    n_trans = arranger.train_on_corpus([tr1, tr2])
    assert arranger.trained is True

    # Test section decision rules
    assert arranger.decide_next_section("INTRO", bars_in_section=8, total_bars=8) == "GROOVE_A"
    assert arranger.decide_next_section("BUILDUP", bars_in_section=8, total_bars=24) == "DROP"

    # Test phrase planning
    p0 = bank.phrases[0]
    next_p, next_sec, is_sec_change = arranger.plan_next_phrase(
        current_phrase=p0,
        current_section="GROOVE_A",
        current_key="C major",
        current_bpm=126.0,
        bars_in_section=8,
        total_bars=8,
    )
    assert next_p is not None
    assert len(next_p.bars) == 8
    assert next_sec in ("GROOVE_A", "BUILDUP")


def test_phrase_composer_model_training():
    tr1 = _make_dance_track("test_tr_01", bpm=126.0, n_bars=8)
    tr2 = _make_dance_track("test_tr_02", bpm=128.0, n_bars=8)
    bank = PatternBank()
    bank.load_from_tracks([tr1, tr2])

    model = PhraseComposerModel()
    n_trans = model.train_on_tracks([tr1, tr2], bank)
    assert n_trans > 0
    assert model.trained is True

    # Test sampling next groove for Bar 8 turnaround
    fill_groove = model.sample_next_groove(phrase_bar=7, stem="drums", current_groove="four_on_floor")
    assert fill_groove == "fill"

    # Test sampling next groove for Bar 5 energy lift
    lift_groove = model.sample_next_groove(phrase_bar=4, stem="drums", current_groove="four_on_floor")
    assert lift_groove == "four_on_floor_hats"


def test_pattern_composer_8_bar_progression():
    tr = _make_dance_track("test_tr_03", bpm=126.0, n_bars=16)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    composer = PatternComposer(bank, seed=42)
    composer.train([tr])

    # Step through all 8 bars of a phrase
    patterns_by_bar = []
    for bar_idx in range(8):
        pats = composer.compose_bar(
            phrase_bar=bar_idx,
            section="GROOVE_A",
            target_energy=0.6,
        )
        assert "drums" in pats
        assert "bass" in pats
        patterns_by_bar.append(pats)

    # Bar 8 (phrase_bar 7) should have a drum fill
    bar_8_drums = patterns_by_bar[7]["drums"]
    assert bar_8_drums is not None
    assert bar_8_drums.groove_type == "fill"


def test_composer_onset_vs_sustain():
    tr = _make_dance_track("test_tr_04", bpm=126.0, n_bars=4)
    bank = PatternBank()
    bank.load_from_tracks([tr])

    composer = PatternComposer(bank, seed=42)
    composer.compose_bar(phrase_bar=0, section="GROOVE_A", target_energy=0.5)

    # Step 0 should be an onset
    slices_step_0 = composer.get_step_slices(step_idx=0, phrase_bar=0, section="GROOVE_A")
    assert slices_step_0["drums"] is not None

    # Step 1 should sustain (return None if the segment is identical)
    slices_step_1 = composer.get_step_slices(step_idx=1, phrase_bar=0, section="GROOVE_A")
    # If the segment didn't change on step 1, it should be None
    assert slices_step_1["drums"] is None
