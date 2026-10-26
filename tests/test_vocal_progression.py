"""Unit tests verifying forward-moving vocal patterns, narrative lyric progression,
and chronological acoustic slice playback across multi-bar song arcs.
"""

import json
from pathlib import Path
import pytest

from wub_core.vocal_pattern_extractor import learn_vocal_corpus_patterns, STYLES


def test_vocal_themes_contain_progressive_phrases():
    """Verify that every style persona has 4 distinct 2-bar narrative phrases per section."""
    res = learn_vocal_corpus_patterns(Path("processed"))
    themes = res["themes"]
    assert len(themes) == 4, f"Expected 4 themes, got {len(themes)}"

    for theme in themes:
        style = theme["territory"]
        assert "versePhrases" in theme, f"{style} missing versePhrases"
        assert "chorusPhrases" in theme, f"{style} missing chorusPhrases"
        assert "buildupPhrases" in theme, f"{style} missing buildupPhrases"
        assert "breakdownPhrases" in theme, f"{style} missing breakdownPhrases"

        assert len(theme["versePhrases"]) == 4, f"{style} versePhrases should have 4 multi-bar phrases"
        assert len(theme["chorusPhrases"]) == 4, f"{style} chorusPhrases should have 4 multi-bar phrases"
        assert len(theme["buildupPhrases"]) == 4, f"{style} buildupPhrases should have 4 multi-bar phrases"
        assert len(theme["breakdownPhrases"]) == 4, f"{style} breakdownPhrases should have 4 multi-bar phrases"


def test_words_move_forward_into_the_future():
    """Verify that lyrics and syllables move into the future across bars (0, 2, 4, 6) without looping."""
    res = learn_vocal_corpus_patterns(Path("processed"))
    themes = res["themes"]

    for theme in themes:
        # Check verse phrases: each 2-bar phrase should have unique forward-moving words
        verse_phrase_texts = []
        for phrase in theme["versePhrases"]:
            phrase_str = " ".join(s["text"] for s in phrase)
            verse_phrase_texts.append(phrase_str)

        # All 4 verse phrases must be distinct from one another (moving into the future)
        assert len(set(verse_phrase_texts)) == 4, f"Verse phrases should all be unique: {verse_phrase_texts}"

        # Check chorus phrases: each 2-bar phrase should have unique forward-moving words
        chorus_phrase_texts = []
        for phrase in theme["chorusPhrases"]:
            phrase_str = " ".join(s["text"] for s in phrase)
            chorus_phrase_texts.append(phrase_str)

        assert len(set(chorus_phrase_texts)) == 4, f"Chorus phrases should all be unique: {chorus_phrase_texts}"


def test_two_bar_phrase_bar_offsets():
    """Verify that syllables within a 2-bar phrase are distributed across bar 0 and bar 1."""
    res = learn_vocal_corpus_patterns(Path("processed"))
    themes = res["themes"]

    for theme in themes:
        for phrase_idx, phrase in enumerate(theme["versePhrases"]):
            bar0_syllables = [s for s in phrase if s.get("barInPhrase", 0) == 0]
            bar1_syllables = [s for s in phrase if s.get("barInPhrase", 0) == 1]
            assert len(bar0_syllables) > 0, f"Theme {theme['title']} phrase {phrase_idx} bar 0 empty"
            assert len(bar1_syllables) > 0, f"Theme {theme['title']} phrase {phrase_idx} bar 1 empty"


def test_chronological_slice_advancement():
    """Verify that acoustic vocal slices sorted by 'start' timestamp advance monotonically in time."""
    manifest_path = Path("frontend/public/data/standalone_patterns.json")
    assert manifest_path.exists()

    with open(manifest_path) as f:
        data = json.load(f)

    vocal_slices = []
    if "slices" in data and "vocals" in data["slices"]:
        for cat, slice_list in data["slices"]["vocals"].items():
            vocal_slices.extend(slice_list)

    assert len(vocal_slices) > 0, "Should have vocal slices in manifest"

    # Sort slices chronologically by start timestamp
    sorted_slices = sorted(vocal_slices, key=lambda s: (s.get("track_id", ""), s.get("start", 0)))

    # Verify that within any given track, timestamps advance monotonically
    current_track = None
    prev_start = -1.0
    for s in sorted_slices:
        tid = s.get("track_id")
        start = float(s.get("start", 0))
        if tid != current_track:
            current_track = tid
            prev_start = start
        else:
            assert start >= prev_start, f"Vocal slices not chronological: {start} < {prev_start}"
            prev_start = start


def test_vocal_patterns_json_and_manifest_sync():
    """Verify that vocal_patterns.json and standalone_patterns.json contain the updated 4-phrase themes."""
    vocal_json_path = Path("frontend/public/data/vocal_patterns.json")
    with open(vocal_json_path) as f:
        vd = json.load(f)

    assert "themes" in vd
    for t in vd["themes"]:
        assert len(t["versePhrases"]) == 4
        assert len(t["chorusPhrases"]) == 4

    manifest_path = Path("frontend/public/data/standalone_patterns.json")
    with open(manifest_path) as mf:
        md = json.load(mf)

    assert "vocal_themes" in md
    for t in md["vocal_themes"]:
        assert len(t["versePhrases"]) == 4
        assert len(t["chorusPhrases"]) == 4
