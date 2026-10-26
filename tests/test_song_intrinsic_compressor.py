"""Unit tests for Song-Intrinsic Vocabulary & Pattern Compression Engine."""
import numpy as np
import pytest

from wub_core.schema import TrackAnalysis, Tatum, Segment, STEMS
from wub_core.song_intrinsic_compressor import (
    SongIntrinsicCompressor,
    SongReducedVocabulary,
    VocabSlice,
    DownConvertedBar,
    IntrinsicPattern,
    CompressedSongRepresentation,
)


def _make_mock_song(n_bars: int = 16) -> TrackAnalysis:
    """Generate a mock song with recurring acoustic patterns."""
    tatums = []
    segments = []
    track_id = "test_song_01"

    # Create 3 distinct drum slices (kick, snare, hat), 2 bass slices, 2 synth slices, 2 vocal slices
    slice_defs = {
        "drums": [
            ("kick", 0.35, -4.0, [15.0] + [0.0]*11, [0.0]*12),
            ("snare", 0.28, -6.0, [2.0] + [0.0]*11, [0.0]*12),
            ("hat", 0.12, -14.0, [0.5] + [0.0]*11, [0.0]*12),
        ],
        "bass": [
            ("sub_c", 1.20, -6.0, [5.0] + [0.0]*11, [1.0] + [0.0]*11),
            ("stab_g", 0.40, -8.0, [8.0] + [0.0]*11, [0.0]*7 + [1.0] + [0.0]*4),
        ],
        "other": [
            ("chord_c", 2.00, -8.0, [4.0] + [0.0]*11, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ("arp_g", 0.80, -10.0, [7.0] + [0.0]*11, [0.0]*7 + [1.0] + [0.0]*4),
        ],
        "vocals": [
            ("phrase_1", 3.00, -8.0, [6.0] + [0.0]*11, [1.0] + [0.0]*11),
            ("chop_ah", 0.45, -12.0, [9.0] + [0.0]*11, [0.0]*5 + [1.0] + [0.0]*6),
        ],
    }

    seg_id_counter = 0
    seg_by_role = {}
    for stem, defs in slice_defs.items():
        for role, dur, loud, timbre, chroma in defs:
            sid = f"seg_{stem}_{seg_id_counter}"
            seg_by_role[(stem, role)] = sid
            segments.append(
                Segment(
                    id=sid,
                    track_id=track_id,
                    stem=stem,  # type: ignore[arg-type]
                    start=float(seg_id_counter * 0.5),
                    duration=dur,
                    is_silent=False,
                    loudness_start=loud - 12.0,
                    loudness_max=loud,
                    loudness_max_time=0.01,
                    pitches=chroma,
                    timbre=timbre,
                    spectral_centroid=1200.0 if stem != "bass" else 250.0,
                    zero_crossing_start_offset=0.0,
                )
            )
            seg_id_counter += 1

    # Build tatums arranged into 2 recurring pattern sections
    for b in range(n_bars):
        is_pattern_a = (b % 4 < 2)  # Alternates 2 bars of A, 2 bars of B
        for step in range(16):
            t_idx = b * 16 + step
            stem_segs = {}

            # Drums: Pattern A is 4-on-floor + snare on 4, 12; Pattern B has ghost kicks
            if step in (0, 4, 8, 12 if is_pattern_a else 10):
                stem_segs["drums"] = [seg_by_role[("drums", "kick")]]
            elif step in (4, 12):
                stem_segs["drums"] = [seg_by_role[("drums", "snare")]]
            elif step % 2 == 0:
                stem_segs["drums"] = [seg_by_role[("drums", "hat")]]

            # Bass: offbeats
            if step in (2, 6, 10, 14):
                role = "sub_c" if is_pattern_a else "stab_g"
                stem_segs["bass"] = [seg_by_role[("bass", role)]]

            # Other: chords on 0, 8
            if step in (0, 8):
                role = "chord_c" if is_pattern_a else "arp_g"
                stem_segs["other"] = [seg_by_role[("other", role)]]

            # Vocals: phrases on bar 0
            if step == 0 and b % 4 == 0:
                stem_segs["vocals"] = [seg_by_role[("vocals", "phrase_1")]]

            tatums.append(
                Tatum(
                    id=f"tat_{t_idx:04d}",
                    index=t_idx,
                    bar_position=b,
                    beat_position=step // 4,
                    tatum_position=step % 4,
                    start=float(t_idx * 0.119),
                    duration=0.119,
                    confidence=0.95,
                    stem_segments=stem_segs,
                )
            )

    return TrackAnalysis(
        track_id=track_id,
        filename=f"{track_id}.mp3",
        duration=float(n_bars * 16 * 0.119),
        sample_rate=44100,
        bpm=126.0,
        bpm_confidence=0.95,
        key="C minor",
        key_confidence=0.90,
        mode=0,
        mode_confidence=0.90,
        time_signature=4,
        time_signature_confidence=0.99,
        bars=[],
        beats=[],
        tatums=tatums,
        segments=segments,
        stems_available=list(STEMS),
    )


def test_extract_song_vocabulary():
    """Verify that extract_song_vocabulary produces a compact, high-quality codebook."""
    song = _make_mock_song(n_bars=8)
    compressor = SongIntrinsicCompressor(target_vocab_per_stem=4)

    vocab = compressor.extract_song_vocabulary(song, k_per_stem=4)

    assert isinstance(vocab, SongReducedVocabulary)
    assert vocab.track_id == song.track_id
    assert vocab.bpm == 126.0
    assert vocab.total_slices > 0
    # Every stem should have at most 4 slices in its reduced vocabulary
    for stem in STEMS:
        assert len(vocab.vocab_by_stem[stem]) <= 4
        assert len(vocab.vocab_by_stem[stem]) > 0

    # Test find_nearest on a 26-D vector
    dummy_feat = np.zeros(26, dtype=np.float32)
    match = vocab.find_nearest("drums", dummy_feat)
    assert match is not None
    assert match.stem == "drums"


def test_downconvert_song_and_signatures():
    """Verify downconverting the song projects every bar onto vocabulary tokens."""
    song = _make_mock_song(n_bars=8)
    compressor = SongIntrinsicCompressor(target_vocab_per_stem=4)
    vocab = compressor.extract_song_vocabulary(song)

    downconverted = compressor.downconvert_song(song, vocab)
    assert len(downconverted) == 8

    b0 = downconverted[0]
    assert isinstance(b0, DownConvertedBar)
    assert len(b0.steps["drums"]) == 16
    assert len(b0.steps["bass"]) == 16

    # Verify signature string generation
    sig_drums = b0.signature("drums")
    assert sig_drums.startswith("drums:")
    sig_full = b0.signature()
    assert "|" in sig_full


def test_mine_intrinsic_patterns():
    """Verify recurring intrinsic patterns are detected with recurrence counts."""
    song = _make_mock_song(n_bars=16)
    compressor = SongIntrinsicCompressor(target_vocab_per_stem=4)
    vocab = compressor.extract_song_vocabulary(song)
    downconverted = compressor.downconvert_song(song, vocab)

    patterns = compressor.mine_intrinsic_patterns(downconverted, song.track_id)

    assert all(s in patterns for s in STEMS)
    # The song was structured with 2 alternating patterns (8 bars each), so top drum pattern must recur >= 4 times
    top_drum_pat = patterns["drums"][0]
    assert isinstance(top_drum_pat, IntrinsicPattern)
    assert top_drum_pat.recurrence_count >= 4
    assert top_drum_pat.recurrence_ratio > 0.20
    assert top_drum_pat.utility_score > 0.30


def test_compress_song_full_pipeline():
    """Verify end-to-end song compression, compression ratio, and transition matrix."""
    song = _make_mock_song(n_bars=16)
    compressor = SongIntrinsicCompressor(target_vocab_per_stem=4)

    compressed = compressor.compress_song(song, k_per_stem=4)
    assert isinstance(compressed, CompressedSongRepresentation)
    assert compressed.total_bars == 16
    assert compressed.compression_ratio >= 0.70  # Substantial musical data reduction
    assert len(compressed.bar_pattern_sequence) == 16

    # Verify transition matrix structure
    for stem in STEMS:
        assert stem in compressed.transition_matrices

    # Serialization test
    d = compressed.to_dict()
    assert d["track_id"] == song.track_id
    assert "vocabulary" in d
    assert "intrinsic_patterns" in d
    assert "bar_pattern_sequence" in d
    assert "transition_matrices" in d
