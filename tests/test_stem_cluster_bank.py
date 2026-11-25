from scripts.layout_stem_clouds import QUADRANTS, layout_point
from scripts.select_cluster_bank import DEFAULT_CLUSTERS, canonical_key, neighbour_keys


def test_canonical_key_merges_enharmonics():
    assert canonical_key("Ab minor") == canonical_key("G# minor")
    assert canonical_key("Bb major") == canonical_key("A# major")
    assert canonical_key("F minor") == "F minor"


def test_camelot_neighbours_include_relative_and_fifths():
    names = neighbour_keys("A minor")
    assert names[0] == "A minor"
    assert "C major" in names       # relative major
    assert "D minor" in names       # subdominant
    assert "E minor" in names       # dominant


def test_default_clusters_are_unique_and_style_balanced():
    ids = {(style, key, bpm) for style, key, bpm in DEFAULT_CLUSTERS}
    assert len(ids) == len(DEFAULT_CLUSTERS)
    assert {style for style, _, _ in DEFAULT_CLUSTERS} == {"breaks", "cyber", "house", "wave"}


def test_stem_clouds_stay_inside_their_quadrants():
    for stem, (x0, x1, y0, y1) in QUADRANTS.items():
        for x, y in ((0.08, 0.08), (0.5, 0.5), (0.92, 0.92)):
            px, py = layout_point(stem, x, y)
            assert x0 - 1e-6 <= px <= x1 + 1e-6, (stem, px)
            assert y0 - 1e-6 <= py <= y1 + 1e-6, (stem, py)
