"""Polar/radial shaders must wrap their seams like seamless wallpaper."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_psychedelic_background_uses_mirror_wrap_at_the_angular_seam():
    src = (ROOT / "frontend/src/psychedelic_bg.ts").read_text(encoding="utf-8")
    assert "float mirrorWrap(float x)" in src
    assert "vec2 mirrorWrap2(vec2 p)" in src
    assert "float wrappedA = (mirrorWrap(na) - 0.5) * sectorAngle;" in src
    # The old fract-only triangle jump must be gone.
    assert "abs(fract(na)" not in src


def test_spacetubes_wraps_the_polar_star_cell_index():
    src = (ROOT / "frontend/src/lightshow.ts").read_text(encoding="utf-8")
    assert "#define TAU 6.28318530718" in src
    assert "float starCell = mod(floor(((ang + PI) / TAU) * 12.0), 12.0);" in src
    assert "hash21(floor(vec2(ang * 12.0" not in src


def test_kaleidoscope_maps_angle_through_a_full_period():
    src = (ROOT / "frontend/src/lightshow.ts").read_text(encoding="utf-8")
    assert "a = abs(mod(a + 6.2831853, sec) - sec * 0.5);" in src
