"""Budget guard for the removed interactive 3D camera subsystems.

The Wub Bear 3D stage (dual offscreen framebuffers, OrbitControls, a full
Three.js mesh scene at device pixel ratio 2) and the eager vocal-tract 3D
construction were the heaviest per-page GPU/memory costs, and the bear stage
stuttered on low-end and software-GL machines.  The bear view is gone; the
vocal-tract 3D view is built only when the user opens that mode.

These tests fail if either regression comes back.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAMA = ROOT / "frontend/src/tamagotchi.ts"
HEAD = ROOT / "frontend/src/singing_head.ts"
BEAR = ROOT / "frontend/src/bear_3d_view.ts"
HTML = ROOT / "frontend/index.html"


def test_bear_3d_stage_is_gone() -> None:
    assert not BEAR.exists(), "the 3D bear stage module must stay removed"

    source = TAMA.read_text(encoding="utf-8")
    for token in ("Bear3DView", "OrbitControls", "3D CAM", "bear3dView"):
        assert token not in source

    html = HTML.read_text(encoding="utf-8")
    for dom_id in ("tamagotchi-3d-host", "btn-tamagotchi-3d", "btn-tamagotchi-cam"):
        assert dom_id not in html
    # 3D-only controls went with the stage.
    for dom_id in ("btn-bear-cage", "btn-bear-visor", "btn-bear-tex"):
        assert dom_id not in html


def test_2d_bear_mouth_controls_survive() -> None:
    source = TAMA.read_text(encoding="utf-8")
    # The 2D kawaii mouth/face articulation is independent of the 3D stage.
    for token in ("uvulaY", "tongueCenterY", "lipGrad", "mouthStyle", "danceGain"):
        assert token in source

    html = HTML.read_text(encoding="utf-8")
    for dom_id in (
        "btn-bear-mouth-style",
        "bear-dance-slider",
        "bear-mouth-slider",
    ):
        assert dom_id in html


def test_tract_3d_is_built_lazily() -> None:
    source = HEAD.read_text(encoding="utf-8")
    assert "private ensureTract3D(): Tract3DView | null" in source
    assert 'if (mode === "3d") this.ensureTract3D();' in source

    init_body = source.split("public bindAudioEngine", 1)[0]
    assert "new Tract3DView(" not in init_body, "3D tract must not be built in initDOM"
