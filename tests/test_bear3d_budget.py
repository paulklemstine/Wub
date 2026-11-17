"""Rendering budget guard for the optional 3D Wub Bear view.

The dual offscreen framebuffer pass (two 512x512 psychedelic shader renders per
frame, wrapped onto the sky sphere and the head sphere) pegged low-end and
software-GL machines, so it is disabled in this build. The Three.js scene must
also stay lazily constructed: building it at page load cost GPU memory and
shader compile time for a view most sessions never open.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEAR = ROOT / "frontend/src/bear_3d_view.ts"
TAMA = ROOT / "frontend/src/tamagotchi.ts"


def test_sphere_texture_pass_is_disabled() -> None:
    source = BEAR.read_text(encoding="utf-8")
    assert "export const SPHERE_TEXTURE_ENABLED = false" in source
    # The FBO creation and the per-frame shader passes must both be gated.
    assert "if (SPHERE_TEXTURE_ENABLED) {" in source
    assert "this.skyTarget = new THREE.WebGLRenderTarget(512, 512" in source
    assert "this.headTarget = new THREE.WebGLRenderTarget(512, 512" in source
    # Texture maps on the head material come only from the gated target.
    assert "SPHERE_TEXTURE_ENABLED ? this.headTarget?.texture ?? null : null" in source
    # The animated gummi palette is the fallback when the pass is off.
    assert "this.gummiMat.color.copy(this.tmpColor)" in source


def test_3d_scene_is_built_lazily() -> None:
    source = TAMA.read_text(encoding="utf-8")
    assert "private ensure3DView(): boolean" in source
    assert "this.view3dActive = active ? this.ensure3DView() : false" in source
    # The constructor must not build the scene; only ensure3DView may.
    init_body = source.split("public set3DMode", 1)[0]
    assert "new Bear3DView(" not in init_body, "3D view must not be built in the constructor"
