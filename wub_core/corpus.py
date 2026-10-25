"""Live corpus registry: everything downstream reads through this.

The pandora folder fills while the analyzer ingests; CorpusStore always
reflects whatever is on disk right now:
- refresh() rescans data/processed for new TrackAnalysis JSONs (cheap: mtime
  cache, only parses new/changed files).
- audio paths resolve via the ingest manifest (track_id -> source file) with
  fallback to scanning library dirs by filename.
- pools (markov SliceRefs), FAISS indices + norm stats, and UMAP points are
  exposed with the same refresh semantics so the server, mashup CLI, and
  trainer stay correct as the library grows to 5000 tracks.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from wub_core.schema import TrackAnalysis

try:
    from wub_core.separator import STEM_FILES
except ImportError:  # keep import light when demucs stack absent
    STEM_FILES = {"drums": "drums.wav", "bass": "bass.wav",
                  "vocals": "vocals.wav", "other": "other.wav"}


@dataclass
class CorpusStore:
    processed_dir: Path
    stems_base: Path = Path("data/stems")
    library_dirs: List[Path] = field(default_factory=list)
    analyses: Dict[str, TrackAnalysis] = field(default_factory=dict)
    mtimes: Dict[str, float] = field(default_factory=dict)
    last_scan: float = 0.0

    # ---- refresh ----

    def refresh(self, force: bool = False, min_interval: float = 5.0, max_tracks: Optional[int] = None) -> int:
        """Load new/changed analyses. Returns number of (re)loaded tracks."""
        from wub_core.schema import TrackAnalysis as TA

        now = time.time()
        if not force and self.last_scan > 0 and (now - self.last_scan < min_interval):
            return 0

        n = 0
        if not self.processed_dir.exists():
            return 0

        # Prioritize files whose stems exist
        stem_tids = set()
        if self.stems_base.exists():
            stem_tids = {d.name for d in self.stems_base.iterdir() if d.is_dir() and not d.name.startswith("_")}

        all_files = [p for p in self.processed_dir.glob("*.json") if not p.name.startswith(("_", "meta_"))]
        # Sort so multi-stem tracks come first (instantaneous string comparison)
        all_files.sort(key=lambda p: 0 if p.stem in stem_tids else 1)

        if max_tracks and max_tracks > 0:
            all_files = all_files[:max_tracks]

        for jf in all_files:
            try:
                mt = jf.stat().st_mtime
            except OSError:
                continue
            if jf.name in self.mtimes and self.mtimes[jf.name] >= mt:
                continue
            try:
                tr = TA.model_validate_json(jf.read_bytes())
            except Exception:
                continue
            self.analyses[tr.track_id] = tr
            self.mtimes[jf.name] = mt
            n += 1

        self.last_scan = time.time()
        return n

    @property
    def tracks(self) -> List[TrackAnalysis]:
        return sorted(self.analyses.values(), key=lambda t: t.filename)

    # ---- audio resolution ----

    def _manifest_paths(self) -> Dict[str, str]:
        mp = self.processed_dir / "_manifest.json"
        if not mp.exists():
            return {}
        try:
            manifest = json.loads(mp.read_text())
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for entry in manifest.values():
            if isinstance(entry, dict) and entry.get("track_id") and entry.get("path"):
                out.setdefault(entry["track_id"], entry["path"])
        return out

    def audio_for(self, track_ids: Set[str]) -> Tuple[Dict[Tuple[str, str], Path], Dict[str, Path]]:
        """(per-stem wav map, fallback source-file map)."""
        audio_for: Dict[Tuple[str, str], Path] = {}
        fallback: Dict[str, Path] = {}
        known = self._manifest_paths()
        for tid in track_ids:
            sdir = self.stems_base / tid
            for stem, fname in STEM_FILES.items():
                p = sdir / fname
                if p.exists():
                    audio_for[(tid, stem)] = p
            src = known.get(tid)
            if src and Path(src).exists():
                fallback[tid] = Path(src)
        missing = [t for t in track_ids if t not in fallback
                   and not any(k[0] == t for k in audio_for)]
        if missing:
            by_name: Dict[str, Path] = {}
            for lib in self.library_dirs:
                if lib.is_file():
                    by_name.setdefault(lib.name, lib)
                elif lib.is_dir():
                    for p in lib.rglob("*"):
                        if p.is_file():
                            by_name.setdefault(p.name, p)
            want = {t: self.analyses[t].filename for t in missing if t in self.analyses}
            for tid, fname in want.items():
                hit = by_name.get(fname)
                if hit:
                    fallback[tid] = hit
        return audio_for, fallback

    # ---- derived views ----

    def pools(self, exclude_silent: bool = True):
        from wub_core import model_markov as mm

        return mm.build_pools(self.tracks, exclude_silent=exclude_silent)

    def index_paths(self) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        idx_dir = self.processed_dir.parent / "indices"
        for p in sorted(idx_dir.glob("index_*.faiss")) if idx_dir.exists() else []:
            out[p.stem.replace("index_", "").replace(".faiss", "")] = p
        return out

    def umap_points(self, max_points: int = 20000) -> List[dict]:
        pts: List[dict] = []
        idx_dir = self.processed_dir.parent / "indices"
        if not idx_dir.exists():
            return pts
        for stem in ("drums", "bass", "vocals", "other"):
            f = idx_dir / f"umap_{stem}.json"
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            for d in data[: max_points // 4]:
                pts.append({"x": d["x"], "y": d["y"], "stem": stem,
                            "segment_id": d.get("segment_id"),
                            "track_id": d.get("track_id")})
        return pts

    def summary(self) -> dict:
        n_seg = sum(len(t.segments) for t in self.analyses.values())
        n_tat = sum(len(t.tatums) for t in self.analyses.values())
        stems: Dict[str, int] = {}
        for t in self.analyses.values():
            for s in t.stems_available:
                stems[s] = stems.get(s, 0) + 1
        return {"tracks": len(self.analyses), "segments": n_seg, "tatums": n_tat,
                "stems": stems, "last_scan": self.last_scan,
                "indices": sorted(self.index_paths().keys())}
