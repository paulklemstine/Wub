"""Check that every step_slice_id in catalog songs resolves to an existing opus file."""
import json
from pathlib import Path

ROOT = Path("/home/raver1975/Wub")
sp = json.loads((ROOT / "frontend/public/data/standalone_patterns.json").read_bytes())
pub = ROOT / "frontend/public"

missing_urls = set()
missing_ids = set()
total_refs = 0
checked_songs = 0
for tid, model in list(sp.get("compressed_song_models", {}).items())[:60]:
    voc = model.get("vocab_by_stem", {}) or {}
    url_of = {}
    for s, vlist in voc.items():
        for vs in vlist or []:
            url_of[vs["id"]] = vs.get("url", "")
            if vs.get("matched_slice_name"):
                url_of[vs["id"]] = url_of.get(vs["id"]) or ""
    pats = model.get("intrinsic_patterns", {})
    seen = set()
    for s, plist in pats.items():
        for p in plist or []:
            for t in (p.get("step_slice_ids") or []):
                if not t or t in seen:
                    continue
                seen.add(t)
                total_refs += 1
                u = url_of.get(t, "")
                if not u or not (pub / u).exists():
                    missing_ids.add(t)
                    if u:
                        missing_urls.add(u)
    checked_songs += 1
print("songs:", checked_songs, "refs:", total_refs)
print("unresolved ids:", len(missing_ids), "missing urls:", len(missing_urls))
for u in sorted(missing_urls)[:10]:
    print("  MISSING:", u)
