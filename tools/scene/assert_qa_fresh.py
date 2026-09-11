"""Fail if the scene changed after the last time anyone LOOKED at it.

    uv run python tools/scene/assert_qa_fresh.py

`docs/QUALITY_GATE.md` requires a rendered, inspected image after every change to the scene. That rule is worth
nothing if it relies on people remembering, so this makes it checkable: `tools/scene/qa_shots.py` stamps
`_artifacts/editor_shots/qa_manifest.json` with the moment it rendered, and this script fails if any scene
script or scene data file is newer than that stamp.

Run it before claiming any environment work is done, and in review of anyone else's. It exits non-zero with the
list of files that changed since the last render, so "did you look at it?" has a factual answer.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "_artifacts" / "editor_shots" / "qa_manifest.json"

#: anything here changes what the camera would see
WATCHED = [
    ("tools/scene", "*.py"),
    ("data/scene", "*.json"),
    ("sim/SightlineSim/Content/Sightline/Maps", "*.umap"),
    ("sim/SightlineSim/Content/Sightline/Terrain", "*.uasset"),
    ("sim/SightlineSim/Content/Sightline/Water", "*.uasset"),
    ("sim/SightlineSim/Content/Sightline/Buildings", "*.uasset"),
]
#: files that do not change the render
IGNORE = {"qa_shots.py", "assert_qa_fresh.py", "flood_level.py", "poses.json", "camera_survey.json"}

#: A script that only OBSERVES the scene cannot change what a render of it shows, so editing one must not
#: invalidate the render. Builders (`build_*`, `gen_*`, `tune_*`, `place_*`) and anything that writes scene
#: data stay watched, because those really can change the pixels. Without this, tightening a checker marks
#: the whole scene stale and pushes people towards re-rendering to silence the gate rather than because
#: anything moved - which is how a freshness gate stops meaning anything.
OBSERVER_PREFIXES = ("check_", "measure_", "verify_", "dump_", "sweep_", "qa_", "gate")


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL  no QA render has ever been taken ({MANIFEST} missing).")
        print("      Run:  ue_python exec tools/scene/qa_shots.py   then OPEN the images.")
        return 1
    m = json.loads(MANIFEST.read_text())
    t = float(m["rendered_utc"])
    age_min = (time.time() - t) / 60.0
    print(f"last QA render: {m['rendered_local']} ({age_min:.0f} min ago)")
    print(f"  level {m['level']}: {m['actor_count']} actors "
          f"({m.get('survivors', '?')} survivors, {m.get('houses', '?')} houses, {m.get('debris', '?')} debris)")
    print(f"  {len(m.get('images', []))} images: {', '.join(i['file'] for i in m.get('images', []))}")

    stale: list[tuple[str, float]] = []
    for rel, pat in WATCHED:
        d = REPO / rel
        if not d.is_dir():
            continue
        for f in d.rglob(pat):
            if f.name in IGNORE or f.name.startswith(OBSERVER_PREFIXES):
                continue
            mt = f.stat().st_mtime
            if mt > t:
                stale.append((str(f.relative_to(REPO)), (mt - t) / 60.0))
    missing = [i["file"] for i in m.get("images", [])
               if not (MANIFEST.parent / i["file"]).exists()]

    if missing:
        print(f"\nFAIL  {len(missing)} rendered image(s) named in the manifest are gone: {missing[:4]}")
        return 1
    if stale:
        stale.sort(key=lambda x: -x[1])
        print(f"\nFAIL  {len(stale)} scene file(s) changed AFTER the last render - nobody has looked at the "
              "current scene:")
        for f, mins in stale[:12]:
            print(f"        {f}  (+{mins:.0f} min)")
        if len(stale) > 12:
            print(f"        ... and {len(stale) - 12} more")
        print("\n      Re-render and INSPECT:  ue_python exec tools/scene/qa_shots.py")
        return 1
    print("\nOK    the last QA render is newer than every scene file - the current scene has been seen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
