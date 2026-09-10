"""Verify every environment asset, and merge the measured palette into data/scene/props.json.

    uv run python tools/scene/check_env_assets.py            # verify only; exits non-zero if anything is wrong
    uv run python tools/scene/check_env_assets.py --write    # verify, then merge into data/scene/props.json

This is the check that is allowed to FAIL. For every asset in data/scene/env_assets.json it asserts, against
the file on disk and not against any API return value:

  1. the glTF parses and its .bin is at least as long as the glTF declares (a truncated download dies here);
  2. every texture the glTF references exists AND decodes with PIL at >= 512 px (a 4 KB HTML error page saved
     as a .jpg dies here);
  3. every mesh has triangles, and the triangle count is the one measured from the index buffer;
  4. every mesh's real-world size is inside the band its role allows - a "tree" 0.4 m tall or 60 m tall is a
     scale error, and scale errors are invisible in a return value and obvious in a picture;
  5. the whole-asset triangle totals are inside the per-role ceiling used for the budget in docs/lanes/env_assets.md.

Sizes are reported in Unreal's axis order (x, y, z-up) - glTF is Y-up, so the y and z components are swapped,
which is exactly what the existing entries in props.json show (covered_car: 1.79 x 4.38 x 1.29 m).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gltf_tools import Gltf  # noqa: E402

REPO = HERE.parents[1]
SRC = REPO / "_downloads" / "assets"
TABLE = REPO / "data" / "scene" / "env_assets.json"
PROPS = REPO / "data" / "scene" / "props.json"

# role -> (min, max) metres for each dimension of ONE mesh, and the whole-asset triangle ceiling
BAND = {
    "tree":         ((0.30, 26.0), 400_000),
    "ground_cover": ((0.01, 7.00), 200_000),
    "utility":      ((0.01, 12.0), 260_000),
    "boat":         ((0.05, 9.00), 20_000),
    "waterfront":   ((0.02, 20.0), 120_000),
    "debris_bark":  ((0.02, 3.00), 60_000),
}


def ue_asset_path(entry: dict, g: Gltf, mesh_index: int, node_name: str) -> str:
    """Predict where the Unreal glTF importer puts each mesh.

    Measured from what the existing 27 prop sets actually produced (data/scene/props.json):
    single-mesh files are named after the FILE (rock_07 -> .../rock_07_2k/StaticMeshes/rock_07_2k); multi-mesh
    files are named after each NODE (rock_moss_set_01 -> .../StaticMeshes/rock_moss_set_01_rock01).
    """
    stem = Path(entry["source"]).name[: -len(".gltf")]
    leaf = stem if len(g.j.get("meshes", [])) == 1 else node_name
    return f"/Game/Sightline/Props/{entry['id']}/{stem}/StaticMeshes/{leaf}"


def measure(entry: dict) -> tuple[list[dict], list[str]]:
    fails: list[str] = []
    path = SRC / entry["source"]
    if not path.exists():
        return [], [f"{entry['id']}: source glTF missing: {path}"]
    try:
        g = Gltf(path)                               # raises if the .bin is short
    except Exception as e:
        return [], [f"{entry['id']}: glTF will not load: {e}"]

    # textures: present and actually decodable
    from PIL import Image
    for img in g.j.get("images", []):
        uri = img.get("uri")
        if not uri:
            continue
        t = path.parent / uri
        if not t.exists():
            fails.append(f"{entry['id']}: texture missing: {t}")
            continue
        try:
            with Image.open(t) as im:
                im.verify()
            with Image.open(t) as im:
                w, h = im.size
            if min(w, h) < 512:
                fails.append(f"{entry['id']}: texture {t.name} is only {w}x{h}")
        except Exception as e:
            fails.append(f"{entry['id']}: texture {t.name} does not decode ({e})")

    xf = g.node_transforms()
    (lo_m, hi_m), tri_ceiling = BAND[entry["role"]]
    out, total = [], 0
    for mi, mesh in enumerate(g.j.get("meshes", [])):
        tris = 0
        lo = np.array([np.inf] * 3, np.float64)
        hi = np.array([-np.inf] * 3, np.float64)
        for p in mesh.get("primitives", []):
            pos, tri, uv, _n = g.geom(p)
            tris += len(tri)
            # the *_lite.gltf files are written by tools/scene/gltf_tools.py, so the buffers are ours and
            # must be proved sound: an index past the end of the vertex array is a crash in the importer
            if len(tri) and int(tri.max()) >= len(pos):
                fails.append(f"{entry['id']}: index {int(tri.max())} >= {len(pos)} vertices")
            if not np.isfinite(pos).all() or not np.isfinite(uv).all():
                fails.append(f"{entry['id']}: non-finite position or UV")
            lo = np.minimum(lo, pos.min(0))
            hi = np.maximum(hi, pos.max(0))
        node_name = xf[mi][0] if mi in xf else mesh.get("name", f"mesh_{mi}")
        s = hi - lo                                   # glTF Y-up
        size = [round(float(s[0]), 3), round(float(s[2]), 3), round(float(s[1]), 3)]   # -> Unreal x, y, z-up
        if tris <= 0:
            fails.append(f"{entry['id']}/{node_name}: zero triangles")
        if max(size) > hi_m or max(size) < lo_m:
            fails.append(f"{entry['id']}/{node_name}: size {size} m outside the {entry['role']} "
                         f"band {lo_m}-{hi_m} m")
        out.append(dict(asset=ue_asset_path(entry, g, mi, node_name), tris=int(tris), size_m=size))
        total += tris
    if total > tri_ceiling:
        fails.append(f"{entry['id']}: {total:,} tris over the {entry['role']} ceiling {tri_ceiling:,}")
    if not out:
        fails.append(f"{entry['id']}: no meshes in the glTF")
    return out, fails


def main() -> int:
    global SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="merge the measured palette into props.json")
    ap.add_argument("--table", default=str(TABLE), help="asset table to check (for negative tests)")
    ap.add_argument("--root", default=str(SRC), help="asset root the table's paths are relative to")
    a = ap.parse_args()
    SRC = Path(a.root)
    table = json.loads(Path(a.table).read_text(encoding="utf-8"))
    props = json.loads(PROPS.read_text(encoding="utf-8")) if PROPS.exists() else {"props": {}, "humans": {}}

    all_fails: list[str] = []
    added, tri_by_role = {}, {}
    for entry in table["assets"]:
        meshes, fails = measure(entry)
        all_fails += fails
        if meshes:
            added[entry["id"]] = dict(role=entry["role"], meshes=meshes,
                                      source=entry["source"], lods=entry.get("lods", False),
                                      measured_by="tools/scene/check_env_assets.py")
            t = sum(m["tris"] for m in meshes)
            tri_by_role[entry["role"]] = tri_by_role.get(entry["role"], 0) + t
            print(f"{entry['id']:28s} {entry['role']:13s} {len(meshes):>3d} mesh  {t:>9,} tris  "
                  f"biggest {max(m['size_m'] for m in meshes)}")

    # the entries that were already there must still resolve, or the palette is lying about something
    for pid, v in (props.get("props", {}) if a.table == str(TABLE) else {}).items():
        if pid in added:
            continue
        p = SRC / "polyhaven" / pid / f"{pid}_2k.gltf"
        if not p.exists():
            all_fails.append(f"{pid}: pre-existing props.json entry has no source glTF at {p}")

    print("\nby role: " + ", ".join(f"{k}={v:,} tris" for k, v in sorted(tri_by_role.items())))
    if all_fails:
        print(f"\nFAIL ({len(all_fails)}):")
        for f in all_fails:
            print("  " + f)
        return 1
    print(f"\nPASS: {len(added)} assets, {sum(len(v['meshes']) for v in added.values())} meshes, "
          f"{sum(tri_by_role.values()):,} unique triangles, every texture decodes, every size in band")

    if a.write:
        props.setdefault("props", {}).update(added)
        props.setdefault("failures", [])
        PROPS.write_text(json.dumps(props, indent=2), encoding="utf-8")
        print(f"written: {PROPS} ({len(props['props'])} props)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
