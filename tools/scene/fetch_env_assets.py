"""Download the environment (canopy / ground-cover / utility / boat) assets from Poly Haven.

    uv run python tools/scene/fetch_env_assets.py            # download + verify everything in WANTED
    uv run python tools/scene/fetch_env_assets.py --verify   # re-verify what is on disk, download nothing

Same layout as the 27 prop sets already in `_downloads/assets/polyhaven/<id>/`: `<id>_2k.gltf`, `<id>.bin`,
`textures/*.jpg`. Every file is checked against the size AND md5 the Poly Haven API reports, so a truncated
download or an HTML error page fails here rather than showing up later as a hole in the scene. All CC0 1.0.

Why these assets: the reference photo is dominated by mature broadleaf canopy (we had none), then dense ground
vegetation, utility poles with wires, small boats and rafted debris. See docs/lanes/env_assets.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "_downloads" / "assets" / "polyhaven"
META = REPO / "_downloads" / "assets" / "_meta"
UA = {"User-Agent": "Sightline-asset-fetch/1.0 (research project)"}
RES = "2k"

# id -> (role for props.json, one-line why)
WANTED: dict[str, tuple[str, str]] = {
    # 1. mature broadleaf canopy - the single biggest gap vs the reference photo
    "island_tree_01": ("tree", "mature tropical broadleaf, wide low canopy"),
    "island_tree_02": ("tree", "mature tropical broadleaf, taller/narrower"),
    "island_tree_03": ("tree", "mature tropical broadleaf, largest crown"),
    "jacaranda_tree": ("tree", "tall mature broadleaf, high open crown"),
    "tree_small_02": ("tree", "medium broadleaf (Burkea africana), yard-scale tree"),
    # 2. dense ground vegetation between the buildings
    "grass_medium_01": ("ground_cover", "grass clump, green+dry variants"),
    "grass_medium_02": ("ground_cover", "grass clump, second silhouette"),
    "grass_bermuda_01": ("ground_cover", "low lawn/verge grass"),
    "shrub_01": ("ground_cover", "flowering shrub (Ageratina adenophora)"),
    "shrub_02": ("ground_cover", "leafy shrub"),
    "shrub_03": ("ground_cover", "leafy shrub"),
    "shrub_04": ("ground_cover", "leafy shrub"),
    "nettle_plant": ("ground_cover", "tall weed, roadside/verge"),
    "weed_plant_02": ("ground_cover", "tall weed, roadside/verge"),
    "anthurium_botany_01": ("ground_cover", "broad glossy tropical leaves (Kerala understorey)"),
    "calathea_orbifolia_01": ("ground_cover", "broad tropical leaves"),
    "pachira_aquatica_01": ("ground_cover", "Malabar chestnut sapling - a wetland species"),
    # 3. utility poles with wires
    "modular_electricity_poles": ("utility", "wooden/concrete power poles, transformers, strung cable"),
    "modular_electric_cables": ("utility", "cable runs and junction boxes"),
    # 4. boats and waterfront structure
    "ship_pinnace": ("boat", "open clinker-built wooden boat (see report: rig is stripped in the lite build)"),
    "modular_wooden_pier": ("waterfront", "plank jetty on poles"),
    "lifebuoy": ("waterfront", "ring buoy - flood-rescue dressing"),
    "life_jacket": ("waterfront", "life jacket - flood-rescue dressing"),
    # 5. debris rafted against trees and walls
    "bark_debris_01": ("woody", "loose bark and woody litter for the wrack line"),
}


def api(path: str, cache: Path) -> dict:
    if not cache.exists():
        req = urllib.request.Request(f"https://api.polyhaven.com/{path}", headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            cache.write_bytes(r.read())
        time.sleep(0.2)
    return json.loads(cache.read_text(encoding="utf-8"))


def md5_of(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, out: Path, size: int, md5: str, verify_only: bool) -> str:
    """Returns 'ok' / 'downloaded'; raises on any mismatch. Never leaves a bad file in place."""
    if out.exists() and out.stat().st_size == size and md5_of(out) == md5:
        return "ok"
    if verify_only:
        raise RuntimeError(f"MISSING or CORRUPT: {out} (want {size} B, md5 {md5})")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
    got_size, got_md5 = tmp.stat().st_size, md5_of(tmp)
    if got_size != size or got_md5 != md5:
        tmp.unlink()
        raise RuntimeError(f"BAD DOWNLOAD {url}: size {got_size}!={size} or md5 {got_md5}!={md5}")
    os.replace(tmp, out)
    return "downloaded"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="check what is on disk; download nothing")
    ap.add_argument("--only", nargs="*", help="subset of asset ids")
    a = ap.parse_args()
    META.mkdir(parents=True, exist_ok=True)

    ids = [i for i in WANTED if not a.only or i in a.only]
    total_bytes = 0
    rows = []
    for aid in ids:
        info = api(f"info/{aid}", META / f"i_{aid}.json")
        files = api(f"files/{aid}", META / f"f_{aid}.json")
        node = files["gltf"][RES]["gltf"]
        d = DEST / aid
        jobs = [(node["url"], d / f"{aid}_{RES}.gltf", node["size"], node["md5"])]
        for rel, v in (node.get("include") or {}).items():
            jobs.append((v["url"], d / rel, v["size"], v["md5"]))
        states = []
        for url, out, size, md5 in jobs:
            states.append(fetch(url, out, size, md5, a.verify))
            total_bytes += size
        mb = sum(j[2] for j in jobs) / 1e6
        authors = ", ".join((info.get("authors") or {}).keys())
        rows.append(dict(id=aid, role=WANTED[aid][0], why=WANTED[aid][1], mb=round(mb, 1),
                         authors=authors, files=len(jobs),
                         state="ok" if set(states) == {"ok"} else "downloaded"))
        print(f"{aid:28s} {rows[-1]['state']:10s} {mb:7.1f} MB  {len(jobs)} files  [{authors}]")
    print(f"\n{len(rows)} assets, {total_bytes/1e6:.1f} MB total, all size+md5 verified against the Poly Haven API")
    (META / "env_assets_fetch.json").write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
