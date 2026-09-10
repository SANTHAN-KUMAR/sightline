"""Generate the Sightline flood-valley terrain (SOLUTION_DOC §2.2) as a STATIC MESH plus zone metadata.

Why a static mesh and not a Landscape: Cosys-AirSim gives Landscape/foliage a single default instance-segmentation
colour (§5.1), which would corrupt the auto-labels the whole data plan rests on. A generated mesh keeps every
component addressable, and lets us hand the spawners exact zone geometry.

Scene (one map, 2 x 2 km, the §4 memory cap; origin = OriginGeopoint 11.4870 N 76.1450 E, 1060 m):
  zone 1 deposit fan       - upstream, 2-7 m of debris over the valley floor, broken terrain (§2.2 zone 1)
  zone 2 flooded settlement- terrace beside the channel where buildings/roofs go (§2.2 zone 2)
  zone 3 channel + banks   - meandering incised channel carrying the flood downstream (§2.2 zone 3)

Outputs (data/scene/):
  flood_valley.obj      terrain mesh, +Z up, metres, origin at the map centre (import into UE as a static mesh)
  flood_valley_zones.png  zone raster (R=fan, G=settlement, B=channel) for inspection
  flood_valley.json     bounds, cell size, water level, channel centreline, per-zone spawn masks as run-length
                        rows, and the height grid path, consumed by the spawner scripts
  flood_valley_height.npy float32 height grid for exact lookups (spawners place actors ON the surface)

Run: uv run python tools/scene/gen_terrain.py [--size-m 2048] [--cell-m 8] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "scene"


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    """Cheap separable box blur (k must be odd); avoids a scipy dependency."""
    pad = k // 2
    p = np.pad(a, pad, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    total = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return total / (k * k)


def build(size_m: float, cell_m: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = int(size_m / cell_m) + 1
    xs = np.linspace(-size_m / 2, size_m / 2, n)          # east
    ys = np.linspace(-size_m / 2, size_m / 2, n)          # north; the valley runs south (downstream)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # --- valley: a V cross-section that deepens downstream, with a meandering thalweg -------------
    t = (size_m / 2 - Y) / size_m                          # 0 upstream (north) .. 1 downstream (south)
    channel_x = 120.0 * np.sin(2.2 * np.pi * t) + 40.0 * np.sin(5.1 * np.pi * t + 1.3)   # meander
    dist = np.abs(X - channel_x)                           # horizontal distance to the channel axis
    # A single flat water plane is only physical over a gentle gradient: 8 m of fall over 2 km (0.4 %), which is
    # what lets one FloodLevel Z cover the channel, the banks and the terrace edge at once. Walls are capped at
    # 110 m so the map stays a hill valley (§2.2) rather than a canyon.
    valley_floor = 1060.0 - 8.0 * t
    wall = np.minimum(0.085 * np.clip(dist - 90.0, 0, None) ** 1.25, 110.0)
    height = valley_floor + wall

    # --- incised channel: 25-45 m wide, 3-6 m deep, deeper downstream ---------------------------
    half_w = 12.5 + 10.0 * t
    depth = 3.0 + 3.0 * t
    in_ch = dist < half_w
    height = np.where(in_ch, height - depth * (1.0 - (dist / half_w) ** 2), height)
    bank = (dist >= half_w) & (dist < half_w + 18.0)       # cut banks where clinging survivors go

    # --- deposit fan: upstream lobe of debris 2-7 m thick over the floor (§2.2 zone 1) -----------
    fan_c = np.array([-60.0, size_m / 2 - 380.0])
    r = np.hypot(X - fan_c[0], Y - fan_c[1])
    fan_lobe = np.clip(1.0 - (r / 420.0) ** 2, 0, None)
    lobe_noise = _smooth(rng.normal(0, 1, (n, n)), 9)
    fan_thick = 7.0 * fan_lobe * (0.75 + 0.25 * lobe_noise)
    fan = fan_thick > 2.0
    height = height + np.where(fan_thick > 0, fan_thick, 0.0)

    # --- settlement terrace: flat-ish bench beside the channel (§2.2 zone 2) ---------------------
    terr_mask = (X - channel_x > 60.0) & (X - channel_x < 400.0) & (Y > -300.0) & (Y < 520.0)
    terrace_h = np.where(terr_mask, valley_floor + 3.2, 0.0)  # low bench: flood stage laps its edge
    blend = _smooth(terr_mask.astype(np.float32), 25)
    height = np.where(terr_mask, terrace_h * blend + height * (1 - blend), height)

    # --- micro-relief: boulders/rubble on the fan, gentler elsewhere -----------------------------
    micro = _smooth(rng.normal(0, 1, (n, n)), 3)
    height += np.where(fan, 0.55, 0.12) * micro
    height = _smooth(height, 3)

    # Flood stage: channel bankfull plus overbank flow across the banks and the terrace edge (§2.2 zones 2-3).
    water_level = float(np.percentile(height[in_ch], 90) + 1.8)
    zones = np.zeros((n, n, 3), dtype=np.uint8)
    zones[..., 0] = (fan * 255).astype(np.uint8)
    zones[..., 1] = (terr_mask * 255).astype(np.uint8)
    zones[..., 2] = ((in_ch | bank) * 255).astype(np.uint8)

    return {"n": n, "cell_m": cell_m, "size_m": size_m, "height": height.astype(np.float32), "zones": zones,
            "water_level": water_level, "channel_x": channel_x[:, 0] if channel_x.ndim > 1 else channel_x,
            "ys": ys, "in_channel": in_ch, "bank": bank, "fan": fan, "terrace": terr_mask}


def write_obj(path: Path, h: np.ndarray, cell_m: float, size_m: float, base_z: float) -> tuple[int, int]:
    """Vertices in metres, +Z up, centred on the map origin (UE import: 1 uu = 1 cm, so scale 100 on import)."""
    n = h.shape[0]
    xs = np.linspace(-size_m / 2, size_m / 2, n)
    lines = ["# Sightline flood valley terrain (generated)"]
    for j in range(n):
        y = xs[j]
        for i in range(n):
            lines.append(f"v {xs[i]:.3f} {y:.3f} {h[j, i] - base_z:.3f}")
    for j in range(n):
        for i in range(n):
            lines.append(f"vt {i / (n - 1):.5f} {j / (n - 1):.5f}")
    for j in range(n - 1):
        for i in range(n - 1):
            a, b = j * n + i + 1, j * n + i + 2
            c, d = (j + 1) * n + i + 2, (j + 1) * n + i + 1
            lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
            lines.append(f"f {a}/{a} {c}/{c} {d}/{d}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return (n * n, (n - 1) * (n - 1) * 2)


def main(size_m: float, cell_m: float, seed: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = build(size_m, cell_m, seed)
    h, n = s["height"], s["n"]
    base_z = float(h.min())
    verts, tris = write_obj(OUT / "flood_valley.obj", h, cell_m, size_m, base_z)
    Image.fromarray(s["zones"]).save(OUT / "flood_valley_zones.png")
    np.save(OUT / "flood_valley_height.npy", h)

    # Hillshade preview so the valley can be judged before it is imported: grey relief, blue = under flood stage,
    # red/green tint = deposit fan / settlement terrace.
    gy, gx = np.gradient(h, cell_m)
    shade = np.clip((gx * 0.5 + gy * 0.5 + 1.0) / 2.0, 0, 1)
    rgb = np.stack([shade, shade, shade], axis=-1)
    flooded = h < s["water_level"]
    rgb[flooded] = rgb[flooded] * 0.35 + np.array([0.10, 0.22, 0.38]) * 0.65
    rgb[s["fan"]] = rgb[s["fan"]] * 0.7 + np.array([0.45, 0.28, 0.16]) * 0.3
    rgb[s["terrace"]] = rgb[s["terrace"]] * 0.7 + np.array([0.20, 0.42, 0.18]) * 0.3
    Image.fromarray((np.flipud(rgb) * 255).astype(np.uint8)).resize((768, 768), Image.NEAREST).save(
        OUT / "flood_valley_preview.png")
    print(f"flooded area: {100.0 * flooded.mean():.1f} % of the map")

    centreline = [{"y_m": float(y), "x_m": float(x)} for y, x in
                  zip(s["ys"][::8], (120.0 * np.sin(2.2 * np.pi * ((size_m / 2 - s["ys"][::8]) / size_m)) +
                                     40.0 * np.sin(5.1 * np.pi * ((size_m / 2 - s["ys"][::8]) / size_m) + 1.3)),
                      strict=True)]
    meta = {
        "generated_by": "tools/scene/gen_terrain.py", "seed": seed,
        "size_m": size_m, "cell_m": cell_m, "grid": n, "vertices": verts, "triangles": tris,
        "origin_geopoint": {"lat": 11.4870, "lon": 76.1450, "alt_m": 1060.0},
        "base_z_m": base_z, "height_min_m": float(h.min()), "height_max_m": float(h.max()),
        "water_level_m": s["water_level"], "water_level_rel_m": s["water_level"] - base_z,
        "ue_import": {"scale": 100.0, "note": "OBJ is in metres, UE units are cm; place at world origin"},
        "zones": {"fan_cells": int(s["fan"].sum()), "settlement_cells": int(s["terrace"].sum()),
                  "channel_cells": int(s["in_channel"].sum()), "bank_cells": int(s["bank"].sum())},
        "channel_centreline": centreline,
        "files": {"mesh": "flood_valley.obj", "zones_png": "flood_valley_zones.png",
                  "height_npy": "flood_valley_height.npy"},
    }
    (OUT / "flood_valley.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("grid", "vertices", "triangles", "height_min_m", "height_max_m",
                                           "water_level_m", "zones")}, indent=2))
    print("written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-m", type=float, default=2048.0)
    ap.add_argument("--cell-m", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.size_m, a.cell_m, a.seed)
