"""Generate the FloodValley settlement (SOLUTION_DOC §2.2 zone 2): Kerala-type building meshes + a seeded layout.

No free, login-free, photoreal house asset exists (see _downloads/assets/MANIFEST.md "Blockers"), so the buildings
are generated here as real geometry with metre-true UVs, and textured in Unreal with the downloaded Poly Haven
plaster / clay-tile / corrugated-iron / concrete / plank scans (tools/scene/build_buildings.py). Archetypes follow
the Wayanad/Chaliyar building stock: single-storey laterite/plaster houses with clay-tile gable roofs, two-storey
RCC houses with flat slab roofs, parapets and a roof stair cabin, and corrugated-sheet sheds. Windows, doors and
concrete sunshades ("chajja") are modelled because oblique views (§2.3 row 13) see walls.

Outputs (data/scene/):
  buildings/<archetype>.obj  one mesh per archetype, CENTIMETRES, +Z up, origin at footprint centre, z = 0 at
                             ground; walls run 1 m below ground so sloping sites never show a gap. `usemtl` groups
                             become UE material slots: wall, roof_tile, roof_sheet, concrete, window, wood.
  settlement.json            seeded layout: per house the archetype, local east/north (m), yaw (deg, long axis
                             measured from east towards north), base ASL (m), wall tint index, flood depth at the
                             default flood stage. Terrace houses stand in 0.5-3.7 m of water (people on roofs,
                             §2.3 row 2); bank houses are dry.

Run: uv run python tools/scene/gen_buildings.py [--seed 11]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_terrain as gt  # noqa: E402  (same seeded terrain, so masks and heights match the imported mesh)

OUT = gt.OUT
TILE_M = {"wall": 2.0, "roof_tile": 1.5, "roof_sheet": 2.0, "concrete": 2.0, "window": 1.2, "wood": 1.0}


class Mesh:
    """Minimal OBJ builder: quads/triangles with per-face material groups and planar metre UVs."""

    def __init__(self) -> None:
        self.v: list[tuple[float, float, float]] = []
        self.vt: list[tuple[float, float]] = []
        self.faces: dict[str, list[tuple[int, ...]]] = {}

    def poly(self, pts, mat: str) -> None:
        """pts counter-clockwise seen from OUTSIDE (right-handed, z up). u runs along pts[0]->pts[1], v along the
        in-plane perpendicular, in texture tiles."""
        p = [np.asarray(q, dtype=float) for q in pts]
        eu = p[1] - p[0]
        eu /= np.linalg.norm(eu)
        n = np.cross(p[1] - p[0], p[-1] - p[0])
        ev = np.cross(n, eu)
        ev /= np.linalg.norm(ev)
        tile = TILE_M[mat]
        base = len(self.v)
        for q in p:
            self.v.append(tuple(q))
            self.vt.append((float(np.dot(q - p[0], eu) / tile), float(np.dot(q - p[0], ev) / tile)))
        idx = list(range(base + 1, base + len(p) + 1))
        for k in range(1, len(p) - 1):
            self.faces.setdefault(mat, []).append((idx[0], idx[k], idx[k + 1]))

    def box(self, x0, y0, z0, x1, y1, z1, mat: str, top: str | None = None, bottom: bool = False) -> None:
        self.poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top or mat)
        if bottom:
            self.poly([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], mat)
        self.poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], mat)
        self.poly([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], mat)
        self.poly([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], mat)
        self.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], mat)

    def write(self, path: Path) -> int:
        lines = ["# Sightline generated building (centimetres)"]
        lines += [f"v {x * 100:.1f} {y * 100:.1f} {z * 100:.1f}" for x, y, z in self.v]
        lines += [f"vt {u:.4f} {v:.4f}" for u, v in self.vt]
        for mat, fs in self.faces.items():
            lines.append(f"usemtl {mat}")
            lines += ["f " + " ".join(f"{i}/{i}" for i in f) for f in fs]
        path.write_text("\n".join(lines), encoding="utf-8")
        return sum(len(f) for f in self.faces.values())


def openings(m: Mesh, L: float, W: float, storeys: int, storey_h: float, plinth: float, door: bool = True) -> None:
    """Windows on every wall, a door on the front (-y) wall, concrete sunshades over openings. 3 cm proud."""
    d = 0.03
    for s in range(storeys):
        z_sill = plinth + s * storey_h + 0.9
        for side in (-1, 1):  # long walls at y = -W/2 and +W/2
            y = side * (W / 2 + d)
            xs = np.arange(-L / 2 + 1.5, L / 2 - 1.0, 3.0)
            for i, xc in enumerate(xs):
                if door and s == 0 and side == -1 and i == len(xs) // 2:
                    x0, x1, z0, z1, mat = xc - 0.5, xc + 0.5, plinth, plinth + 2.1, "wood"
                else:
                    x0, x1, z0, z1, mat = xc - 0.55, xc + 0.55, z_sill, z_sill + 1.2, "window"
                if side == -1:
                    m.poly([(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)], mat)
                else:
                    m.poly([(x1, y, z0), (x0, y, z0), (x0, y, z1), (x1, y, z1)], mat)
                ys = sorted((side * W / 2, side * (W / 2 + 0.5)))
                m.box(x0 - 0.15, ys[0], z1 + 0.15, x1 + 0.15, ys[1], z1 + 0.25, "concrete")
        for side in (-1, 1):  # end walls
            x = side * (L / 2 + d)
            y0, y1, z0, z1 = -0.55, 0.55, z_sill, z_sill + 1.2
            if side == 1:
                m.poly([(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)], "window")
            else:
                m.poly([(x, y1, z0), (x, y0, z0), (x, y0, z1), (x, y1, z1)], "window")


def gable_roof(m: Mesh, L: float, W: float, h: float, pitch_deg: float, over: float, mat: str) -> None:
    t = math.tan(math.radians(pitch_deg))
    he, hr, yc = h - over * t, h + (W / 2) * t, 0.0
    x0, x1, y0, y1 = -L / 2 - over, L / 2 + over, -W / 2 - over, W / 2 + over
    m.poly([(x0, y0, he), (x1, y0, he), (x1, yc, hr), (x0, yc, hr)], mat)          # south slope
    m.poly([(x1, y1, he), (x0, y1, he), (x0, yc, hr), (x1, yc, hr)], mat)          # north slope
    tk = 0.12                                                                      # underside, seen obliquely
    m.poly([(x0, yc, hr - tk), (x1, yc, hr - tk), (x1, y0, he - tk), (x0, y0, he - tk)], "wood")
    m.poly([(x1, yc, hr - tk), (x0, yc, hr - tk), (x0, y1, he - tk), (x1, y1, he - tk)], "wood")
    for xs, sgn in ((-L / 2, -1), (L / 2, 1)):                                    # gable triangles
        if sgn < 0:
            m.poly([(xs, W / 2, h), (xs, -W / 2, h), (xs, 0.0, h + (W / 2) * t)], "wall")
        else:
            m.poly([(xs, -W / 2, h), (xs, W / 2, h), (xs, 0.0, h + (W / 2) * t)], "wall")


def mono_roof(m: Mesh, L: float, W: float, h: float, pitch_deg: float, over: float, mat: str) -> None:
    t = math.tan(math.radians(pitch_deg))
    x0, x1, y0, y1 = -L / 2 - over, L / 2 + over, -W / 2 - over, W / 2 + over
    z0, z1 = h - over * t, h + (W + over) * t
    m.poly([(x0, y0, z0), (x1, y0, z0), (x1, y1, z1), (x0, y1, z1)], mat)
    m.poly([(x0, y1, z1 - 0.05), (x1, y1, z1 - 0.05), (x1, y0, z0 - 0.05), (x0, y0, z0 - 0.05)], "wood")
    rise = W * t                                                                   # wall infill under the slope
    m.poly([(-L / 2, W / 2, h), (-L / 2, -W / 2, h), (-L / 2, W / 2, h + rise)], "wall")
    m.poly([(L / 2, -W / 2, h), (L / 2, W / 2, h), (L / 2, W / 2, h + rise)], "wall")
    m.poly([(L / 2, W / 2, h), (-L / 2, W / 2, h), (-L / 2, W / 2, h + rise), (L / 2, W / 2, h + rise)], "wall")


def flat_roof(m: Mesh, L: float, W: float, h: float, cabin: bool) -> None:
    m.box(-L / 2 - 0.3, -W / 2 - 0.3, h, L / 2 + 0.3, W / 2 + 0.3, h + 0.2, "concrete", bottom=True)
    p, ph, top = 0.15, 0.9, h + 0.2
    for x0, y0, x1, y1 in ((-L / 2 - 0.3, -W / 2 - 0.3, L / 2 + 0.3, -W / 2 - 0.3 + p),
                           (-L / 2 - 0.3, W / 2 + 0.3 - p, L / 2 + 0.3, W / 2 + 0.3),
                           (-L / 2 - 0.3, -W / 2 - 0.3, -L / 2 - 0.3 + p, W / 2 + 0.3),
                           (L / 2 + 0.3 - p, -W / 2 - 0.3, L / 2 + 0.3, W / 2 + 0.3)):
        m.box(x0, y0, top, x1, y1, top + ph, "wall", top="concrete")
    if cabin:  # stair cabin ("mumty") on the roof slab, a common flat-roof feature
        cx0, cy0 = L / 2 - 3.6, W / 2 - 3.2
        m.box(cx0, cy0, top, cx0 + 3.0, cy0 + 2.8, top + 2.4, "wall")
        m.box(cx0 - 0.2, cy0 - 0.2, top + 2.4, cx0 + 3.2, cy0 + 3.0, top + 2.55, "concrete")
        m.poly([(cx0 + 1.0, cy0 - 0.03, top), (cx0 + 1.9, cy0 - 0.03, top), (cx0 + 1.9, cy0 - 0.03, top + 2.0),
                (cx0 + 1.0, cy0 - 0.03, top + 2.0)], "wood")


ARCHETYPES = {  # name: (length, width, storeys, roof, weight in the mix)
    "A_tile_1s": (10.0, 8.0, 1, "gable", 0.32),
    "B_flat_2s": (11.0, 9.0, 2, "flat_cabin", 0.24),
    "C_sheet_1s": (8.0, 6.0, 1, "mono", 0.16),
    "D_tile_2s": (12.0, 8.5, 2, "gable", 0.14),
    "E_flat_1s": (9.0, 7.0, 1, "flat", 0.14),
}
STOREY_H, PLINTH = 3.0, 0.45


def build_archetype(name: str) -> Mesh:
    L, W, storeys, roof, _ = ARCHETYPES[name]
    m = Mesh()
    h = PLINTH + storeys * STOREY_H
    m.box(-L / 2, -W / 2, -1.0, L / 2, W / 2, h, "wall")
    m.box(-L / 2 - 0.15, -W / 2 - 0.15, -1.0, L / 2 + 0.15, W / 2 + 0.15, PLINTH, "concrete")  # plinth band
    openings(m, L, W, storeys, STOREY_H, PLINTH)
    if storeys == 2:  # floor band between storeys
        m.box(-L / 2 - 0.08, -W / 2 - 0.08, PLINTH + STOREY_H - 0.1, L / 2 + 0.08, W / 2 + 0.08,
              PLINTH + STOREY_H + 0.1, "concrete")
    if roof == "gable":
        gable_roof(m, L, W, h, 26.0, 0.6, "roof_tile")
    elif roof == "mono":
        mono_roof(m, L, W, h, 11.0, 0.5, "roof_sheet")
    else:
        flat_roof(m, L, W, h, cabin=(roof == "flat_cabin"))
    return m


def layout(seed: int, meta: dict) -> list[dict]:
    rng = np.random.default_rng(seed)
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, c, S = s["height"], s["n"], s["cell_m"], s["size_m"]
    wl = s["water_level"]
    xs = np.linspace(-S / 2, S / 2, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    gy, gx = np.gradient(h, c)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    wet = s["in_channel"] | s["bank"]
    # keep 25 m clear of the channel and banks (box-dilate the mask)
    k = int(25 / c)
    wet_d = np.zeros_like(wet)
    for dj in range(-k, k + 1, 2):
        for di in range(-k, k + 1, 2):
            wet_d |= np.roll(np.roll(wet, dj, 0), di, 1)
    L = meta["launch_site"]
    far_from_pad = np.hypot(X - L["east_m"], Y - L["north_m"]) > 70.0
    terrace = s["terrace"] & ~wet_d & (slope < 8.0) & far_from_pad
    # dry bank houses above flood stage (the valley walls are steep: 10 deg left only 3 sites of 26, seed 11)
    bank = (~s["terrace"]) & ~wet_d & ~s["fan"] & (slope < 14.0) & (h > wl + 1.0) & (h < wl + 16.0) & far_from_pad
    bank &= (np.abs(Y) < 700)
    cands = [("terrace", np.argwhere(terrace)), ("bank", np.argwhere(bank))]
    quota = {"terrace": 70, "bank": 26}
    names = list(ARCHETYPES)
    weights = np.array([ARCHETYPES[a][4] for a in names])
    weights /= weights.sum()
    houses: list[dict] = []
    placed = np.empty((0, 2))
    for zone, idx in cands:
        order = rng.permutation(len(idx))
        count = 0
        for o in order:
            if count >= quota[zone]:
                break
            j, i = idx[o]
            e, nn = float(X[j, i]), float(Y[j, i])
            if len(placed) and np.min(np.hypot(placed[:, 0] - e, placed[:, 1] - nn)) < 17.0:
                continue
            arch = str(rng.choice(names, p=weights))
            Lh, Wh = ARCHETYPES[arch][:2]
            # long axis parallel to the valley (the channel runs roughly north-south), jittered; some turned 90 deg
            tt = (S / 2 - nn) / S
            dchx = (120 * 2.2 * np.pi * np.cos(2.2 * np.pi * tt) + 40 * 5.1 * np.pi * np.cos(5.1 * np.pi * tt + 1.3)) * (-1 / S)
            theta = math.degrees(math.atan2(1.0, dchx)) + float(rng.normal(0, 10)) + (90.0 if rng.random() < 0.3 else 0.0)
            th = math.radians(theta)
            ux, uy, vx, vy = math.cos(th), math.sin(th), -math.sin(th), math.cos(th)
            corners = [(e + a * Lh / 2 * ux + b * Wh / 2 * vx, nn + a * Lh / 2 * uy + b * Wh / 2 * vy)
                       for a in (-1, 1) for b in (-1, 1)] + [(e, nn)]
            hs = []
            for ce, cn in corners:
                ci, cj = int(round((ce + S / 2) / c)), int(round((cn + S / 2) / c))
                if not (0 <= ci < n and 0 <= cj < n):
                    break
                hs.append(float(h[cj, ci]))
            if len(hs) < 5 or max(hs) - min(hs) > 1.2:   # too steep for a level footprint
                continue
            base = min(hs)
            houses.append({"id": len(houses), "archetype": arch, "zone": zone, "east_m": round(e, 2),
                           "north_m": round(nn, 2), "yaw_deg": round(theta % 360.0, 1), "base_asl_m": round(base, 3),
                           "wall_tint": int(rng.integers(0, 4)), "flood_depth_m": round(max(0.0, wl - base), 2)})
            placed = np.vstack([placed, [e, nn]])
            count += 1
    return houses


def main(seed: int) -> None:
    meta = json.loads((OUT / "flood_valley.json").read_text())
    bdir = OUT / "buildings"
    bdir.mkdir(parents=True, exist_ok=True)
    tris = {}
    for name in ARCHETYPES:
        tris[name] = build_archetype(name).write(bdir / f"{name}.obj")
    houses = layout(seed, meta)
    flooded = [x for x in houses if x["flood_depth_m"] > 0]
    out = {"generated_by": "tools/scene/gen_buildings.py", "seed": seed, "terrain_seed": meta["seed"],
           "water_level_m": meta["water_level_m"], "archetypes": {k: {"length_m": v[0], "width_m": v[1],
           "storeys": v[2], "roof": v[3], "triangles": tris[k]} for k, v in ARCHETYPES.items()},
           "houses": houses}
    (OUT / "settlement.json").write_text(json.dumps(out, indent=1))
    depths = [x["flood_depth_m"] for x in flooded]
    print(json.dumps({"archetype_tris": tris, "houses": len(houses),
                      "by_zone": {z: sum(1 for x in houses if x["zone"] == z) for z in ("terrace", "bank")},
                      "flooded": len(flooded), "flood_depth_m_min_med_max":
                      [min(depths), float(np.median(depths)), max(depths)] if depths else None}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    main(ap.parse_args().seed)
