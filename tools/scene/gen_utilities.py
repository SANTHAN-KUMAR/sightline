"""Generate the FloodValley street utilities and small craft: LV poles, overhead wires, service drops, boats.

    uv run python tools/scene/gen_utilities.py [--seed 83]
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_utilities.py').read())"

Writes, all in CENTIMETRES with `usemtl` groups that become UE material slots (the convention
`gen_buildings.py` established for the 76 generated houses):

  data/scene/utilities/network.obj    ONE mesh in WORLD coordinates holding every pole, crossarm, insulator,
                                      conductor and service drop. World coordinates because a catenary spans
                                      two poles and cannot be expressed in a per-pole local frame; the actor
                                      goes to the origin with yaw +90, exactly like `Ground` and `FloodFoam`.
  data/scene/utilities/boat_*.obj     one mesh per boat archetype, LOCAL coordinates, x along the keel,
                                      z = 0 at the DESIGN WATERLINE so a boat placed at the flood level floats
                                      at the right draught with no per-boat fudge.
  data/scene/utilities.json           the layout, bounds and counts
  data/scene/utilities/preview_*.png  orthographic renders of every generated mesh - LOOK AT THESE

No CC0 asset exists for a Kerala concrete distribution pole or a country boat (see
`_downloads/assets/MANIFEST.md` "Blockers"), so they are generated as real geometry, which is exactly how the
73 houses and the 14 damaged variants were built.

WHY THE WIRES ARE 3.6 cm THICK
------------------------------
The survey camera is calibrated at f_px 2548.7 (`data/scene/camera_survey.json`), so at the 45 m survey
altitude the ground sample distance is 45 / 2548.7 = **1.766 cm/px**. A bare 8 mm LV conductor would be
0.45 px wide: TAA and the 4K downsample would erase it, and we would ship wires that exist in the mesh and not
in the image. Kerala LV distribution is overwhelmingly aerial bundled cable (ABC) - three or four 50 mm2 cores
twisted into one sheathed bundle 30-38 mm across - so 3.6 cm is both the physically right diameter and
**2.04 px** at survey altitude: a thin dark line that survives resampling. Service drops are 2.4 cm (1.36 px),
deliberately fainter. The conductors on a crossarm are 0.30 m apart = 17 px, so they resolve as separate lines
rather than a blur.

WHY THE POLES FOLLOW A CHAIN AND NOT A SCATTER
----------------------------------------------
A distribution line is a sequence, not a cloud: consecutive spans are 28-44 m and nearly collinear. Poles are
therefore chained greedily from the south end of the settlement, scoring each next site by how close its span
is to the ideal 36 m AND how little it turns the line (a pure nearest-neighbour walk produces a zig-zag that
reads as scattered posts from the air). A site is eligible when it is 7-26 m from the nearest house (a line
follows the street, not the hillside), clear of every house footprint, out of the channel and on ground
between 4 m below and 8 m above flood stage - which puts most of the poles standing IN the flood, which is
what the reference photograph shows.

BOATS
-----
Three archetypes, lofted from stations rather than boxed:
  vallam       6.2 m Kerala country boat, pointed both ends, planked, three thwarts
  dinghy       3.6 m hard-chine dinghy with a transom and a stern bench
  rescue_punt  4.8 m flat-bottomed sheet-metal punt with a squared bow - the 2018 Kerala rescue boat
Every hull carries an INWARD-facing copy of its own surface, coincident with the outer skin. UE materials are
single-sided, so without it a nadir camera looks straight through an open boat to the terrain (the same fix
`gen_damage.py` needed for opened roofs). Zero thickness means exactly one of each coincident pair faces the
camera, so there is no z-fighting.
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
import gen_terrain as gt  # noqa: E402

OUT = gt.OUT
UDIR = OUT / "utilities"

# UV tile size in metres per material slot. wall/roof_sheet/concrete/wood/window match gen_buildings.TILE_M so
# the shared material instances tile identically; `wire` is new and is deliberately tiny.
TILE_M = {"wall": 2.0, "roof_sheet": 2.0, "concrete": 2.0, "wood": 1.0, "window": 1.2, "wire": 0.4}

GSD_CM_PER_PX = 45.0 * 100.0 / 2548.723   # survey altitude / f_px, from data/scene/camera_survey.json
WIRE_D_M = 0.036                       # LV aerial bundled cable, 2.04 px at survey GSD
DROP_D_M = 0.024                       # service drop, 1.36 px
POLE_H_M = (8.4, 9.2)                  # above ground; Kerala PSC poles are 8-9.1 m
POLE_BURY_M = 1.2                      # geometry continues below ground so sloping sites never show a gap
SPAN_MIN_M, SPAN_IDEAL_M, SPAN_MAX_M = 28.0, 36.0, 44.0
CROSSARM_M = 1.9
CONDUCTOR_DY_M = 0.30
CLEAR_SURVIVOR_M = 2.5
MAX_POLES = 60            # 76 houses need one LV line, not a lattice: keep the longest chains only
MIN_CHAIN = 4


# ==========================================================================================================
# a tiny OBJ builder with vertex normals
# ==========================================================================================================
class Mesh:
    """Polygons with per-face material groups, planar metre UVs and real vertex normals.

    `gen_buildings.Mesh` writes no `vn`, which is right for a box but wrong for a tapered pole or a lofted
    hull: UE flat-shades an OBJ without normals, and a 12-station hull then reads as a faceted gem. Normals
    are accumulated per unique position so the hull smooths and the boxes stay crisp.
    """

    def __init__(self) -> None:
        self.v: list[tuple[float, float, float]] = []
        self.vt: list[tuple[float, float]] = []
        self.vn: list[np.ndarray] = []
        self.faces: dict[str, list[tuple[int, ...]]] = {}

    def poly(self, pts, mat: str, smooth: bool = False) -> None:
        """pts counter-clockwise seen from OUTSIDE (right-handed, z up)."""
        p = [np.asarray(q, dtype=float) for q in pts]
        eu = p[1] - p[0]
        nu = np.linalg.norm(eu)
        nrm = np.cross(p[1] - p[0], p[-1] - p[0])
        ln = np.linalg.norm(nrm)
        if nu < 1e-9 or ln < 1e-9:
            return                                   # degenerate (a hull station can collapse at the stem)
        eu = eu / nu
        nrm = nrm / ln
        ev = np.cross(nrm, eu)
        tile = TILE_M[mat]
        base = len(self.v)
        for q in p:
            self.v.append(tuple(q))
            self.vt.append((float(np.dot(q - p[0], eu) / tile), float(np.dot(q - p[0], ev) / tile)))
            self.vn.append(nrm.copy())
        idx = list(range(base + 1, base + len(p) + 1))
        for k in range(1, len(p) - 1):
            # A lofted hull collapses to a point at the stem, so a quad there has two coincident corners and
            # one of its two triangles has zero area. UE welds those away silently; keeping them would make
            # the imported triangle count disagree with the generator's and trip build_utilities' assertion.
            if np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) < 1e-8:
                continue
            self.faces.setdefault(mat, []).append((idx[0], idx[k], idx[k + 1]))
        if smooth:
            self._smooth_marks = getattr(self, "_smooth_marks", [])
            self._smooth_marks.extend(range(base, base + len(p)))

    def box(self, c, half, mat: str, yaw: float = 0.0) -> None:
        """Axis-aligned box of half-extents `half` at centre `c`, optionally yawed about z."""
        cy, sy = math.cos(yaw), math.sin(yaw)
        def P(sx, sy_, sz):
            x, y, z = sx * half[0], sy_ * half[1], sz * half[2]
            return (c[0] + x * cy - y * sy, c[1] + x * sy + y * cy, c[2] + z)
        self.poly([P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1)], mat)
        self.poly([P(-1, -1, -1), P(-1, 1, -1), P(1, 1, -1), P(1, -1, -1)], mat)
        self.poly([P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), P(1, -1, 1)], mat)
        self.poly([P(-1, 1, -1), P(-1, -1, -1), P(-1, -1, 1), P(-1, 1, 1)], mat)
        self.poly([P(1, 1, -1), P(-1, 1, -1), P(-1, 1, 1), P(1, 1, 1)], mat)
        self.poly([P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), P(-1, -1, 1)], mat)

    def tube(self, path, radii, sides: int, mat: str, caps: bool = True) -> None:
        """Sweep a regular `sides`-gon of radius radii[i] along `path` (a list of 3-vectors)."""
        path = [np.asarray(p, dtype=float) for p in path]
        if len(path) < 2:
            return
        rings = []
        for i, p in enumerate(path):
            t = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
            nt = np.linalg.norm(t)
            t = t / nt if nt > 1e-9 else np.array([0.0, 0.0, 1.0])
            ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            a = np.cross(t, ref)
            a /= np.linalg.norm(a)
            b = np.cross(t, a)
            r = radii[i] if isinstance(radii, (list, tuple, np.ndarray)) else radii
            rings.append([p + r * (math.cos(2 * math.pi * k / sides) * a
                                   + math.sin(2 * math.pi * k / sides) * b) for k in range(sides)])
        for i in range(len(rings) - 1):
            for k in range(sides):
                k2 = (k + 1) % sides
                self.poly([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]], mat, smooth=True)
        if caps:
            self.poly(list(reversed(rings[0])), mat)
            self.poly(rings[-1], mat)

    def bbox(self):
        a = np.asarray(self.v)
        return a.min(axis=0), a.max(axis=0)

    def triangles(self) -> int:
        return sum(len(f) for f in self.faces.values())

    def write(self, path: Path, header: str) -> int:
        # average the normals of coincident vertices that were marked smooth, so lofted surfaces shade smooth
        marks = set(getattr(self, "_smooth_marks", []))
        if marks:
            key = {}
            for i in marks:
                key.setdefault(tuple(round(c, 4) for c in self.v[i]), []).append(i)
            for group in key.values():
                acc = np.sum([self.vn[i] for i in group], axis=0)
                ln = np.linalg.norm(acc)
                if ln > 1e-9:
                    for i in group:
                        self.vn[i] = acc / ln
        lines = [f"# {header} (generated, centimetres)"]
        lines += [f"v {x * 100:.2f} {y * 100:.2f} {z * 100:.2f}" for x, y, z in self.v]
        lines += [f"vt {u:.4f} {v:.4f}" for u, v in self.vt]
        lines += [f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}" for n in self.vn]
        for mat, fs in self.faces.items():
            lines.append(f"usemtl {mat}")
            lines += ["f " + " ".join(f"{i}/{i}/{i}" for i in f) for f in fs]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return self.triangles()


# ==========================================================================================================
# preview renderer - so the shape is LOOKED AT before it ships
# ==========================================================================================================
def render_obj(mesh: Mesh, path: Path, title: str, size: int = 420) -> None:
    """Three flat-shaded orthographic views (side / front / top) by painter's algorithm.

    A wireframe would hide the failure this is here to catch - a pole that generated as a flat quad, or a hull
    whose stations collapsed - so the faces are filled and lit, and the three views make a degenerate axis
    obvious at a glance.
    """
    from PIL import Image, ImageDraw
    V = np.asarray(mesh.v, dtype=float)
    tris, cols = [], []
    palette = {"concrete": (176, 172, 165), "wood": (150, 112, 74), "roof_sheet": (140, 116, 96),
               "wire": (36, 36, 38), "wall": (198, 192, 180), "window": (40, 46, 52)}
    for mat, fs in mesh.faces.items():
        for f in fs:
            tris.append([V[i - 1] for i in f])
            cols.append(palette.get(mat, (170, 170, 170)))
    tris = np.asarray(tris)
    lo, hi = V.min(axis=0), V.max(axis=0)
    ctr, span = (lo + hi) / 2, max(float((hi - lo).max()), 1e-3)
    light = np.array([0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    views = [("side  (x,z)", 0, 2, 1), ("front (y,z)", 1, 2, 0), ("top   (x,y)", 0, 1, 2)]
    img = Image.new("RGB", (size * 3, size + 18), (250, 250, 248))
    dr = ImageDraw.Draw(img)
    for vi, (name, ax, ay, az) in enumerate(views):
        depth = tris[:, :, az].mean(axis=1)
        order = np.argsort(depth)
        ox = vi * size
        for k in order:
            t = tris[k]
            e1, e2 = t[1] - t[0], t[2] - t[0]
            nrm = np.cross(e1, e2)
            ln = np.linalg.norm(nrm)
            if ln < 1e-12:
                continue
            shade = 0.35 + 0.65 * max(0.0, float(np.dot(nrm / ln, light)))
            col = tuple(int(min(255, c * shade)) for c in cols[k])
            pts = [(ox + size / 2 + (p[ax] - ctr[ax]) / span * size * 0.88,
                    size / 2 - (p[ay] - ctr[ay]) / span * size * 0.88) for p in t]
            dr.polygon(pts, fill=col)
        dr.rectangle([ox + 1, 0, ox + size - 1, size - 1], outline=(200, 200, 195))
        dr.text((ox + 6, size + 3), name, fill=(60, 60, 60))
    dr.text((6, size + 3), "", fill=(0, 0, 0))
    dr.text((size * 3 - 240, size + 3),
            f"{title}  {mesh.triangles()} tris  bbox_m "
            f"({hi[0] - lo[0]:.2f},{hi[1] - lo[1]:.2f},{hi[2] - lo[2]:.2f})", fill=(60, 60, 60))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ==========================================================================================================
# boats
# ==========================================================================================================
def hull(L, B, D, draught, stern_frac, bow_frac, fullness, stations, mat, gunwale_mat=None):
    """Loft a hull from stations. z = 0 is the DESIGN WATERLINE, so the boat floats correctly when it is
    placed at the flood surface; the keel sits `draught` below it and the sheer `D - draught` above."""
    m = Mesh()
    gunwale_mat = gunwale_mat or mat
    xs = np.linspace(-L / 2, L / 2, stations)
    sec = []
    for x in xs:
        t = 2 * x / L                                        # -1 at the stern, +1 at the bow
        end = stern_frac if t < 0 else bow_frac              # 0 = pointed, >0 = transom / squared
        bw = (B / 2) * ((1 - abs(t) ** fullness) ** 0.55 * (1 - end) + end)
        z_top = (D - draught) * (0.72 + 0.28 * t * t)
        z_bot = -draught * max(0.06, (1 - t * t) ** 0.55)
        # four girth points from gunwale to keel
        sec.append([(x, bw, z_top),
                    (x, bw * 0.97, z_top - 0.42 * D),
                    (x, bw * 0.70, z_bot * 0.30),
                    (x, 0.0, z_bot)])
    for i in range(len(sec) - 1):
        for g in range(3):
            for side in (1, -1):
                a = sec[i][g]
                b = sec[i][g + 1]
                c = sec[i + 1][g + 1]
                d = sec[i + 1][g]
                q = [(a[0], side * a[1], a[2]), (b[0], side * b[1], b[2]),
                     (c[0], side * c[1], c[2]), (d[0], side * d[1], d[2])]
                if side < 0:
                    q = list(reversed(q))
                m.poly(q, mat, smooth=True)                    # outer skin
                m.poly(list(reversed(q)), mat, smooth=True)    # inward-facing copy: UE is single-sided
    # gunwale cap: the rim is what a nadir camera actually sees of an open boat
    cap = 0.055
    for i in range(len(sec) - 1):
        for side in (1, -1):
            a, d = sec[i][0], sec[i + 1][0]
            q = [(a[0], side * a[1], a[2]), (a[0], side * max(a[1] - cap, 0.0), a[2]),
                 (d[0], side * max(d[1] - cap, 0.0), d[2]), (d[0], side * d[1], d[2])]
            if side < 0:
                q = list(reversed(q))
            m.poly(q, gunwale_mat)
    # transom / stem faces where the section does not close to a point
    for i, t_end in ((0, stern_frac), (len(sec) - 1, bow_frac)):
        if t_end <= 0.01:
            continue
        s = sec[i]
        pts = [(s[g][0], s[g][1], s[g][2]) for g in range(4)] + \
              [(s[g][0], -s[g][1], s[g][2]) for g in (2, 1, 0)]
        m.poly(pts if i else list(reversed(pts)), mat)
    # thwarts
    for f in (-0.28, 0.05, 0.36):
        x = f * L
        t = 2 * x / L
        bw = (B / 2) * ((1 - abs(t) ** fullness) ** 0.55 * (1 - (stern_frac if t < 0 else bow_frac))
                        + (stern_frac if t < 0 else bow_frac))
        z = (D - draught) * (0.72 + 0.28 * t * t) - 0.20 * D
        m.box((x, 0.0, z), (0.11, max(bw * 0.94, 0.05), 0.035), gunwale_mat)
    return m


BOATS = {
    # name:        L,    B,    D,   draught, stern, bow, fullness, material
    "vallam": (6.20, 1.28, 0.62, 0.20, 0.00, 0.00, 1.60, "wood"),
    "dinghy": (3.60, 1.46, 0.58, 0.17, 0.42, 0.00, 2.40, "wood"),
    "rescue_punt": (4.80, 1.60, 0.62, 0.16, 0.62, 0.42, 3.60, "roof_sheet"),
}


# ==========================================================================================================
# the distribution network
# ==========================================================================================================
def add_pole(m: Mesh, e, n, z_ground, height, run_dir, crossarms: int, light: bool):
    """A tapered octagonal PSC pole with 1-2 steel crossarms, insulators and an optional street-light arm."""
    top = z_ground + height
    m.tube([(e, n, z_ground - POLE_BURY_M), (e, n, z_ground + 0.4), (e, n, top)],
           [0.135, 0.125, 0.085], 8, "concrete")
    ax = math.atan2(run_dir[1], run_dir[0]) + math.pi / 2      # crossarm across the line
    out = []
    for c in range(crossarms):
        z = top - 0.30 - 0.62 * c
        m.box((e, n, z), (CROSSARM_M / 2, 0.045, 0.045), "roof_sheet", yaw=ax)
        for k in (-1, 0, 1):
            d = k * CONDUCTOR_DY_M * 2.0
            px, py = e + d * math.cos(ax), n + d * math.sin(ax)
            m.box((px, py, z + 0.10), (0.055, 0.055, 0.060), "concrete")   # ceramic pin insulator
            out.append((px, py, z + 0.16))
    if light:
        lx, ly = e + 0.75 * math.cos(ax + math.pi), n + 0.75 * math.sin(ax + math.pi)
        m.tube([(e, n, top - 0.9), (lx, ly, top - 0.62)], 0.030, 6, "roof_sheet")
        m.box((lx, ly, top - 0.70), (0.20, 0.10, 0.055), "roof_sheet")
    return out


def catenary(p0, p1, sag, segments):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    ts = np.linspace(0.0, 1.0, segments + 1)
    return [tuple(p0 + (p1 - p0) * t - np.array([0.0, 0.0, 4.0 * sag * t * (1.0 - t)])) for t in ts]


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    actors = json.loads((OUT / "actors.json").read_text())["actors"]
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    base_z = float(h.min())
    xs = np.linspace(-size / 2, size / 2, n)
    gy, gx = np.gradient(h, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))

    def ground(e, nn):
        i = int(round((e + size / 2) / cell))
        j = int(round((nn + size / 2) / cell))
        i, j = min(max(i, 0), n - 1), min(max(j, 0), n - 1)
        return float(h[j, i]), float(slope[j, i]), bool(s["in_channel"][j, i])

    houses = town["houses"]
    arch = town["archetypes"]
    hc = np.array([[b["east_m"], b["north_m"]] for b in houses])
    sv = np.array([[a["east_m"], a["north_m"]] for a in actors])

    def clear_of_survivors(e, nn, r=CLEAR_SURVIVOR_M):
        return not bool(np.any(np.hypot(sv[:, 0] - e, sv[:, 1] - nn) < r))

    def inside_house(e, nn, margin):
        for b in houses:
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = e - b["east_m"], nn - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 + margin and abs(v) < a["width_m"] / 2 + margin:
                return True
        return False

    # --- candidate pole sites --------------------------------------------------------------------------
    lo_e, hi_e = hc[:, 0].min() - 30, hc[:, 0].max() + 30
    lo_n, hi_n = hc[:, 1].min() - 30, hc[:, 1].max() + 30
    cands = []
    for nn in np.arange(lo_n, hi_n, 6.0):
        for e in np.arange(lo_e, hi_e, 6.0):
            d = float(np.min(np.hypot(hc[:, 0] - e, hc[:, 1] - nn)))
            if not (7.0 <= d <= 26.0):
                continue
            gz, sl, in_ch = ground(e, nn)
            if in_ch or sl > 20.0 or not (water - 4.0 <= gz <= water + 8.0):
                continue
            if inside_house(e, nn, 1.6) or not clear_of_survivors(e, nn, 2.0):
                continue
            cands.append((float(e), float(nn), gz))
    cands.sort(key=lambda c: (c[1], c[0]))                    # deterministic order

    # --- chain them into lines --------------------------------------------------------------------------
    used = np.zeros(len(cands), dtype=bool)
    pts = np.array([[c[0], c[1]] for c in cands]) if cands else np.zeros((0, 2))
    chains: list[list[int]] = []
    while True:
        free = np.flatnonzero(~used)
        if not len(free):
            break
        start = int(free[0])                                  # southernmost unused site
        used[start] = True
        used |= np.hypot(pts[:, 0] - pts[start][0], pts[:, 1] - pts[start][1]) < SPAN_MIN_M * 0.92
        chain = [start]
        direction = None
        while True:
            cur = pts[chain[-1]]
            free = np.flatnonzero(~used)
            if not len(free):
                break
            d = np.hypot(pts[free, 0] - cur[0], pts[free, 1] - cur[1])
            ok = (d >= SPAN_MIN_M) & (d <= SPAN_MAX_M)
            if not ok.any():
                break
            cand_idx = free[ok]
            dd = d[ok]
            u = (pts[cand_idx] - cur) / dd[:, None]
            score = np.abs(dd - SPAN_IDEAL_M) / SPAN_IDEAL_M
            if direction is not None:
                score = score + 2.2 * (1.0 - u @ direction)   # keep the line straight, not zig-zag
            k = int(np.argmin(score))
            nxt = int(cand_idx[k])
            direction = u[k]
            used[nxt] = True
            chain.append(nxt)
            # a real line does not have a second pole 10 m away: retire nearby sites
            near = np.hypot(pts[:, 0] - pts[nxt][0], pts[:, 1] - pts[nxt][1]) < SPAN_MIN_M * 0.92
            used |= near
        if len(chain) >= MIN_CHAIN:
            chains.append(chain)
        else:
            for i in chain:
                used[i] = True
    # Keep the longest lines and stop at MAX_POLES. Every extra short chain reads as scattered posts rather
    # than a distribution line, and 76 houses do not carry 20 separate feeders.
    chains.sort(key=lambda ch: (-len(ch), pts[ch[0]][1], pts[ch[0]][0]))
    kept, total = [], 0
    for ch in chains:
        if total + len(ch) > MAX_POLES:
            continue
        kept.append(ch)
        total += len(ch)
    chains = kept

    # --- geometry ----------------------------------------------------------------------------------------
    m = Mesh()
    poles, spans, drops = [], 0, 0
    tops: dict[int, list[tuple[float, float, float]]] = {}
    for chain in chains:
        for pos, idx in enumerate(chain):
            e, nn, gz = cands[idx]
            nxt = chain[min(pos + 1, len(chain) - 1)]
            prv = chain[max(pos - 1, 0)]
            run = pts[nxt] - pts[prv]
            if np.linalg.norm(run) < 1e-6:
                run = np.array([0.0, 1.0])
            run = run / np.linalg.norm(run)
            height = float(rng.uniform(*POLE_H_M))
            crossarms = 2 if rng.random() < 0.35 else 1
            light = bool(rng.random() < 0.28)
            tops[idx] = add_pole(m, e, nn, gz, height, run, crossarms, light)
            poles.append({"east_m": round(e, 2), "north_m": round(nn, 2), "ground_asl_m": round(gz, 3),
                          "height_m": round(height, 2), "crossarms": crossarms, "street_light": light,
                          "flood_depth_m": round(max(0.0, water - gz), 2)})
        for a, b in zip(chain, chain[1:], strict=False):
            pa, pb = tops[a], tops[b]
            L = float(np.hypot(pts[b][0] - pts[a][0], pts[b][1] - pts[a][1]))
            sag = float(np.clip(0.020 * L, 0.15, 1.6))
            for ca, cb in zip(pa[:3], pb[:3], strict=False):
                m.tube(catenary(ca, cb, sag, 10), WIRE_D_M / 2, 3, "wire", caps=False)
            # neutral / messenger, slung 0.55 m below the crossarm
            na = (pa[0][0], pa[0][1], pa[0][2] - 0.55)
            nb = (pb[0][0], pb[0][1], pb[0][2] - 0.55)
            m.tube(catenary(na, nb, sag * 1.15, 10), WIRE_D_M / 2, 3, "wire", caps=False)
            spans += 1

    # --- service drops to the houses ---------------------------------------------------------------------
    for idx, top in sorted(tops.items()):
        e, nn, _gz = cands[idx]
        d = np.hypot(hc[:, 0] - e, hc[:, 1] - nn)
        for hi in np.argsort(d)[:2]:
            if d[hi] > 22.0:
                continue
            b = houses[int(hi)]
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = e - b["east_m"], nn - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            k = min(1.0, min((a["length_m"] / 2 + 0.15) / max(abs(u), 1e-3),
                             (a["width_m"] / 2 + 0.15) / max(abs(v), 1e-3)))
            wu, wv = u * k, v * k                              # the point where the ray leaves the footprint
            ax_ = b["east_m"] + wu * math.cos(th) - wv * math.sin(th)
            an_ = b["north_m"] + wu * math.sin(th) + wv * math.cos(th)
            eaves = b["base_asl_m"] + 0.45 + 3.0 * a["storeys"] - 0.35
            src = (top[0][0], top[0][1], top[0][2] - 0.95)
            m.tube(catenary(src, (ax_, an_, eaves), 0.22, 8), DROP_D_M / 2, 3, "wire", caps=False)
            drops += 1

    # The network is built in metres ASL because the poles stand on the height grid; the OBJ must carry
    # z = (asl - base_z) * 100 like flood_valley.obj and foam.obj, or the actor at the origin would float
    # base_z = 1046 m above the terrain. Shift once, here, after all the geometry exists.
    m.v = [(x, y, z - base_z) for x, y, z in m.v]
    net_tris = m.write(UDIR / "network.obj", "Sightline utility network")
    render_obj(m, UDIR / "preview_network.png", "network")
    lo, hi = m.bbox()          # already shifted to (asl - base_z)
    # A single pole, rendered alone, is the only way to see that the shaft is a real tapered prism and not a
    # flat quad - it is 3 poles wide in the full-network render and invisible.
    one = Mesh()
    add_pole(one, 0.0, 0.0, 0.0, 8.8, (0.0, 1.0), 2, True)
    render_obj(one, UDIR / "preview_pole.png", "pole")
    pole_tris = one.triangles()
    # ...and three ideal spans alone, because the sag of a 36 m catenary is 0.72 m in a 540 m wide render:
    # invisible in preview_network.png and the one thing that says "power line" rather than "fence".
    demo = Mesh()
    dtops = [add_pole(demo, 0.0, k * SPAN_IDEAL_M, 0.0, 8.8, (0.0, 1.0), 1, k == 1) for k in range(4)]
    for ta, tb in zip(dtops, dtops[1:], strict=False):
        for ca, cb in zip(ta[:3], tb[:3], strict=False):
            demo.tube(catenary(ca, cb, 0.020 * SPAN_IDEAL_M, 10), WIRE_D_M / 2, 3, "wire", caps=False)
    render_obj(demo, UDIR / "preview_span.png", "3 spans @ 36 m")

    # --- boats -------------------------------------------------------------------------------------------
    variants = {}
    for name, (L, B, D, dr_, st, bo, fu, mat) in BOATS.items():
        hm = hull(L, B, D, dr_, st, bo, fu, 15, mat, gunwale_mat="wood")
        tris = hm.write(UDIR / f"boat_{name}.obj", f"Sightline boat {name}")
        render_obj(hm, UDIR / f"preview_boat_{name}.png", f"boat_{name}")
        blo, bhi = hm.bbox()
        variants[name] = {"obj": f"utilities/boat_{name}.obj", "triangles": tris,
                          "length_m": round(float(bhi[0] - blo[0]), 3),
                          "beam_m": round(float(bhi[1] - blo[1]), 3),
                          "depth_m": round(float(bhi[2] - blo[2]), 3),
                          "freeboard_m": round(float(bhi[2]), 3), "draught_m": round(float(-blo[2]), 3),
                          "materials": sorted(hm.faces)}

    names = sorted(variants)
    items = []
    flooded = [b for b in houses if b["flood_depth_m"] > 0.8]
    order = rng.permutation(len(flooded))
    for oi in order:
        if len(items) >= 13:
            break
        b = flooded[int(oi)]
        a = arch[b["archetype"]]
        th = math.radians(b["yaw_deg"])
        side = 1.0 if rng.random() < 0.5 else -1.0
        v = side * (a["width_m"] / 2 + float(rng.uniform(2.4, 4.6)))
        u = float(rng.uniform(-0.35, 0.35)) * a["length_m"]
        e = b["east_m"] + u * math.cos(th) - v * math.sin(th)
        nn = b["north_m"] + u * math.sin(th) + v * math.cos(th)
        gz, _sl, in_ch = ground(e, nn)
        if in_ch or water - gz < 0.55 or inside_house(e, nn, 0.6) or not clear_of_survivors(e, nn):
            continue
        name = names[int(rng.integers(len(names)))]
        items.append({"id": len(items), "name": f"Boat_{len(items):03d}", "variant": name,
                      "east_m": round(e, 2), "north_m": round(nn, 2),
                      "base_asl_m": round(water + float(rng.uniform(-0.03, 0.03)), 3),
                      "yaw_deg": round((b["yaw_deg"] + float(rng.normal(0, 14))) % 360.0, 1),
                      "pitch_deg": round(float(rng.normal(0, 2.0)), 2),
                      "roll_deg": round(float(rng.normal(0, 3.0)), 2),
                      "flood_depth_m": round(water - gz, 2), "note": f"moored against house {b['id']}"})
    # a few adrift in open water, where the wrack line already puts floating debris
    props = json.loads((OUT / "props_layout.json").read_text()) if (OUT / "props_layout.json").is_file() else None
    if props:
        floats = [p for p in props["items"] if p["floats"] and p["role"] == "household"]
        for k in rng.permutation(len(floats))[:40]:
            if len(items) >= 19:
                break
            p = floats[int(k)]
            e = p["east_m"] + float(rng.uniform(2.0, 5.0)) * (1 if rng.random() < 0.5 else -1)
            nn = p["north_m"] + float(rng.uniform(2.0, 5.0)) * (1 if rng.random() < 0.5 else -1)
            gz, _sl, in_ch = ground(e, nn)
            if in_ch or water - gz < 0.6 or inside_house(e, nn, 1.0) or not clear_of_survivors(e, nn):
                continue
            if any(math.hypot(e - it["east_m"], nn - it["north_m"]) < 12.0 for it in items):
                continue
            name = names[int(rng.integers(len(names)))]
            items.append({"id": len(items), "name": f"Boat_{len(items):03d}", "variant": name,
                          "east_m": round(e, 2), "north_m": round(nn, 2),
                          "base_asl_m": round(water + float(rng.uniform(-0.03, 0.03)), 3),
                          "yaw_deg": round(float(rng.uniform(0, 360)), 1),
                          "pitch_deg": round(float(rng.normal(0, 2.5)), 2),
                          "roll_deg": round(float(rng.normal(0, 4.0)), 2),
                          "flood_depth_m": round(water - gz, 2), "note": "adrift in the wrack"})

    boat_tris = sum(variants[i["variant"]]["triangles"] for i in items)
    return {
        "generated_by": "tools/scene/gen_utilities.py", "seed": seed, "terrain_seed": meta["seed"],
        "settlement_seed": town["seed"], "water_level_m": water, "base_z_m": base_z,
        "survey": {"gsd_cm_per_px": round(GSD_CM_PER_PX, 4),
                   "wire_px": round(WIRE_D_M * 100 / GSD_CM_PER_PX, 2),
                   "drop_px": round(DROP_D_M * 100 / GSD_CM_PER_PX, 2),
                   "conductor_spacing_px": round(CONDUCTOR_DY_M * 100 / GSD_CM_PER_PX, 1)},
        "network": {
            "obj": "utilities/network.obj", "triangles": net_tris, "vertices": len(m.v),
            "poles": len(poles), "chains": len(chains), "chain_lengths": [len(c) for c in chains],
            "spans": spans, "conductors_per_span": 4, "service_drops": drops,
            "pole_triangles_each": pole_tris,
            "wire_diameter_m": WIRE_D_M, "drop_diameter_m": DROP_D_M,
            "span_m": {"min": SPAN_MIN_M, "ideal": SPAN_IDEAL_M, "max": SPAN_MAX_M},
            "poles_in_water": sum(1 for p in poles if p["flood_depth_m"] > 0.1),
            "ue_actor": {"location_cm": [0.0, 0.0, 0.0], "yaw_deg": 90.0,
                         "bounds_cm": {"x_north": [round(lo[1] * 100, 1), round(hi[1] * 100, 1)],
                                       "y_east": [round(lo[0] * 100, 1), round(hi[0] * 100, 1)],
                                       "z": [round(lo[2] * 100, 1), round(hi[2] * 100, 1)]}},
            "materials": sorted(m.faces),
        },
        "boats": {"variants": variants, "items": items, "triangles": boat_tris},
        "counts": {"poles": len(poles), "spans": spans, "service_drops": drops, "boats": len(items),
                   "network_triangles": net_tris, "boat_triangles": boat_tris,
                   "total_triangles": net_tris + boat_tris},
        "pole_list": poles,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=83)
    a = ap.parse_args()
    d = build(a.seed)
    (OUT / "utilities.json").write_text(json.dumps(d, indent=1))
    net, c, sv = d["network"], d["counts"], d["survey"]
    print(f"network: {net['poles']} poles in {net['chains']} lines {net['chain_lengths']}, "
          f"{net['spans']} spans x 4 conductors, {net['service_drops']} service drops")
    print(f"  {net['poles_in_water']} / {net['poles']} poles stand in the flood; "
          f"one pole = {net['pole_triangles_each']} tris; whole network = {net['triangles']:,} tris, "
          f"{net['vertices']:,} verts")
    print(f"  wire {net['wire_diameter_m'] * 100:.1f} cm = {sv['wire_px']} px at the {sv['gsd_cm_per_px']} "
          f"cm/px survey GSD; drops {sv['drop_px']} px; conductors {sv['conductor_spacing_px']} px apart")
    print(f"boats: {c['boats']} placed, {c['boat_triangles']:,} tris")
    for k, v in d["boats"]["variants"].items():
        print(f"  {k:12s} {v['triangles']:4d} tris  L={v['length_m']} B={v['beam_m']} "
              f"draught={v['draught_m']} freeboard={v['freeboard_m']} slots={v['materials']}")
    print(f"TOTAL added triangles: {c['total_triangles']:,}")
    print(f"written to {UDIR} and {OUT / 'utilities.json'}")
    print("NOW LOOK at", UDIR / "preview_pole.png", "and the boat previews")


if __name__ == "__main__":
    main()
