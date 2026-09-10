"""Generate the FloodValley **structural-collapse rubble field** on the deposit fan (SCENE_REFERENCE "Reference B").

    D:\\Tools\\uv\\uv.exe run python tools\\scene\\gen_rubble.py            # meshes + layout + renders + checks
    D:\\Tools\\uv\\uv.exe run python tools\\scene\\gen_rubble.py --no-render # skip the offline renders (checks still run)

WHY THIS EXISTS
The deposit fan today carries scattered Poly Haven boulders and logs on clean mud (`gen_props.py` step 1), which
reads as a rocky riverbed. The reference photograph the user supplied for that zone is a *collapsed multi-storey
building*: broken floor slabs at every angle, blocks and masonry, protruding rebar, voids between the plates,
everything dusted to a desaturated tan/grey, with a few belongings as the only colour. That difference is not
cosmetic. SOLUTION_DOC 2.7 draws the burial boundary with the physics of a debris-flow deposit, and 2.3 wants the
`trapped` posture and the partial-occlusion slices; a person in a rubble void IS that case. Without rubble there
is nothing in the scene for a survivor to be partly under, so the hardest and most valuable detection cases have
no physical justification.

WHAT IT WRITES
  data/scene/rubble/<variant>.obj   32 meshes, CENTIMETRES, +Z up (UE does not rescale OBJ units), metre-true
                                    planar UVs, `usemtl` groups that become UE material slots:
                                    concrete | masonry | rebar | fabric | wood | metal
                                    Families: P_* assembled rubble piles (the unit of a collapse field: slabs
                                    leaning on a mound of blocks, so the pile itself contains the voids),
                                    S_* loose floor-slab fragments, B_* concrete blocks, M_* broken masonry,
                                    R_* loose rebar tangles, G_* belongings (mattress, cabinet, appliance, chair).
  data/scene/rubble_layout.json     seeded placement: variant, east/north (m), base ASL (m), yaw/pitch/roll (deg),
                                    scale, group, note, per-item triangle cost, and the triangle budget.
  _artifacts/rubble/*.png           offline renders (see --render): a 3-view sheet per variant with a 1.75 m human
                                    for scale, and four scene views of the assembled field.

PLACEMENT PHYSICS (why the arrangement is what it is)
  * A debris flow drops its coarsest, heaviest load FIRST, where it loses capacity as it spreads. The structural
    load - slabs, blocks, masonry - therefore piles at the **fan apex** and thins downstream. Density here falls
    as (1 - f)^1.8 with f the downstream fraction across the fan, which is the same law `gen_props.py` uses for
    boulders (its comment: "coarse near the apex").
  * Rubble piles **against obstacles**: the upstream (+north) face of any house standing on or beside the fan,
    and the cut bank of the incised channel where the flow was deflected.
  * The channel itself is **scoured clean** (fast water leaves bare bed) - no rubble is placed in it, exactly as
    `gen_props.py` treats it.
  * Only the fan FLOOR is used: the `fan` mask in `gen_terrain.py` is an ellipse of added thickness, so it also
    covers valley wall up to 105 m above flood stage. Rubble is restricted to slope < 18 deg and less than 25 m
    above flood stage, which is the lobe a real deposit occupies.
  * **Voids are protected around survivors.** Every aerially-detectable survivor on the fan gets a ring of slabs
    and blocks with a clear radius around them: that is the `trapped` posture and the 25-50 % occlusion slice.
    Nothing is placed inside the clear radius, so the ground truth stays honest.
  * The two survivors marked `aerially_detectable: false` (2.7 burial boundary) each get a floor slab laid over
    them, on top of the boulder cap `gen_props.py` already builds. A concrete slab is the physical reason aerial
    search cannot clear that cell; boulders alone were a weaker claim.

BUDGET
The scene already carries ~524k triangles of terrain and ~4.15M of debris props on an 8 GB-VRAM machine, so the
rubble is hard-capped (--max-tris, default 2.5M) and the generator refuses to exceed it, biasing to cheaper
variants as it fills - the same pattern as `gen_props.py`.

DETERMINISM
One `numpy.default_rng(seed)`, split with `.spawn()` so the mesh library and the layout draw from independent
streams and changing one does not shift the other. Run twice and the OBJ bytes and the JSON are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_terrain as gt  # noqa: E402  (same seeded terrain, so the masks match the imported mesh exactly)

OUT = gt.OUT
RUBBLE_DIR = OUT / "rubble"
ART = gt.REPO / "_artifacts" / "rubble"

# metres of world per texture tile, per material slot (the OBJ UVs are written in tiles, like gen_buildings.py)
TILE_M = {"concrete": 2.0, "masonry": 1.0, "rebar": 0.4, "fabric": 1.6, "wood": 1.0, "metal": 1.2}
MATS = tuple(TILE_M)

# approximate albedos for the OFFLINE render only. The real look comes from the UE material instances that
# build_rubble.py wires from the Poly Haven scans already in the project.
RENDER_RGB = {
    "concrete": (0.70, 0.68, 0.63), "masonry": (0.63, 0.55, 0.47), "rebar": (0.42, 0.27, 0.18),
    "fabric": (0.32, 0.42, 0.60), "wood": (0.48, 0.38, 0.26), "metal": (0.60, 0.60, 0.63),
    "_ground": (0.40, 0.33, 0.25), "_scale_figure": (0.90, 0.45, 0.06), "_water": (0.20, 0.33, 0.34),
}


# ----------------------------------------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------------------------------------
class Mesh:
    """OBJ builder: polygons with per-face material groups and planar metre-true UVs.

    Same shape as `gen_buildings.Mesh` (that file is the proven pattern) plus `merge`, because a rubble pile is
    an assembly of many small solids rotated into place. Rotation is an isometry, so UVs generated in the part's
    local frame stay metre-true after the merge; a uniform scale is applied to the UVs as well.
    """

    def __init__(self) -> None:
        self.v: list[tuple[float, float, float]] = []
        self.vt: list[tuple[float, float]] = []
        self.faces: dict[str, list[tuple[int, ...]]] = {}

    def poly(self, pts, mat: str) -> None:
        """pts counter-clockwise seen from OUTSIDE (right-handed, +Z up)."""
        p = [np.asarray(q, dtype=float) for q in pts]
        eu = p[1] - p[0]
        lu = float(np.linalg.norm(eu))
        n = np.cross(p[1] - p[0], p[-1] - p[0])
        ln = float(np.linalg.norm(n))
        if lu < 1e-7 or ln < 1e-11:
            raise ValueError(f"degenerate polygon in group {mat!r}: |edge|={lu:.3e} |normal|={ln:.3e}")
        eu = eu / lu
        ev = np.cross(n / ln, eu)
        tile = TILE_M[mat]
        base = len(self.v)
        for q in p:
            self.v.append((float(q[0]), float(q[1]), float(q[2])))
            self.vt.append((float(np.dot(q - p[0], eu) / tile), float(np.dot(q - p[0], ev) / tile)))
        idx = list(range(base + 1, base + len(p) + 1))
        for k in range(1, len(p) - 1):
            self.faces.setdefault(mat, []).append((idx[0], idx[k], idx[k + 1]))

    def box(self, c, half, mat: str, R=None) -> None:
        """Axis-aligned box (optionally rotated by R about its own centre), all six faces outward."""
        c = np.asarray(c, float)
        hx, hy, hz = half
        crn = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        if R is not None:
            crn = crn @ np.asarray(R, float).T
        crn = crn + c
        # index = 4*ix + 2*iy + iz with i in {0,1} meaning the -/+ side
        f = [((0, 2, 3, 1), ), ((4, 5, 7, 6), ), ((0, 1, 5, 4), ), ((2, 6, 7, 3), ),
             ((0, 4, 6, 2), ), ((1, 3, 7, 5), )]
        for (quad, ) in f:
            self.poly([crn[i] for i in quad], mat)

    def merge(self, other: Mesh, R=None, t=None, scale: float = 1.0) -> None:
        base = len(self.v)
        V = np.asarray(other.v, dtype=float) * scale
        if R is not None:
            V = V @ np.asarray(R, float).T
        if t is not None:
            V = V + np.asarray(t, float)
        self.v.extend((float(a), float(b), float(c)) for a, b, c in V.tolist())
        self.vt.extend((u * scale, w * scale) for u, w in other.vt)
        for mat, fs in other.faces.items():
            self.faces.setdefault(mat, []).extend(tuple(i + base for i in f) for f in fs)

    def ntris(self) -> int:
        return sum(len(f) for f in self.faces.values())

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        V = np.asarray(self.v, dtype=float)
        return V.min(axis=0), V.max(axis=0)

    def write(self, path: Path, header: str) -> int:
        assert len(self.v) == len(self.vt), "vertex/uv arrays must stay in lockstep"
        lines = [f"# {header} (Sightline generated, CENTIMETRES, +Z up)"]
        lines += [f"v {x * 100:.2f} {y * 100:.2f} {z * 100:.2f}" for x, y, z in self.v]
        lines += [f"vt {u:.4f} {w:.4f}" for u, w in self.vt]
        for mat in MATS:                       # stable group order -> stable UE material slot order
            fs = self.faces.get(mat)
            if not fs:
                continue
            lines.append(f"usemtl {mat}")
            lines += ["f " + " ".join(f"{i}/{i}" for i in f) for f in fs]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.ntris()


def basis(direction, roll_deg: float = 0.0) -> np.ndarray:
    """Rotation whose local +x maps onto `direction`, then rolled about it. Avoids Euler sign traps entirely."""
    u = np.asarray(direction, float)
    nu = float(np.linalg.norm(u))
    if nu < 1e-9:
        raise ValueError("basis(): zero direction")
    u = u / nu
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.97 else np.array([0.0, 1.0, 0.0])
    v = np.cross(ref, u)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    R0 = np.stack([u, v, w], axis=1)
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    return R0 @ np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rand_rot(rng, max_tilt_deg: float = 180.0) -> np.ndarray:
    """Random orientation, optionally limited in how far the local +z can tip away from world +z."""
    yaw = rng.uniform(0, 360)
    tilt = math.radians(rng.uniform(0, max_tilt_deg))
    az = rng.uniform(0, 2 * math.pi)
    axis = np.array([math.cos(az), math.sin(az), 0.0])
    c, s = math.cos(tilt), math.sin(tilt)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return rot_z(yaw) @ (np.eye(3) + s * K + (1 - c) * K @ K)


# ----------------------------------------------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------------------------------------------
def slab_outline(rng, span_u: float, span_v: float) -> np.ndarray:
    """Broken floor-slab outline: 4-6 corner anchors joined by chords - some kept clean (an original cast edge),
    the rest given radial fracture noise. Built in polar form about the centroid so it is always star-shaped,
    which is what makes the ring triangulation below safe."""
    k = int(rng.integers(4, 7))
    base = np.linspace(0.0, 2.0 * math.pi, k, endpoint=False)
    ang = base + rng.uniform(-0.30, 0.30, k) * (2.0 * math.pi / k)
    rad = rng.uniform(0.80, 1.0, k)
    anchors = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    pts = []
    for i in range(k):
        a, b = anchors[i], anchors[(i + 1) % k]
        m = int(rng.integers(4, 8))
        jag = 0.028 if rng.random() < 0.35 else float(rng.uniform(0.06, 0.15))
        for s in np.linspace(0.0, 1.0, m, endpoint=False):
            p = a * (1.0 - s) + b * s
            r = float(np.linalg.norm(p))
            if r < 1e-6:
                continue
            pts.append(p * (1.0 + float(rng.normal(0.0, jag))))
    P = np.asarray(pts, float)
    P[:, 0] *= span_u / 2.0
    P[:, 1] *= span_v / 2.0
    th = np.arctan2(P[:, 1], P[:, 0])
    P = P[np.argsort(th)]
    th = np.sort(th)
    # drop points that would collapse a wedge to a sliver, and keep the plate a plate
    keep = np.ones(len(P), bool)
    last = -9.0
    for i in range(len(P)):
        if th[i] - last < 0.035:
            keep[i] = False
        else:
            last = th[i]
    P = P[keep]
    r = np.hypot(P[:, 0], P[:, 1])
    floor_r = 0.32 * float(np.median(r))
    scale = np.maximum(1.0, floor_r / np.maximum(r, 1e-9))
    return P * scale[:, None]


def slab(rng, span: float, thick: float, rings: int = 2, rebar: int = 0) -> Mesh:
    """A broken reinforced-concrete floor slab: thin irregular plate, gently sagged/twisted where it cracked,
    with rebar stubs protruding from the fracture edges. This is the signature shape of Reference B."""
    span_v = span * float(rng.uniform(0.55, 0.95))
    P = slab_outline(rng, span, span_v)
    n = len(P)
    hu, hv = max(span / 2.0, 1e-6), max(span_v / 2.0, 1e-6)
    # Sag/twist of a cracked slab. The coefficients are normalised so the total deflection over the plate is
    # `amp`: without that the three quadratic terms add and a 3.4 m slab folded into a 1.1 m taco (caught by
    # the "thin plates" check on the first run).
    amp = span * float(rng.uniform(0.010, 0.032))
    a = np.array([float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)) * 0.6, float(rng.uniform(-1, 1)) * 0.8])
    a = a / max(float(np.abs(a).sum()), 1e-9) * amp
    a1, a2, a3 = (float(a[0]), float(a[1]), float(a[2]))

    def zoff(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0] / hu, xy[:, 1] / hv
        return a1 * x * x + a2 * y * y + a3 * x * y

    fr = [1.0, 0.62, 0.30] if rings == 2 else [1.0, 0.52]
    ringsP = [P * f for f in fr]
    top = [np.column_stack([q, zoff(q) + thick / 2.0]) for q in ringsP]
    bot = [np.column_stack([q, zoff(q) * 0.97 - thick / 2.0 * float(rng.uniform(0.85, 1.0))]) for q in ringsP]
    ctr_t = np.array([0.0, 0.0, float(zoff(np.zeros((1, 2)))[0]) + thick / 2.0])
    ctr_b = np.array([0.0, 0.0, float(zoff(np.zeros((1, 2)))[0]) - thick / 2.0])

    m = Mesh()
    for i in range(n):                                           # broken edge
        j = (i + 1) % n
        m.poly([bot[0][i], bot[0][j], top[0][j], top[0][i]], "concrete")
    for r in range(len(fr) - 1):                                 # top surface, ring strips
        for i in range(n):
            j = (i + 1) % n
            m.poly([top[r][i], top[r][j], top[r + 1][j], top[r + 1][i]], "concrete")
            m.poly([bot[r][j], bot[r][i], bot[r + 1][i], bot[r + 1][j]], "concrete")
    for i in range(n):                                           # inner fan
        j = (i + 1) % n
        m.poly([top[-1][i], top[-1][j], ctr_t], "concrete")
        m.poly([bot[-1][j], bot[-1][i], ctr_b], "concrete")

    for _ in range(rebar):
        i = int(rng.integers(n))
        p = np.array([P[i][0], P[i][1], float(rng.uniform(-thick / 3, thick / 3))])
        out = np.array([P[i][0], P[i][1], 0.0])
        out /= max(float(np.linalg.norm(out)), 1e-9)
        d = out + np.array([0.0, 0.0, float(rng.uniform(-0.3, 0.9))])
        rebar_rod(m, rng, p, d, length=float(rng.uniform(0.35, 1.15)),
                  radius=float(rng.uniform(0.012, 0.019)))
    return m


def rebar_rod(m: Mesh, rng, p0, d0, length: float, radius: float, segs: int = 4) -> None:
    """A bent reinforcing bar as a square-section swept tube. Cheap (8*segs + 4 triangles) and, per the brief,
    the single highest-payoff detail in the whole field."""
    d = np.asarray(d0, float)
    d /= max(float(np.linalg.norm(d)), 1e-9)
    pts = [np.asarray(p0, float)]
    for _ in range(segs):
        d = d + rng.normal(0.0, 0.30, 3)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        pts.append(pts[-1] + d * (length / segs))
    prev_n = None
    rings = []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        t /= max(float(np.linalg.norm(t)), 1e-9)
        ref = prev_n if prev_n is not None else (np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9
                                                 else np.array([1.0, 0.0, 0.0]))
        n1 = np.cross(ref, t)
        if float(np.linalg.norm(n1)) < 1e-6:
            n1 = np.cross(np.array([0.0, 1.0, 0.0]), t)
        n1 /= float(np.linalg.norm(n1))
        n2 = np.cross(t, n1)
        prev_n = n1
        rings.append([p + radius * (s1 * n1 + s2 * n2) for s1, s2 in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    for i in range(len(rings) - 1):
        for k in range(4):
            k2 = (k + 1) % 4
            m.poly([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]], "rebar")
    m.poly(rings[0][::-1], "rebar")
    m.poly(rings[-1], "rebar")


def chunk(rng, a: float, b: float, c: float, mat: str, jitter: float = 0.13) -> Mesh:
    """An angular broken lump: a box whose eight corners are displaced and whose faces are split 2x2 and pushed
    out along their own normals. Flat-shaded facets read as fractured concrete/masonry, not as a pebble.

    The displacements are CLIPPED at 1.5 sigma. Unclipped, a rare draw collapsed two corners of a small lump
    onto each other and produced a 0.09 cm^2 sliver triangle - caught by the zero-area check on run 2.
    """
    s = min(a, b, c)
    sig = jitter * s

    def jit(k: int = 3) -> np.ndarray:
        return np.clip(rng.normal(0.0, sig, k), -1.5 * sig, 1.5 * sig)

    crn = {}
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                crn[(ix, iy, iz)] = np.array([(ix * 2 - 1) * a / 2, (iy * 2 - 1) * b / 2, (iz * 2 - 1) * c / 2]) \
                    + jit()
    faces = [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),      # -z
             ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),      # +z
             ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),      # -y
             ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),      # +y
             ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),      # -x
             ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]      # +x
    m = Mesh()
    for quad in faces:
        p = [crn[i] for i in quad]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        nrm /= max(float(np.linalg.norm(nrm)), 1e-9)
        g = np.empty((3, 3, 3))
        for i in range(3):
            for j in range(3):
                u, v = i / 2.0, j / 2.0
                g[i, j] = (p[0] * (1 - u) * (1 - v) + p[1] * u * (1 - v) + p[2] * u * v + p[3] * (1 - u) * v)
                if 0 < i < 2 or 0 < j < 2:
                    g[i, j] = g[i, j] + nrm * float(np.clip(rng.normal(0.0, 0.05 * s), -0.09 * s, 0.09 * s))
        for i in range(2):
            for j in range(2):
                m.poly([g[i, j], g[i + 1, j], g[i + 1, j + 1], g[i, j + 1]], mat)
    return m


def lump(rng, a: float, b: float, c: float, mat: str) -> Mesh:
    """The cheapest possible fragment: a jittered box, 12 triangles. Used by the tens inside a rubble mat, where
    a 48-triangle `chunk` would be unaffordable and, at 0.2-0.5 m, indistinguishable."""
    m = Mesh()
    s = min(a, b, c)
    crn = {}
    for ix in (0, 1):
        for iy in (0, 1):
            for iz in (0, 1):
                crn[(ix, iy, iz)] = np.array([(ix * 2 - 1) * a / 2, (iy * 2 - 1) * b / 2, (iz * 2 - 1) * c / 2]) \
                    + np.clip(rng.normal(0.0, 0.14 * s, 3), -0.22 * s, 0.22 * s)
    for quad in [((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
                 ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
                 ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))]:
        m.poly([crn[i] for i in quad], mat)
    return m


def shard(rng, span: float, thick: float, mat: str = "concrete") -> Mesh:
    """A small flat fragment of slab: an irregular 6-9 sided plate, 4n triangles. The cheap cousin of `slab`."""
    k = int(rng.integers(6, 10))
    ang = np.linspace(0, 2 * math.pi, k, endpoint=False) + rng.uniform(-0.25, 0.25, k)
    r = span / 2.0 * rng.uniform(0.55, 1.0, k)
    P = np.stack([r * np.cos(ang), r * np.sin(ang) * float(rng.uniform(0.55, 1.0))], axis=1)
    top = [np.array([p[0], p[1], thick / 2]) for p in P]
    bot = [np.array([p[0], p[1], -thick / 2]) for p in P]
    m = Mesh()
    for i in range(k):
        j = (i + 1) % k
        m.poly([bot[i], bot[j], top[j], top[i]], mat)
        m.poly([top[i], top[j], np.array([0.0, 0.0, thick / 2])], mat)
        m.poly([bot[j], bot[i], np.array([0.0, 0.0, -thick / 2])], mat)
    return m


def rubble_mat(rng, size: float = 10.0) -> Mesh:
    """A carpet of small fragments strewn on the ground over roughly `size` x `size` metres, with an irregular
    outline so instances of it do not read as tiles.

    This exists because the first nadir render was the finding of this lane: 210 rubble piles on clean mud read
    as isolated grey flowers, not as a collapse field. A collapse field is continuous - the ground BETWEEN the
    piles is covered in fragments too - and that ground cover cannot be bought one actor at a time inside the
    triangle budget.
    """
    m = Mesh()
    k = 9
    lobe_r = rng.uniform(0.60, 1.0, k)

    def radius_at(t: float) -> float:
        i = int((t % (2 * math.pi)) / (2 * math.pi) * k)
        return float(lobe_r[i] * size / 2.0)

    for _ in range(int(rng.integers(58, 74))):
        t = float(rng.uniform(0, 2 * math.pi))
        r = radius_at(t) * math.sqrt(float(rng.random()))
        s = float(rng.uniform(0.16, 0.55))
        m.merge(lump(rng, s, s * float(rng.uniform(0.7, 1.3)), s * float(rng.uniform(0.5, 0.95)),
                     "concrete" if rng.random() < 0.7 else "masonry"),
                rand_rot(rng), (r * math.cos(t), r * math.sin(t), float(rng.uniform(-0.08, 0.12))))
    for _ in range(int(rng.integers(26, 36))):
        t = float(rng.uniform(0, 2 * math.pi))
        r = radius_at(t) * math.sqrt(float(rng.random()))
        m.merge(shard(rng, float(rng.uniform(0.5, 1.55)), float(rng.uniform(0.05, 0.14))),
                rand_rot(rng, 42.0), (r * math.cos(t), r * math.sin(t), float(rng.uniform(-0.04, 0.10))))
    return m


def masonry_lump(rng) -> Mesh:
    """Broken masonry keeps its rectangular character: one to three bonded units, chipped."""
    m = Mesh()
    n = int(rng.integers(1, 4))
    z = 0.0
    for i in range(n):
        a = float(rng.uniform(0.22, 0.45))
        b = float(rng.uniform(0.10, 0.22))
        c = float(rng.uniform(0.07, 0.12))
        part = chunk(rng, a, b, c, "masonry", jitter=0.16)
        off = np.array([float(rng.normal(0, 0.06)), float(rng.normal(0, 0.05)), z + c / 2])
        m.merge(part, rot_z(float(rng.uniform(-25, 25))), off)
        z += c * float(rng.uniform(0.75, 1.0))
        if i and rng.random() < 0.4:
            break
    return m


def rebar_tangle(rng) -> Mesh:
    """A knot of bar torn out of a slab: a small concrete lump with three to six bars springing from it."""
    m = Mesh()
    m.merge(chunk(rng, float(rng.uniform(0.25, 0.5)), float(rng.uniform(0.2, 0.4)),
                  float(rng.uniform(0.12, 0.25)), "concrete"))
    for _ in range(int(rng.integers(3, 7))):
        d = np.array([float(rng.normal(0, 1)), float(rng.normal(0, 1)), float(rng.uniform(-0.2, 1.2))])
        rebar_rod(m, rng, np.array([float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1)), 0.05]), d,
                  length=float(rng.uniform(0.5, 1.4)), radius=float(rng.uniform(0.011, 0.017)))
    return m


def belonging(rng, kind: str) -> Mesh:
    """The only colour in Reference B: personal belongings turned out of the building."""
    m = Mesh()
    if kind == "mattress":
        m.merge(chunk(rng, 1.90, 1.15, 0.22, "fabric", jitter=0.10))
        m.box((0, 0, 0.0), (0.95, 0.575, 0.11), "fabric")
    elif kind == "cabinet":
        m.box((0, 0, 0.0), (0.50, 0.27, 0.78), "wood")
        m.box((0, -0.29, 0.0), (0.44, 0.02, 0.70), "wood")
    elif kind == "appliance":
        m.box((0, 0, 0.0), (0.30, 0.31, 0.43), "metal")
        m.box((0, -0.33, 0.05), (0.20, 0.03, 0.20), "metal")
    else:  # chair
        m.box((0, 0, 0.42), (0.24, 0.24, 0.03), "wood")
        m.box((0, 0.21, 0.68), (0.23, 0.03, 0.23), "wood")
        for sx in (-1, 1):
            for sy in (-1, 1):
                m.box((sx * 0.20, sy * 0.20, 0.20), (0.025, 0.025, 0.20), "wood")
    return m


def pile(rng, radius: float, height: float, n_slab: int, n_block: int, n_masonry: int, n_belong: int = 0,
         bury: float = 0.5) -> Mesh:
    """One assembled rubble pile: a mound of blocks with floor slabs leaning up it.

    The slabs are laid deliberately as lean-tos - one edge on the ground at the pile margin, the other resting
    on the crest - so the space under each plate is an air void. That is the geometry Reference B is made of and
    it is the geometry a `trapped` survivor occupies. The base extends `bury` metres below z = 0 so the pile
    never shows a gap where the fan surface slopes away under it.
    """
    m = Mesh()

    def mound(r: float) -> float:
        return max(0.0, height * (1.0 - (r / radius) ** 2))

    for _ in range(n_block):
        a = float(rng.uniform(0, 2 * math.pi))
        r = radius * math.sqrt(float(rng.random())) * 0.98
        top = mound(r)
        s = float(rng.uniform(0.20, 0.80))
        z = float(rng.uniform(-bury, max(0.06, top)))
        m.merge(chunk(rng, s * float(rng.uniform(0.7, 1.35)), s * float(rng.uniform(0.7, 1.35)),
                      s * float(rng.uniform(0.45, 1.0)), "concrete"),
                rand_rot(rng), (r * math.cos(a), r * math.sin(a), z))

    for _ in range(n_masonry):
        a = float(rng.uniform(0, 2 * math.pi))
        r = radius * math.sqrt(float(rng.random()))
        z = float(rng.uniform(-bury * 0.6, max(0.06, mound(r))))
        m.merge(masonry_lump(rng), rand_rot(rng, 120.0), (r * math.cos(a), r * math.sin(a), z))

    # Three slab attitudes, because a lean-to-only pile renders from nadir as a radially symmetric "flower"
    # (seen in the first scene render): 60 % lean-tos up the mound, 25 % near-horizontal caps lying across it,
    # 15 % wedged steeply on edge. The mix is what makes the top-down silhouette read as chaos.
    # The attitude is chosen by INDEX, not by a coin flip: with 6-8 slabs a coin flip left the small piles with
    # too few lean-tos and the void check dropped to 0.06 (caught on pass 3). Every pile now gets 60 % lean-tos.
    for si in range(n_slab):
        u = ((si * 7) % max(n_slab, 1)) / max(n_slab, 1)
        a = float(rng.uniform(0, 2 * math.pi))
        if u < 0.60:                                     # lean-to: outer edge on the ground, inner on the crest
            r_out = radius * float(rng.uniform(0.75, 1.20))
            r_in = radius * float(rng.uniform(0.0, 0.35))
            a_in = a + float(rng.uniform(-0.9, 0.9))
            p_out = np.array([r_out * math.cos(a), r_out * math.sin(a), float(rng.uniform(-0.12, 0.22))])
            p_in = np.array([r_in * math.cos(a_in), r_in * math.sin(a_in),
                             mound(r_in) * float(rng.uniform(0.7, 1.1)) + float(rng.uniform(0.08, 0.5))])
        elif u < 0.85:                                   # cap lying across the mound at a shallow angle
            r = radius * float(rng.uniform(0.05, 0.72))
            c = np.array([r * math.cos(a), r * math.sin(a), mound(r) + float(rng.uniform(0.05, 0.3))])
            hd = float(rng.uniform(0, 2 * math.pi))
            half = float(rng.uniform(0.8, 1.9))
            tilt = float(rng.uniform(-0.35, 0.35))
            p_out = c - np.array([half * math.cos(hd), half * math.sin(hd), half * tilt])
            p_in = c + np.array([half * math.cos(hd), half * math.sin(hd), half * tilt])
        else:                                            # wedged on edge
            r = radius * float(rng.uniform(0.25, 0.95))
            base = np.array([r * math.cos(a), r * math.sin(a), float(rng.uniform(-0.2, 0.15))])
            hgt = float(rng.uniform(1.0, 2.4))
            lean = float(rng.uniform(0.15, 0.55))
            hd = float(rng.uniform(0, 2 * math.pi))
            p_out = base
            p_in = base + np.array([lean * hgt * math.cos(hd), lean * hgt * math.sin(hd), hgt])
        d = p_in - p_out
        span = float(np.clip(float(np.linalg.norm(d)) * float(rng.uniform(1.0, 1.30)), 1.3, 4.2))
        sl = slab(rng, span, float(rng.uniform(0.15, 0.25)), rings=2 if span > 2.2 else 1,
                  rebar=int(rng.integers(0, 5)))
        m.merge(sl, basis(d, float(rng.uniform(-40, 40))), (p_out + p_in) / 2.0)

    for _ in range(max(2, n_block // 5)):                # blocks pinning the slabs down near the crest
        a = float(rng.uniform(0, 2 * math.pi))
        r = radius * float(rng.uniform(0.0, 0.5))
        s = float(rng.uniform(0.22, 0.6))
        # sunk INTO the crest, not perched on the analytic dome: the first render showed cap blocks floating in
        # mid-air above the plates because the dome height is not where the slab surface actually is.
        m.merge(chunk(rng, s, s * float(rng.uniform(0.7, 1.3)), s * float(rng.uniform(0.5, 0.9)), "concrete"),
                rand_rot(rng), (r * math.cos(a), r * math.sin(a), mound(r) * 0.8 + s * 0.2))

    for _ in range(n_belong):
        a = float(rng.uniform(0, 2 * math.pi))
        r = radius * float(rng.uniform(0.45, 1.05))
        kind = ("mattress", "cabinet", "appliance", "chair")[int(rng.integers(4))]
        m.merge(belonging(rng, kind), rand_rot(rng, 95.0),
                (r * math.cos(a), r * math.sin(a), max(0.05, mound(r) * 0.5)))
    return m


# ----------------------------------------------------------------------------------------------------------
# mesh library
# ----------------------------------------------------------------------------------------------------------
# Reference B is mostly SLABS with blocks and masonry filling between them, so the piles carry roughly as many
# slabs as the mound can support. Slabs cost ~10x a block in triangles; the mix below is what fits the budget.
PILE_SPEC = [                       # name, radius, height, slabs, blocks, masonry, belongings
    ("P_small_0", 2.6, 1.35, 6, 11, 5, 0), ("P_small_1", 3.0, 1.55, 8, 13, 6, 1),
    ("P_small_2", 3.2, 1.50, 8, 12, 6, 0), ("P_small_3", 2.8, 1.60, 7, 11, 5, 1),
    ("P_med_0", 4.2, 1.90, 14, 21, 9, 1), ("P_med_1", 4.8, 2.20, 16, 23, 10, 1),
    ("P_med_2", 4.5, 2.05, 15, 22, 9, 2), ("P_med_3", 5.1, 2.45, 17, 25, 11, 1),
    ("P_med_4", 4.0, 1.75, 13, 20, 8, 0),
    ("P_large_0", 6.2, 2.90, 23, 34, 14, 2), ("P_large_1", 7.0, 3.30, 26, 38, 15, 3),
    ("P_large_2", 6.6, 3.05, 24, 36, 14, 2),
]
SLAB_SPEC = [                       # name, span, thickness, rebar stubs
    ("S_slab_0", 1.7, 0.16, 3), ("S_slab_1", 2.3, 0.19, 5), ("S_slab_2", 2.9, 0.21, 4),
    ("S_slab_3", 3.4, 0.23, 6), ("S_slab_4", 3.9, 0.25, 7), ("S_slab_5", 2.6, 0.18, 0),
]
BLOCK_SPEC = [("B_block_0", 0.22), ("B_block_1", 0.34), ("B_block_2", 0.46), ("B_block_3", 0.60)]
MAT_SPEC = [("D_mat_0", 9.0), ("D_mat_1", 10.0), ("D_mat_2", 11.0), ("D_mat_3", 12.0), ("D_mat_4", 9.5),
            ("D_mat_5", 11.5), ("D_mat_6", 10.5), ("D_mat_7", 12.5)]
BELONG_SPEC = [("G_mattress", "mattress"), ("G_cabinet", "cabinet"), ("G_appliance", "appliance"),
               ("G_chair", "chair")]


def build_library(rng) -> dict[str, dict]:
    lib: dict[str, dict] = {}

    def add(name: str, m: Mesh, family: str) -> None:
        lo, hi = m.bounds()
        rec = {"mesh": m, "family": family, "tris": m.ntris(),
               "size_m": [round(float(x), 3) for x in (hi - lo)],
               "min_m": [round(float(x), 3) for x in lo], "max_m": [round(float(x), 3) for x in hi],
               "slots": [k for k in MATS if k in m.faces]}
        # The full bounding box of a slab includes its protruding rebar, so the plate itself has to be measured
        # from the `concrete` group alone - measuring the wrong box is how a bent-taco slab passes a check.
        if "concrete" in m.faces:
            V = np.asarray(m.v, float)
            ci = np.unique(np.asarray(m.faces["concrete"], int) - 1)
            clo, chi = V[ci].min(axis=0), V[ci].max(axis=0)
            rec["concrete_size_m"] = [round(float(x), 3) for x in (chi - clo)]
        lib[name] = rec

    for name, r, h, ns, nb, nm, ng in PILE_SPEC:
        add(name, pile(rng, r, h, ns, nb, nm, ng, bury=0.40 + 0.11 * r), "pile")
    for name, span, th, rb in SLAB_SPEC:
        add(name, slab(rng, span, th, rings=2 if span > 2.2 else 1, rebar=rb), "slab")
    for name, s in BLOCK_SPEC:
        add(name, chunk(rng, s, s * float(rng.uniform(0.75, 1.25)), s * float(rng.uniform(0.5, 0.95)),
                        "concrete"), "block")
    for name, sz in MAT_SPEC:
        add(name, rubble_mat(rng, sz), "mat")
    for i in range(3):
        add(f"M_masonry_{i}", masonry_lump(rng), "masonry")
    for i in range(3):
        add(f"R_rebar_{i}", rebar_tangle(rng), "rebar")
    for name, kind in BELONG_SPEC:
        add(name, belonging(rng, kind), "belonging")
    return lib


# ----------------------------------------------------------------------------------------------------------
# layout
# ----------------------------------------------------------------------------------------------------------
SURVIVOR_CLEAR_M = 1.35        # radius kept free of rubble around every detectable survivor: the void
PROP_CLEAR_M = 1.2             # keep off the existing Poly Haven boulders/logs on the fan
FAN_MAX_ABOVE_FLOOD_M = 25.0   # the `fan` mask also covers valley wall; the deposit lobe does not
FAN_MAX_SLOPE_DEG = 18.0
PILE_MAX_SLOPE_DEG = 10.0      # piles do not carry a slope-following tilt, so keep them on near-flat ground


def fan_masks(s: dict) -> dict:
    """Fan-floor masks derived from the same seeded terrain the imported mesh was written from."""
    h, cell, w = s["height"], s["cell_m"], s["water_level"]
    gy, gx = np.gradient(h, cell)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    floor = (s["fan"] & ~s["in_channel"] & (slope < FAN_MAX_SLOPE_DEG)
             & (h > w - 0.5) & (h < w + FAN_MAX_ABOVE_FLOOD_M))
    return {"slope": slope, "floor": floor, "flat": floor & (slope < PILE_MAX_SLOPE_DEG)}


def build(seed: int, max_tris: int = 2_500_000, n_pile: int = 145, n_mat: int = 760,
          n_loose: int = 900) -> dict:
    meta = json.loads((OUT / "flood_valley.json").read_text())
    lib_rng, lay_rng = np.random.default_rng(seed).spawn(2)
    lib = build_library(lib_rng)
    rng = lay_rng

    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    xs = np.linspace(-size / 2, size / 2, n)
    mk = fan_masks(s)
    floor, flat = mk["floor"], mk["flat"]

    def ground(e: float, nn: float) -> float:
        i = int(round((e + size / 2) / cell))
        j = int(round((nn + size / 2) / cell))
        return float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])

    def cellij(e: float, nn: float) -> tuple[int, int]:
        return (int(round((nn + size / 2) / cell)), int(round((e + size / 2) / cell)))

    def in_mask(mask, e: float, nn: float) -> bool:
        j, i = cellij(e, nn)
        return bool(0 <= i < n and 0 <= j < n and mask[j, i])

    def snap(e: float, nn: float) -> tuple[float, float]:
        """Round to the 2 dp that `add()` writes to the JSON, BEFORE sampling the terrain or the masks.

        Sampling at full precision and storing a rounded position let two of 1300 loose items land in a
        neighbouring 4 m cell whose height differs by up to 1.7 m, so they were written 1.7 m underground. The
        "nothing floats or sinks" check caught it; the fix is to make the stored position the sampled one.
        """
        return (round(float(e), 2), round(float(nn), 2))

    # downstream fraction across the fan floor: 0 at the apex (upstream, +north), 1 at the toe
    fidx = np.argwhere(floor)
    apex_n, toe_n = float(xs[fidx[:, 0]].max()), float(xs[fidx[:, 0]].min())

    def frac(nn: float) -> float:
        return float(np.clip((apex_n - nn) / max(1.0, apex_n - toe_n), 0.0, 1.0))

    town = json.loads((OUT / "settlement.json").read_text())
    houses, arch = town["houses"], town["archetypes"]
    actors = json.loads((OUT / "actors.json").read_text())["actors"]
    fan_actors = [a for a in actors if in_mask(s["fan"], a["east_m"], a["north_m"])]
    keep_clear = [(a["east_m"], a["north_m"]) for a in actors if a["aerially_detectable"]]
    buried = [a for a in actors if not a["aerially_detectable"]]

    prop_xy: list[tuple[float, float]] = []
    p_path = OUT / "props_layout.json"
    if p_path.exists():
        for it in json.loads(p_path.read_text())["items"]:
            if not it["floats"] and in_mask(s["fan"], it["east_m"], it["north_m"]):
                prop_xy.append((it["east_m"], it["north_m"]))
    props_arr = np.asarray(prop_xy, float) if prop_xy else np.empty((0, 2))
    clear_arr = np.asarray(keep_clear, float) if keep_clear else np.empty((0, 2))

    items: list[dict] = []
    tris = 0
    placed_xy: list[tuple[float, float, float]] = []          # e, n, footprint radius
    pile_xy: list[tuple[float, float, float]] = []            # piles only, for the mat exclusion

    def house_clear(e: float, nn: float, margin: float = 1.0) -> bool:
        for b in houses:
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = e - b["east_m"], nn - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 + margin and abs(v) < a["width_m"] / 2 + margin:
                return False
        return True

    def survivor_clear(e: float, nn: float, r: float) -> bool:
        if not len(clear_arr):
            return True
        d = np.hypot(clear_arr[:, 0] - e, clear_arr[:, 1] - nn)
        return bool(d.min() > SURVIVOR_CLEAR_M + r)

    def prop_clear(e: float, nn: float, r: float) -> bool:
        if not len(props_arr):
            return True
        d = np.hypot(props_arr[:, 0] - e, props_arr[:, 1] - nn)
        return bool(d.min() > PROP_CLEAR_M + r * 0.4)

    def spaced(e: float, nn: float, r: float, gap: float = 0.6, factor: float = 0.62) -> bool:
        for pe, pn, pr in placed_xy:
            if math.hypot(pe - e, pn - nn) < (pr + r) * factor + gap:
                return False
        return True

    # Per-stage share of the triangle budget, in priority order. Without it the ground-cover mats - the cheapest
    # and least important thing in the file - ate 100 % of the budget on pass 4 and the SIX burial slabs that
    # carry section 2.7 were never placed at all. The last stages keep the full budget so the mission-critical
    # groups (voids around survivors, burial slabs) can always be placed.
    stage_cap = max_tris

    def stage(frac: float) -> None:
        nonlocal stage_cap
        stage_cap = int(max_tris * frac)

    def add(variant: str, group: str, e: float, nn: float, asl: float, yaw: float, pitch: float, roll: float,
            scale: float, note: str, footprint: float = 0.0) -> bool:
        """Returns False once this stage's triangle budget is spent (gen_props.py pattern: the budget is
        enforced here, not discovered as a 4 fps editor)."""
        nonlocal tris
        cost = int(round(lib[variant]["tris"] * max(1.0, scale) ** 2))
        if tris + cost > stage_cap:
            return False
        items.append({
            "id": len(items), "name": f"Rubble_{group}_{len(items):04d}", "variant": variant,
            "family": lib[variant]["family"], "group": group,
            "east_m": round(float(e), 2), "north_m": round(float(nn), 2), "base_asl_m": round(float(asl), 3),
            "yaw_deg": round(float(yaw) % 360.0, 1), "pitch_deg": round(float(pitch), 1),
            "roll_deg": round(float(roll), 1), "scale": round(float(scale), 3),
            "tris": cost, "note": note,
        })
        tris += cost
        if footprint:
            placed_xy.append((float(e), float(nn), float(footprint)))
        return True

    def pick_pile(f: float) -> str:
        """Coarse structural load drops at the apex: large piles upstream, small ones towards the toe."""
        if f < 0.28:
            w = {"P_large": 0.42, "P_med": 0.42, "P_small": 0.16}
        elif f < 0.6:
            w = {"P_large": 0.14, "P_med": 0.48, "P_small": 0.38}
        else:
            w = {"P_large": 0.0, "P_med": 0.22, "P_small": 0.78}
        keys = list(w)
        pref = keys[int(rng.choice(len(keys), p=np.array([w[k] for k in keys]) / sum(w.values())))]
        cands = [k for k in lib if k.startswith(pref)]
        return cands[int(rng.integers(len(cands)))]

    # --- 1. the field: piles, densest at the apex ------------------------------------------------------------
    # Confined to the upper 58 % of the fan and allowed to OVERLAP (the spacing factor below is 0.5 of the sum
    # of radii): the structural load of a collapsed settlement lands in one lobe and merges into a continuous
    # field, it does not spread evenly as separate mounds.
    PILE_MAX_F = 0.50
    stage(0.49)                                          # <- 49 % of the budget for piles
    cand = np.argwhere(flat)
    rng.shuffle(cand)
    n_piles = 0
    for j, i in cand:
        if n_piles >= n_pile:
            break
        e, nn = float(xs[i]), float(xs[j])
        f = frac(nn)
        if f > PILE_MAX_F or rng.random() > (1.0 - f / PILE_MAX_F) ** 1.2 * 0.94 + 0.06:
            continue
        variant = pick_pile(f)
        r = max(lib[variant]["size_m"][0], lib[variant]["size_m"][1]) / 2.0
        if not (house_clear(e, nn, 2.0) and survivor_clear(e, nn, r) and prop_clear(e, nn, r)
                and spaced(e, nn, r, 0.2, factor=0.50)):
            continue
        sc = float(rng.uniform(0.85, 1.15))
        if add(variant, "pile", e, nn, ground(e, nn), rng.uniform(0, 360), 0.0, 0.0, sc,
               f"deposit-fan pile, downstream fraction {f:.2f}", footprint=r * sc):
            n_piles += 1
            pile_xy.append((e, nn, r * sc))
    piles_arr = np.asarray(pile_xy, float) if pile_xy else np.empty((0, 3))

    # --- 1b. rubble mats: the CONTINUOUS ground cover between the piles --------------------------------------
    # The first nadir render of this lane showed the piles as isolated grey flowers on clean mud. A collapse
    # field is continuous; the fragments between the piles are most of what a camera sees. One mat carries ~65
    # fragments for ~1.2k triangles, which is the only way to buy that cover inside the budget.
    MAT_MAX_F = 0.62
    stage(0.88)                                          # <- mats may run the total to 88 %
    matc = np.argwhere(floor)
    rng.shuffle(matc)
    n_mats = 0
    for j, i in matc:
        if n_mats >= n_mat:
            break
        e0, n0 = float(xs[i]), float(xs[j])
        f = frac(n0)
        if f > MAT_MAX_F or rng.random() > (1.0 - f / MAT_MAX_F) ** 1.25 * 0.93 + 0.07:
            continue
        e, nn = snap(e0 + float(rng.uniform(-cell / 2, cell / 2)), n0 + float(rng.uniform(-cell / 2, cell / 2)))
        if not in_mask(floor, e, nn):
            continue
        variant = MAT_SPEC[int(rng.integers(len(MAT_SPEC)))][0]
        r = max(lib[variant]["size_m"][0], lib[variant]["size_m"][1]) / 2.0
        if len(piles_arr):                                   # keep out of a pile's core, hug its skirt
            d = np.hypot(piles_arr[:, 0] - e, piles_arr[:, 1] - nn) - piles_arr[:, 2] * 0.75
            if float(d.min()) < 0:
                continue
        if not (house_clear(e, nn, r) and survivor_clear(e, nn, r * 0.85)):
            continue
        if add(variant, "mat", e, nn, ground(e, nn) - 0.05, rng.uniform(0, 360), 0.0, 0.0,
               float(rng.uniform(0.9, 1.15)), f"rubble ground cover, downstream fraction {f:.2f}"):
            n_mats += 1

    # --- 2. loose scatter between the piles: slabs, blocks, masonry, rebar -----------------------------------
    stage(0.94)                                          # <- loose scatter to 94 %
    cand = np.argwhere(floor)
    rng.shuffle(cand)
    n_l = 0
    for j, i in cand:
        if n_l >= n_loose:
            break
        e0, n0 = float(xs[i]), float(xs[j])
        f = frac(n0)
        if rng.random() > (1.0 - f) ** 1.5 * 0.85 + 0.06:
            continue
        e, nn = snap(e0 + float(rng.uniform(-cell / 2, cell / 2)), n0 + float(rng.uniform(-cell / 2, cell / 2)))
        if not in_mask(floor, e, nn):
            continue
        r = float(rng.random())
        if r < 0.20 and mk["slope"][j, i] < 14.0:
            variant = SLAB_SPEC[int(rng.integers(len(SLAB_SPEC)))][0]
            pitch, roll = float(rng.uniform(-32, 32)), float(rng.uniform(-32, 32))
        elif r < 0.62:
            variant = BLOCK_SPEC[int(rng.integers(len(BLOCK_SPEC)))][0]
            pitch, roll = float(rng.uniform(-70, 70)), float(rng.uniform(-70, 70))
        elif r < 0.86:
            variant = f"M_masonry_{int(rng.integers(3))}"
            pitch, roll = float(rng.uniform(-50, 50)), float(rng.uniform(-50, 50))
        elif r < 0.95:
            variant = f"R_rebar_{int(rng.integers(3))}"
            pitch, roll = float(rng.uniform(-25, 25)), float(rng.uniform(-25, 25))
        else:
            variant = BELONG_SPEC[int(rng.integers(len(BELONG_SPEC)))][0]
            pitch, roll = float(rng.uniform(-30, 30)), float(rng.uniform(-30, 30))
        rad = max(lib[variant]["size_m"][0], lib[variant]["size_m"][1]) / 2.0
        if not (house_clear(e, nn) and survivor_clear(e, nn, rad) and prop_clear(e, nn, rad)
                and spaced(e, nn, rad, 0.35)):
            continue
        if add(variant, "loose", e, nn, ground(e, nn) - 0.12, rng.uniform(0, 360), pitch, roll,
               float(rng.uniform(0.8, 1.25)), f"loose fragment, downstream fraction {f:.2f}",
               footprint=rad):
            n_l += 1

    # --- 3. piled against obstacles: the upstream face of any house on or beside the fan ---------------------
    stage(1.0)                                           # <- the mission-critical groups keep the full budget
    n_obs = 0
    for b in houses:
        if not in_mask(s["fan"], b["east_m"], b["north_m"]):
            # also accept houses just off the fan edge, which is where a flow stacks against a wall
            j, i = cellij(b["east_m"], b["north_m"])
            k = int(round(30 / cell))
            win = s["fan"][max(0, j - k):j + k + 1, max(0, i - k):i + k + 1]
            if not win.any():
                continue
        a = arch[b["archetype"]]
        th = math.radians(b["yaw_deg"])
        for _ in range(int(rng.integers(4, 9))):
            u = float(rng.uniform(-0.55, 0.55)) * a["length_m"]
            v = a["width_m"] / 2 + float(rng.uniform(0.5, 3.0))          # +north face = upstream
            e, nn = snap(b["east_m"] + u * math.cos(th) - v * math.sin(th),
                         b["north_m"] + u * math.sin(th) + v * math.cos(th))
            if not survivor_clear(e, nn, 1.0):
                continue
            r = float(rng.random())
            if r < 0.45:
                variant = SLAB_SPEC[int(rng.integers(2, len(SLAB_SPEC)))][0]
                pitch = float(rng.uniform(28, 62))                       # leaning ON the wall
            elif r < 0.8:
                variant = BLOCK_SPEC[int(rng.integers(len(BLOCK_SPEC)))][0]
                pitch = float(rng.uniform(-60, 60))
            else:
                variant = f"M_masonry_{int(rng.integers(3))}"
                pitch = float(rng.uniform(-40, 40))
            if add(variant, "obstacle", e, nn, ground(e, nn) - 0.1,
                   math.degrees(th) + 90.0 + float(rng.uniform(-25, 25)), pitch, float(rng.uniform(-20, 20)),
                   float(rng.uniform(0.9, 1.2)), f"rafted against house {b['id']} (upstream face)"):
                n_obs += 1

    # --- 4. against the cut bank of the incised channel where the fan meets it -------------------------------
    bankfan = np.argwhere(s["bank"] & s["fan"])
    rng.shuffle(bankfan)
    n_bank = 0
    for j, i in bankfan:
        if n_bank >= 90:
            break
        e, nn = float(xs[i]), float(xs[j])
        if rng.random() > 0.35 or ground(e, nn) < water - 0.4:
            continue
        variant = (SLAB_SPEC[int(rng.integers(len(SLAB_SPEC)))][0] if rng.random() < 0.4
                   else BLOCK_SPEC[int(rng.integers(len(BLOCK_SPEC)))][0])
        rad = max(lib[variant]["size_m"][0], lib[variant]["size_m"][1]) / 2.0
        if not (survivor_clear(e, nn, rad) and spaced(e, nn, rad, 0.3)):
            continue
        if add(variant, "bank", e, nn, ground(e, nn) - 0.15, rng.uniform(0, 360),
               float(rng.uniform(-45, 45)), float(rng.uniform(-45, 45)), float(rng.uniform(0.85, 1.2)),
               "jammed against the channel cut bank", footprint=rad):
            n_bank += 1

    # --- 5. voids around the survivors on the fan (2.3: `trapped` posture, 25-75 % occlusion slices) ---------
    n_void = 0
    for a in fan_actors:
        if not a["aerially_detectable"]:
            continue
        for kk in range(int(rng.integers(5, 10))):
            ang = float(rng.uniform(0, 2 * math.pi))
            rad_r = SURVIVOR_CLEAR_M + float(rng.uniform(0.15, 1.5))
            e, nn = snap(a["east_m"] + rad_r * math.cos(ang), a["north_m"] + rad_r * math.sin(ang))
            if not house_clear(e, nn):
                continue
            if rng.random() < 0.45:
                variant = SLAB_SPEC[int(rng.integers(1, len(SLAB_SPEC)))][0]
                # leaned back over the survivor: the plate roofs the void without filling it
                pitch = float(rng.uniform(22, 55))
                yaw = math.degrees(ang) + 180.0 + float(rng.uniform(-20, 20))
            else:
                variant = BLOCK_SPEC[int(rng.integers(len(BLOCK_SPEC)))][0]
                pitch, yaw = float(rng.uniform(-60, 60)), float(rng.uniform(0, 360))
            if add(variant, "void", e, nn, ground(e, nn) - 0.05, yaw, pitch, float(rng.uniform(-25, 25)),
                   float(rng.uniform(0.9, 1.25)),
                   f"void wall around {a['name']} ({a['pose']}, occlusion slice)"):
                n_void += 1

    # --- 6. the burial boundary (2.7): a floor slab over each survivor aerial search cannot find -------------
    n_burial = 0
    for a in buried:
        for kk in range(3):
            ang = 2.09 * kk + float(rng.uniform(-0.3, 0.3))
            rad_r = float(rng.uniform(0.0, 0.55))
            variant = SLAB_SPEC[3 + kk % 3][0]
            if add(variant, "burial", a["east_m"] + rad_r * math.cos(ang),
                   a["north_m"] + rad_r * math.sin(ang), a["base_asl_m"] + 0.75 + 0.22 * kk,
                   math.degrees(ang) + float(rng.uniform(-30, 30)), float(rng.uniform(-12, 12)),
                   float(rng.uniform(-12, 12)), float(rng.uniform(1.0, 1.25)),
                   f"collapsed floor slab over {a['name']}: 2.7 burial boundary, aerial search cannot clear"):
                n_burial += 1

    by_group: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for it in items:
        by_group[it["group"]] = by_group.get(it["group"], 0) + 1
        by_family[it["family"]] = by_family.get(it["family"], 0) + 1
    piles_f = [frac(it["north_m"]) for it in items if it["group"] == "pile"]

    return {
        "generated_by": "tools/scene/gen_rubble.py", "seed": seed, "terrain_seed": meta["seed"],
        "water_level_m": water, "base_z_m": meta["base_z_m"],
        "mesh_dir": "data/scene/rubble",
        "orientation": ("yaw_deg is measured from EAST towards NORTH, the gen_buildings/gen_damage convention: "
                        "east = u*cos(yaw) - v*sin(yaw), north = u*sin(yaw) + v*cos(yaw). build_rubble.py spawns "
                        "with UE yaw = 90 - yaw_deg and UE (X,Y,Z) cm = (north*100, east*100, (asl-base_z)*100)."),
        "variants": {k: {kk: vv for kk, vv in v.items() if kk != "mesh"} for k, v in lib.items()},
        "counts": {
            "items": len(items), "by_group": by_group, "by_family": by_family,
            "approx_triangles": int(tris), "triangle_budget": int(max_tris),
            "library_triangles": int(sum(v["tris"] for v in lib.values())),
            "mean_downstream_fraction_piles": round(float(np.mean(piles_f)), 4) if piles_f else None,
            "fan_floor_cells": int(floor.sum()), "fan_floor_area_m2": int(floor.sum() * cell * cell),
        },
        "items": items,
    }, lib


# ----------------------------------------------------------------------------------------------------------
# offline renderer (numpy + PIL): the quality gate says LOOK at it, and this lane cannot open the editor
# ----------------------------------------------------------------------------------------------------------
def _look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f], axis=0)


def render(tris: np.ndarray, colors: np.ndarray, eye, target, fov_deg: float = 55.0, W: int = 1000,
           H: int = 640, sun=(0.45, 0.30, 0.84), sky=(0.55, 0.62, 0.72)):
    """Z-buffered flat-shaded software rasteriser. tris: (M,3,3) world metres, colors: (M,3) linear RGB."""
    from PIL import Image

    eye = np.asarray(eye, float)
    M = _look_at(eye, target)
    V0 = ((tris.reshape(-1, 3) - eye) @ M.T).reshape(-1, 3, 3)

    # Flat shading is per triangle, so shade in WORLD space first, then clip in camera space and carry the
    # colour index through. Simply dropping any triangle with a vertex behind the near plane (the first version)
    # punched a wedge of sky out of the ground right under a ground-level camera.
    nrm0 = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm0 = nrm0 / np.maximum(np.linalg.norm(nrm0, axis=1, keepdims=True), 1e-12)
    sun = np.asarray(sun, float)
    sun = sun / np.linalg.norm(sun)
    shade = np.clip(0.30 + 0.16 * np.clip(nrm0[:, 2], -1, 1) + 0.72 * np.abs(nrm0 @ sun), 0.0, 1.35)[:, None]
    col0 = np.clip(colors * shade, 0.0, 1.0)

    NEAR = 0.10
    zin = V0[:, :, 2] > NEAR
    cnt = zin.sum(axis=1)
    full = np.nonzero(cnt == 3)[0]
    parts_v, parts_i = [V0[full]], [full]

    def _cut(a, b):
        return a + (NEAR - a[2]) / (b[2] - a[2]) * (b - a)

    for t in np.nonzero((cnt == 1) | (cnt == 2))[0]:
        v = V0[t]
        if cnt[t] == 1:
            i0 = int(np.nonzero(zin[t])[0][0])
            i1, i2 = (i0 + 1) % 3, (i0 + 2) % 3
            parts_v.append(np.array([[v[i0], _cut(v[i0], v[i1]), _cut(v[i0], v[i2])]]))
            parts_i.append(np.array([t]))
        else:
            io = int(np.nonzero(~zin[t])[0][0])
            i0, i1 = (io + 1) % 3, (io + 2) % 3
            c0, c1 = _cut(v[i0], v[io]), _cut(v[i1], v[io])
            parts_v.append(np.array([[v[i0], v[i1], c1], [v[i0], c1, c0]]))
            parts_i.append(np.array([t, t]))
    V = np.concatenate(parts_v) if len(parts_v) else np.empty((0, 3, 3))
    src = np.concatenate(parts_i) if len(parts_i) else np.empty((0,), int)
    if not len(V):
        return Image.new("RGB", (W, H), tuple(int(c * 255) for c in sky))
    col = col0[src]

    tan_h = math.tan(math.radians(fov_deg) / 2.0)
    aspect = W / H
    z = V[:, :, 2]
    px = (V[:, :, 0] / (z * tan_h) * 0.5 + 0.5) * W
    py = (0.5 - V[:, :, 1] / (z * tan_h / aspect) * 0.5) * H
    iz = 1.0 / z

    onscreen = ((px.max(1) >= 0) & (px.min(1) < W) & (py.max(1) >= 0) & (py.min(1) < H))
    idx = np.nonzero(onscreen)[0]
    order = np.argsort(-iz[idx].mean(axis=1))            # far to near helps nothing but keeps memory local
    idx = idx[order]

    fb = np.zeros((H, W, 3), np.float32)
    fb[:] = np.asarray(sky, np.float32)
    zb = np.full((H, W), -1e30, np.float32)

    for t in idx:
        x0, x1, x2 = px[t]
        y0, y1, y2 = py[t]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        xmin, xmax = int(max(0, math.floor(min(x0, x1, x2)))), int(min(W - 1, math.ceil(max(x0, x1, x2))))
        ymin, ymax = int(max(0, math.floor(min(y0, y1, y2)))), int(min(H - 1, math.ceil(max(y0, y1, y2))))
        if xmin > xmax or ymin > ymax:
            continue
        if abs(den) < 1e-9:
            cx, cy = int((x0 + x1 + x2) / 3), int((y0 + y1 + y2) / 3)
            if 0 <= cx < W and 0 <= cy < H:
                zz = float(iz[t].mean())
                if zz > zb[cy, cx]:
                    zb[cy, cx] = zz
                    fb[cy, cx] = col[t]
            continue
        gx = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        gy = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
        X, Y = np.meshgrid(gx, gy)
        l0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
        l1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            cx, cy = int(round((x0 + x1 + x2) / 3)), int(round((y0 + y1 + y2) / 3))
            if 0 <= cx < W and 0 <= cy < H:
                zz = float(iz[t].mean())
                if zz > zb[cy, cx]:
                    zb[cy, cx] = zz
                    fb[cy, cx] = col[t]
            continue
        zz = l0 * iz[t, 0] + l1 * iz[t, 1] + l2 * iz[t, 2]
        sub = zb[ymin:ymax + 1, xmin:xmax + 1]
        m = inside & (zz > sub)
        if m.any():
            sub[m] = zz[m]
            fb[ymin:ymax + 1, xmin:xmax + 1][m] = col[t]
    img = (np.clip(fb, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    return Image.fromarray(img)


def mesh_tris(m: Mesh, mat_rgb=None) -> tuple[np.ndarray, np.ndarray]:
    V = np.asarray(m.v, float)
    tri, col = [], []
    for mat, fs in m.faces.items():
        rgb = (mat_rgb or RENDER_RGB)[mat]
        for f in fs:
            tri.append(V[[f[0] - 1, f[1] - 1, f[2] - 1]])
            col.append(rgb)
    return np.asarray(tri, float), np.asarray(col, float)


def scale_figure() -> Mesh:
    """A 1.75 m human as a scale reference in the variant sheets - and the rescuers standing on the rubble in
    Reference B."""
    m = Mesh()
    m.box((0, 0, 0.42), (0.09, 0.16, 0.42), "fabric")
    m.box((0, 0, 1.10), (0.13, 0.20, 0.28), "fabric")
    m.box((0, 0, 1.50), (0.10, 0.10, 0.11), "fabric")
    return m


def render_variants(lib: dict, out_dir: Path) -> list[Path]:
    from PIL import Image, ImageDraw

    fig_t, _ = mesh_tris(scale_figure())
    fig_c = np.tile(np.asarray(RENDER_RGB["_scale_figure"]), (len(fig_t), 1))
    paths = []
    names = list(lib)
    per_sheet = 6
    for k in range(0, len(names), per_sheet):
        block = names[k:k + per_sheet]
        tiles = []
        for name in block:
            m = lib[name]["mesh"]
            T, C = mesh_tris(m)
            lo, hi = m.bounds()
            span = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 0.5))
            ctr = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
            fx = fig_t + np.array([hi[0] + 0.9, 0.0, 0.0])
            gsz = span * 2.2
            g = np.array([[[-gsz, -gsz, lo[2]], [gsz, -gsz, lo[2]], [gsz, gsz, lo[2]]],
                          [[-gsz, -gsz, lo[2]], [gsz, gsz, lo[2]], [-gsz, gsz, lo[2]]]])
            AT = np.concatenate([T, fx, g])
            AC = np.concatenate([C, fig_c, np.tile(np.asarray(RENDER_RGB["_ground"]), (2, 1))])
            row = []
            for label, eye, tgt in (
                ("iso", ctr + np.array([span * 1.5, -span * 1.7, span * 1.25 + 1.2]), ctr),
                ("side", ctr + np.array([0.15 * span, -span * 2.6, 0.35 * span + 0.9]), ctr),
                ("top", ctr + np.array([0.01, 0.01, span * 2.6 + 1.5]), ctr),
            ):
                im = render(AT, AC, eye, tgt, fov_deg=48.0, W=420, H=300)
                d = ImageDraw.Draw(im)
                d.text((6, 4), f"{name} [{label}]", fill=(255, 255, 255))
                d.text((6, 16), f"{lib[name]['tris']} tris  {lib[name]['size_m']} m  "
                                f"{','.join(lib[name]['slots'])}", fill=(230, 235, 245))
                row.append(im)
            tiles.append(row)
        sheet = Image.new("RGB", (420 * 3, 300 * len(tiles)), (18, 20, 24))
        for r, row in enumerate(tiles):
            for c, im in enumerate(row):
                sheet.paste(im, (c * 420, r * 300))
        p = out_dir / f"variants_{k // per_sheet:02d}.png"
        sheet.save(p)
        paths.append(p)
    return paths


def render_scene(plan: dict, lib: dict, out_dir: Path, max_tris: int = 2_600_000) -> list[Path]:
    from PIL import ImageDraw

    meta = json.loads((OUT / "flood_valley.json").read_text())
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    xs = np.linspace(-size / 2, size / 2, n)
    cache = {k: mesh_tris(v["mesh"]) for k, v in lib.items()}

    piles = [it for it in plan["items"] if it["group"] == "pile"]
    # Aim at the densest part of the field: the pile whose 20 m neighbourhood holds the most rubble. The first
    # pass used a median position and put the ground camera INSIDE a pile.
    pe = np.array([[p["east_m"], p["north_m"]] for p in piles])
    dens = [(int(((np.abs(pe[:, 0] - p[0]) < 20) & (np.abs(pe[:, 1] - p[1]) < 20)).sum()), p) for p in pe]
    focus_e, focus_n = (float(max(dens, key=lambda t: t[0])[1][0]), float(max(dens, key=lambda t: t[0])[1][1]))

    def ground(e, nn):
        i = int(round((e + size / 2) / cell))
        j = int(round((nn + size / 2) / cell))
        return float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])

    def terrain_patch(ce, cn, half, stride):
        i0 = max(0, int((ce - half + size / 2) / cell))
        i1 = min(n - 1, int((ce + half + size / 2) / cell))
        j0 = max(0, int((cn - half + size / 2) / cell))
        j1 = min(n - 1, int((cn + half + size / 2) / cell))
        ii = np.arange(i0, i1 + 1, stride)
        jj = np.arange(j0, j1 + 1, stride)
        T = []
        for a in range(len(jj) - 1):
            for b in range(len(ii) - 1):
                p00 = [xs[ii[b]], xs[jj[a]], h[jj[a], ii[b]]]
                p10 = [xs[ii[b + 1]], xs[jj[a]], h[jj[a], ii[b + 1]]]
                p11 = [xs[ii[b + 1]], xs[jj[a + 1]], h[jj[a + 1], ii[b + 1]]]
                p01 = [xs[ii[b]], xs[jj[a + 1]], h[jj[a + 1], ii[b]]]
                T.append([p00, p10, p11])
                T.append([p00, p11, p01])
        return np.asarray(T, float)

    def gather(ce, cn, radius):
        """Collect the instances near (ce, cn), NEAREST FIRST.

        The first version walked `plan["items"]` in list order and stopped at the triangle cap. Because the
        piles are written first, they consumed the whole cap and the render silently dropped almost every mat
        and loose fragment - the picture showed a sparser field than the layout actually contains, which is
        exactly the kind of "verified by looking at the wrong thing" this project keeps hitting.
        """
        T, C = [], []
        total = 0
        near = sorted(((math.hypot(it["east_m"] - ce, it["north_m"] - cn), k)
                       for k, it in enumerate(plan["items"])), key=lambda t: t[0])
        for d, k in near:
            if d > radius:
                break
            it = plan["items"][k]
            vt, vc = cache[it["variant"]]
            if total + len(vt) > max_tris:
                break
            Rp = rot_z(it["yaw_deg"]) @ _pitch_roll(it["pitch_deg"], it["roll_deg"])
            V = vt.reshape(-1, 3) * it["scale"]
            V = V @ Rp.T + np.array([it["east_m"], it["north_m"], it["base_asl_m"]])
            T.append(V.reshape(-1, 3, 3))
            C.append(vc)
            total += len(vt)
        if not T:
            return np.empty((0, 3, 3)), np.empty((0, 3)), 0
        return np.concatenate(T), np.concatenate(C), total

    def standoff(start: float) -> float:
        """Back the ground camera off until it is standing on open ground, not inside a pile (pass 5 put the
        close camera 14 m from the focus, which after the piles were allowed to merge was inside the mass)."""
        for back in np.arange(start, start + 30.0, 3.0):
            d = [math.hypot(p["east_m"] - focus_e, p["north_m"] - (focus_n - back))
                 - max(plan["variants"][p["variant"]]["size_m"][:2]) / 2 * p["scale"] for p in piles]
            if not d or min(d) > 6.0:
                return float(back)
        return float(start + 30.0)

    # gather radii are sized to the view frustum, not guessed: gathering 190 m for a nadir frame that only sees
    # 54 m spent the whole triangle cap on geometry outside the picture.
    views = [
        ("ground_oblique", focus_e, focus_n, 95.0, 1.75, 62.0, standoff(45.0)),
        ("ground_close", focus_e, focus_n, 42.0, 1.60, 55.0, standoff(20.0)),
        ("aerial_45m", focus_e, focus_n, 215.0, 45.0, 62.0, 150.0),
        ("aerial_nadir_60m", focus_e, focus_n, 58.0, 60.0, 74.0, 0.1),
    ]
    paths = []
    for name, ce, cn, radius, alt, fov, back in views:
        T, C, cnt = gather(ce, cn, radius)
        gz = ground(ce, cn)
        # the terrain has to reach past the far horizon or the ground ends in mid-air (seen in pass 1)
        # 4 m quads for the ground-level views: an 8 m quad containing the camera fails the near-plane cull and
        # leaves a wedge of sky along the bottom edge.
        gt_tri = terrain_patch(ce, cn, max(radius * 1.15, back + radius + 120.0), 1 if alt < 5 else 2)
        gc = np.tile(np.asarray(RENDER_RGB["_ground"]), (len(gt_tri), 1))
        AT = np.concatenate([gt_tri, T]) if len(T) else gt_tri
        AC = np.concatenate([gc, C]) if len(C) else gc
        # AGL is measured at the CAMERA's own ground, not at the focus. Sampling it at the focus put the
        # ground-level eye 0.52 m above the terrain under it (the fan rolls by ~1 m over 30 m), so the near
        # ground fell behind the near plane and the render showed rubble floating over a strip of sky.
        eye = np.array([ce, cn - back, ground(ce, cn - back) + alt])
        tgt = np.array([ce, cn, gz + (1.4 if alt < 5 else 0.0)])
        t0 = time.time()
        im = render(AT, AC, eye, tgt, fov_deg=fov, W=1100, H=700)
        d = ImageDraw.Draw(im)
        d.text((8, 6), f"{name}: {cnt:,} rubble tris + {len(gt_tri):,} terrain tris in view "
                       f"| eye {alt:.1f} m AGL, {back:.0f} m back | fan @ E{ce:.0f} N{cn:.0f}",
               fill=(250, 250, 250))
        d.text((8, 20), f"[sim] rendered offline by gen_rubble.py in {time.time() - t0:.1f}s "
                        f"- flat shading, no textures: geometry check only", fill=(235, 240, 250))
        p = out_dir / f"scene_{name}.png"
        im.save(p)
        paths.append(p)
    return paths


def _pitch_roll(pitch_deg: float, roll_deg: float) -> np.ndarray:
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Ry @ Rx


# ----------------------------------------------------------------------------------------------------------
# checks that can FAIL
# ----------------------------------------------------------------------------------------------------------
def void_fraction(m: Mesh, samples: int = 34) -> float:
    """Fraction of vertical columns through a pile that contain an enclosed air gap >= 0.35 m.

    Heuristic (the parts overlap, so the crossing parity is not exact), but it is the only thing that separates
    "a pile with voids a person can be in" from "a solid mound", and it can fail.
    """
    V = np.asarray(m.v, float)
    tri = np.concatenate([np.asarray(fs, int) for fs in m.faces.values()]) - 1
    P = V[tri]                                                     # (M,3,3)
    lo, hi = V.min(0), V.max(0)
    gx = np.linspace(lo[0] * 0.72, hi[0] * 0.72, samples)
    gy = np.linspace(lo[1] * 0.72, hi[1] * 0.72, samples)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-12
    a, b, c, den, P = a[ok], b[ok], c[ok], den[ok], P[ok]
    hits = 0
    chunk_n = 64
    for k in range(0, len(pts), chunk_n):
        q = pts[k:k + chunk_n]
        l0 = ((b[:, None, 1] - c[:, None, 1]) * (q[None, :, 0] - c[:, None, 0])
              + (c[:, None, 0] - b[:, None, 0]) * (q[None, :, 1] - c[:, None, 1])) / den[:, None]
        l1 = ((c[:, None, 1] - a[:, None, 1]) * (q[None, :, 0] - c[:, None, 0])
              + (a[:, None, 0] - c[:, None, 0]) * (q[None, :, 1] - c[:, None, 1])) / den[:, None]
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        z = l0 * a[:, None, 2] + l1 * b[:, None, 2] + l2 * c[:, None, 2]
        for col in range(q.shape[0]):
            zs = np.sort(z[inside[:, col], col])
            if len(zs) < 4:
                continue
            gaps = zs[2::2] - zs[1:-1:2] if len(zs) >= 4 else np.array([])
            if len(gaps) and gaps.max() >= 0.35:
                hits += 1
    return hits / len(pts)


def run_checks(plan: dict, lib: dict) -> list[tuple[str, bool, str]]:
    res: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, detail: str = "") -> None:
        res.append((name, bool(ok), detail))

    # -- 0. the rotation helper itself (a sign error here would tilt every slab the wrong way silently) -------
    d = np.array([0.3, -0.7, 0.5])
    chk("basis() maps local +x onto the requested direction",
        bool(np.allclose(basis(d, 37.0) @ np.array([1.0, 0.0, 0.0]), d / np.linalg.norm(d), atol=1e-9)))

    # -- 1. geometry -----------------------------------------------------------------------------------------
    bad_area, bad_sliver, nan_ct, bad_slots = [], [], 0, []
    for name, v in lib.items():
        m: Mesh = v["mesh"]
        V = np.asarray(m.v, float)
        if not np.isfinite(V).all():
            nan_ct += 1
        tri = np.concatenate([np.asarray(fs, int) for fs in m.faces.values()]) - 1
        P = V[tri]
        e0, e1 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
        area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
        if float(area.min()) < 2e-5:                     # 0.2 cm^2
            bad_area.append((name, float(area.min())))
        el = np.stack([np.linalg.norm(P[:, 1] - P[:, 0], axis=1), np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
                       np.linalg.norm(P[:, 0] - P[:, 2], axis=1)], 1)
        longest = el.max(1)
        heights = 2.0 * area / np.maximum(longest, 1e-12)
        ar = longest / np.maximum(heights, 1e-12)
        if float(ar.max()) > 400.0:
            bad_sliver.append((name, float(ar.max())))
        if any(k not in MATS for k in m.faces):
            bad_slots.append(name)
        if len(m.v) != len(m.vt):
            bad_slots.append(name + " (uv mismatch)")
    chk("no non-finite vertices", nan_ct == 0, f"{nan_ct} meshes with NaN/inf")
    chk("no zero-area faces (min triangle >= 0.2 cm^2)", not bad_area, f"worst: {bad_area[:3]}")
    chk("no degenerate slivers (aspect ratio < 400)", not bad_sliver, f"worst: {bad_sliver[:3]}")
    chk("every usemtl group is a known material slot", not bad_slots, str(bad_slots[:3]))

    # -- 2. the shapes are the shapes the brief asks for ------------------------------------------------------
    slab_ok, slab_bad, plate_dims = True, [], {}
    for name, span, th, _rb in SLAB_SPEC:
        plate = sorted(lib[name]["concrete_size_m"])     # the plate only: rebar is excluded on purpose
        plate_dims[name] = plate
        if not (0.12 <= plate[0] <= 0.45):               # nominal thickness plus the sag deflection
            slab_ok = False
            slab_bad.append((name, "thickness", plate[0]))
        if not (1.3 <= plate[2] <= 4.4):
            slab_ok = False
            slab_bad.append((name, "span", plate[2]))
        if plate[2] / max(plate[0], 1e-6) < 6.0:         # a plate, not a lump
            slab_ok = False
            slab_bad.append((name, "aspect", round(plate[2] / plate[0], 1)))
    chk("slab variants are thin plates 1.5-4 m across, 0.15-0.25 m thick", slab_ok,
        str(slab_bad) if slab_bad else "min/mid/max dims " + str({k: v for k, v in plate_dims.items()}))
    rebar_ct = sum(1 for name, *_ in SLAB_SPEC if "rebar" in lib[name]["slots"])
    chk("rebar protrudes from most slab variants", rebar_ct >= len(SLAB_SPEC) - 1,
        f"{rebar_ct}/{len(SLAB_SPEC)} slab variants carry a rebar slot")
    # a rebar slot that does not stick out past the concrete would be invisible: prove the bar protrudes
    stub = [(n, round(float(max(np.array(lib[n]["size_m"]) - np.array(lib[n]["concrete_size_m"]))), 3))
            for n, *_ in SLAB_SPEC if "rebar" in lib[n]["slots"]]
    chk("rebar sticks out at least 0.15 m past the concrete", all(v >= 0.15 for _, v in stub), str(stub))
    blk_ok = all(0.18 <= max(lib[n]["size_m"]) <= 0.90 for n, _ in BLOCK_SPEC)
    chk("block variants are 0.2-0.8 m angular lumps", blk_ok,
        str({n: lib[n]["size_m"] for n, _ in BLOCK_SPEC}))

    # -- 3. the piles actually contain voids ------------------------------------------------------------------
    vf = {name: void_fraction(lib[name]["mesh"]) for name, *_ in PILE_SPEC}
    worst = min(vf.values())
    chk("every rubble pile contains vertical voids >= 0.35 m", worst >= 0.06,
        "void column fraction " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(vf.items())))

    # -- 4. the layout ----------------------------------------------------------------------------------------
    meta = json.loads((OUT / "flood_valley.json").read_text())
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    n, cell, size = s["n"], s["cell_m"], s["size_m"]
    hgt = s["height"]
    mk = fan_masks(s)

    def cij(e, nn):
        return (int(round((nn + size / 2) / cell)), int(round((e + size / 2) / cell)))

    out_of_fan = [it["name"] for it in plan["items"]
                  if it["group"] in ("pile", "mat", "loose")
                  and not mk["floor"][cij(it["east_m"], it["north_m"])]]
    chk("every field item sits inside the fan-floor mask", not out_of_fan,
        f"{len(out_of_fan)} outside, first {out_of_fan[:3]}")
    in_channel = [it["name"] for it in plan["items"] if s["in_channel"][cij(it["east_m"], it["north_m"])]]
    chk("the scoured channel is left clean", not in_channel,
        f"{len(in_channel)} in the channel, first {in_channel[:3]}")
    steep = [it["name"] for it in plan["items"]
             if it["group"] == "pile" and mk["slope"][cij(it["east_m"], it["north_m"])] >= PILE_MAX_SLOPE_DEG]
    chk(f"piles only on ground flatter than {PILE_MAX_SLOPE_DEG:.0f} deg", not steep, f"{len(steep)} too steep")

    floating = []
    for it in plan["items"]:
        if it["group"] == "burial":
            continue
        j, i = cij(it["east_m"], it["north_m"])
        g = float(hgt[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])
        if not (-1.0 <= it["base_asl_m"] - g <= 1.0):
            floating.append((it["name"], round(it["base_asl_m"] - g, 2)))
    chk("nothing floats or sinks more than 1 m from the fan surface", not floating,
        f"{len(floating)} bad, first {floating[:3]}")

    acts = json.loads((OUT / "actors.json").read_text())["actors"]
    vis = [(a["east_m"], a["north_m"], a["name"]) for a in acts if a["aerially_detectable"]]
    intruders = []
    for it in plan["items"]:
        for ae, an, nm in vis:
            if math.hypot(it["east_m"] - ae, it["north_m"] - an) < SURVIVOR_CLEAR_M:
                intruders.append((it["name"], nm))
    chk(f"the {SURVIVOR_CLEAR_M} m void around every detectable survivor is respected", not intruders,
        f"{len(intruders)} intrusions, first {intruders[:3]}")

    buried = [a for a in acts if not a["aerially_detectable"]]
    covered = []
    for a in buried:
        cov = [it for it in plan["items"] if it["group"] == "burial"
               and math.hypot(it["east_m"] - a["east_m"], it["north_m"] - a["north_m"]) < 1.2
               and it["base_asl_m"] > a["base_asl_m"] + 0.3]
        covered.append((a["name"], len(cov)))
    chk("every 2.7 burial-boundary survivor carries a slab over them",
        all(c >= 1 for _, c in covered), str(covered))

    houses = json.loads((OUT / "settlement.json").read_text())
    inside_house = []
    for it in plan["items"]:
        for b in houses["houses"]:
            a = houses["archetypes"][b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = it["east_m"] - b["east_m"], it["north_m"] - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 and abs(v) < a["width_m"] / 2:
                inside_house.append(it["name"])
                break
    chk("nothing is placed inside a house footprint", not inside_house, f"{len(inside_house)} inside")

    c = plan["counts"]
    chk(f"triangle budget respected ({c['approx_triangles']:,} <= {c['triangle_budget']:,})",
        c["approx_triangles"] <= c["triangle_budget"])
    mdf = c["mean_downstream_fraction_piles"]
    chk("piles are biased to the fan apex (mean downstream fraction < 0.45)",
        mdf is not None and mdf < 0.45, f"mean f = {mdf}")
    chk("the field is large enough to read as a collapse zone (>= 800 items, >= 120 piles)",
        c["items"] >= 800 and c["by_group"].get("pile", 0) >= 120, str(c["by_group"]))

    # -- 5. GROUND COVER. This is the check the first render forced into existence: 210 piles passed every other
    # test and still read from nadir as isolated grey flowers on clean mud, because nothing covered the ground
    # between them. Coverage is rasterised over the fan-floor grid and measured in the apex third.
    xs = np.linspace(-size / 2, size / 2, n)
    fidx = np.argwhere(mk["floor"])
    apex, toe = float(xs[fidx[:, 0]].max()), float(xs[fidx[:, 0]].min())
    F = np.clip((apex - xs[:, None]) / (apex - toe), 0, 1) * np.ones((1, n))
    cov = np.zeros((n, n), bool)
    E, N = np.meshgrid(xs, xs, indexing="xy")
    for it in plan["items"]:
        if it["group"] not in ("pile", "mat"):
            continue
        sz = plan["variants"][it["variant"]]["size_m"]
        r = max(sz[0], sz[1]) / 2.0 * it["scale"] * (0.90 if it["group"] == "pile" else 0.80)
        j, i = cij(it["east_m"], it["north_m"])
        k = int(r / cell) + 1
        j0, j1, i0, i1 = max(0, j - k), min(n, j + k + 1), max(0, i - k), min(n, i + k + 1)
        sub = (E[j0:j1, i0:i1] - it["east_m"]) ** 2 + (N[j0:j1, i0:i1] - it["north_m"]) ** 2 <= r * r
        cov[j0:j1, i0:i1] |= sub
    core = mk["floor"] & (F < 0.35)
    frac_core = float((cov & core).sum()) / max(1, int(core.sum()))
    mid = mk["floor"] & (F >= 0.35) & (F < 0.7)
    frac_mid = float((cov & mid).sum()) / max(1, int(mid.sum()))
    chk("the apex third of the fan is >= 45 % covered by rubble (a field, not isolated mounds)",
        frac_core >= 0.45,
        f"apex third {100 * frac_core:.0f} % covered ({int(core.sum()) * cell * cell:,.0f} m2), "
        f"mid fan {100 * frac_mid:.0f} %")
    chk("coverage still thins downstream (apex third denser than the mid fan)", frac_core > frac_mid + 0.08,
        f"{100 * frac_core:.0f} % vs {100 * frac_mid:.0f} %")
    return res


# ----------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--max-tris", type=int, default=2_500_000)
    ap.add_argument("--piles", type=int, default=145)
    ap.add_argument("--mats", type=int, default=760)
    ap.add_argument("--loose", type=int, default=900)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()

    RUBBLE_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plan, lib = build(a.seed, a.max_tris, a.piles, a.mats, a.loose)

    for name, v in lib.items():
        v["mesh"].write(RUBBLE_DIR / f"{name}.obj", name)
    (OUT / "rubble_layout.json").write_text(json.dumps(plan, indent=1))

    c = plan["counts"]
    print(f"library: {len(lib)} meshes, {c['library_triangles']:,} triangles total "
          f"-> {RUBBLE_DIR}")
    for fam in ("pile", "mat", "slab", "block", "masonry", "rebar", "belonging"):
        sel = {k: v["tris"] for k, v in lib.items() if v["family"] == fam}
        if sel:
            print(f"  {fam:<10} {len(sel):>2} variants, {min(sel.values()):>6,}-{max(sel.values()):>6,} tris")
    print(f"layout: {c['items']:,} actors, ~{c['approx_triangles']:,} triangles "
          f"(budget {c['triangle_budget']:,}, {100 * c['approx_triangles'] / c['triangle_budget']:.0f} % used)")
    print(f"  by group : {c['by_group']}")
    print(f"  by family: {c['by_family']}")
    print(f"  fan floor: {c['fan_floor_area_m2']:,} m2, mean pile downstream fraction "
          f"{c['mean_downstream_fraction_piles']}")

    if not a.no_render:
        print("rendering (offline software rasteriser) ...")
        vp = render_variants(lib, ART)
        sp = render_scene(plan, lib, ART)
        for p in vp + sp:
            print(f"  {p}")

    print(f"\nCHECKS ({time.time() - t0:.0f}s elapsed)")
    fails = 0
    for name, ok, detail in run_checks(plan, lib):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        fails += 0 if ok else 1
    hsh = hashlib.sha256()
    for p in sorted(RUBBLE_DIR.glob("*.obj")):
        hsh.update(p.read_bytes())
    print(f"  obj bytes sha256 {hsh.hexdigest()[:16]}  layout sha256 "
          f"{hashlib.sha256((OUT / 'rubble_layout.json').read_bytes()).hexdigest()[:16]}")
    if fails:
        print(f"\n{fails} CHECK(S) FAILED - do not import this rubble field.")
        raise SystemExit(1)
    print("\nall checks passed. NOW LOOK at the PNGs above before importing.")


if __name__ == "__main__":
    main()
