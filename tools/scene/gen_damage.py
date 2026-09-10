"""Generate seeded flood damage for the FloodValley settlement: damaged archetype meshes + a per-house plan.

    uv run python tools/scene/gen_damage.py [--seed 41]
    ue_python code="exec(open(r'D:\\Sightline\\tools\\scene\\build_damage.py').read())"

Every one of the 73 houses from `gen_buildings.py` is pristine, which reads as a village in a puddle rather than a
disaster. There is no damaged-building asset anywhere in `_downloads/assets` (MANIFEST "Blockers"), so the damage
is generated here as real geometry, from the same archetype dimensions, and textured in Unreal with the material
slots `build_buildings.py` already set up. Nothing here needs a new texture download.

WHAT IT PRODUCES (data/scene/):
  damage/<variant>_[a|b].obj   damaged variants of five archetypes, CENTIMETRES, `usemtl` groups -> UE slots
                               (wall, roof_tile, roof_sheet, concrete, window, wood, rubble). `_a` is as built,
                               `_b` is mirrored in the house's local +v axis.
  damage.json                  which house gets which variant, plus the debris rafted onto the flat roofs.

THE DAMAGE, in the order a nadir camera notices it:
  * missing / peeled roof sheeting on the corrugated sheds (`C_sheet_1s`), with purlins and a silt-filled
    interior exposed. This is also the partial-occlusion case the detector should be tested against.
  * holes torn in the clay-tile slopes of `A_tile_1s` / `D_tile_2s`.
  * a collapsed gable end on some tile houses, a collapsed slab corner on the two-storey RCC houses, and a
    slumped wall with a sagging slab on the single-storey ones - a minority, as in a real event.
  * debris rafted onto the flat roofs (existing Poly Haven props, so no new assets and no new materials).
The silt tide line and the mud staining below it are NOT here: they are a material effect and live in
`build_roof_materials.py`, because `M_PBR_Master` must have exactly one owner - two scripts clearing and
rebuilding the same graph would silently undo each other.

ORIENTATION (exact - there is no mirror ambiguity to worry about). `build_buildings.py` imports the OBJ (UE flips
it to UE X = u, UE Y = -v) and then spawns the actor with UE yaw = 90 - theta. Composing the two rotations:

    east  = u * cos(theta) - v * sin(theta)
    north = u * sin(theta) + v * cos(theta)

which is exactly the local->world convention `gen_actors.py` and `gen_props.py` already use. So local +v points
along world (-sin theta, cos theta), and the `_a` / `_b` mirror lets every torn roof face UPSTREAM (+north):
`_a` when cos(theta) > 0, `_b` otherwise.

SAFETY: a house that carries a survivor is never damaged. Opening its roof or dropping its slab corner would
leave a Rocketbox actor standing in mid-air or falling through the hole, and those actors are the ground truth
the whole data plan rests on. The test is geometric (any actor inside the footprint + 1 m), so it does not depend
on `gen_actors.py` recording a house id.

DETERMINISM: one `numpy.default_rng(seed)` drives every choice, and the per-variant rubble uses `zlib.crc32`
rather than `hash()` (Python randomises string hashes per process, which would have made the meshes differ
between runs).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_buildings as gb  # noqa: E402  (same archetype dimensions, Mesh builder and UV convention)

OUT = gb.OUT
DMG_DIR = OUT / "damage"

# `gb.Mesh.poly` looks the texture tile size up in the module-level TILE_M. Adding a key is purely additive:
# gen_buildings never emits a "rubble" group, so its own output is bit-identical with or without this line.
gb.TILE_M.setdefault("rubble", 1.6)

PLINTH, STOREY_H = gb.PLINTH, gb.STOREY_H
INNER_WALL, INNER_FLOOR = "wood", "rubble"   # a dark timbered interior over a silt-filled floor: reads as a hole
ROOF_DEBRIS_TRI_BUDGET = 220_000


def sid(tag: str) -> int:
    """A STABLE hash: `hash()` on a str is salted per process, which would break determinism across runs."""
    return zlib.crc32(tag.encode()) % 100_000


# --------------------------------------------------------------------------------------------------------
# geometry helpers (metres, local house frame: +u = long axis, +v = short axis, z = 0 at ground)
# --------------------------------------------------------------------------------------------------------
def side_walls(m, x0, y0, x1, y1, z0, z1, mat, sides=(True, True, True, True)):
    """The four OUTWARD-facing side quads of a box (same winding as gb.Mesh.box): no top lid, no floor."""
    if sides[0]:
        m.poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], mat)          # +u
    if sides[1]:
        m.poly([(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], mat)          # -u
    if sides[2]:
        m.poly([(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)], mat)          # +v
    if sides[3]:
        m.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], mat)          # -v


def inner_room(m, x0, y0, x1, y1, z0, z1, wall_mat=INNER_WALL, floor_mat=INNER_FLOOR):
    """Inward-facing liner + floor, COINCIDENT with the outer walls.

    The walls are zero-thickness surfaces, so a coincident quad with the opposite winding never z-fights: from
    any viewpoint exactly one of the pair is front-facing and the other is backface-culled. Without this an
    opened roof shows straight through the far walls to the terrain, because UE materials are single-sided.
    """
    m.poly([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], floor_mat)        # floor, faces +z
    m.poly([(x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0)], wall_mat)
    m.poly([(x0, y1, z1), (x0, y0, z1), (x0, y0, z0), (x0, y1, z0)], wall_mat)
    m.poly([(x1, y1, z1), (x0, y1, z1), (x0, y1, z0), (x1, y1, z0)], wall_mat)
    m.poly([(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)], wall_mat)


def two_sided(m, pts, mat):
    """A single surface visible from both sides (a peeled sheet, a dangling slab, a leaning wall)."""
    m.poly(pts, mat)
    m.poly(list(reversed(pts)), mat)


def holed_plane(m, x0, y0, x1, y1, hx0, hy0, hx1, hy1, zfun, mat, flip=False):
    """A plane z = zfun(y) with a rectangular hole punched in it, as four border quads.

    `flip` reverses the winding, which is what the roof UNDERSIDE needs: an up-facing underside would be
    backface-culled from below and would still block the view down through the hole from above.
    """
    def q(ax, ay, bx, by):
        if bx - ax < 1e-6 or by - ay < 1e-6:
            return
        pts = [(ax, ay, zfun(ay)), (bx, ay, zfun(ay)), (bx, by, zfun(by)), (ax, by, zfun(by))]
        m.poly(list(reversed(pts)) if flip else pts, mat)
    q(x0, y0, x1, hy0)          # before the hole
    q(x0, hy1, x1, y1)          # after the hole
    q(x0, hy0, hx0, hy1)        # left of it
    q(hx1, hy0, x1, hy1)        # right of it


def purlins(m, x0, x1, y_from, y_to, zfun, n=3, mat="wood"):
    """Horizontal purlins across an opened roof: the strongest 'this roof is gone' cue from directly above."""
    for k in range(n):
        y = y_from + (y_to - y_from) * (k + 0.5) / n
        z = zfun(y) - 0.10
        m.box(x0, y - 0.06, z - 0.12, x1, y + 0.06, z, mat)


def rubble_heap(m, cx, cy, z0, rx, ry, h, tag):
    """A crude heap of broken plaster and blockwork: three stacked boxes, deterministic in `tag`."""
    r = np.random.default_rng(sid(tag))
    for k in range(3):
        f = 1.0 - 0.28 * k
        dx, dy = float(r.uniform(-0.25, 0.25)), float(r.uniform(-0.25, 0.25))
        m.box(cx + dx - rx * f, cy + dy - ry * f, z0, cx + dx + rx * f, cy + dy + ry * f,
              z0 + h * (k + 1) / 3.0, "rubble")


def cull(m, inside):
    """Drop every face whose centroid satisfies `inside(x, y, z)`.

    Used where a wall moves: `gb.openings` draws windows, doors and sunshades on the ORIGINAL wall plane, so a
    leaning wall would leave them hanging in mid-air. Faces index into m.v (1-based), so this is exact.
    """
    for mat in list(m.faces):
        kept = []
        for f in m.faces[mat]:
            pts = [m.v[i - 1] for i in f]
            c = [sum(p[k] for p in pts) / len(pts) for k in range(3)]
            if not inside(*c):
                kept.append(f)
        if kept:
            m.faces[mat] = kept
        else:
            del m.faces[mat]


def mirror_v(mesh):
    """Mirror a finished mesh in the local v axis (v -> -v) and flip every winding so normals still point out."""
    out = gb.Mesh()
    out.v = [(x, -y, z) for x, y, z in mesh.v]
    out.vt = list(mesh.vt)
    out.faces = {mat: [tuple(reversed(f)) for f in fs] for mat, fs in mesh.faces.items()}
    return out


def shell(m, name, sides=(True, True, True, True)):
    """Walls, plinth band, openings and floor band - everything gen_buildings builds below the roof, but with NO
    top lid, so the interior can be exposed. Returns (L, W, storeys, eaves height, interior floor z)."""
    L, W, storeys, _roof, _w = gb.ARCHETYPES[name]
    h = PLINTH + storeys * STOREY_H
    side_walls(m, -L / 2, -W / 2, L / 2, W / 2, -1.0, h, "wall", sides)
    m.box(-L / 2 - 0.15, -W / 2 - 0.15, -1.0, L / 2 + 0.15, W / 2 + 0.15, PLINTH, "concrete")
    gb.openings(m, L, W, storeys, STOREY_H, PLINTH)
    if storeys == 2:
        m.box(-L / 2 - 0.08, -W / 2 - 0.08, PLINTH + STOREY_H - 0.1, L / 2 + 0.08, W / 2 + 0.08,
              PLINTH + STOREY_H + 0.1, "concrete")
    # the room the camera looks into is the TOP storey: its floor is the slab under it, silted up ~12 cm
    return L, W, storeys, h, PLINTH + (storeys - 1) * STOREY_H + 0.12


# --------------------------------------------------------------------------------------------------------
# the damaged archetypes
# --------------------------------------------------------------------------------------------------------
def v_sheet(name, keep_frac, peeled, tag):
    """C_sheet_1s with the mono-pitch sheeting torn off from the UPSTREAM (+v) eave."""
    m = gb.Mesh()
    L, W, _st, h, fz = shell(m, name)
    pitch, over = 11.0, 0.5
    t = math.tan(math.radians(pitch))
    x0, x1, y0, y1 = -L / 2 - over, L / 2 + over, -W / 2 - over, W / 2 + over
    z0, z1 = h - over * t, h + (W + over) * t

    def zf(y):
        return z0 + (z1 - z0) * (y - y0) / (y1 - y0)

    ykeep = y0 + (y1 - y0) * keep_frac
    m.poly([(x0, y0, zf(y0)), (x1, y0, zf(y0)), (x1, ykeep, zf(ykeep)), (x0, ykeep, zf(ykeep))], "roof_sheet")
    m.poly([(x0, ykeep, zf(ykeep) - 0.05), (x1, ykeep, zf(ykeep) - 0.05),
            (x1, y0, zf(y0) - 0.05), (x0, y0, zf(y0) - 0.05)], "wood")
    purlins(m, -L / 2, L / 2, ykeep, y1, zf, n=3)
    rise = W * t
    for pts in ([(-L / 2, W / 2, h), (-L / 2, -W / 2, h), (-L / 2, W / 2, h + rise)],
                [(L / 2, -W / 2, h), (L / 2, W / 2, h), (L / 2, W / 2, h + rise)],
                [(L / 2, W / 2, h), (-L / 2, W / 2, h), (-L / 2, W / 2, h + rise), (L / 2, W / 2, h + rise)]):
        two_sided(m, pts, "wall")
    inner_room(m, -L / 2, -W / 2, L / 2, W / 2, fz, h)
    if peeled:
        # one sheet still hinged at the tear line, folded back over the intact roof at ~55 deg
        a = math.radians(55.0)
        xa, xb = -L / 2 + 0.6, -L / 2 + 3.0
        zt = zf(ykeep)
        two_sided(m, [(xa, ykeep, zt), (xb, ykeep, zt),
                      (xb, ykeep - 1.7 * math.cos(a), zt + 1.7 * math.sin(a)),
                      (xa, ykeep - 1.7 * math.cos(a), zt + 1.7 * math.sin(a))], "roof_sheet")
    rubble_heap(m, 0.32 * L, 0.0, fz, 0.9, 0.8, 0.7, tag)
    return m


def v_tile_holed(name, hole_u, hole_v, tag):
    """A_tile_1s / D_tile_2s with a hole torn in the upstream (+v) slope of the clay-tile gable roof."""
    m = gb.Mesh()
    L, W, _st, h, fz = shell(m, name)
    pitch, over, tk = 26.0, 0.6, 0.12
    t = math.tan(math.radians(pitch))
    he, hr = h - over * t, h + (W / 2) * t
    x0, x1, y0, y1 = -L / 2 - over, L / 2 + over, -W / 2 - over, W / 2 + over

    def zn(y):                                    # +v slope: hr at the ridge (y = 0) down to he at the eave
        return hr + (he - hr) * y / y1

    hx0, hx1 = -hole_u / 2, hole_u / 2
    hy0, hy1 = 0.35 * y1 - hole_v / 2, 0.35 * y1 + hole_v / 2
    holed_plane(m, x0, 0.0, x1, y1, hx0, hy0, hx1, hy1, zn, "roof_tile")
    holed_plane(m, x0, 0.0, x1, y1, hx0, hy0, hx1, hy1, lambda y: zn(y) - tk, "wood", flip=True)
    purlins(m, hx0 - 0.2, hx1 + 0.2, hy0, hy1, zn, n=2)
    # the intact -v slope and its underside, exactly as gen_buildings builds them
    m.poly([(x0, y0, he), (x1, y0, he), (x1, 0.0, hr), (x0, 0.0, hr)], "roof_tile")
    m.poly([(x0, 0.0, hr - tk), (x1, 0.0, hr - tk), (x1, y0, he - tk), (x0, y0, he - tk)], "wood")
    for xs, sgn in ((-L / 2, -1), (L / 2, 1)):
        pts = ([(xs, W / 2, h), (xs, -W / 2, h), (xs, 0.0, h + (W / 2) * t)] if sgn < 0 else
               [(xs, -W / 2, h), (xs, W / 2, h), (xs, 0.0, h + (W / 2) * t)])
        two_sided(m, pts, "wall")
    # the liner stops at the eaves: above that the attic is closed by the two gable triangles above
    inner_room(m, -L / 2, -W / 2, L / 2, W / 2, fz, h)
    rubble_heap(m, 0.0, 0.28 * W, fz, 1.0, 0.9, 0.8, tag)
    return m


def v_tile_endgone(name, tag):
    """A_tile_1s with the -u end wrecked: the roof gone over ~3 m, the end wall broken down to a jagged top.

    The wall tops stay above 2.9 m where the openings are (`gb.openings` puts window heads at 2.55 m and
    sunshades at 2.80 m): a lower break would leave window quads hanging in mid-air over the ruin.
    """
    m = gb.Mesh()
    L, W, storeys, h, fz = shell(m, name, sides=(True, False, True, True))
    gone = 3.0
    xg = -L / 2 + gone
    for k, ztop in enumerate((2.25, 2.95, 3.20, 1.95)):                     # jagged -u end wall
        ya, yb = -W / 2 + k * W / 4, -W / 2 + (k + 1) * W / 4
        two_sided(m, [(-L / 2, ya, -1.0), (-L / 2, yb, -1.0), (-L / 2, yb, ztop), (-L / 2, ya, ztop)], "wall")

    pitch, over, tk = 26.0, 0.6, 0.12
    t = math.tan(math.radians(pitch))
    he, hr = h - over * t, h + (W / 2) * t
    x1, y0, y1 = L / 2 + over, -W / 2 - over, W / 2 + over
    m.poly([(xg, y0, he), (x1, y0, he), (x1, 0.0, hr), (xg, 0.0, hr)], "roof_tile")
    m.poly([(x1, y1, he), (xg, y1, he), (xg, 0.0, hr), (x1, 0.0, hr)], "roof_tile")
    m.poly([(xg, 0.0, hr - tk), (x1, 0.0, hr - tk), (x1, y0, he - tk), (xg, y0, he - tk)], "wood")
    m.poly([(x1, 0.0, hr - tk), (xg, 0.0, hr - tk), (xg, y1, he - tk), (x1, y1, he - tk)], "wood")
    two_sided(m, [(L / 2, -W / 2, h), (L / 2, W / 2, h), (L / 2, 0.0, h + (W / 2) * t)], "wall")
    two_sided(m, [(xg, W / 2, h), (xg, -W / 2, h), (xg, 0.0, h + (W / 2) * t)], "wall")   # torn roof section
    for k in range(3):                                                      # rafters left over the ruined end
        x = -L / 2 + 0.7 + k * (gone - 1.2) / 2.0
        m.box(x - 0.06, -W / 2 + 0.2, h - 0.35, x + 0.06, W / 2 - 0.2, h - 0.23, "wood")
    inner_room(m, -L / 2, -W / 2, L / 2, W / 2, fz, h)
    rubble_heap(m, -L / 2 - 1.5, 0.0, -0.05, 1.9, 1.6, 1.5, tag + "out")
    rubble_heap(m, -L / 2 + 1.2, 0.6, fz, 1.2, 1.0, 1.0, tag + "in")
    return m


def v_flat_corner(name, tag):
    """B_flat_2s with the +u/+v corner of the roof slab and its parapet collapsed into the storey below."""
    m = gb.Mesh()
    L, W, _st, h, fz = shell(m, name)
    sx0, sy0, sx1, sy1, top = -L / 2 - 0.3, -W / 2 - 0.3, L / 2 + 0.3, W / 2 + 0.3, h + 0.2
    hx0, hy0 = sx1 - 3.8, sy1 - 3.2                              # the missing corner
    # the surviving slab as two closed boxes, so no unlit backface is ever exposed
    m.box(sx0, sy0, h, sx1, hy0, top, "concrete", bottom=True)
    m.box(sx0, hy0, h, hx0, sy1, top, "concrete", bottom=True)
    p, ph = 0.15, 0.9
    for x0, y0, x1, y1 in ((sx0, sy0, sx1, sy0 + p), (sx0, sy1 - p, hx0, sy1),
                           (sx0, sy0, sx0 + p, sy1), (sx1 - p, sy0, sx1, hy0)):
        m.box(x0, y0, top, x1, y1, top + ph, "wall", top="concrete")
    # the two broken slab edges, and a fragment still hanging into the hole
    two_sided(m, [(hx0, hy0, h), (sx1, hy0, h), (sx1, hy0, top), (hx0, hy0, top)], "rubble")
    two_sided(m, [(hx0, hy0, h), (hx0, sy1, h), (hx0, sy1, top), (hx0, hy0, top)], "rubble")
    two_sided(m, [(hx0 + 0.1, hy0 + 0.1, h), (hx0 + 2.2, hy0 + 0.2, h - 0.15),
                  (hx0 + 2.0, hy0 + 1.9, h - 1.35), (hx0 + 0.2, hy0 + 1.7, h - 1.15)], "concrete")
    cx0, cy0 = L / 2 - 3.6, -W / 2 + 0.4                         # stair cabin, moved clear of the collapse
    m.box(cx0, cy0, top, cx0 + 3.0, cy0 + 2.8, top + 2.4, "wall")
    m.box(cx0 - 0.2, cy0 - 0.2, top + 2.4, cx0 + 3.2, cy0 + 3.0, top + 2.55, "concrete")
    inner_room(m, -L / 2, -W / 2, L / 2, W / 2, fz, h)
    rubble_heap(m, hx0 + 1.6, hy0 + 1.4, fz, 1.5, 1.3, 1.2, tag)
    return m


def v_flat_slump(name, tag):
    """E_flat_1s with the +v wall pushed out by the flow and the slab sagging over it."""
    m = gb.Mesh()
    L, W, _st, h, fz = shell(m, name, sides=(True, True, False, True))
    # gb.openings drew windows and sunshades on the +v wall plane; that wall has moved, so drop them. The rule
    # is above the plinth band (whose +v face centroid sits at z = -0.275) so the plinth survives intact.
    cull(m, lambda x, y, z: y > W / 2 - 0.05 and z > PLINTH + 0.6)
    two_sided(m, [(-L / 2, W / 2, -1.0), (L / 2, W / 2, -1.0),
                  (L / 2, W / 2 + 1.1, h - 0.9), (-L / 2, W / 2 + 1.1, h - 0.9)], "wall")
    sx0, sy0, sx1, top = -L / 2 - 0.3, -W / 2 - 0.3, L / 2 + 0.3, h + 0.2
    two_sided(m, [(sx0, sy0, top), (sx1, sy0, top), (sx1, W / 2 + 0.9, h - 0.65), (sx0, W / 2 + 0.9, h - 0.65)],
              "concrete")
    p, ph = 0.15, 0.9
    for x0, y0, x1, y1 in ((sx0, sy0, sx1, sy0 + p), (sx0, sy0, sx0 + p, W / 2 - 1.0),
                           (sx1 - p, sy0, sx1, W / 2 - 1.0)):
        m.box(x0, y0, top, x1, y1, top + ph, "wall", top="concrete")
    inner_room(m, -L / 2, -W / 2, L / 2, W / 2, fz, h - 0.4)
    rubble_heap(m, -0.2 * L, W / 2 + 1.6, -0.05, 2.0, 1.1, 1.1, tag)
    return m


VARIANTS = {
    "C_sheet_1s_open": ("C_sheet_1s", lambda t: v_sheet("C_sheet_1s", 0.42, True, t)),
    "C_sheet_1s_peel": ("C_sheet_1s", lambda t: v_sheet("C_sheet_1s", 0.74, True, t)),
    "A_tile_1s_holed": ("A_tile_1s", lambda t: v_tile_holed("A_tile_1s", 4.2, 2.6, t)),
    "A_tile_1s_endgone": ("A_tile_1s", lambda t: v_tile_endgone("A_tile_1s", t)),
    "D_tile_2s_holed": ("D_tile_2s", lambda t: v_tile_holed("D_tile_2s", 5.4, 3.0, t)),
    "B_flat_2s_corner": ("B_flat_2s", lambda t: v_flat_corner("B_flat_2s", t)),
    "E_flat_1s_slump": ("E_flat_1s", lambda t: v_flat_slump("E_flat_1s", t)),
}
CHOICES = {
    "C_sheet_1s": ["C_sheet_1s_open", "C_sheet_1s_open", "C_sheet_1s_peel"],
    "A_tile_1s": ["A_tile_1s_holed", "A_tile_1s_holed", "A_tile_1s_endgone"],
    "D_tile_2s": ["D_tile_2s_holed"],
    "B_flat_2s": ["B_flat_2s_corner"],
    "E_flat_1s": ["E_flat_1s_slump"],
}
# Base probability that a house of this archetype is damaged. Corrugated sheds are the flimsiest structures in
# the stock and lose their roofs first; RCC frames mostly survive, so their collapses stay a minority.
DAMAGE_P = {"C_sheet_1s": 0.85, "A_tile_1s": 0.52, "D_tile_2s": 0.46, "B_flat_2s": 0.34, "E_flat_1s": 0.40}

# The local (u_min, u_max, v_min, v_max) rectangle each variant REMOVES or moves, for the `_a` flavour. A house
# is only eligible for a variant if no survivor stands in it (plus a margin): otherwise the mesh swap would drop
# a Rocketbox actor through an opened roof or leave it standing on a slab that is no longer there. Blocking the
# whole house instead would rule out most of the flooded terrace, which is exactly where the damage belongs -
# 26 of the 76 houses carry roof survivors.
_BIG = 99.0
UNSAFE = {
    "C_sheet_1s_open": (-_BIG, _BIG, -1.36, _BIG),      # sheeting kept only up to v = -0.56
    "C_sheet_1s_peel": (-_BIG, _BIG, 0.88, _BIG),       # sheeting kept up to v = 1.68
    "A_tile_1s_holed": (-2.9, 2.9, -0.5, 3.7),          # hole u +/-2.1, v 0.31..2.91
    "D_tile_2s_holed": (-3.5, 3.5, -0.6, 4.0),          # hole u +/-2.7, v 0.20..3.20
    "A_tile_1s_endgone": (-_BIG, -1.2, -_BIG, _BIG),    # everything past u = -2.0
    "B_flat_2s_corner": (0.9, _BIG, 0.8, _BIG),         # slab corner u > 1.7, v > 1.6
    "E_flat_1s_slump": (-_BIG, _BIG, -_BIG, _BIG),      # the whole slab is replaced
}
SURVIVOR_MARGIN_M = 0.8


def local_to_world(b, u, v):
    th = math.radians(b["yaw_deg"])
    return (b["east_m"] + u * math.cos(th) - v * math.sin(th),
            b["north_m"] + u * math.sin(th) + v * math.cos(th))


def build(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    arch = town["archetypes"]
    actors = json.loads((OUT / "actors.json").read_text())["actors"] if (OUT / "actors.json").exists() else []
    props = json.loads((OUT / "props.json").read_text())["props"] if (OUT / "props.json").exists() else {}

    # --- meshes ------------------------------------------------------------------------------------------
    DMG_DIR.mkdir(parents=True, exist_ok=True)
    variants = {}
    for vname, (aname, fn) in VARIANTS.items():
        base = fn(vname)
        for suffix, mesh in (("a", base), ("b", mirror_v(base))):
            tris = mesh.write(DMG_DIR / f"{vname}_{suffix}.obj")
            variants[f"{vname}_{suffix}"] = {"archetype": aname, "obj": f"damage/{vname}_{suffix}.obj",
                                             "triangles": tris, "slots": list(mesh.faces)}

    # --- who is standing on or in each house --------------------------------------------------------------
    def occupants(b):
        """Survivors inside the footprint + 1 m, in the house's local (u, v) frame."""
        a = arch[b["archetype"]]
        th = math.radians(b["yaw_deg"])
        out = []
        for act in actors:
            de, dn = act["east_m"] - b["east_m"], act["north_m"] - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 + 1.0 and abs(v) < a["width_m"] / 2 + 1.0:
                out.append((u, v))
        return out

    def safe(vname, suffix, occ):
        u0, u1, v0, v1 = UNSAFE[vname]
        if suffix == "b":                                    # the mirrored mesh damages the mirrored region
            v0, v1 = -v1, -v0
        m_ = SURVIVOR_MARGIN_M
        return not any(u0 - m_ < u < u1 + m_ and v0 - m_ < v < v1 + m_ for u, v in occ)

    # --- assign variants ----------------------------------------------------------------------------------
    houses, skipped_occupied, occupied_houses = [], 0, 0
    for b in town["houses"]:
        if b["archetype"] not in CHOICES:
            continue
        d = b["flood_depth_m"]
        # deeper water = more damage; the dry bank houses were never in the flow, so they stay almost pristine
        p = DAMAGE_P[b["archetype"]] * (0.45 + 0.55 * min(1.0, d / 2.5)) * (0.20 if d <= 0.05 else 1.0)
        if rng.random() > p:
            continue
        # `_a` tears the roof on local +v, which faces upstream (+north) exactly when cos(theta) > 0
        suffix = "a" if math.cos(math.radians(b["yaw_deg"])) > 0 else "b"
        occ = occupants(b)
        if occ:
            occupied_houses += 1
        options = [v for v in dict.fromkeys(CHOICES[b["archetype"]]) if safe(v, suffix, occ)]
        if not options:
            skipped_occupied += 1
            continue
        # keep the CHOICES weighting among whatever is still safe
        weighted = [v for v in CHOICES[b["archetype"]] if v in options]
        vname = str(rng.choice(weighted))
        houses.append({"id": b["id"], "archetype": b["archetype"], "variant": f"{vname}_{suffix}",
                       "wall_tint": b["wall_tint"], "zone": b["zone"], "flood_depth_m": b["flood_depth_m"],
                       "north_m": b["north_m"], "east_m": b["east_m"], "yaw_deg": b["yaw_deg"],
                       "survivors_on_house": len(occ)})

    variant_of = {h["id"]: h["variant"] for h in houses}

    # --- debris rafted onto the flat roofs ----------------------------------------------------------------
    pool = []
    for pname, p in props.items():
        if p["role"] in ("woody", "household"):
            pool.extend((pname, mm["asset"], mm.get("tris", 0)) for mm in p["meshes"])
    pool.sort(key=lambda t: t[2])                    # cheap meshes first: at 45 m a 53k rock is a few pixels
    debris, tris_used = [], 0
    for b in town["houses"]:
        a = arch[b["archetype"]]
        if a["roof"] not in ("flat", "flat_cabin") or b["flood_depth_m"] <= 0.3 or not pool:
            continue
        top = b["base_asl_m"] + PLINTH + a["storeys"] * STOREY_H + 0.2
        L, W = a["length_m"], a["width_m"]
        collapsed = variant_of.get(b["id"], "").startswith("B_flat_2s_corner")
        for _ in range(int(rng.integers(1, 5))):
            u = float(rng.uniform(-1, 1)) * (L / 2 - 0.5)
            v = float(rng.uniform(-1, 1)) * (W / 2 - 0.5)
            if a["roof"] == "flat_cabin" and u > L / 2 - 3.9:
                if W / 2 - 3.4 < v < W / 2 - 0.2 or -W / 2 + 0.2 < v < -W / 2 + 3.4:
                    continue                                                # the stair cabin, either position
            if collapsed and u > L / 2 - 3.5 and v > W / 2 - 2.9:
                continue                                                    # the missing corner
            e, nn = local_to_world(b, u, v)
            if any((act["east_m"] - e) ** 2 + (act["north_m"] - nn) ** 2 < 2.25 for act in actors):
                continue                                                    # never occlude a survivor
            kind, asset, mt = pool[int(rng.integers(0, max(1, int(len(pool) * 0.6))))]
            if tris_used + mt > ROOF_DEBRIS_TRI_BUDGET:
                continue
            debris.append({"id": len(debris), "name": f"RoofDebris_{kind}_{len(debris):03d}", "asset": asset,
                           "house_id": b["id"], "east_m": round(e, 2), "north_m": round(nn, 2),
                           "base_asl_m": round(top, 3), "yaw_deg": round(float(rng.uniform(0, 360)), 1),
                           "pitch_deg": round(float(rng.normal(0, 8)), 1),
                           "roll_deg": round(float(rng.normal(0, 8)), 1),
                           "scale": round(float(rng.uniform(0.85, 1.35)), 3), "tris": mt})
            tris_used += mt

    by_variant = {}
    for h in houses:
        by_variant[h["variant"]] = by_variant.get(h["variant"], 0) + 1
    mesh_tris = sum(variants[h["variant"]]["triangles"] for h in houses)
    pristine_tris = sum(arch[h["archetype"]]["triangles"] for h in houses)
    return {
        "generated_by": "tools/scene/gen_damage.py", "seed": seed, "terrain_seed": meta["seed"],
        "settlement_seed": town["seed"], "water_level_m": meta["water_level_m"], "base_z_m": meta["base_z_m"],
        "variants": variants, "houses": houses, "roof_debris": debris,
        "counts": {
            "houses_total": len(town["houses"]), "houses_damaged": len(houses),
            "skipped_because_occupied": skipped_occupied, "damaged_houses_with_survivors": occupied_houses,
            "damaged_and_occupied": sum(1 for h in houses if h["survivors_on_house"]),
            "by_variant": dict(sorted(by_variant.items())),
            "by_archetype": {a: f"{sum(1 for h in houses if h['archetype'] == a)}/"
                                f"{sum(1 for b in town['houses'] if b['archetype'] == a)}" for a in CHOICES},
            "by_zone": {z: sum(1 for h in houses if h["zone"] == z) for z in ("terrace", "bank")},
            "roof_debris": len(debris), "roof_debris_triangles": int(tris_used),
            "roof_debris_triangle_budget": ROOF_DEBRIS_TRI_BUDGET,
            "variant_mesh_triangles_in_scene": int(mesh_tris),
            "pristine_mesh_triangles_replaced": int(pristine_tris),
            "net_added_triangles": int(mesh_tris - pristine_tris + tris_used),
        },
    }


def main(seed: int) -> None:
    d = build(seed)
    (OUT / "damage.json").write_text(json.dumps(d, indent=1))
    c = d["counts"]
    tri = [v["triangles"] for v in d["variants"].values()]
    print(f"variants: {len(d['variants'])} meshes ({min(tri)}-{max(tri)} tris each)")
    print(f"damaged houses: {c['houses_damaged']} / {c['houses_total']} "
          f"({c['skipped_because_occupied']} rolled damage but no variant was clear of their survivors; "
          f"{c['damaged_and_occupied']} are damaged AND carry survivors, well away from the damage)")
    print(f"  by variant:   {c['by_variant']}")
    print(f"  by archetype: {c['by_archetype']}")
    print(f"  by zone:      {c['by_zone']}")
    print(f"roof debris: {c['roof_debris']} items, {c['roof_debris_triangles']:,} tris "
          f"(budget {c['roof_debris_triangle_budget']:,})")
    print(f"NET added triangles: {c['net_added_triangles']:,} "
          f"(variant meshes {c['variant_mesh_triangles_in_scene']:,} replace "
          f"{c['pristine_mesh_triangles_replaced']:,} pristine)")
    print(f"written to {OUT / 'damage.json'} and {DMG_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=41)
    main(ap.parse_args().seed)
