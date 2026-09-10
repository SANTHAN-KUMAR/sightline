"""Validate everything the scene-realism lane produced, and FAIL if it is wrong. Exits non-zero.

    uv run python tools/scene/check_realism.py [--strict]

Two halves:

A. THE DATA. Re-derives the terrain, re-reads every generated OBJ and every source glTF, and asserts the
   claims the two generators make: clearances actually hold, geometry is not degenerate, the OBJ indices are
   in range, the lean encoding round-trips through UE's own FRotationMatrix formulas, the boats float, the
   wire is the width it says it is at the calibrated survey GSD, and both generators are byte-deterministic
   across two runs in the same process.

B. THE EDITOR SCRIPTS' CONTROL FLOW. `build_vegetation.py`, `build_utilities.py` and `tune_water.py` cannot
   be executed here (this lane may not open the editor), so they are exec'd against a mock `unreal` module
   under several configurations - Nanite refusing to turn on, the plume-field probe failing, a wrong actor
   footprint - and each configuration asserts the script either completes or raises for the right reason.
   This validates our own control flow, the JSON handling and the bounds arithmetic. It CANNOT validate UE
   API names or shader compilation; see docs/lanes/scene_realism.md for that list.

Everything here is written so that it can fail. `--self-test` deliberately corrupts a copy of the data in
four ways and asserts that the checks catch all four; run it if you ever doubt that this file is a test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_terrain as gt  # noqa: E402
import gen_utilities as gu  # noqa: E402
import gen_vegetation as gv  # noqa: E402

OUT = gt.OUT
FAILS: list[str] = []
CHECKS = [0]


def check(cond: bool, msg: str) -> bool:
    CHECKS[0] += 1
    if not cond:
        FAILS.append(msg)
    return bool(cond)


# ==========================================================================================================
# OBJ reader
# ==========================================================================================================
def read_obj(path: Path) -> dict:
    v, vt, vn, faces, cur = [], [], [], {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            v.append(tuple(float(x) for x in line[2:].split()))
        elif line.startswith("vt "):
            vt.append(tuple(float(x) for x in line[3:].split()))
        elif line.startswith("vn "):
            vn.append(tuple(float(x) for x in line[3:].split()))
        elif line.startswith("usemtl "):
            cur = line[7:].strip()
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line[2:].split()]
            faces.setdefault(cur, []).append(idx)
    return {"v": np.asarray(v, dtype=float), "vt": vt, "vn": vn, "faces": faces,
            "tris": sum(len(f) for f in faces.values())}


def obj_geometry_checks(name: str, o: dict, known_slots: set[str]) -> None:
    V = o["v"]
    check(len(V) > 0, f"{name}: no vertices")
    check(len(o["vn"]) == len(V), f"{name}: {len(o['vn'])} normals for {len(V)} vertices (UE flat-shades "
                                  f"an OBJ without vn)")
    check(len(o["vt"]) == len(V), f"{name}: {len(o['vt'])} UVs for {len(V)} vertices")
    bad_slot = sorted(set(o["faces"]) - known_slots)
    check(not bad_slot, f"{name}: unknown usemtl groups {bad_slot} - build_*.py maps slots by name")
    flat = [i for f in o["faces"].values() for tri in f for i in tri]
    check(min(flat) >= 1 and max(flat) <= len(V),
          f"{name}: face index out of range (1..{len(V)} expected, got {min(flat)}..{max(flat)})")
    tri = np.array([[V[i - 1] for i in t] for f in o["faces"].values() for t in f])
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    check((area > 1e-4).all(), f"{name}: {(area <= 1e-4).sum()} degenerate (zero-area) triangles")
    # a flat quad masquerading as a solid is the exact failure the brief warns about
    ext = V.max(axis=0) - V.min(axis=0)
    check((ext > 1.0).sum() >= 3, f"{name}: bounding box {ext.round(1)} cm is degenerate in an axis - "
                                  f"this mesh is flat, not solid")


# ==========================================================================================================
# UE rotator maths, straight from FRotationMatrix, so the lean encoding is verified and not assumed
# ==========================================================================================================
def ue_up_axis(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """World direction of an actor's local +Z. UE's FRotationMatrix third row:
        Z = ( -(CR*SP*CY + SR*SY),  CY*SR - CR*SP*SY,  CR*CP )"""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    return np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])


# ==========================================================================================================
# A. data checks
# ==========================================================================================================
def check_vegetation(strict: bool) -> dict:
    d = json.loads((OUT / "vegetation.json").read_text())
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    actors = json.loads((OUT / "actors.json").read_text())["actors"]
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water = s["water_level"]
    arch, houses = town["archetypes"], town["houses"]
    sv = np.array([[a["east_m"], a["north_m"]] for a in actors])
    L = meta["launch_site"]

    check(d["terrain_seed"] == meta["seed"],
          f"vegetation was generated for terrain seed {d['terrain_seed']}, the level uses {meta['seed']}")
    check(abs(d["water_level_m"] - meta["water_level_m"]) < 1e-6,
          "vegetation water level does not match flood_valley.json")

    # source integrity: a half-finished download must not be shipped as a placement
    for pid, e in sorted(d["catalogue"].items()):
        m = gv.measure_gltf(pid)
        check(m["ok"], f"catalogue species {pid}: {m.get('why')}")
        if m["ok"] and pid in gv.SPECIES:
            check(m["tris"] == e.get("tris_gltf", e["tris"]),
                  f"{pid}: the {m['variant']} glTF now has {m['tris']} tris, the layout measured "
                  f"{e.get('tris_gltf')} - regenerate")

    items = d["items"]
    check(len(items) == d["counts"]["trees"], "tree count does not match the items list")
    inside = near_sv = near_pad = bad_scale = bad_z = bad_ground = bad_asset = 0
    for t in items:
        e, nn = t["east_m"], t["north_m"]
        for b in houses:
            a = arch[b["archetype"]]
            th = math.radians(b["yaw_deg"])
            de, dn = e - b["east_m"], nn - b["north_m"]
            u = de * math.cos(th) + dn * math.sin(th)
            v = -de * math.sin(th) + dn * math.cos(th)
            if abs(u) < a["length_m"] / 2 + gv.CLEAR_HOUSE_M - 1e-6 and \
                    abs(v) < a["width_m"] / 2 + gv.CLEAR_HOUSE_M - 1e-6:
                inside += 1
                break
        if len(sv) and float(np.min(np.hypot(sv[:, 0] - e, sv[:, 1] - nn))) < gv.CLEAR_SURVIVOR_M - 1e-6:
            near_sv += 1
        if math.hypot(e - L["east_m"], nn - L["north_m"]) < gv.CLEAR_PAD_M - 1e-6:
            near_pad += 1
        lo, hi = gv.SPECIES[t["pid"]]["scale"]
        if not (lo - 1e-6 <= t["scale"] <= hi + 1e-6):
            bad_scale += 1
        # The base must equal ground - undercut - sink EXACTLY (rounding aside); and the ground the
        # generator recorded must be a real height of the terrain near that point. Checking against a
        # freshly-looked-up cell alone is not enough: east/north are rounded to 1 cm, so a tree within a
        # centimetre of a 4 m cell boundary legitimately falls in the neighbouring cell.
        want = t["ground_asl_m"] - t["undercut_m"] - gv.SINK_CM / 100.0
        # 0.008 m of slack: base is stored to 3 dp, ground to 3 dp and undercut to 2 dp
        if abs(t["base_asl_m"] - want) > 0.008:
            bad_z += 1
        i = int((e + size / 2) / cell)
        j = int((nn + size / 2) / cell)
        i, j = min(max(i, 0), n - 2), min(max(j, 0), n - 2)
        loc = h[j:j + 2, i:i + 2]
        if not (loc.min() - 0.30 <= t["ground_asl_m"] <= loc.max() + 0.30):
            bad_ground += 1
        if not str(t["asset_hint"]).startswith("/Game/Sightline/Props/"):
            bad_asset += 1
    check(inside == 0, f"{inside} trees stand inside a house footprint + {gv.CLEAR_HOUSE_M} m")
    check(near_sv == 0, f"{near_sv} trees are within {gv.CLEAR_SURVIVOR_M} m of a survivor")
    check(near_pad == 0, f"{near_pad} trees are within {gv.CLEAR_PAD_M} m of the launch pad")
    check(bad_scale == 0, f"{bad_scale} trees carry a scale outside their species range")
    check(bad_z == 0, f"{bad_z} trees have base != ground - undercut - sink")
    check(bad_ground == 0, f"{bad_ground} trees record a ground height that is not the terrain there")
    check(bad_asset == 0, f"{bad_asset} trees have an asset hint outside /Game/Sightline/Props/")

    # the lean encoding must round-trip through UE's own rotation matrix
    worst_tilt = worst_azim = 0.0
    for t in items:
        if t["tilt_deg"] <= 0.01:
            continue
        z = ue_up_axis(t["roll_deg"], t["pitch_deg"], t["yaw_deg"])
        worst_tilt = max(worst_tilt, abs(math.degrees(math.acos(min(1.0, z[2]))) - t["tilt_deg"]))
        az = math.degrees(math.atan2(z[1], z[0])) % 360.0
        worst_azim = max(worst_azim, min(abs(az - t["yaw_deg"]), 360 - abs(az - t["yaw_deg"])))
    # tolerance 0.06 deg: tilt_deg is stored to 1 dp and pitch_deg to 2 dp, so 0.05 deg is rounding
    check(worst_tilt < 0.06, f"lean encoding: worst tilt error {worst_tilt:.3f} deg (UE FRotationMatrix)")
    check(worst_azim < 0.06, f"lean encoding: worst azimuth error {worst_azim:.3f} deg")

    # drowned trees must actually lean downstream (towards -north = azimuth 180 in UE X=north/Y=east)
    lean = [t for t in items if t["tilt_deg"] > 5.5]
    if lean:
        az = np.array([t["yaw_deg"] for t in lean])
        south = np.mean(np.cos(np.radians(az - 180.0)) > 0.5)
        check(south > 0.75, f"only {south * 100:.0f} % of the {len(lean)} scoured trees lean downstream "
                            f"(azimuth 180); the flow runs north -> south")

    # the "crowns still emerging from shallow water" claim, as a number
    em = [t["emergent_m"] for t in items if t["emergent_m"] is not None]
    shallow = [x for x in em if x is not None and x < 3.0]
    check(len(em) > 200, f"only {len(em)} trees stand in water; the flooded terrace should carry hundreds")
    check(len(shallow) > 50, f"only {len(shallow)} drowned trees show less than 3 m above the surface - the "
                             f"'crowns emerging' look needs small trees in deep water")
    check(min(em) > 0.30, f"a tree crown reaches only {min(em):.2f} m above the water: it is 1-4 M "
                          f"triangles of invisible geometry")

    check(d["counts"]["instanced_source_triangles"] == sum(t["tris"] for t in items),
          "instanced triangle total does not match the sum over items")
    check(d["canopy"]["closure_roi"] > 0.30,
          f"canopy closure over the ROI is only {d['canopy']['closure_roi'] * 100:.1f} %; the reference is "
          f"roughly half canopy")
    if strict:
        check(d["canopy"]["closure_settlement"] > 0.35,
              f"canopy closure over the settlement is {d['canopy']['closure_settlement'] * 100:.1f} %")
    check((OUT / "vegetation_canopy.png").is_file(), "vegetation_canopy.png was not written - LOOK at it")
    _ = water
    return d


def check_utilities(strict: bool) -> dict:
    d = json.loads((OUT / "utilities.json").read_text())
    meta = json.loads((OUT / "flood_valley.json").read_text())
    town = json.loads((OUT / "settlement.json").read_text())
    actors = json.loads((OUT / "actors.json").read_text())["actors"]
    s = gt.build(meta["size_m"], meta["cell_m"], meta["seed"])
    h, n, cell, size = s["height"], s["n"], s["cell_m"], s["size_m"]
    water, base_z = s["water_level"], float(h.min())
    sv = np.array([[a["east_m"], a["north_m"]] for a in actors])
    known = set(gu.TILE_M)

    net = d["network"]
    o = read_obj(OUT / "utilities" / "network.obj")
    obj_geometry_checks("network.obj", o, known)
    check(o["tris"] == net["triangles"],
          f"network.obj holds {o['tris']} triangles, utilities.json claims {net['triangles']}")
    V = o["v"]
    b = net["ue_actor"]["bounds_cm"]
    check(abs(V[:, 0].min() - b["y_east"][0]) < 1.0 and abs(V[:, 1].min() - b["x_north"][0]) < 1.0,
          "network.obj bounds do not match the bounds build_utilities.py will assert against")
    check(abs(V).max() < size * 100.0, "network.obj leaves the 2 x 2 km map")
    check(V[:, 2].min() > -50.0, f"network.obj dips to z = {V[:, 2].min():.0f} cm: it was not shifted to "
                                 f"(asl - base_z), so the actor at the origin would float 1046 m up")

    heights = np.array([p["height_m"] for p in d["pole_list"]])
    check(len(heights) == net["poles"], "pole_list length does not match the pole count")
    check(heights.min() >= gu.POLE_H_M[0] - 1e-6 and heights.max() <= gu.POLE_H_M[1] + 1e-6,
          f"pole heights {heights.min():.2f}-{heights.max():.2f} m are outside {gu.POLE_H_M}")
    pe = np.array([[p["east_m"], p["north_m"]] for p in d["pole_list"]])
    dd = np.hypot(pe[:, None, 0] - pe[None, :, 0], pe[:, None, 1] - pe[None, :, 1])
    np.fill_diagonal(dd, 1e9)
    check(dd.min() > gu.SPAN_MIN_M * 0.9 - 1e-6,
          f"two poles are {dd.min():.1f} m apart; the minimum separation is {gu.SPAN_MIN_M * 0.9:.1f} m")
    if len(sv):
        near = min(float(np.min(np.hypot(sv[:, 0] - p[0], sv[:, 1] - p[1]))) for p in pe)
        check(near >= 2.0 - 1e-6, f"a pole stands {near:.2f} m from a survivor")
    for p in d["pole_list"]:
        i = int(round((p["east_m"] + size / 2) / cell))
        j = int(round((p["north_m"] + size / 2) / cell))
        gz = float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])
        if not check(abs(gz - p["ground_asl_m"]) < 0.01,
                     f"pole at ({p['east_m']}, {p['north_m']}) records ground {p['ground_asl_m']} m but the "
                     f"terrain is {gz:.3f} m"):
            break

    # the wire has to survive the survey resampling: this is the number the brief asks to be measured
    gsd = 45.0 * 100.0 / json.loads((OUT / "camera_survey.json").read_text())["f_px"]
    check(abs(gsd - d["survey"]["gsd_cm_per_px"]) < 1e-3,
          f"utilities.json GSD {d['survey']['gsd_cm_per_px']} != {gsd:.4f} cm/px from camera_survey.json")
    px = net["wire_diameter_m"] * 100.0 / gsd
    check(px >= 1.5, f"a {net['wire_diameter_m'] * 100:.1f} cm wire is {px:.2f} px at {gsd:.3f} cm/px - "
                     f"below ~1.5 px TAA and the 4K resample will erase it")

    # boats
    for name, v in sorted(d["boats"]["variants"].items()):
        ob = read_obj(OUT / "utilities" / f"boat_{name}.obj")
        obj_geometry_checks(f"boat_{name}.obj", ob, known)
        check(ob["tris"] == v["triangles"], f"boat_{name}.obj holds {ob['tris']} tris, json says "
                                            f"{v['triangles']}")
        z = ob["v"][:, 2] / 100.0
        check(abs(z.min() + v["draught_m"]) < 0.02 and abs(z.max() - v["freeboard_m"]) < 0.02,
              f"boat_{name}: z range {z.min():.2f}..{z.max():.2f} m does not match the declared "
              f"draught/freeboard - z = 0 must be the design waterline")
        check(0.08 < v["draught_m"] < 0.45, f"boat_{name} draught {v['draught_m']} m is not a small craft")
        check(v["freeboard_m"] > 0.25, f"boat_{name} freeboard {v['freeboard_m']} m is too low to read as a "
                                       f"boat from 45 m")
    for it in d["boats"]["items"]:
        i = int(round((it["east_m"] + size / 2) / cell))
        j = int(round((it["north_m"] + size / 2) / cell))
        gz = float(h[min(max(j, 0), n - 1), min(max(i, 0), n - 1)])
        if not check(water - gz > 0.5,
                     f"{it['name']} floats in {water - gz:.2f} m of water; it would sit on the ground"):
            break
        if not check(abs(it["base_asl_m"] - water) < 0.05,
                     f"{it['name']} is {it['base_asl_m'] - water:+.2f} m off the flood surface"):
            break
        if len(sv) and not check(
                float(np.min(np.hypot(sv[:, 0] - it["east_m"], sv[:, 1] - it["north_m"]))) >= gu.CLEAR_SURVIVOR_M,
                f"{it['name']} is within {gu.CLEAR_SURVIVOR_M} m of a survivor"):
            break
    check(d["counts"]["total_triangles"] == net["triangles"] + d["boats"]["triangles"],
          "utilities triangle total does not add up")
    for f in ("preview_pole.png", "preview_span.png", "preview_network.png"):
        check((OUT / "utilities" / f).is_file(), f"{f} was not rendered - it is the only look at the shape")
    _ = base_z, town
    return d


def check_determinism() -> None:
    a = gv.build(67, 4600, 110.0, True)
    b = gv.build(67, 4600, 110.0, True)
    a.pop("_cover_png", None)
    b.pop("_cover_png", None)
    check(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True),
          "gen_vegetation is NOT deterministic across two runs at the same seed")
    c = gu.build(83)
    e = gu.build(83)
    check(json.dumps(c, sort_keys=True) == json.dumps(e, sort_keys=True),
          "gen_utilities is NOT deterministic across two runs at the same seed")


# ==========================================================================================================
# B. dry-run the editor scripts against a mock `unreal`
# ==========================================================================================================
class MockMeta(type):
    def __instancecheck__(cls, obj):
        return getattr(cls, "_answer", True)


class Obj(metaclass=MockMeta):
    """Permissive stand-in for any UE object. Records set properties so a script's own logic can read them
    back, which is what makes the idempotency and assertion paths real rather than vacuous."""

    def __init__(self, name="Obj", **props):
        self._name = name
        self._p = dict(props)

    def get_editor_property(self, k):
        return self._p.get(k, Obj(k))

    def set_editor_property(self, k, v):
        self._p[k] = v

    def get_name(self):
        return self._name

    def get_path_name(self):
        return f"/Game/Mock/{self._name}"

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        if k in self._p:                       # props set at construction read back as values, not callables
            return self._p[k]
        return lambda *a, **kw: Obj(k)


def make_unreal(cfg):
    u = types.ModuleType("unreal")
    world = {"actors": []}
    imported = {"n": 0}

    class Cls(Obj):
        pass

    def klass(name):
        answer = True
        if name == "MaterialInstanceConstant":
            answer = cfg.get("is_instance", True)
        if name == "StaticMesh":
            answer = True
        if name == "MaterialExpressionLinearInterpolate":
            answer = cfg.get("lerp_ok", True)
        if name == "MaterialExpressionSingleLayerWaterMaterialOutput":
            answer = cfg.get("slw_ok", True)
        if name == "MaterialExpression":
            answer = cfg.get("expr_ok", True)
        if name in ("MaterialExpressionVectorParameter", "MaterialExpressionScalarParameter"):
            answer = cfg.get("param_ok", True)
        return MockMeta(name, (Obj,), {"_answer": answer})

    cache: dict[str, object] = {}

    class Stats:
        num_pixel_shader_instructions = cfg.get("instructions", 240)
        num_pixel_texture_samples = cfg.get("samples", 4)

    class Vector:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Rotator(Vector):
        pass

    class LinearColor:
        def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
            self.r, self.g, self.b, self.a = r, g, b, a

    def slot(nm):
        return Obj("slot", material_slot_name=nm, material_interface=Obj(f"MI_{nm}"))

    def static_mesh(name, slots, tris):
        ns = Obj("ns", enabled=cfg.get("nanite_sticks", True))
        if not cfg.get("nanite_sticks", True):
            # the engine refusing to enable Nanite looks exactly like this: the property write is accepted
            # and the value read back afterwards is still False
            ns.set_editor_property = lambda k, v: None
        return Obj(name, nanite_settings=ns, static_materials=[slot(x) for x in slots],
                   body_setup=Obj("bs"), _tris=tris)

    def make_mesh(name, slots, tris, bbox=(500.0, 500.0, 1000.0)):
        sm = static_mesh(name, slots, tris)
        sm.get_num_triangles = lambda lod=0: tris
        sm.get_bounding_box = lambda: Obj("bb", min=Vector(-bbox[0] / 2, -bbox[1] / 2, 0),
                                          max=Vector(bbox[0] / 2, bbox[1] / 2, bbox[2]))
        return sm

    veg = json.loads((OUT / "vegetation.json").read_text())
    uti = json.loads((OUT / "utilities.json").read_text())
    tri_by_name = {"SM_Utilities": uti["network"]["triangles"]}
    for k, v in uti["boats"]["variants"].items():
        tri_by_name[f"SM_Boat_{k}"] = v["triangles"]

    def load_asset(path):
        if path in cache:
            return cache[path]
        nm = str(path).rsplit("/", 1)[-1]
        if nm in tri_by_name:
            o = make_mesh(nm, ["concrete", "roof_sheet", "wood", "wire"], tri_by_name[nm])
        elif "/Props/" in str(path):
            o = make_mesh(nm, ["trunk", "leaves", "branches"], 400)
        elif nm == "M_FloodWater":
            o = Obj(nm)
        elif cfg.get("missing_material") and nm == cfg["missing_material"]:
            return None
        else:
            o = Obj(nm)
        cache[path] = o
        cache[o.get_path_name()] = o     # re-loading by the asset's own path must return the SAME object,
        return o                         # or a script that re-reads a property it just set gets a fresh mock

    class EAL:
        @staticmethod
        def does_asset_exist(p):
            return cfg.get("assets_exist", True)

        @staticmethod
        def does_directory_exist(p):
            # a folder that did not exist before the import does exist after it
            return cfg.get("dir_exists", True) or imported["n"] > 0

        @staticmethod
        def list_assets(p, recursive=True, include_folder=False):
            return [f"{p}/mesh_LOD0"]

        @staticmethod
        def save_asset(p):
            return True

    exprs: list[Obj] = []

    class MEL:
        @staticmethod
        def get_material_expressions(m):
            return list(exprs)

        @staticmethod
        def delete_material_expression(m, e):
            if e in exprs:
                exprs.remove(e)
            return True

        @staticmethod
        def get_num_material_expressions(m):
            return len(exprs)

        @staticmethod
        def create_material_expression(m, cls, x=0, y=0):
            e = Obj(getattr(cls, "__name__", "Expr"))
            if not cfg.get("desc_ok", True):
                def no_desc(k, v=None):
                    if k == "desc":
                        raise Exception("Desc is not exposed")
                    return None
                e.get_editor_property = lambda k: (_ for _ in ()).throw(Exception("no desc")) \
                    if k == "desc" else Obj(k)
                e.set_editor_property = no_desc
            exprs.append(e)
            return e

        @staticmethod
        def connect_material_expressions(a, ao, b, bi):
            return True

        @staticmethod
        def connect_material_property(a, ao, p):
            return True

        @staticmethod
        def recompile_material(m):
            return True

        @staticmethod
        def get_statistics(m):
            return Stats()

        @staticmethod
        def set_material_instance_parent(i, p):
            return True

        @staticmethod
        def set_material_instance_texture_parameter_value(i, n, t):
            return True

        @staticmethod
        def get_material_instance_texture_parameter_value(i, n):
            return None if cfg.get("no_wood_tex") else Obj(f"T_{n}")

        @staticmethod
        def set_material_instance_vector_parameter_value(i, n, v):
            return True

        @staticmethod
        def update_material_instance(i):
            return True

        @staticmethod
        def get_material_property_input_node(m, p):
            return None if cfg.get("no_input_node") else Obj("Lerp", alpha=Obj("in", expression=Obj("Sat")))

    class EAS:
        @staticmethod
        def get_all_level_actors():
            return list(world["actors"])

        @staticmethod
        def destroy_actor(a):
            if a in world["actors"]:
                world["actors"].remove(a)

        @staticmethod
        def spawn_actor_from_object(mesh, loc, rot):
            a = Obj("actor", _folder="")
            a.set_actor_label = lambda s: None
            a.set_folder_path = lambda f: a.set_editor_property("_folder", f)
            a.get_folder_path = lambda: a.get_editor_property("_folder")
            a.static_mesh_component = Obj("smc")
            a.set_actor_scale3d = lambda v: None
            a.get_actor_location = lambda: Vector(0, 0, 0)
            a.set_actor_location = lambda v, a1, a2: None
            f = cfg.get("bounds_factor", 1.0)
            net = uti["network"]["ue_actor"]["bounds_cm"]
            ox = (net["x_north"][0] + net["x_north"][1]) / 2
            oy = (net["y_east"][0] + net["y_east"][1]) / 2
            hx = (net["x_north"][1] - net["x_north"][0]) / 2 * f
            hy = (net["y_east"][1] - net["y_east"][0]) / 2 * f
            a.get_actor_bounds = lambda simple: (Vector(ox, oy, 0), Vector(hx, hy, 500))
            world["actors"].append(a)
            return a

    class LES:
        @staticmethod
        def is_in_play_in_editor():
            return False

        @staticmethod
        def save_current_level():
            return True

    class Tools:
        @staticmethod
        def import_asset_tasks(t):
            imported["n"] += 1
            return True

        @staticmethod
        def create_asset(n, d, c, f):
            return Obj(n)

    def get_editor_subsystem(cls):
        nm = getattr(cls, "__name__", "")
        return {"LevelEditorSubsystem": LES, "EditorActorSubsystem": EAS}.get(nm, Obj(nm))

    u.EditorAssetLibrary = EAL
    u.MaterialEditingLibrary = MEL
    u.AssetToolsHelpers = Obj("ath")
    u.AssetToolsHelpers.get_asset_tools = lambda: Tools
    u.get_editor_subsystem = get_editor_subsystem
    u.Vector, u.Rotator, u.LinearColor = Vector, Rotator, LinearColor
    u.AssetImportTask = lambda: Obj("task", imported_object_paths=["/Game/Mock/mesh_LOD0"])
    u.SystemLibrary = Obj("sys")
    u.SystemLibrary.collect_garbage = lambda: None

    def getattr_(name):
        if name not in cache:
            cache[name] = klass(name)
        return cache[name]

    u.__getattr__ = getattr_
    u.load_asset = load_asset
    u.LevelEditorSubsystem = klass("LevelEditorSubsystem")
    u.EditorActorSubsystem = klass("EditorActorSubsystem")
    u.StaticMeshEditorSubsystem = klass("StaticMeshEditorSubsystem")
    u.ComponentMobility = Obj("mob", STATIC="STATIC", MOVABLE="MOVABLE")
    u.MaterialProperty = Obj("mp")
    u.BlendMode = Obj("bm", BM_OPAQUE="BM_OPAQUE")
    u.Material = klass("Material")
    u.MaterialInstanceConstant = klass("MaterialInstanceConstant")
    u.StaticMesh = klass("StaticMesh")
    u.MaterialExpression = klass("MaterialExpression")
    u.MaterialExpressionLinearInterpolate = klass("MaterialExpressionLinearInterpolate")
    u.MaterialExpressionSingleLayerWaterMaterialOutput = klass(
        "MaterialExpressionSingleLayerWaterMaterialOutput")

    # the water parameters build_materials.py leaves behind, so tune_water can find them
    if cfg.get("water_params", True):
        names = ["Scattering", "Absorption", "PhaseG", "ColorScaleBehindWater", "BaseColor",
                 "BaseColorSilt", "Roughness", "RoughnessSilt", "Specular", "NormalStrength"]
        if cfg.get("drop_param"):
            names.remove(cfg["drop_param"])
        for nm in names:
            e = Obj(nm, parameter_name=nm, desc="")
            exprs.append(e)
    if cfg.get("slw_present", True):
        exprs.append(Obj("SLWOut", desc=""))
    _ = veg
    return u


def dry_run(script: str, cfg: dict):
    u = make_unreal(cfg)
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = u
    try:
        src = (HERE / script).read_text(encoding="utf-8")
        g = {"__name__": "__dry_run__", "__file__": str(HERE / script)}
        exec(compile(src, script, "exec"), g)  # noqa: S102 - deliberately exec'ing our own script
        return None
    except BaseException as exc:  # noqa: BLE001
        return exc
    finally:
        if saved is None:
            sys.modules.pop("unreal", None)
        else:
            sys.modules["unreal"] = saved


def check_editor_scripts() -> None:
    import io
    import contextlib
    cases = [
        ("build_vegetation.py", {}, None),
        ("build_vegetation.py", {"is_instance": False}, None),            # plain Material branch
        ("build_vegetation.py", {"assets_exist": False}, None),           # resolve by folder listing
        ("build_vegetation.py", {"assets_exist": False, "dir_exists": False}, None),   # import path
        ("build_vegetation.py", {"nanite_sticks": False}, "Nanite is OFF"),
        ("build_vegetation.py", {"instructions": 0}, "FAILED TO COMPILE"),
        ("build_utilities.py", {}, None),
        ("build_utilities.py", {"bounds_factor": 0.5}, "wrong footprint"),
        ("build_utilities.py", {"no_wood_tex": True}, "no BaseColor texture override"),
        ("tune_water.py", {}, None),
        ("tune_water.py", {"no_input_node": True}, None),                 # plume probe fails -> constants
        ("tune_water.py", {"drop_param": "PhaseG"}, "does not carry the parameters"),
        ("tune_water.py", {"slw_present": False, "slw_ok": False}, "no SingleLayerWaterMaterialOutput"),
        ("tune_water.py", {"instructions": 0}, "FAILED TO COMPILE"),
    ]
    for script, cfg, expect in cases:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exc = dry_run(script, cfg)
        label = f"{script} {cfg or '{}'}"
        if expect is None:
            check(exc is None, f"dry run {label}: unexpected {type(exc).__name__}: {exc}")
        else:
            check(exc is not None and expect in str(exc),
                  f"dry run {label}: expected a failure containing {expect!r}, got {exc!r}")


# ==========================================================================================================
def self_test() -> int:
    """Prove the checks can fail: corrupt the data four ways and require each to be caught."""
    veg = OUT / "vegetation.json"
    orig = veg.read_text()
    bad = 0
    try:
        for name, mutate in (
            ("tree inside a house", lambda d: d["items"].__setitem__(
                0, {**d["items"][0], "east_m": json.loads((OUT / "settlement.json").read_text())
                    ["houses"][0]["east_m"], "north_m": json.loads((OUT / "settlement.json").read_text())
                    ["houses"][0]["north_m"]})),
            ("tree floating in the air", lambda d: d["items"].__setitem__(
                1, {**d["items"][1], "base_asl_m": d["items"][1]["base_asl_m"] + 12.0})),
            ("scale outside the species range", lambda d: d["items"].__setitem__(
                2, {**d["items"][2], "scale": 9.9})),
            ("triangle total does not add up", lambda d: d["counts"].__setitem__(
                "instanced_source_triangles", 1)),
        ):
            d = json.loads(orig)
            mutate(d)
            veg.write_text(json.dumps(d))
            FAILS.clear()
            check_vegetation(False)
            if FAILS:
                bad += 1
                print(f"  caught: {name} -> {FAILS[0]}")
            else:
                print(f"  MISSED: {name}")
    finally:
        veg.write_text(orig)
    FAILS.clear()
    print(f"self-test: {bad}/4 deliberate corruptions caught")
    return 0 if bad == 4 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="also enforce the settlement-closure target")
    ap.add_argument("--self-test", action="store_true", help="corrupt the data and prove the checks fire")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    v = check_vegetation(a.strict)
    u = check_utilities(a.strict)
    check_determinism()
    check_editor_scripts()
    print(f"vegetation: {v['counts']['trees']:,} trees + {v['counts']['ground_cover']:,} understorey, "
          f"closure {v['canopy']['closure_roi'] * 100:.1f} % ROI / "
          f"{v['canopy']['closure_settlement'] * 100:.1f} % settlement")
    print(f"utilities:  {u['counts']['poles']} poles, {u['counts']['spans']} spans, "
          f"{u['counts']['service_drops']} drops, {u['counts']['boats']} boats, "
          f"{u['counts']['total_triangles']:,} tris")
    print(f"{CHECKS[0]} checks run")
    if FAILS:
        print(f"\nFAILED {len(FAILS)} check(s) - DO NOT BUILD THIS INTO THE LEVEL:")
        for f in FAILS:
            print("  *", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
