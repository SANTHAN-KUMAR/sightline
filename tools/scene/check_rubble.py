"""Judge the rubble + structural-damage placement, and FAIL if it is wrong. Exits non-zero.

    uv run python tools/scene/check_rubble.py --no-report   # before the editor run: data + dry runs only
    uv run python tools/scene/check_rubble.py               # after it: everything, including the level report
    uv run python tools/scene/check_rubble.py --self-test   # prove this file can actually fail

Three halves, none of which trusts a return value:

A. THE PLAN. `data/scene/rubble_layout.json` and `data/scene/damage.json` are re-derived far enough to be
   trusted as input: every variant an item names has an OBJ on disk and a triangle count, the per-group and
   per-family totals add up, the triangle budget is respected, and every roof-debris record points at a real
   asset path.

B. THE LEVEL, as measured by `tools/scene/verify_rubble.py` inside the running editor. This is the half that
   matters: the placed instance count PER VARIANT must equal the layout exactly, the meshes must still have
   Nanite off and the triangle count the generator recorded, every material a rubble or debris component uses
   must report NON-ZERO shader instructions and texture samples (a material that fails to compile renders as
   the grey WorldGridMaterial checker and reports zeroes and nothing else), no component may be bound to
   WorldGridMaterial or DefaultMaterial, the first instance of each component must read back in WORLD space
   within a metre of the record it came from, the 18 planned houses must carry the damaged mesh the plan names
   and every other house the pristine one, and the level must not have grown an actor per item - the previous
   attempt died at ~5,500 actors on the Windows commit limit.

C. THE SCRIPTS' CONTROL FLOW, exec'd against a strict mock `unreal` (unknown enum members RAISE, because
   `unreal.BlendMode.BM_OPAQUE` - a name UE 5.8 does not define - passed a permissive mock and then failed in
   the editor). Each configuration asserts the script either completes or raises for the right reason:
   Nanite refusing to turn off, a material reporting zero instructions, a normal map left as sRGB, a component
   that cannot be created, and instance transforms that read back at the origin.

`--self-test` corrupts a copy of the level report in six ways and asserts that B catches all six. Run it if you
ever doubt that this file is a test rather than a formality.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "scene"
REPORT = REPO / "_artifacts" / "rubble" / "placement_report.json"
UMAP = REPO / "sim" / "SightlineSim" / "Content" / "Sightline" / "Maps" / "FloodValley.umap"

#: this lane's hard ceiling. The vegetation lane's actor-per-item run died at ~5,500 actors with
#: "AvailableVirtual 0.00 GiB"; 1,799 items + 67 debris pieces must not cost more than 400 actors.
MAX_LANE_ACTORS = 400

FAILS: list[str] = []
CHECKS = [0]


def check(cond: bool, msg: str) -> bool:
    CHECKS[0] += 1
    if not cond:
        FAILS.append(msg)
    return bool(cond)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ==========================================================================================================
# A. the plan
# ==========================================================================================================
def check_plans() -> tuple[dict, dict]:
    rub = load(DATA / "rubble_layout.json")
    dmg = load(DATA / "damage.json")

    items = rub["items"]
    check(len(items) == rub["counts"]["items"],
          f"rubble_layout: {len(items)} items but counts says {rub['counts']['items']}")
    check(len(items) == 1799, f"rubble_layout: expected 1799 items (the lane's brief), got {len(items)}")

    per_variant: dict[str, int] = {}
    tri_total = 0
    for it in items:
        per_variant[it["variant"]] = per_variant.get(it["variant"], 0) + 1
        tri_total += it["tris"]
    check(set(per_variant) <= set(rub["variants"]),
          f"rubble_layout: items name variants that do not exist: {sorted(set(per_variant) - set(rub['variants']))[:5]}")
    check(len(rub["variants"]) == 40, f"rubble_layout: expected 40 variants, got {len(rub['variants'])}")

    missing = [v for v in rub["variants"] if not (DATA / "rubble" / f"{v}.obj").exists()]
    check(not missing, f"rubble meshes missing on disk: {missing[:5]}")

    for v, info in rub["variants"].items():
        check(info["tris"] > 0, f"variant {v} claims {info['tris']} triangles")
        check(bool(info["slots"]), f"variant {v} has no material slots")

    check(abs(tri_total - rub["counts"]["approx_triangles"]) <= max(64, 0.001 * tri_total),
          f"rubble_layout: item triangles sum to {tri_total}, counts says {rub['counts']['approx_triangles']}")
    check(tri_total <= rub["counts"]["triangle_budget"],
          f"rubble_layout: {tri_total} triangles exceeds the budget {rub['counts']['triangle_budget']}")

    for key, field in (("by_group", "group"), ("by_family", "family")):
        got: dict[str, int] = {}
        for it in items:
            got[it[field]] = got.get(it[field], 0) + 1
        check(got == rub["counts"][key], f"rubble_layout: {key} is {got}, counts says {rub['counts'][key]}")

    # damage
    check(len(dmg["houses"]) == 18, f"damage.json: expected 18 damaged houses, got {len(dmg['houses'])}")
    check(len(dmg["roof_debris"]) == 67,
          f"damage.json: expected 67 roof-debris pieces, got {len(dmg['roof_debris'])}")
    bad = [h["variant"] for h in dmg["houses"] if h["variant"] not in dmg["variants"]]
    check(not bad, f"damage.json: houses name unknown variants {bad[:5]}")
    miss = [v["obj"] for v in dmg["variants"].values() if not (DATA / v["obj"]).exists()]
    check(not miss, f"damaged-house meshes missing on disk: {miss[:5]}")
    bad = [r["name"] for r in dmg["roof_debris"] if not str(r["asset"]).startswith("/Game/")]
    check(not bad, f"damage.json: roof debris with a non-/Game asset path {bad[:3]}")
    ids = [h["id"] for h in dmg["houses"]]
    check(len(set(ids)) == len(ids), "damage.json: the same house is damaged twice")
    check_objs(rub)
    return rub, dmg


def check_objs(rub: dict) -> None:
    """Read every rubble OBJ and assert it is in CENTIMETRES and carries the slots it claims.

    UE's OBJ importer does not rescale: a file written in metres arrives 100x too small, and a field of
    5 cm slabs renders as a gravel bed that passes every count, every triangle check and every material
    assertion. `gen_rubble.py` says centimetres; this is where that claim is checked rather than believed.
    """
    for v, info in sorted(rub["variants"].items()):
        p = DATA / "rubble" / f"{v}.obj"
        if not p.exists():
            continue
        lo = [1e18, 1e18, 1e18]
        hi = [-1e18, -1e18, -1e18]
        groups, nv = set(), 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("v "):
                xyz = [float(x) for x in line[2:].split()[:3]]
                for i in range(3):
                    lo[i] = min(lo[i], xyz[i])
                    hi[i] = max(hi[i], xyz[i])
                nv += 1
            elif line.startswith("usemtl "):
                groups.add(line.split(None, 1)[1].strip())
        if not check(nv > 0, f"{v}.obj has no vertices"):
            continue
        got = [hi[i] - lo[i] for i in range(3)]
        want = [s * 100.0 for s in info["size_m"]]        # size_m is metres; the file must be centimetres
        worst = max(abs(g - w) for g, w in zip(got, want))
        check(worst <= max(2.0, 0.02 * max(want)),
              f"{v}.obj spans {[round(g, 1) for g in got]} file units but the layout says "
              f"{[round(w, 1) for w in want]} cm. A factor of ~100 here means the OBJ is in METRES and the "
              f"whole field will import as gravel.")
        check(groups == set(info["slots"]),
              f"{v}.obj has usemtl groups {sorted(groups)}, the layout declares {sorted(info['slots'])} - a "
              f"slot build_rubble.py cannot map raises, an extra one silently keeps the default material")


# ==========================================================================================================
# B. the level, as measured in the editor
# ==========================================================================================================
def check_level(rub: dict, dmg: dict, rep: dict) -> None:
    base_z = rub["base_z_m"]
    check(not rep.get("pie"), "the report was taken while PIE was running; saves and spawns are unreliable there")

    # --- rubble instances, per variant -------------------------------------------------------------------
    want: dict[str, int] = {}
    first: dict[str, dict] = {}
    for it in rub["items"]:
        want[it["variant"]] = want.get(it["variant"], 0) + 1
        first.setdefault(it["variant"], it)

    got: dict[str, int] = {}
    for label, c in rep["rubble"].items():
        if not check(c.get("mesh"), f"rubble component {label!r} has NO static mesh bound ({c.get('note')})"):
            continue
        v = c["mesh"][3:] if c["mesh"].startswith("SM_") else c["mesh"]
        check(v in want, f"rubble component {label!r} holds mesh {c['mesh']} which no layout item asks for")
        got[v] = got.get(v, 0) + c["instances"]
        check(str(c.get("class", "")).endswith("StaticMeshComponent") and "Instanced" in str(c.get("class")),
              f"{label}: component class is {c.get('class')!r}, not an instanced static mesh component")
        check("STATIC" in str(c.get("mobility", "")).upper(),
              f"{label}: mobility is {c.get('mobility')!r}, expected STATIC")

    check(got == want,
          "placed rubble instances per variant DO NOT match rubble_layout.json: "
          + str({k: (got.get(k, 0), want.get(k, 0)) for k in sorted(set(got) | set(want))
                 if got.get(k, 0) != want.get(k, 0)}))
    total = sum(got.values())
    check(total == len(rub["items"]),
          f"placed {total} rubble instances, the layout has {len(rub['items'])}")

    # --- the instances are where the layout says ---------------------------------------------------------
    for label, c in rep["rubble"].items():
        if not c.get("instances"):
            continue
        v = c["mesh"][3:] if c["mesh"].startswith("SM_") else c["mesh"]
        rec = first.get(v)
        if rec is None or "instance0_world_cm" not in c:
            continue
        # a component may hold either colour bucket of a belonging, so only compare when it holds them all
        if got.get(v) != want.get(v) or c["instances"] != want.get(v):
            continue
        exp = [rec["north_m"] * 100.0, rec["east_m"] * 100.0, (rec["base_asl_m"] - base_z) * 100.0]
        d = max(abs(a - b) for a, b in zip(c["instance0_world_cm"], exp))
        check(d <= 100.0,
              f"{label}: instance 0 is at {[round(x) for x in c['instance0_world_cm']]} cm but the layout "
              f"record {rec['name']} says {[round(x) for x in exp]} cm ({d / 100:.1f} m out)")
        bb = c.get("sample_bbox_cm")
        if bb and c["instances"] > 8:
            spread = max(bb[1] - bb[0], bb[3] - bb[2])
            check(spread > 500.0,
                  f"{label}: {c['instances']} instances span only {spread / 100:.1f} m - they are stacked, "
                  f"not spread over the fan")

    # --- meshes: Nanite off, triangle counts intact ------------------------------------------------------
    for v, info in rub["variants"].items():
        m = rep["meshes"].get(f"SM_{v}")
        if not check(m is not None, f"SM_{v} is not in the project - the rubble mesh was never imported"):
            continue
        check(m["nanite"] is False,
              f"SM_{v}: Nanite is {m['nanite']!r}; it must be OFF (with it on, get_num_triangles reports the "
              f"fallback mesh and every triangle claim in this project becomes a lie)")
        check(abs(m["triangles"] - info["tris"]) <= max(8, 0.02 * info["tris"]),
              f"SM_{v} has {m['triangles']} triangles, the generator recorded {info['tris']}")
    for v, info in dmg["variants"].items():
        m = rep["meshes"].get(f"SM_{v}")
        if not check(m is not None, f"SM_{v} (damaged house) was never imported"):
            continue
        check(m["nanite"] is False, f"SM_{v}: Nanite is {m['nanite']!r}, must be OFF")
        check(abs(m["triangles"] - info["triangles"]) <= max(4, 0.02 * info["triangles"]),
              f"SM_{v} has {m['triangles']} triangles, the generator recorded {info['triangles']}")

    # --- materials: a failed compile reports zeroes and renders as the grey checker -----------------------
    checked_mats = 0
    for p, s in rep["materials"].items():
        if not check(s.get("loaded"), f"material {p} does not load"):
            continue
        if "error" in s:
            check(False, f"material {p} statistics failed: {s['error']}")
            continue
        checked_mats += 1
        check(s.get("instructions", 0) > 0 and s.get("texture_samples", 0) > 0,
              f"material {p} FAILED TO COMPILE: {s.get('instructions')} instructions, "
              f"{s.get('texture_samples')} texture samples - it renders as the grey WorldGridMaterial checker")
    check(checked_mats >= 8,
          f"only {checked_mats} materials were probed; the 7 rubble instances + master + MI_Rubble are 9")

    for grp, name in ((rep["rubble"], "rubble"), (rep["damage_debris"], "debris")):
        for label, c in grp.items():
            for m in c.get("materials") or []:
                check(m not in ("WorldGridMaterial", "DefaultMaterial", "DefaultDeferredDecalMaterial"),
                      f"{name} component {label!r} is bound to {m} - that IS the grey checker")

    # --- roof debris --------------------------------------------------------------------------------------
    dwant: dict[str, int] = {}
    for r in dmg["roof_debris"]:
        nm = r["asset"].rsplit("/", 1)[-1]
        dwant[nm] = dwant.get(nm, 0) + 1
    dgot: dict[str, int] = {}
    for label, c in rep["damage_debris"].items():
        if not check(c.get("mesh"), f"debris component {label!r} has NO static mesh bound"):
            continue
        dgot[c["mesh"]] = dgot.get(c["mesh"], 0) + c["instances"]
    check(dgot == dwant,
          "roof-debris instances per mesh DO NOT match damage.json: "
          + str({k: (dgot.get(k, 0), dwant.get(k, 0)) for k in sorted(set(dgot) | set(dwant))
                 if dgot.get(k, 0) != dwant.get(k, 0)}))
    check(sum(dgot.values()) == len(dmg["roof_debris"]),
          f"placed {sum(dgot.values())} roof-debris instances, the plan has {len(dmg['roof_debris'])}")

    # --- damaged houses -----------------------------------------------------------------------------------
    houses = rep["houses"]
    check(len(houses) >= len(dmg["houses"]),
          f"only {len(houses)} House_### actors in the level, the plan damages {len(dmg['houses'])}")
    planned = {f"House_{h['id']:03d}": h for h in dmg["houses"]}
    for lbl, h in planned.items():
        a = houses.get(lbl)
        if not check(a is not None, f"{lbl} is planned as damaged but no such actor exists"):
            continue
        check(a["mesh"] == f"SM_{h['variant']}",
              f"{lbl} is bound to {a['mesh']}, the plan says SM_{h['variant']} - the mesh swap did not take")
        check("Damaged" in a["tags"], f"{lbl} carries no Damaged tag ({a['tags']})")
    stray = [lbl for lbl, a in houses.items() if lbl not in planned and "Damaged" in a["tags"]]
    check(not stray, f"{len(stray)} houses are tagged Damaged but are not in the plan: {stray[:5]}")
    dmg_meshes = {f"SM_{v}" for v in dmg["variants"]}
    stray = [lbl for lbl, a in houses.items() if lbl not in planned and a["mesh"] in dmg_meshes]
    check(not stray, f"{len(stray)} houses outside the plan still carry a damaged mesh: {stray[:5]}")

    # --- the constraint that killed the last attempt -------------------------------------------------------
    lane = rep["actors_by_folder"].get("Rubble", 0) + rep["actors_by_folder"].get("Damage", 0)
    check(lane <= MAX_LANE_ACTORS,
          f"this lane created {lane} actors; the hard ceiling is {MAX_LANE_ACTORS} (the previous attempt died "
          f"at ~5,500 actors with AvailableVirtual 0.00 GiB)")
    check(lane >= len(rep["rubble"]) - 4,
          f"the Rubble/Damage folders hold {lane} actors but {len(rep['rubble'])} rubble components were "
          f"measured - components are hiding on actors outside those folders")
    check(rep["actor_count"] < 5000,
          f"the level holds {rep['actor_count']} actors; the editor died at ~5,500 last time")


# Mean LINEAR albedo of each source scan, measured from the Poly Haven files in _downloads (see the command in
# the report): sum over the whole 2K diffuse map, sRGB-decoded. Applying one material's albedo to another is
# how a check lies, so a material whose scan is not measured here is skipped rather than guessed at.
SOURCE_LINEAR = {
    "T_dirty_concrete_D": (0.257, 0.236, 0.196),        # sRGB std 0.0880
    "T_brown_mud_rocks_01_D": (0.105, 0.081, 0.050),    # sRGB std 0.1357
}
CONCRETE_GRAIN_SRGB = 0.088
#: Reference B: "dust-covered, desaturated tan/grey, very low colour contrast".
#: The upper bound is the blowout guard - linear 0.42 is already sRGB 0.68, the top of "dusty concrete"; above
#: it the slabs read as white paper, which is exactly what the first run produced. The lower bound is a sanity
#: floor: below it the piece reads as wet mud rather than dusted rubble. Broken masonry is legitimately darker
#: than dusty concrete, so the band is wide and it is CHROMA that carries "desaturated", not brightness.
ALBEDO_BAND = (0.10, 0.42)
#: chroma relative to brightness, so the same standard applies to a dark brick and a light slab
MAX_RELATIVE_CHROMA = 0.35
#: how much of the scan's own grain must survive the dust wash. Every point of dust is a point of grain lost,
#: and a slab with no grain renders as paper - which is exactly what the first run produced.
MIN_GRAIN_FRACTION = 0.55


def check_palette(rep: dict) -> None:
    """Judge the rubble palette against Reference B, from the parameters the editor reports.

    This exists because the first run of this lane passed EVERY other check in this file - 856 of them - and
    still rendered the slabs as featureless near-white. The material compiled, the textures were bound, the
    counts were exact; the bug was that `unreal.LinearColor(0.74, 0.70, 0.62)` is consumed LINEARLY while it
    was authored as an sRGB tan. Only a picture showed it. This turns that picture into arithmetic.
    """
    probed = 0
    for p, s in sorted(rep["materials"].items()):
        if "MI_Rubble_" not in p or "DustAmount" not in s or "DustColor" not in s:
            continue
        # belongings are meant to be the one colour in a grey field - Reference B: "scattered colour"
        if "Fabric" in p or "Rebar" in p:
            continue
        src = SOURCE_LINEAR.get(str(s.get("BaseColorTexture")))
        if src is None:
            continue                                  # unmeasured scan: skip rather than guess
        probed += 1
        name = p.rsplit("/", 1)[-1]
        dust = float(s["DustAmount"])
        dc = s["DustColor"]
        tint = s.get("Tint", [1.0, 1.0, 1.0])
        base = [src[i] * tint[i] + dust * (dc[i] - src[i] * tint[i]) for i in range(3)]
        mean = sum(base) / 3.0
        rel_chroma = (max(base) - min(base)) / mean if mean else 9.9
        print(f"   {name:22s} scan {s['BaseColorTexture']:24s} -> linear albedo "
              f"{[round(b, 3) for b in base]} mean {mean:.3f} rel-chroma {rel_chroma:.3f} dust {dust:.2f}")
        check(ALBEDO_BAND[0] <= mean <= ALBEDO_BAND[1],
              f"{name}: resulting linear albedo {mean:.3f} is outside the Reference B band "
              f"{ALBEDO_BAND} (DustAmount {dust:.2f}, DustColor {[round(c, 3) for c in dc]}). Remember "
              f"LinearColor is LINEAR: linear 0.74 is sRGB 0.885, i.e. near white.")
        check(rel_chroma <= MAX_RELATIVE_CHROMA,
              f"{name}: chroma is {rel_chroma:.3f} of its brightness (max {MAX_RELATIVE_CHROMA}) - "
              f"Reference B is 'desaturated tan/grey, very low colour contrast'")
        check(1.0 - dust >= MIN_GRAIN_FRACTION,
              f"{name}: DustAmount {dust:.2f} leaves only {(1 - dust) * 100:.0f}% of the "
              f"scan's grain (floor {MIN_GRAIN_FRACTION * 100:.0f}%); the slabs render as blank paper")
    check(probed >= 2, f"only {probed} grey rubble materials carried a MEASURED source scan; expected concrete "
                       f"and masonry at least (is verify_rubble.py stale?)")


def check_rendered_slab(shot: Path) -> None:
    """Measure the actual rendered pixels of a slab filling the frame. The parameter check above is arithmetic;
    this is the picture. It fails on a blown-out or a saturated slab, which is what the first run produced."""
    try:
        from PIL import Image
    except ImportError:
        check(False, "Pillow is not available, so the rendered slab cannot be measured")
        return
    if not check(shot.exists(), f"{shot} is missing - run tools/scene/qa_rubble.py in the editor"):
        return
    im = Image.open(shot).convert("RGB")
    w, h = im.size
    crop = im.crop((int(w * 0.30), int(h * 0.55), int(w * 0.70), int(h * 0.95)))
    px = list(crop.getdata())
    n = len(px)
    mean = [sum(p[i] for p in px) / n / 255.0 for i in range(3)]
    lum = [(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / 255.0 for p in px]
    mlum = sum(lum) / n
    std = (sum((x - mlum) ** 2 for x in lum) / n) ** 0.5
    sat = max(mean) - min(mean)
    print(f"   rendered slab crop: mean sRGB {[round(c, 3) for c in mean]}  luminance {mlum:.3f}  "
          f"saturation {sat:.3f}  grain(std) {std:.4f}")
    check(0.25 <= mlum <= 0.80,
          f"the rendered slab's mean luminance is {mlum:.3f}; outside 0.25-0.80 it is either blown out to "
          f"white or in full shadow")
    check(sat <= 0.10,
          f"the rendered slab's saturation is {sat:.3f} (max 0.10) - Reference B is very low colour contrast")
    # The scan's own sRGB standard deviation is 0.088. A dust wash of at most 0.45 must leave >= 0.048 of that
    # albedo variation; uniform lighting and 8-bit quantisation eat some, so 0.030 is a conservative floor. The
    # first run measured 0.0201 here - visibly flat paper - so this threshold is set where it FAILS that, not
    # where it passes it.
    check(std >= 0.030,
          f"the rendered slab has almost no local variation (std {std:.4f}, floor 0.030, source scan "
          f"{CONCRETE_GRAIN_SRGB}) - it is rendering as flat paper, not as a concrete scan")


def check_umap_persistence() -> None:
    """The instances are real only if they survive the save. The level is a single .umap (no external actors),
    so its name table must mention the component class; if the components were transient, it will not."""
    if not check(UMAP.exists(), f"{UMAP} does not exist"):
        return
    b = UMAP.read_bytes()
    needle = b"HierarchicalInstancedStaticMeshComponent"
    check(needle in b or needle.decode().encode("utf-16-le") in b,
          f"{UMAP.name} ({len(b):,} bytes) contains no HierarchicalInstancedStaticMeshComponent name - the "
          f"instanced components did not serialise into the saved level")


# ==========================================================================================================
# C. dry-run the editor scripts against a STRICT mock `unreal`
# ==========================================================================================================
class MockMeta(type):
    def __instancecheck__(cls, obj):
        return getattr(obj, "_is", None) == cls.__name__ or getattr(cls, "_answer", True)


class Obj(metaclass=MockMeta):
    def __init__(self, name="Obj", **props):
        self._name = name
        self._is = props.pop("_is", None)      # a real attribute: MockMeta.__instancecheck__ reads it
        self._p = dict(props)

    def get_editor_property(self, k):
        if k in self._p:
            return self._p[k]
        return Obj(k)

    def set_editor_property(self, k, v):
        self._p[k] = v

    def get_name(self):
        return self._name

    def get_path_name(self):
        return self._p.get("_path", f"/Game/Mock/{self._name}")

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        if k in self._p:
            return self._p[k]
        return lambda *a, **kw: Obj(k)


class Enum(Obj):
    """Unknown members RAISE. `unreal.BlendMode.BM_OPAQUE` does not exist in UE 5.8 and a permissive mock
    answered it with a lambda; the editor answered with AttributeError."""

    def __init__(self, name, *members):
        super().__init__(name, **{m: f"{name}.{m}" for m in members})

    def __getattr__(self, k):
        if k.startswith("_") or k in self._p:
            return super().__getattr__(k)
        raise AttributeError(f"unreal.{self._name} has no member {k!r}; UE 5.8 defines {sorted(self._p)}")


def make_unreal(cfg):                                                    # noqa: C901 - it is a fake engine
    u = types.ModuleType("unreal")
    rub = load(DATA / "rubble_layout.json")
    dmg = load(DATA / "damage.json")
    town = load(DATA / "settlement.json")
    world: list = []
    cache: dict[str, object] = {}

    class Vector:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

    class Rotator:
        def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
            self.roll, self.pitch, self.yaw = float(roll), float(pitch), float(yaw)

        def quaternion(self):
            return Quat(self)

    class LinearColor:
        def __init__(self, r=0.0, g=0.0, b=0.0, a=1.0):
            self.r, self.g, self.b, self.a = r, g, b, a

    class Quat:
        """UE 5.8's FTransform.Rotation is a QUAT, not a Rotator. The mock models that faithfully, because a
        mock that accepted a Rotator is exactly how the first version of build_rubble.py passed this checker
        and then produced a field of perfectly flat slabs in the editor."""

        def __init__(self, rot):
            self._rot = rot

        def rotator(self):
            return self._rot

    class Transform:
        def __init__(self, rotation=None, translation=None, scale=None):
            self._p = {"rotation": rotation or Quat(Rotator()), "translation": translation or Vector(),
                       "scale3d": scale or Vector(1, 1, 1)}

        def set_editor_property(self, k, v):
            if k not in self._p:
                raise AttributeError(f"unreal.Transform has no property {k!r}; UE 5.8 defines "
                                     f"{sorted(self._p)}")
            if k == "rotation" and not isinstance(v, Quat):
                raise TypeError("Transform.rotation is a Quat in UE 5.8, not a "
                                f"{type(v).__name__} - convert with Rotator.quaternion()")
            self._p[k] = v

        def get_editor_property(self, k):
            return self._p[k]

    class Stats:
        num_pixel_shader_instructions = cfg.get("instructions", 240)
        num_pixel_texture_samples = cfg.get("samples", 6)

    def slot(nm):
        return Obj("slot", material_slot_name=nm, material_interface=Obj(f"MI_{nm}"))

    def make_mesh(name, slots, tris, origin=(0.0, 0.0, 0.0)):
        ns = Obj("ns", enabled=True)
        if cfg.get("nanite_sticks"):
            ns.set_editor_property = lambda k, v: None          # the write is accepted and does nothing
        sm = Obj(name, nanite_settings=ns, static_materials=[slot(s) for s in slots], body_setup=Obj("bs"),
                 _is="StaticMesh", _path=f"/Game/Sightline/Mock/{name}")
        sm.get_num_triangles = lambda lod=0: (2 if ns.get_editor_property("enabled") else tris)
        sm.get_num_lods = lambda: 1
        sm.set_material = lambda i, m: None
        sm.get_bounds = lambda: Obj("b", origin=Vector(*origin))
        return sm

    def load_asset(path):
        p = str(path)
        if p in cache:
            return cache[p]
        nm = p.rsplit("/", 1)[-1]
        if cfg.get("missing_asset") and nm == cfg["missing_asset"]:
            return None
        if nm.startswith("T_"):
            kind = "arm" if nm.endswith("_ARM") else ("nrm" if nm.endswith(("_N", "_NRM")) else "col")
            srgb = kind == "col"
            comp = {"col": "TC_DEFAULT", "nrm": "TC_NORMALMAP", "arm": "TC_MASKS"}[kind]
            if cfg.get("srgb_normal") and kind == "nrm":
                srgb, comp = True, "TC_DEFAULT"
            o = Obj(nm, srgb=srgb, compression_settings=f"TextureCompressionSettings.{comp}")
        elif nm.startswith("SM_") and nm[3:] in rub["variants"]:
            v = rub["variants"][nm[3:]]
            o = make_mesh(nm, v["slots"], v["tris"])
        elif nm.startswith("SM_") and nm[3:] in dmg["variants"]:
            v = dmg["variants"][nm[3:]]
            cx, cy, cz = v["bbox_centre_obj_cm"]
            o = make_mesh(nm, v["slots"], v["triangles"], origin=(cx, -cy, cz))
        elif nm.startswith("SM_"):
            o = make_mesh(nm, ["wall", "roof_tile", "concrete", "wood", "window"], 900)
        elif "/Props/" in p:
            o = make_mesh(nm, ["prop"], 4000)
        else:
            o = Obj(nm, _path=p)
        cache[p] = o
        cache[o.get_path_name()] = o
        return o

    class EAL:
        @staticmethod
        def does_asset_exist(p):
            nm = str(p).rsplit("/", 1)[-1]
            if nm.endswith("_NRM"):                 # this project writes terrain normals as _N, buildings _NRM
                return False
            return cfg.get("assets_exist", True)

        @staticmethod
        def save_asset(p):
            return True

        @staticmethod
        def list_assets(p, recursive=True, include_folder=False):
            return []

    exprs: list = []

    class MEL:
        get_material_expressions = staticmethod(lambda m: list(exprs))
        delete_unused_expressions = staticmethod(lambda m: True)
        connect_material_expressions = staticmethod(lambda a, ao, b, bi: True)
        connect_material_property = staticmethod(lambda a, ao, p: True)
        recompile_material = staticmethod(lambda m: True)
        get_statistics = staticmethod(lambda m: Stats())
        set_material_instance_parent = staticmethod(lambda i, p: True)
        set_material_instance_texture_parameter_value = staticmethod(lambda i, n, t: True)
        set_material_instance_vector_parameter_value = staticmethod(lambda i, n, v: True)
        set_material_instance_scalar_parameter_value = staticmethod(lambda i, n, v: True)
        update_material_instance = staticmethod(lambda i: True)

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
            exprs.append(e)
            return e

    class ISM(Obj):
        def __init__(self):
            super().__init__("HISM")
            self._x: list = []
            self._mesh = None

        def set_static_mesh(self, sm):
            self._mesh = sm
            self._p["static_mesh"] = sm          # so get_editor_property("static_mesh") reads it back
            return True

        def set_mobility(self, m):
            return None

        def set_collision_enabled(self, m):
            return None

        def set_material(self, i, m):
            return None

        def set_collision_profile_name(self, n):
            return None

        def add_instances(self, xf, ret=False, world_space=False):
            self._x.extend(xf)
            return list(range(len(self._x)))

        def add_instance(self, t, world_space=False):
            self._x.append(t)
            return len(self._x) - 1

        def clear_instances(self):
            self._x = []

        def get_instance_count(self):
            return len(self._x) - (1 if cfg.get("short_count") else 0)

        def get_instance_transform(self, i, world_space=False):
            return Transform() if cfg.get("instances_at_origin") else self._x[i]

    def make_actor():
        a = Obj("actor", _folder="", tags=[])
        a.set_actor_label = lambda s: a.set_editor_property("_label", s)
        a.get_actor_label = lambda: a.get_editor_property("_label")
        a.set_folder_path = lambda f: a.set_editor_property("_folder", f)
        a.get_folder_path = lambda: a.get_editor_property("_folder")
        a.set_actor_scale3d = lambda v: None
        a.set_actor_location = lambda v, s=False, t=False: None
        a.get_actor_location = lambda: Vector(0, 0, 0)
        a.get_actor_rotation = lambda: Rotator(0, 0, 0)
        a.get_actor_scale3d = lambda: Vector(1, 1, 1)
        a.static_mesh_component = Obj("smc")
        a.static_mesh_component.set_static_mesh = lambda m: None
        a.static_mesh_component.set_mobility = lambda m: None
        a.static_mesh_component.set_material = lambda i, m: None
        a.static_mesh_component.set_collision_enabled = lambda m: None
        a._p["_comps"] = []
        a.get_component_by_class = lambda cls: (a._p["_comps"][0] if a._p["_comps"] else None)
        a.get_components_by_class = lambda cls: list(a._p["_comps"])
        world.append(a)
        return a

    class SDS:
        """UE 5.8 has no Actor.add_component_by_class; components are added through this subsystem, the same
        path the Details panel's '+ Add Component' takes."""

        @staticmethod
        def k2_gather_subobject_data_for_instance(act):
            return [Obj("root_handle", _actor=act)]

        @staticmethod
        def add_new_subobject(params):
            parent = params.get_editor_property("parent_handle")
            act = parent.get_editor_property("_actor")
            if cfg.get("no_component"):
                return Obj("handle"), "the component could not be created"
            c = ISM()
            act._p["_comps"].append(c)
            return Obj("handle", _comp=c), ""

        @staticmethod
        def rename_subobject(handle, name):
            return True

    class EAS:
        get_all_level_actors = staticmethod(lambda: list(world))

        @staticmethod
        def destroy_actor(a):
            if a in world:
                world.remove(a)

        @staticmethod
        def spawn_actor_from_class(cls, loc, rot=None):
            return make_actor()

        @staticmethod
        def spawn_actor_from_object(mesh, loc, rot=None):
            return make_actor()

    class LES:
        is_in_play_in_editor = staticmethod(lambda: False)
        save_current_level = staticmethod(lambda: True)

    class Tools:
        @staticmethod
        def create_asset(n, d, c, f):
            return Obj(n, _path=f"{d}/{n}")

        @staticmethod
        def import_asset_tasks(tasks):
            for t in tasks:
                dest = t.get_editor_property("destination_path")
                name = t.get_editor_property("destination_name")
                sm = load_asset(f"{dest}/{name}")
                t.set_editor_property("imported_object_paths", [f"{dest}/{name}"])
                cache[f"{dest}/{name}"] = sm
            return True

    def klass(name):
        return MockMeta(name, (Obj,), {"_answer": False})

    def get_editor_subsystem(cls):
        nm = getattr(cls, "__name__", "")
        return {"LevelEditorSubsystem": LES, "EditorActorSubsystem": EAS}.get(nm, Obj(nm))

    # pre-seed the pristine archetypes so build_damage.py finds them
    for arch in town["archetypes"]:
        load_asset(f"/Game/Sightline/Buildings/SM_{arch}")
    # ... and the House_### actors it is supposed to modify
    for h in town["houses"]:
        a = make_actor()
        a.set_actor_label(f"House_{h['id']:03d}")

    u.EditorAssetLibrary = EAL
    u.MaterialEditingLibrary = MEL
    u.AssetToolsHelpers = Obj("ath", get_asset_tools=lambda: Tools)
    u.get_editor_subsystem = get_editor_subsystem
    u.get_engine_subsystem = lambda cls: SDS
    u.load_asset = load_asset
    u.Vector, u.Rotator, u.LinearColor, u.Transform, u.Quat = Vector, Rotator, LinearColor, Transform, Quat
    u.AssetImportTask = lambda: Obj("task", imported_object_paths=[])
    u.SystemLibrary = Obj("sys", collect_garbage=lambda: None)
    u.MathLibrary = Obj("math", conv_rotator_to_quaternion=Quat,
                        conv_quaternion_to_rotator=lambda q: q.rotator())
    u.AddNewSubobjectParams = lambda **kw: Obj("params", **kw)
    # Enums are spelled EXACTLY as UE 5.8 spells them, and an unknown member raises.
    u.ComponentMobility = Enum("ComponentMobility", "STATIC", "STATIONARY", "MOVABLE")
    u.CollisionEnabled = Enum("CollisionEnabled", "NO_COLLISION", "QUERY_ONLY", "PHYSICS_ONLY",
                              "QUERY_AND_PHYSICS", "PROBE_ONLY", "QUERY_AND_PROBE")
    u.CollisionTraceFlag = Enum("CollisionTraceFlag", "CTF_USE_DEFAULT", "CTF_USE_SIMPLE_AND_COMPLEX",
                                "CTF_USE_SIMPLE_AS_COMPLEX", "CTF_USE_COMPLEX_AS_SIMPLE")
    u.TextureCompressionSettings = Enum(
        "TextureCompressionSettings", "TC_DEFAULT", "TC_NORMALMAP", "TC_MASKS", "TC_GRAYSCALE",
        "TC_DISPLACEMENTMAP", "TC_VECTOR_DISPLACEMENTMAP", "TC_HDR", "TC_EDITOR_ICON", "TC_ALPHA",
        "TC_DISTANCE_FIELD_FONT", "TC_HDR_COMPRESSED", "TC_BC7", "TC_HALF_FLOAT", "TC_ENCODED_REFLECTION_CAPTURE",
        "TC_SINGLE_FLOAT", "TC_HDR_F32")
    u.MaterialSamplerType = Enum(
        "MaterialSamplerType", "SAMPLERTYPE_COLOR", "SAMPLERTYPE_GRAYSCALE", "SAMPLERTYPE_ALPHA",
        "SAMPLERTYPE_NORMAL", "SAMPLERTYPE_MASKS", "SAMPLERTYPE_DISTANCE_FIELD_FONT", "SAMPLERTYPE_LINEAR_COLOR",
        "SAMPLERTYPE_LINEAR_GRAYSCALE", "SAMPLERTYPE_DATA", "SAMPLERTYPE_EXTERNAL", "SAMPLERTYPE_VIRTUAL_COLOR",
        "SAMPLERTYPE_VIRTUAL_GRAYSCALE", "SAMPLERTYPE_VIRTUAL_ALPHA", "SAMPLERTYPE_VIRTUAL_NORMAL",
        "SAMPLERTYPE_VIRTUAL_MASKS", "SAMPLERTYPE_VIRTUAL_LINEAR_COLOR", "SAMPLERTYPE_VIRTUAL_LINEAR_GRAYSCALE")
    u.SamplerSourceMode = Enum("SamplerSourceMode", "SSM_FROM_TEXTURE_ASSET", "SSM_WRAP_WORLD_GROUP_SETTINGS",
                               "SSM_CLAMP_WORLD_GROUP_SETTINGS", "SSM_TERRAIN_WEIGHTMAP_GROUP_SETTINGS")
    u.MaterialProperty = Enum(
        "MaterialProperty", "MP_EMISSIVE_COLOR", "MP_OPACITY", "MP_OPACITY_MASK", "MP_DIFFUSE_COLOR",
        "MP_SPECULAR_COLOR", "MP_BASE_COLOR", "MP_METALLIC", "MP_SPECULAR", "MP_ROUGHNESS", "MP_ANISOTROPY",
        "MP_NORMAL", "MP_TANGENT", "MP_WORLD_POSITION_OFFSET", "MP_SUBSURFACE_COLOR", "MP_AMBIENT_OCCLUSION",
        "MP_REFRACTION", "MP_PIXEL_DEPTH_OFFSET", "MP_SHADING_MODEL", "MP_DISPLACEMENT")
    u.BlendMode = Enum("BlendMode", "BLEND_OPAQUE", "BLEND_MASKED", "BLEND_TRANSLUCENT", "BLEND_ADDITIVE",
                       "BLEND_MODULATE", "BLEND_ALPHA_COMPOSITE", "BLEND_ALPHA_HOLDOUT",
                       "BLEND_TRANSLUCENT_COLORED_TRANSMITTANCE")
    for nm in ("LevelEditorSubsystem", "EditorActorSubsystem", "UnrealEditorSubsystem", "Material",
               "MaterialInstanceConstant", "StaticMesh", "Actor", "HierarchicalInstancedStaticMeshComponent",
               "InstancedStaticMeshComponent", "StaticMeshActor", "SubobjectDataSubsystem"):
        setattr(u, nm, klass(nm))
    u.StaticMesh = MockMeta("StaticMesh", (Obj,), {"_answer": False})

    def getattr_(name):
        if name.startswith("MaterialExpression") or name.endswith("FactoryNew"):
            return klass(name)
        raise AttributeError(f"the mock does not model unreal.{name}; add it rather than guessing")

    u.__getattr__ = getattr_
    return u


def dry_run(script: str, cfg: dict):
    u = make_unreal(cfg)
    saved = sys.modules.get("unreal")
    sys.modules["unreal"] = u
    try:
        src = (HERE / script).read_text(encoding="utf-8")
        exec(compile(src, script, "exec"), {"__name__": "__dry_run__", "__file__": str(HERE / script)})  # noqa: S102
        return None
    except BaseException as exc:                                         # noqa: BLE001
        return exc
    finally:
        if saved is None:
            sys.modules.pop("unreal", None)
        else:
            sys.modules["unreal"] = saved


def check_editor_scripts() -> None:
    cases = [
        ("build_rubble.py", {}, None),
        ("build_rubble.py", {"nanite_sticks": True}, "triangle counts do not match"),
        ("build_rubble.py", {"instructions": 0}, "FAILED TO COMPILE"),
        ("build_rubble.py", {"srgb_normal": True}, "wrong sampler settings"),
        ("build_rubble.py", {"no_component": True}, "add_new_subobject refused"),
        ("build_rubble.py", {"short_count": True}, "component reports"),
        ("build_rubble.py", {"instances_at_origin": True}, "misplaced"),
        ("build_damage.py", {}, None),
        ("build_damage.py", {"instructions": 0}, "FAILED TO COMPILE"),
        ("build_damage.py", {"nanite_sticks": True}, "Nanite fallback"),
        ("build_damage.py", {"no_component": True}, "add_new_subobject refused"),
        ("build_damage.py", {"instances_at_origin": True}, "instance 0 reads back"),
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
                  f"dry run {label}: expected a failure containing {expect!r}, got {type(exc).__name__}: {exc}")


# ==========================================================================================================
def self_test() -> int:
    """Corrupt the level report six ways and assert every corruption is caught."""
    if not REPORT.exists():
        print(f"self-test needs a real report first: {REPORT} is missing")
        return 2
    rub, dmg = load(DATA / "rubble_layout.json"), load(DATA / "damage.json")
    good = load(REPORT)

    def run(mut, name):
        rep = copy.deepcopy(good)
        mut(rep)
        FAILS.clear()
        check_level(rub, dmg, rep)
        ok = bool(FAILS)
        print(f"  {'caught' if ok else 'MISSED'}: {name}" + (f"  -> {FAILS[0][:90]}" if ok else ""))
        return ok

    def drop_instances(rep):
        k = next(iter(rep["rubble"]))
        rep["rubble"][k]["instances"] -= 1

    def zero_material(rep):
        k = next(iter(rep["materials"]))
        rep["materials"][k] = {"loaded": True, "instructions": 0, "texture_samples": 0}

    def nanite_on(rep):
        k = next(k for k in rep["meshes"] if rep["meshes"][k]["group"] == "rubble")
        rep["meshes"][k]["nanite"] = True

    def checker_material(rep):
        k = next(iter(rep["rubble"]))
        rep["rubble"][k]["materials"] = ["WorldGridMaterial"]

    def undamage_house(rep):
        lbl = f"House_{dmg['houses'][0]['id']:03d}"
        if lbl in rep["houses"]:
            rep["houses"][lbl]["mesh"] = "SM_A_tile_1s"
            rep["houses"][lbl]["tags"] = []

    def actor_per_item(rep):
        rep["actors_by_folder"]["Rubble"] = 1799
        rep["actor_count"] = 6000

    def drop_debris(rep):
        k = next(iter(rep["damage_debris"]))
        rep["damage_debris"][k]["instances"] = 0

    results = [run(drop_instances, "one rubble instance missing"),
               run(zero_material, "a material that failed to compile"),
               run(nanite_on, "Nanite left on a rubble mesh"),
               run(checker_material, "a component bound to WorldGridMaterial"),
               run(undamage_house, "a planned house left pristine"),
               run(actor_per_item, "an actor per item (the crash)"),
               run(drop_debris, "roof debris missing")]
    FAILS.clear()
    FAILS.extend([] if all(results) else ["self-test: a corruption was NOT caught"])
    print(f"\nself-test: {sum(results)}/{len(results)} corruptions caught")
    return 0 if all(results) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-report", action="store_true",
                    help="skip section B (use before the editor run has happened)")
    ap.add_argument("--no-dry-run", action="store_true", help="skip section C")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    rub, dmg = check_plans()
    print(f"A. plan: {len(rub['items'])} rubble items / {len(rub['variants'])} variants, "
          f"{len(dmg['houses'])} damaged houses, {len(dmg['roof_debris'])} roof-debris pieces")

    if a.no_report:
        print("B. level: SKIPPED (--no-report)")
    elif not REPORT.exists():
        check(False, f"no level report at {REPORT} - run verify_rubble.py inside the editor first")
    else:
        rep = load(REPORT)
        check_level(rub, dmg, rep)
        check_palette(rep)
        check_rendered_slab(REPO / "_artifacts" / "editor_shots" / "qa_rubble_6_scale.png")
        check_umap_persistence()
        ri = sum(c["instances"] for c in rep["rubble"].values())
        di = sum(c["instances"] for c in rep["damage_debris"].values())
        dh = sum(1 for h in rep["houses"].values() if "Damaged" in h["tags"])
        print(f"B. level {rep['level']}: {rep['actor_count']} actors | {ri} rubble instances in "
              f"{len(rep['rubble'])} components | {di} debris instances in {len(rep['damage_debris'])} | "
              f"{dh} damaged houses | {len(rep['materials'])} materials probed")

    if a.no_dry_run:
        print("C. dry runs: SKIPPED (--no-dry-run)")
    else:
        check_editor_scripts()
        print("C. dry runs: build_rubble.py and build_damage.py exec'd against a strict mock unreal")

    print(f"\n{CHECKS[0]} checks run")
    if FAILS:
        print(f"\nFAIL  {len(FAILS)} problem(s):")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("OK    rubble and structural damage are placed exactly as the plans say")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
