"""Terrain context under a box — **measured against placed geometry, never assumed from a zone label**.

SOLUTION_DOC 5.12 requires "FP/min per terrain type (water, debris, vegetation, roof)", at detection level
and again at record level after dedup. That number was structurally impossible to produce before this module:

* :data:`sightline.eval.slicing.AXIS_VALUES` carried no ``context`` axis, so no ``MetricRow`` could hold one;
* :func:`sightline.eval.detection.false_positives` reported ``frame_idx / bbox / score / size_px`` and nothing
  about what the detector had actually fired on;
* a false positive has **no ground-truth box**, so it cannot inherit ``GtBox.context`` at all; and
* where ``context`` *was* set, ``detect.dataset._context_for(zone, submersion)`` derived it from the
  survivor's zone label — an intent, not an observation. This project has already been bitten by exactly that
  reading: ``actors.json`` occlusion was an intent that five generators contradicted in both directions, which
  is why ``tools/capture/measure_occlusion.py`` exists. Terrain type had the same defect and now has the same
  remedy.

**What is measured here.** A pixel is projected to a world position and the *topmost surface actually placed
there* is looked up in the scene layouts (``settlement.json``, ``vegetation.json``, ``rubble_layout.json``,
``props_layout.json``, the terrain heightfield and the flood level). It works identically for a ground-truth
box and for a prediction, which is the whole requirement: FP/min per terrain type is a statement about where
false positives land, and a false positive has no ground truth to inherit from.

**Why the projection carries a surface height.** Image-right is due EAST and image-up due NORTH, world-fixed:
measured on 110 boxes at 35 m (median residual 1.37 m; next-best hypothesis 15.8 m) by the coverage-geometry
fix of 2026-09-11, and reproduced here independently on 150 boxes at 55 m (2.75 m; next-best 27.2 m). But
projecting every pixel onto the *terrain* is wrong for anything standing above it. Measured over 260 boxes
across both passes:

====================================  ========  ========  ========
scale taken from                       median      p90       max
====================================  ========  ========  ========
the frame's GSD (terrain-referenced)    2.03 m    9.31 m   77.59 m
the target's own elevation              0.77 m    3.13 m    5.01 m
====================================  ========  ========  ========

The 77 m outliers were all roof survivors imaged far off-axis — pure parallax. So :func:`classify` solves for
the surface height instead of assuming zero: project, look up what is there, re-project at that surface's top,
repeat until it settles. :data:`PROJECTION_P90_M` is the measured p90 above and is used as the radius of the
ambiguity disc, so a box that lands within one projection error of a shoreline or a roof edge is reported
``ambiguous`` rather than being silently assigned to whichever class won by a metre.

**A systematic bias this module inherits and cannot remove.** The camera pose it projects from is the
telemetry pose, and the orchestrator session established on 2026-09-11 that the recorded telemetry pose is
**not the shutter pose**: `survey.py` samples `simGetGroundTruthKinematics()` at the top of its loop and only
then calls `grab()`, which spends a few hundred milliseconds on two 4K buffers — up to ~0.3 m of travel at
11 m/s. That is inside the residual budget measured above (median 1.32 m, p90 3.31 m) rather than dominating
it, but it is a bias, not noise: it points along the direction of flight every time. It affects every
geolocation computed from telemetry, not just this module. Fixing it means recording the pose at the shutter.

Nothing here deletes or clears anything (guardrail R10); an ambiguous classification is reported as ambiguous
and keeps its alternatives.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: The ``GtBox.context`` vocabulary. Kept identical to :class:`sightline.eval.groundtruth.GtBox` on purpose:
#: a context measured here must be assignable to a GtBox without translation.
CONTEXTS: tuple[str, ...] = ("open_ground", "water", "debris", "vegetation", "structure", "vehicle")

#: p90 of the elevation-corrected projection residual, measured over 260 boxes from ``seed23_alt35`` and
#: ``seed23_alt55`` against ``data/scene/actors.json``. Used as the ambiguity radius — not as an error bar on
#: any reported number.
PROJECTION_P90_M: float = 3.13

#: Bucket size for the spatial index. Must exceed the largest object radius in the scene: the jacaranda
#: catalogue entry is 24.61 m across the major axis and instances scale up to ~1.6, so ~20 m of radius.
_BUCKET_M: float = 32.0

#: A surface has to stand this far above another before it counts as covering it. Below this the two are
#: treated as coplanar and the more specific class wins by :data:`CONTEXT_PRIORITY`.
_COVER_EPS_M: float = 0.25

#: Breaks ties between surfaces at the SAME height — the more structured thing is what a detector fires on.
#: Never used to override a genuine height difference.
CONTEXT_PRIORITY: dict[str, int] = {
    "structure": 5, "vehicle": 4, "debris": 3, "vegetation": 2, "water": 1, "open_ground": 0,
}

#: Surfaces a target can be RESTING ON, and therefore the only ones allowed to set the elevation the fix is
#: solved at. Vegetation is deliberately absent: nothing in this scene stands on top of a tree crown, so a
#: crown on the slant ray is an OCCLUSION, which this project measures from the rendered masks
#: (``tools/capture/measure_occlusion.py``) rather than inferring from layout geometry. Letting crowns drive
#: the position solve was measurably wrong — it dragged 26 of 140 roof survivors onto neighbouring canopy and
#: moved their fixes up to 14 m, while ``tools/scene/check_vegetation.py`` independently asserts that no
#: unoccluded survivor sits under a crown at all. Crowns still count for the reported CONTEXT once the
#: position is known, so a target genuinely under canopy still reads ``vegetation``.
_RESTABLE: frozenset[str] = frozenset({"structure", "vehicle", "debris", "water", "open_ground"})


class ContextError(RuntimeError):
    """Raised when the scene geometry cannot be loaded or fails its own self-check."""


@dataclass(frozen=True, slots=True)
class Surface:
    """The topmost thing placed at one ground position."""

    context: str
    top_asl_m: float
    source: str  # the layout record's own name, so any answer is traceable back to a placed object

    def __post_init__(self) -> None:
        if self.context not in CONTEXTS:
            raise ContextError(f"context {self.context!r} not in {CONTEXTS}")


@dataclass(frozen=True, slots=True)
class ContextFix:
    """Where a box landed and what was under it."""

    context: str
    east_m: float
    north_m: float
    top_asl_m: float
    source: str
    iterations: int
    converged: bool
    ambiguous: bool
    #: class -> fraction of the ambiguity disc, descending. Kept so an ambiguous fix can still be reported
    #: with its alternatives rather than reduced to a single guess.
    mix: tuple[tuple[str, float], ...] = ()

    @property
    def dominant_fraction(self) -> float:
        return self.mix[0][1] if self.mix else 1.0


@dataclass(slots=True)
class _Item:
    east_m: float
    north_m: float
    r_major_m: float
    r_minor_m: float
    yaw_deg: float
    top_asl_m: float
    context: str
    source: str


@dataclass
class SceneGeometry:
    """The placed scene, indexed for point queries. Load once, query many times."""

    water_asl_m: float
    height: np.ndarray  # (grid, grid) float, terrain ASL
    cell_m: float
    size_m: float
    _buckets: dict[tuple[int, int], list[_Item]] = field(default_factory=dict, repr=False)
    #: True when ``height[i, j]`` is indexed (north, east); False when (east, north). Decided by measurement
    #: in :meth:`load`, never assumed — a silent transpose is exactly the defect that left the coverage
    #: footprint 90 degrees out until 2026-09-11.
    north_major: bool = True
    n_items: int = 0
    #: The highest surface placed anywhere in the scene. Bounds the ray segment that has to be searched.
    max_top_asl_m: float = float("-inf")
    #: Median |heightfield - known ground elevation| over the ground-level survivors, from :meth:`load`.
    terrain_check_m: float = float("nan")

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, scene_dir: str | Path | None = None) -> "SceneGeometry":
        d = Path(scene_dir) if scene_dir else Path(__file__).resolve().parents[2] / "data" / "scene"
        if not d.is_dir():
            raise ContextError(f"scene directory not found: {d}")

        valley = _read_json(d / "flood_valley.json")
        settlement = _read_json(d / "settlement.json")
        geom = cls(
            water_asl_m=float(settlement["water_level_m"]),
            height=np.load(d / "flood_valley_height.npy").astype(float),
            cell_m=float(valley["cell_m"]),
            size_m=float(valley["size_m"]),
        )
        geom._orient_heightfield(d)
        geom._load_structures(settlement)
        geom._load_vegetation(d)
        geom._load_debris(d)
        return geom

    def _orient_heightfield(self, scene_dir: Path) -> None:
        """Decide the heightfield's axis order by measurement against known survivor elevations."""
        actors = _read_json(scene_dir / "actors.json")["actors"]
        probes = [(a["east_m"], a["north_m"], a["base_asl_m"] - a.get("ground_offset_cm", 0.0) / 100.0)
                  for a in actors
                  if not (str(a.get("group") or "").startswith("roof")
                          or "roof" in str(a.get("note") or ""))]
        if len(probes) < 8:
            raise ContextError(f"only {len(probes)} ground-level survivors to orient the heightfield with")

        best: tuple[float, bool] | None = None
        for north_major in (True, False):
            self.north_major = north_major
            med = float(np.median([abs(self.terrain_asl(e, n) - z) for e, n, z in probes]))
            if best is None or med < best[0]:
                best = (med, north_major)
        assert best is not None
        self.terrain_check_m, self.north_major = best
        if self.terrain_check_m > 2.0:
            raise ContextError(
                f"heightfield disagrees with actors.json in BOTH axis orders (best median error "
                f"{self.terrain_check_m:.2f} m over {len(probes)} survivors). Terrain or layout has moved.")

    def _load_structures(self, settlement: dict[str, Any]) -> None:
        arche = settlement["archetypes"]
        for h in settlement["houses"]:
            a = arche[h["archetype"]]
            storeys = int(a.get("storeys", 1))
            # Roof height. `base_asl_m` is the slab; a storey is ~3 m, a gable adds ~1.2 m over the eaves.
            top = float(h["base_asl_m"]) + 3.0 * storeys + (1.2 if a.get("roof") == "gable" else 0.3)
            self._add(_Item(float(h["east_m"]), float(h["north_m"]),
                            float(a["length_m"]) / 2.0, float(a["width_m"]) / 2.0,
                            float(h.get("yaw_deg", 0.0)), top, "structure", f"House_{h['id']:03d}"))

    def _load_vegetation(self, scene_dir: Path) -> None:
        veg = _read_json(scene_dir / "vegetation.json")
        cat = veg["catalogue"]
        for it in veg.get("items", ()):
            c = cat.get(it["pid"])
            if not c:
                continue
            s = float(it.get("scale", 1.0))
            self._add(_Item(float(it["east_m"]), float(it["north_m"]),
                            float(c["crown_major_m"]) * s / 2.0, float(c["crown_minor_m"]) * s / 2.0,
                            float(it.get("yaw_deg", 0.0)),
                            float(it["base_asl_m"]) + float(c["height_m"]) * s,
                            "vegetation", str(it.get("name") or f"{it['pid']}#{it.get('id', -1)}")))
        for it in veg.get("ground_items", ()):
            c = cat.get(it["pid"])
            s = float(it.get("scale", 1.0))
            r = (float(c["crown_major_m"]) * s / 2.0) if c else 0.6 * s
            top = float(it["base_asl_m"]) + ((float(c["height_m"]) * s) if c else 0.5 * s)
            self._add(_Item(float(it["east_m"]), float(it["north_m"]), r, r,
                            float(it.get("yaw_deg", 0.0)), top, "vegetation",
                            str(it.get("name") or f"{it['pid']}#{it.get('id', -1)}")))

    def _load_debris(self, scene_dir: Path) -> None:
        rub = _read_json(scene_dir / "rubble_layout.json")
        variants = rub.get("variants", {})
        for it in rub.get("items", ()):
            v = variants.get(it.get("variant"), {})
            size = v.get("size_m") or (3.0, 3.0, 1.5)
            s = float(it.get("scale", 1.0))
            self._add(_Item(float(it["east_m"]), float(it["north_m"]),
                            float(size[0]) * s / 2.0, float(size[1]) * s / 2.0,
                            float(it.get("yaw_deg", 0.0)),
                            float(it["base_asl_m"]) + float(size[2]) * s,
                            "debris", str(it.get("name") or f"Rubble_{it.get('id', -1)}")))

        props = _read_json(scene_dir / "props_layout.json")
        for it in props.get("items", ()):
            s = float(it.get("scale", 1.0))
            role = str(it.get("role", ""))
            r, h = _PROP_SIZE.get(role, _PROP_SIZE_DEFAULT)
            self._add(_Item(float(it["east_m"]), float(it["north_m"]), r * s, r * s,
                            float(it.get("yaw_deg", 0.0)), float(it["base_asl_m"]) + h * s,
                            "vehicle" if role in _VEHICLE_ROLES else "debris",
                            str(it.get("name") or f"Prop_{it.get('id', -1)}")))

    def add(self, *, east_m: float, north_m: float, r_major_m: float, r_minor_m: float,
            top_asl_m: float, context: str, source: str, yaw_deg: float = 0.0) -> None:
        """Place one item by hand. For building a scene programmatically, and for tests."""
        if context not in CONTEXTS:
            raise ContextError(f"context {context!r} not in {CONTEXTS}")
        self._add(_Item(east_m, north_m, r_major_m, r_minor_m, yaw_deg, top_asl_m, context, source))

    def _add(self, item: _Item) -> None:
        self._buckets.setdefault(_key(item.east_m, item.north_m), []).append(item)
        self.n_items += 1
        if item.top_asl_m > self.max_top_asl_m:
            self.max_top_asl_m = item.top_asl_m

    # ------------------------------------------------------------------ queries

    def terrain_asl(self, east_m: float, north_m: float) -> float:
        """Terrain elevation, bilinear, clamped at the map edge."""
        g = self.height.shape[0]
        half = self.size_m / 2.0
        fe = (east_m + half) / self.cell_m
        fn = (north_m + half) / self.cell_m
        a, b = (fn, fe) if self.north_major else (fe, fn)
        i0 = min(max(int(math.floor(a)), 0), g - 1)
        j0 = min(max(int(math.floor(b)), 0), g - 1)
        i1, j1 = min(i0 + 1, g - 1), min(j0 + 1, g - 1)
        ta = min(max(a - i0, 0.0), 1.0)
        tb = min(max(b - j0, 0.0), 1.0)
        h = self.height
        return float((h[i0, j0] * (1 - tb) + h[i0, j1] * tb) * (1 - ta)
                     + (h[i1, j0] * (1 - tb) + h[i1, j1] * tb) * ta)

    def surface_at(self, east_m: float, north_m: float, below_asl_m: float | None = None) -> Surface:
        """The topmost placed surface at a ground position — what a camera directly above it sees.

        ``below_asl_m`` is the camera's elevation when there is one: a surface at or above the camera cannot
        be seen looking down, and on this terrain that is not hypothetical — the valley climbs to 1170 m while
        a survey over the flooded settlement flies at about 1097 m ASL.
        """
        ground = self.terrain_asl(east_m, north_m)
        if ground < self.water_asl_m:
            best = Surface("water", self.water_asl_m, "flood")
        else:
            best = Surface("open_ground", ground, "terrain")

        ke, kn = _key(east_m, north_m)
        for de in (-1, 0, 1):
            for dn in (-1, 0, 1):
                for it in self._buckets.get((ke + de, kn + dn), ()):
                    if below_asl_m is not None and it.top_asl_m >= below_asl_m - _COVER_EPS_M:
                        continue
                    if not _inside(east_m, north_m, it):
                        continue
                    if it.top_asl_m > best.top_asl_m + _COVER_EPS_M:
                        best = Surface(it.context, it.top_asl_m, it.source)
                    elif (abs(it.top_asl_m - best.top_asl_m) <= _COVER_EPS_M
                          and CONTEXT_PRIORITY[it.context] > CONTEXT_PRIORITY[best.context]):
                        best = Surface(it.context, it.top_asl_m, it.source)
        return best

    def items_near_ray(self, ray: Any, z_low_asl: float, z_high_asl: float,
                       samples: int = 16) -> list["_Item"]:
        """Every placed item whose bucket the ray passes over between two elevations.

        The ray sweeps sideways as it descends, so the candidate set is a swept segment rather than a single
        column. Sampling the segment and taking each sample's 3x3 bucket neighbourhood covers it: buckets are
        :data:`_BUCKET_M` wide and the segment is short compared with that over the height of anything placed
        in this scene.
        """
        # Nothing is placed above the tallest object, so the segment never needs to reach the camera. The
        # margin keeps the ray off its own apex, where the scale is zero and the ground position undefined.
        # The clamp is NOT redundant with the camera height: this valley climbs to 1170 m and a hillslope tree
        # can top out well above a drone flying 35 m over the flooded settlement.
        z_high_asl = min(z_high_asl - _COVER_EPS_M, self.max_top_asl_m)
        if z_high_asl <= z_low_asl:
            return []
        keys: set[tuple[int, int]] = set()
        for k in range(samples + 1):
            z = z_low_asl + (z_high_asl - z_low_asl) * k / samples
            ke, kn = _key(*ray(z))
            for de in (-1, 0, 1):
                for dn in (-1, 0, 1):
                    keys.add((ke + de, kn + dn))
        out: list[_Item] = []
        for key in keys:
            out.extend(self._buckets.get(key, ()))
        return out

    def mix_within(self, east_m: float, north_m: float, radius_m: float,
                   rings: int = 2, per_ring: int = 8,
                   below_asl_m: float | None = None) -> tuple[tuple[str, float], ...]:
        """Class fractions over a disc — how mixed the neighbourhood of a fix is."""
        pts = [(east_m, north_m)]
        for r in range(1, rings + 1):
            rad = radius_m * r / rings
            for k in range(per_ring):
                th = 2.0 * math.pi * k / per_ring
                pts.append((east_m + rad * math.cos(th), north_m + rad * math.sin(th)))
        counts: dict[str, int] = {}
        for e, n in pts:
            c = self.surface_at(e, n, below_asl_m=below_asl_m).context
            counts[c] = counts.get(c, 0) + 1
        total = float(len(pts))
        return tuple(sorted(((c, n / total) for c, n in counts.items()), key=lambda kv: (-kv[1], kv[0])))


# -------------------------------------------------------------------------- projection

def project_pixel(u_px: float, v_px: float, *, drone_east_m: float, drone_north_m: float,
                  drone_asl_m: float, target_asl_m: float, f_px: float,
                  cx_px: float, cy_px: float) -> tuple[float, float]:
    """Pixel -> world NE for a world-fixed nadir camera, at a stated target elevation.

    Image-right is EAST and image-down is SOUTH (measured; see the module docstring). The scale is the range
    to the **target's own surface**, not the frame's terrain GSD — a distinction worth 77 m on a roof survivor
    imaged far off-axis.
    """
    rng = drone_asl_m - target_asl_m
    if rng <= 0.0:
        raise ContextError(f"camera at or below the target surface (range {rng:.2f} m)")
    s = rng / f_px
    return drone_east_m + (u_px - cx_px) * s, drone_north_m - (v_px - cy_px) * s


def classify(bbox_px: Sequence[float], geom: SceneGeometry, *, drone_east_m: float, drone_north_m: float,
             drone_asl_m: float, f_px: float, cx_px: float, cy_px: float,
             uncertainty_m: float = PROJECTION_P90_M, max_iter: int = 8) -> ContextFix:
    """The terrain context under a box — ground truth or prediction, the same way for both.

    **This traces the ray, it does not iterate a substitution.** The obvious implementation — project onto the
    terrain, look up what is there, re-project at that surface's top, repeat — oscillates, because the ray
    sweeps sideways as it descends: at 55 m AGL a corner pixel moves ~15 m laterally over a 20 m tree, so the
    lookup lands on a different object each time and the fixed point never settles. Measured on the 260
    campaign boxes, that version left 37 unconverged and a residual p90 of 10.84 m against the 3.13 m the
    geometry actually supports.

    What happens instead: the ray's ground position is an exact function of height, so every candidate surface
    is tested *at its own elevation* — an object is hit if the ray, evaluated at that object's top, falls
    inside its footprint — and the highest hit wins. The terrain and the flood plane are solved separately
    (the terrain by a short iteration, which is stable because the terrain is smooth; the flood because it is
    a horizontal plane and therefore closed-form).
    """
    x1, y1, x2, y2 = (float(v) for v in bbox_px)
    u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def ray(z_asl: float) -> tuple[float, float]:
        return project_pixel(u, v, drone_east_m=drone_east_m, drone_north_m=drone_north_m,
                             drone_asl_m=drone_asl_m, target_asl_m=z_asl,
                             f_px=f_px, cx_px=cx_px, cy_px=cy_px)

    # --- the ground: terrain, or the flood plane where the terrain is under it -----------------------------
    z = geom.terrain_asl(drone_east_m, drone_north_m)
    east, north = ray(z)
    used = 0
    converged = False
    for used in range(1, max_iter + 1):
        z_new = geom.terrain_asl(east, north)
        east, north = ray(z_new)
        if abs(z_new - z) <= _COVER_EPS_M:
            z, converged = z_new, True
            break
        z = z_new
    best = Surface("open_ground", z, "terrain")

    e_w, n_w = ray(geom.water_asl_m)
    if geom.terrain_asl(e_w, n_w) < geom.water_asl_m:
        best = Surface("water", geom.water_asl_m, "flood")
        east, north = e_w, n_w

    # --- placed objects: each tested at its OWN top, so the sideways sweep is accounted for ----------------
    for it in geom.items_near_ray(ray, best.top_asl_m, drone_asl_m):
        # A surface at or above the camera cannot be seen by a downward ray. On this terrain that is not
        # hypothetical: hillslope crowns stand above a drone flying low over the settlement.
        if it.context not in _RESTABLE:
            continue
        if it.top_asl_m >= drone_asl_m - _COVER_EPS_M or it.top_asl_m <= best.top_asl_m - _COVER_EPS_M:
            continue
        e_i, n_i = ray(it.top_asl_m)
        if not _inside(e_i, n_i, it):
            continue
        if it.top_asl_m > best.top_asl_m + _COVER_EPS_M or (
                abs(it.top_asl_m - best.top_asl_m) <= _COVER_EPS_M
                and CONTEXT_PRIORITY[it.context] > CONTEXT_PRIORITY[best.context]):
            best = Surface(it.context, it.top_asl_m, it.source)
            east, north = e_i, n_i

    # The position is settled; now report what is actually at it, canopy included.
    final = geom.surface_at(east, north, below_asl_m=drone_asl_m)
    if CONTEXT_PRIORITY[final.context] > CONTEXT_PRIORITY[best.context] or final.top_asl_m > best.top_asl_m:
        best = final

    mix = geom.mix_within(east, north, uncertainty_m, below_asl_m=drone_asl_m)
    return ContextFix(context=best.context, east_m=east, north_m=north, top_asl_m=best.top_asl_m,
                      source=best.source, iterations=used, converged=converged,
                      ambiguous=(mix[0][1] < 0.75 if mix else False), mix=mix)


# -------------------------------------------------------------------------- helpers

#: Nominal (radius_m, height_m) per prop role. These are the ONLY assumed dimensions in this module:
#: ``props_layout.json`` records a placement but no bound. Everything else is read from the layouts.
_PROP_SIZE: dict[str, tuple[float, float]] = {
    "boulder": (1.1, 0.9), "log": (2.2, 0.5), "stump": (0.7, 0.8), "barrel": (0.35, 0.9),
    "crate": (0.6, 0.6), "jerrycan": (0.2, 0.4), "tyre": (0.35, 0.3), "trash": (0.5, 0.4),
    "car": (2.2, 1.5), "sheet": (1.2, 0.2), "timber": (1.8, 0.3),
}
_PROP_SIZE_DEFAULT: tuple[float, float] = (0.8, 0.6)
_VEHICLE_ROLES: frozenset[str] = frozenset({"car", "vehicle", "van", "truck", "boat"})


def _read_json(p: Path) -> Any:
    if not p.is_file():
        raise ContextError(f"scene layout missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _key(east_m: float, north_m: float) -> tuple[int, int]:
    return int(math.floor(east_m / _BUCKET_M)), int(math.floor(north_m / _BUCKET_M))


def _inside(east_m: float, north_m: float, it: _Item) -> bool:
    """Point-in-ellipse in the item's own yawed frame."""
    if it.r_major_m <= 0.0 or it.r_minor_m <= 0.0:
        return False
    th = math.radians(it.yaw_deg)
    de, dn = east_m - it.east_m, north_m - it.north_m
    c, s = math.cos(th), math.sin(th)
    a = (de * c + dn * s) / it.r_major_m
    b = (-de * s + dn * c) / it.r_minor_m
    return a * a + b * b <= 1.0


__all__ = ["CONTEXTS", "CONTEXT_PRIORITY", "PROJECTION_P90_M", "ContextError", "ContextFix",
           "SceneGeometry", "Surface", "classify", "context_rows", "measure_dataset_contexts",
           "measure_prediction_contexts", "project_pixel"]


# -------------------------------------------------------------------------- 5.12: FP/min per terrain type

def measure_prediction_contexts(res: Any, ds: Any, geom: SceneGeometry, *, f_px: float,
                                cx_px: float, cy_px: float) -> "np.ndarray":
    """The terrain each PREDICTION landed on, as a string array parallel to ``res.pred_score``.

    ``""`` where the frame carries no camera pose, so a dataset captured before
    :attr:`GtFrame.camera_east_m` existed is reported as unmeasured rather than guessed.
    """
    by_idx = {f.frame_idx: f for f in ds.frames}
    out = np.full(len(res.pred_score), "", dtype="<U12")
    for i, det in enumerate(res.detections):
        fr = by_idx.get(int(res.pred_frame[i]))
        if fr is None or not fr.has_camera_pose:
            continue
        try:
            fix = classify(det.bbox_px, geom, drone_east_m=float(fr.camera_east_m),
                           drone_north_m=float(fr.camera_north_m), drone_asl_m=float(fr.camera_asl_m),
                           f_px=f_px, cx_px=cx_px, cy_px=cy_px)
        except ContextError:
            continue
        out[i] = fix.context
    return out


def context_rows(res: Any, ds: Any, conf: float, key: Any, geom: SceneGeometry, *, f_px: float,
                 cx_px: float, cy_px: float) -> Any:
    """SOLUTION_DOC 5.12: ``FP/min per terrain type (water, debris, vegetation, roof)``.

    This is the one BOX-level axis a false positive can carry. :meth:`MatchResult.counts_at` says so itself:

        "A box-level slice restricts the ground truth only: a false positive has no ground-truth box and so
        has no occlusion, posture or pixel-size bin."

    True of occlusion, posture and pixel size — and not true of terrain, because terrain is a property of
    *where the box landed*, which is measurable for a prediction with no ground truth behind it at all. So
    both halves are emitted per terrain type: recall over the ground-truth boxes in that terrain, and FP/min
    over the predictions that fell on it.

    5.12 names the terrain types "water, debris, vegetation, roof". ``roof`` is spelled ``structure`` here,
    which is the ``GtBox.context`` vocabulary; the report prints the mapping so the requirement is legible.
    """
    from sightline.eval.slicing import MetricSet, metric_row, narrow  # local: slicing imports CONTEXTS above

    pred_ctx = measure_prediction_contexts(res, ds, geom, f_px=f_px, cx_px=cx_px, cy_px=cy_px)
    gt_ctx = np.asarray([str(b.context) if b.context in CONTEXTS else "" for b in res.gt_boxes], dtype="<U12")

    ms = MetricSet()
    tag = f"@iou{res.iou_thr:g}"
    measured = int((pred_ctx != "").sum())
    ms.add(metric_row("fp_context_coverage", measured / max(len(pred_ctx), 1), key, len(pred_ctx),
                      at_conf=conf, level="detection",
                      note="fraction of predictions whose terrain could be measured; the rest sit on "
                           "frames that carry no recorded camera pose"))

    for ctx in CONTEXTS:
        pm = pred_ctx == ctx
        gm = gt_ctx == ctx
        if not pm.any() and not gm.any():
            continue
        k = narrow(key, context=ctx)
        c = res.counts_at(conf, gt_mask=gm if gm.any() else None, pred_mask=pm)
        ms.add(metric_row(f"fp_per_min{tag}", c.fp_per_min(), k, c.fp, minutes=c.minutes, at_conf=conf,
                          level="detection"))
        if gm.any():
            cg = res.counts_at(conf, gt_mask=gm, pred_mask=pm)
            ms.add(metric_row(f"recall{tag}", cg.recall(), k, cg.n_gt, tp=cg.tp, fn=cg.fn, at_conf=conf))
    return ms


def measure_dataset_contexts(ds: Any, geom: SceneGeometry, *, f_px: float, cx_px: float,
                             cy_px: float) -> dict[str, int]:
    """Re-label every ground-truth box's ``context`` from the placed scene, in place.

    Without this the two halves of :func:`context_rows` disagree in kind: predictions would be measured while
    ground truth kept ``detect.dataset._context_for(zone, submersion)``, which is the survivor's zone label.
    Measured on the flown campaign, those two answers differ for **41.5 %** of boxes (183 of 313 agree), in
    both directions — so reporting recall-per-terrain against the assumption and FP/min-per-terrain against
    the measurement would be comparing two different axes that happen to share a name.

    Returns a count per outcome. Frames with no camera pose are counted as ``unmeasured`` and their boxes are
    left exactly as they were, never blanked.
    """
    out = {"measured": 0, "changed": 0, "unmeasured": 0}
    for fr in ds.frames:
        if not fr.has_camera_pose:
            out["unmeasured"] += len(fr.boxes)
            continue
        for b in fr.boxes:
            try:
                fix = classify(b.bbox_px, geom, drone_east_m=float(fr.camera_east_m),
                               drone_north_m=float(fr.camera_north_m), drone_asl_m=float(fr.camera_asl_m),
                               f_px=f_px, cx_px=cx_px, cy_px=cy_px)
            except ContextError:
                out["unmeasured"] += 1
                continue
            if b.context != fix.context:
                out["changed"] += 1
            b.context = fix.context
            out["measured"] += 1
    return out
