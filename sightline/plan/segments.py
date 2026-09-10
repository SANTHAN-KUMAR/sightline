"""Search segments and their ICS-204-style assignment records (SOLUTION_DOC §5.3 "Outputs", §2.6 Wayanad).

    "A vector layer of segments with an ICS-204-style assignment record (segment, priority, assigned asset, passes
     flown, current POD, recommended next action). The recommendation text is generated from the numbers
     ('Segment B: POD 0.41 after two passes at 60 m; night thermal pass recommended before 06:00 to beat
     crossover'), and the commander accepts, edits or ignores it."

Segment polygons are in **scene-NE metres** (the same frame as waypoints); they are converted to grid-NE only when
a coverage mask is needed. Nothing here closes a segment: `recommend()` proposes, and the strongest thing it can
say about a burial region is that aerial search cannot clear it (§2.7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from sightline.coverage.accumulate import CoverageMap
from sightline.coverage.grid import SceneFrame, grid_ne_to_latlon, scene_poly_to_grid
from sightline.coverage.presentation import CameraModel, mix_for_zone
from sightline.coverage.quality import CROSSOVER_WINDOWS_LOCAL_H
from sightline.plan.patterns import polygon_area_m2, polygon_centroid_ne, rect_polygon
from sightline.schemas import SCHEMA_VERSION

SEGMENT_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted: they read as 1 and 0 on a radio


@dataclass(slots=True)
class Segment:
    """One assignment. `poly_ne` is scene-NE; `priority` is 1 (highest) upward, as ICS assignments are numbered."""

    seg_id: str
    poly_ne: np.ndarray
    priority: int = 1
    assigned_asset: str = ""
    passes_flown: int = 0
    last_agl_m: float = 0.0
    last_band: str = ""
    zone: str = "unknown"
    poa_mass: float = 0.0
    notes: str = ""
    _mask: np.ndarray | None = field(default=None, repr=False)

    def area_m2(self) -> float:
        return polygon_area_m2(self.poly_ne)

    def centroid_ne(self) -> tuple[float, float]:
        return polygon_centroid_ne(self.poly_ne)

    def mask(self, cmap: CoverageMap, scene: SceneFrame) -> np.ndarray:
        """Boolean cell mask over the coverage grid. Cached: a segment's geometry does not change."""
        if self._mask is None or self._mask.shape != cmap.shape:
            from sightline.coverage.grid import polygon_cell_weights

            win, w = polygon_cell_weights(cmap.any_grid, scene_poly_to_grid(cmap.any_grid, scene, self.poly_ne), 2)
            m = np.zeros(cmap.shape, dtype=bool)
            if not win.empty:
                m[win.slice()] = w > 0.5
            self._mask = m
        return self._mask

    def to_feature(self, scene: SceneFrame) -> dict[str, Any]:
        ring = [[lon, lat] for lat, lon in (scene.to_latlon(float(n), float(e)) for n, e in self.poly_ne)]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        return {"type": "Feature", "id": self.seg_id,
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {"seg_id": self.seg_id, "priority": self.priority, "zone": self.zone,
                               "assigned_asset": self.assigned_asset, "passes_flown": self.passes_flown,
                               "area_m2": round(self.area_m2(), 1), "poa": round(self.poa_mass, 5),
                               "notes": self.notes, "domain": "sim"}}


def auto_segments(cmap: CoverageMap, scene: SceneFrame, poa: np.ndarray | None = None, n_rows: int = 4,
                  n_cols: int = 4, min_poa: float = 0.0, aoi_ne: np.ndarray | None = None) -> list[Segment]:
    """Tile the area of operations into segments and rank them by prior mass (§5.3: "or accept the auto-segmentation").

    Kept deliberately simple: a regular tiling is what a commander redraws by hand anyway, and the planner does not
    care whether the segments came from a grid or a pen.
    """
    grid = cmap.any_grid
    if aoi_ne is None:
        sw = scene.to_scene_ne(grid.origin_lat, grid.origin_lon)
        ne_corner = scene.to_scene_ne(*grid_ne_to_latlon(grid, grid.n_north * grid.cell_m,
                                                         grid.n_east * grid.cell_m))
        aoi_ne = np.array([[sw[0], sw[1]], [ne_corner[0], sw[1]], [ne_corner[0], ne_corner[1]],
                           [sw[0], ne_corner[1]]], dtype=float)
    n0, n1 = float(aoi_ne[:, 0].min()), float(aoi_ne[:, 0].max())
    e0, e1 = float(aoi_ne[:, 1].min()), float(aoi_ne[:, 1].max())
    dn, de = (n1 - n0) / n_rows, (e1 - e0) / n_cols
    segs: list[Segment] = []
    for i in range(n_rows):
        for j in range(n_cols):
            centre = (n0 + (i + 0.5) * dn, e0 + (j + 0.5) * de)
            poly = rect_polygon(centre, dn, de)
            s = Segment(seg_id="", poly_ne=poly)
            m = s.mask(cmap, scene)
            if not m.any():
                continue
            s.poa_mass = float(np.asarray(poa)[m].sum()) if poa is not None else float(m.sum()) / max(m.size, 1)
            codes = cmap.zone_codes[m]
            if codes.size:
                from sightline.coverage.accumulate import ZONE_NAMES

                s.zone = ZONE_NAMES[int(np.bincount(codes).argmax())]
            segs.append(s)
    segs = [s for s in segs if s.poa_mass >= min_poa]
    segs.sort(key=lambda s: -s.poa_mass)
    for rank, s in enumerate(segs):
        s.seg_id = f"Segment {SEGMENT_LETTERS[rank]}" if rank < len(SEGMENT_LETTERS) else f"Segment {rank + 1}"
        s.priority = rank + 1
    return segs


def segments_geojson(segments: Sequence[Segment], scene: SceneFrame) -> dict[str, Any]:
    return {"type": "FeatureCollection", "schema_version": SCHEMA_VERSION,
            "features": [s.to_feature(scene) for s in segments]}


# --- the assignment record --------------------------------------------------------------------------------
@dataclass(slots=True)
class AssignmentRecord:
    """ICS-204-shaped: what to search, who has it, how well it has been searched, and what to do next."""

    seg_id: str
    priority: int
    zone: str
    assigned_asset: str
    passes_flown: int
    area_m2: float
    poa: float
    pod: dict[str, float]
    pod_effective: float
    cannot_clear_fraction: float
    recommendation: str
    domain: str = "sim"

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def __str__(self) -> str:
        pods = ", ".join(f"{k} {v:.2f}" for k, v in self.pod.items())
        return f"{self.seg_id} (priority {self.priority}, {self.zone}): POD {pods}. {self.recommendation}"


def assignment_record(seg: Segment, cmap: CoverageMap, scene: SceneFrame, poa: np.ndarray | None = None,
                      camera: CameraModel | None = None, local_hour: float | None = None,
                      thermal_flown: bool = False) -> AssignmentRecord:
    """Build the record and generate its recommendation text from the numbers, never from a template guess."""
    m = seg.mask(cmap, scene)
    n = int(m.sum())
    pod = {p: float(cmap.pod(p)[m].mean()) if n else 0.0 for p in cmap.presentations}
    eff = float(cmap.pod_effective()[m].mean()) if n else 0.0
    cc = float(cmap.cannot_clear[m].mean()) if n else 0.0
    rec = recommend(seg, pod, cc, camera, local_hour, thermal_flown)
    return AssignmentRecord(seg.seg_id, seg.priority, seg.zone, seg.assigned_asset, seg.passes_flown,
                            seg.area_m2(), float(np.asarray(poa)[m].sum()) if poa is not None else seg.poa_mass,
                            pod, eff, cc, rec)


def recommend(seg: Segment, pod: dict[str, float], cannot_clear_fraction: float,
              camera: CameraModel | None = None, local_hour: float | None = None,
              thermal_flown: bool = False, thin_threshold: float = 0.30, good_threshold: float = 0.60) -> str:
    """The generated "recommended next action" sentence. Every clause is triggered by a number, and it never
    says a segment is cleared."""
    parts: list[str] = []
    if seg.passes_flown == 0:
        parts.append("not yet flown: a first detection pass at 45-60 m nadir is the highest-value action")
    else:
        body = pod.get("body", pod.get("prone", 0.0))
        thin = {k: v for k, v in pod.items() if v < thin_threshold and k != "buried"}
        if body >= good_threshold and thin and camera is not None:
            worst = min(thin, key=lambda k: thin[k])
            alt = camera.ceiling_m(worst)
            parts.append(f"POD {body:.2f} for body but only {thin[worst]:.2f} for {worst}; one pass at "
                         f"{alt:.0f} m is recommended before it is treated as searched for {worst} presentations")
        elif body < good_threshold:
            parts.append(f"POD {body:.2f} after {seg.passes_flown} pass(es): another pass at the same altitude "
                         f"still adds materially")
        else:
            parts.append(f"POD {body:.2f} after {seg.passes_flown} pass(es): a repeat pass in identical "
                         f"conditions would add little; a different band or altitude adds more")
    if not thermal_flown:
        window = CROSSOVER_WINDOWS_LOCAL_H[0]
        # Suppressed only while the clock is actually inside a crossover window, where "fly before 06.40" is
        # already stale advice. It is emitted through the working day on purpose: that is when a commander
        # schedules the pre-dawn pass (§2.7 "pre-dawn is thermal's strongest window").
        in_crossover = local_hour is not None and any(lo <= local_hour <= hi
                                                      for lo, hi in CROSSOVER_WINDOWS_LOCAL_H)
        if not in_crossover:
            parts.append(f"a pre-dawn thermal pass before {window[0]:.2f} local beats the thermal crossover window "
                         f"{window[0]:.2f}-{window[1]:.2f}")
    if cannot_clear_fraction > 0.01:
        parts.append(f"{cannot_clear_fraction * 100:.0f} % of this segment is inside a burial polygon: aerial "
                     f"search cannot clear it, and a close-in radar or canine check is the only next step there")
    return "; ".join(parts) + " (simulation)"


def presentation_mix_note(zone: str, presentations: Sequence[str]) -> str:
    mix = mix_for_zone(zone, tuple(presentations))
    return ", ".join(f"{k} {v:.2f}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))


def segment_of_point(segments: Sequence[Segment], ne: Sequence[float]) -> Segment | None:
    from sightline.coverage.grid import points_in_polygon

    pt = np.array([[float(ne[0]), float(ne[1])]])
    for s in segments:
        if bool(points_in_polygon(s.poly_ne, pt)[0]):
            return s
    return None


def nearest_segment(segments: Sequence[Segment], ne: Sequence[float]) -> Segment | None:
    if not segments:
        return None
    return min(segments, key=lambda s: math.dist(tuple(ne), s.centroid_ne()))
