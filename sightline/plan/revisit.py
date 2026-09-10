"""F2 behaviour 3 — the revisit queue (SOLUTION_DOC §5.2).

    "Records marked `stale` and cells whose search quality is below a threshold are queued as revisit waypoints
     after the pattern completes."

Priority is the probability of success still on the table, `POS = POA x (1 - POD)` (Appendix B): the queue is
ordered by what is still *unfound*, not by what is old. Because POD is clamped below 1, `1 - POD` never reaches
zero, so a cell can always come back onto the queue — which is the queue-side expression of "no segment is ever
closed" (guardrail R10). Nothing here removes a record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from sightline.coverage.accumulate import CoverageMap
from sightline.coverage.grid import SceneFrame, grid_ne_to_latlon
from sightline.coverage.presentation import SIM_RGB_4K, CameraModel
from sightline.plan.patterns import orbit_route
from sightline.plan.waypoints import Route, TerrainFn, make_waypoint
from sightline.schemas import Record

DEFAULT_POD_THRESHOLD = 0.5
DEFAULT_STALE_AFTER_S = 20.0 * 60.0


@dataclass(slots=True)
class RevisitItem:
    kind: str  # "low_pod" | "stale_record"
    ne: tuple[float, float]  # scene-NE
    priority: float  # residual POS
    reason: str
    record_id: str = ""
    agl_m: float = 45.0
    radius_m: float = 40.0
    presentation: str = ""
    n_cells: int = 0

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(slots=True)
class RevisitQueue:
    items: list[RevisitItem] = field(default_factory=list)

    def push(self, item: RevisitItem) -> None:
        self.items.append(item)
        self.items.sort(key=lambda it: -it.priority)

    def extend(self, items: Iterable[RevisitItem]) -> None:
        self.items.extend(items)
        self.items.sort(key=lambda it: -it.priority)

    def pop(self) -> RevisitItem | None:
        return self.items.pop(0) if self.items else None

    def __len__(self) -> int:
        return len(self.items)

    def to_route(self, scene: SceneFrame, camera: CameraModel = SIM_RGB_4K, terrain: TerrainFn | None = None,
                 max_items: int = 8, speed_ms: float = 8.0, orbit_stale_records: bool = True) -> Route:
        """Waypoints for the top `max_items`, flown in priority order. Stale records get a confirmation orbit."""
        route = Route(pattern="revisit", scene=scene,
                      params={"camera": camera.name, "max_items": max_items, "n_queued": len(self.items)})
        for it in self.items[:max_items]:
            if it.kind == "stale_record" and orbit_stale_records:
                sub = orbit_route(it.ne, it.radius_m, it.agl_m, scene, n_points=6, dwell_s=2.0,
                                  terrain=terrain, record_id=it.record_id)
                for wp in sub.waypoints:
                    wp.reason = f"revisit: {it.reason}"
                    route.waypoints.append(wp)
            else:
                route.waypoints.append(make_waypoint(
                    len(route.waypoints), it.ne[0], it.ne[1], it.agl_m, scene, terrain, speed_ms=speed_ms,
                    action="goto", dwell_s=0.0, reason=f"revisit: {it.reason}"))
        route.notes.append(f"{len(self.items)} items queued, {min(max_items, len(self.items))} scheduled; "
                           f"ordered by residual POS = POA x (1 - POD) (simulation)")
        return route.renumber()


def from_coverage(cmap: CoverageMap, scene: SceneFrame, poa: np.ndarray | None = None,
                  presentation: str | None = None, pod_threshold: float = DEFAULT_POD_THRESHOLD,
                  min_cells: int = 4, max_items: int = 20, agl_m: float = 45.0) -> list[RevisitItem]:
    """Cluster the cells whose POD is below `pod_threshold` and emit one revisit item per cluster.

    `cannot_clear` cells are excluded: they are not under-searched, they are unsearchable from the air (§2.7), and
    queueing them would be exactly the false promise the guardrail exists to prevent.
    """
    layer = presentation or cmap.presentations[0]
    pod = cmap.pod(layer)
    poa_arr = np.asarray(poa, dtype=np.float32) if poa is not None else np.full(cmap.shape, 1.0 / pod.size,
                                                                               dtype=np.float32)
    mask = (pod < pod_threshold) & (~cmap.cannot_clear)
    if not mask.any():
        return []
    labels, n = _label(mask)
    grid = cmap.any_grid
    nn, ee = cmap.cell_centres()
    residual = poa_arr * (1.0 - pod)
    items: list[RevisitItem] = []
    for lab in range(1, n + 1):
        sel = labels == lab
        cells = int(sel.sum())
        if cells < min_cells:
            continue
        gn, ge = float(nn[sel].mean()), float(ee[sel].mean())
        lat, lon = grid_ne_to_latlon(grid, gn, ge)
        sn, se = scene.to_scene_ne(lat, lon)
        pos = float(residual[sel].sum())
        radius = math.sqrt(cells * grid.cell_m**2 / math.pi)
        items.append(RevisitItem("low_pod", (sn, se), pos,
                                 f"{cells} cells below POD {pod_threshold:.2f} for '{layer}' "
                                 f"(mean {float(pod[sel].mean()):.2f}); residual POS {pos:.4f}",
                                 agl_m=agl_m, radius_m=radius, presentation=layer, n_cells=cells))
    items.sort(key=lambda it: -it.priority)
    return items[:max_items]


def from_records(records: Sequence[Record], scene: SceneFrame, now_utc: float,
                 stale_after_s: float = DEFAULT_STALE_AFTER_S, agl_m: float = 35.0,
                 radius_m: float = 30.0) -> list[RevisitItem]:
    """Queue records the dedup lane marked `stale`, or that have not been re-seen inside `stale_after_s`.

    A dismissed record is NOT queued and NOT deleted; it simply stops attracting effort (guardrail R10).
    """
    out: list[RevisitItem] = []
    for r in records:
        if r.status == "dismissed":
            continue
        age = now_utc - r.last_seen_utc if r.last_seen_utc else 0.0
        if r.status != "stale" and age < stale_after_s:
            continue
        n, e = scene.to_scene_ne(r.lat, r.lon)
        pri = max(r.score, 1e-6) * (1.0 - min(r.confidence, 0.99))
        out.append(RevisitItem("stale_record", (n, e), float(pri),
                               f"record {r.record_id[:8]} status={r.status}, last seen {age / 60.0:.1f} min ago, "
                               f"score {r.score:.2f}", record_id=r.record_id, agl_m=agl_m, radius_m=radius_m))
    out.sort(key=lambda it: -it.priority)
    return out


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected connected components. Uses scipy when present, else a small union-find."""
    try:
        from scipy import ndimage

        lab, n = ndimage.label(mask)
        return np.asarray(lab), int(n)
    except ImportError:  # pragma: no cover - scipy is a pinned dependency
        lab = np.zeros(mask.shape, dtype=np.int32)
        cur = 0
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and lab[i, j] == 0:
                    cur += 1
                    stack = [(i, j)]
                    while stack:
                        a, b = stack.pop()
                        if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and lab[a, b] == 0:
                            lab[a, b] = cur
                            stack += [(a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)]
        return lab, cur
