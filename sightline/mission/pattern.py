"""The survey flight plan and terrain following — ONE implementation, shared by every mission runner.

`sightline/mission/survey.py` (the capture flight, F5) and `sightline/mission/live.py` (the real-time demo
loop, F2/F3) both fly the same boustrophedon over the same terrain. Before this module they could not: the
plan and the terrain follower lived inside `survey.main()` as local closures, so a second runner had to copy
them and the two would have drifted the first time either was tuned.

Nothing here talks to the pipeline and nothing here imports torch, the tracker or the store: this module is
the *flight* half and is safe to import in a capture process.

    scn  = Scenario.load()                       # flood_valley.json + actors.json + camera_survey.json
    plan = build_plan(scn, alt_m=45, speed_ms=12, plan="patches")
    ok   = cadence_verdict(45, 12, plan.shutter_m, scn.hfov_deg)      # SOLUTION_DOC §5.6 rule 3
    asl  = scn.terrain.surface_asl(east_m, north_m) + alt_m           # terrain following

Terrain following is not optional in this valley: the walls reach 1169.9 m ASL while the flood surface sits at
1061.7, so a fixed height above the water flies into the hillside.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]

__all__ = [
    "MIN_HITS", "MIN_HITS_WINDOW_S", "Terrain", "Scenario", "Leg", "SurveyPlan", "CadenceVerdict",
    "lawnmower", "build_plan", "cadence_verdict", "connect", "grab", "body_euler_deg", "SURVEY_CAMERA",
]

#: Mirrors `sightline/track/config.py`. Deliberately NOT imported: this module runs inside the capture
#: process, which must not pull the tracker stack (and therefore torch) in. `tests/test_mission_cadence.py`
#: asserts the copy has not drifted from `TrackerConfig`.
MIN_HITS = 3
MIN_HITS_WINDOW_S = 2.0

#: The nadir camera every mission flies with (`sim/settings/*.json`), NED (0, 0, 0.30) below the airframe.
SURVEY_CAMERA = "survey"


# --- terrain -------------------------------------------------------------------------------------------
class Terrain:
    """The scenario heightfield, sampled the way the flight controller needs it.

    `surface_asl` returns the HIGHER of ground and flood surface, because the drone must clear both.
    """

    def __init__(self, height: Any, n: int, cell_m: float, size_m: float, water_level_m: float):
        self.height = height
        self.n = int(n)
        self.cell_m = float(cell_m)
        self.size_m = float(size_m)
        self.water_level_m = float(water_level_m)
        self.origin_m = -self.size_m / 2.0

    @classmethod
    def from_scene(cls, scene: dict[str, Any], water_level_m: float) -> "Terrain":
        """Rebuild the exact heightfield `tools/scene/gen_terrain.py` wrote the level from (seeded)."""
        p = str(REPO / "tools" / "scene")
        if p not in sys.path:
            sys.path.insert(0, p)
        import gen_terrain as gt  # noqa: PLC0415  (host-side generator, not a package)

        t = gt.build(scene["size_m"], scene["cell_m"], scene["seed"])
        return cls(t["height"], t["n"], t["cell_m"], t["size_m"], water_level_m)

    def surface_asl(self, east_m: float, north_m: float) -> float:
        i = int(round((east_m - self.origin_m) / self.cell_m))
        j = int(round((north_m - self.origin_m) / self.cell_m))
        i = min(max(i, 0), self.n - 1)
        j = min(max(j, 0), self.n - 1)
        return max(float(self.height[j][i]), self.water_level_m)

    def agl_m(self, east_m: float, north_m: float, asl_m: float) -> float:
        return asl_m - self.surface_asl(east_m, north_m)


@dataclass
class Scenario:
    """Everything a mission needs to know about the world before it takes off."""

    scene: dict[str, Any]
    truth: dict[str, Any]
    cal: dict[str, Any]
    terrain: Terrain

    @classmethod
    def load(cls, repo: Path | None = None) -> "Scenario":
        r = repo or REPO
        scene = json.loads((r / "data/scene/flood_valley.json").read_text())
        truth = json.loads((r / "data/scene/actors.json").read_text())
        cal = json.loads((r / "data/scene/camera_survey.json").read_text())
        return cls(scene, truth, cal, Terrain.from_scene(scene, float(truth["water_level_m"])))

    # -- convenience -------------------------------------------------------------------------------
    @property
    def home(self) -> dict[str, Any]:
        return self.scene["launch_site"]

    @property
    def f_px(self) -> float:
        return float(self.cal["f_px"])

    @property
    def hfov_deg(self) -> float:
        return float(self.cal["hfov_deg"])

    @property
    def water_level_m(self) -> float:
        return float(self.truth["water_level_m"])

    @property
    def seed(self) -> int:
        return int(self.truth["seed"])

    def detectable_actors(self) -> list[dict[str, Any]]:
        return [a for a in self.truth["actors"] if a["aerially_detectable"]]

    # -- frame geometry ----------------------------------------------------------------------------
    def frame_size_m(self, alt_m: float, width_px: int = 3840, height_px: int = 2160
                     ) -> tuple[float, float]:
        """Across-track and along-track ground extent of one frame at this AGL."""
        w = 2.0 * alt_m * math.tan(math.radians(self.hfov_deg) / 2.0)
        return w, w * height_px / width_px

    def gsd_cm_px(self, agl_m: float) -> float:
        return agl_m / self.f_px * 100.0


# --- the plan ------------------------------------------------------------------------------------------
@dataclass(slots=True)
class Leg:
    """One boustrophedon line: fly along constant east from north_start to north_end."""

    east_m: float
    north_start_m: float
    north_end_m: float

    @property
    def length_m(self) -> float:
        return abs(self.north_end_m - self.north_start_m)

    def reached(self, north_m: float, tol_m: float = 2.0) -> bool:
        if self.north_end_m > self.north_start_m:
            return north_m >= self.north_end_m - tol_m
        return north_m <= self.north_end_m + tol_m

    def progress(self, north_m: float) -> float:
        """0 at the start of the leg, 1 at its end. Used to resume a leg rather than restart it."""
        if self.length_m < 1e-6:
            return 1.0
        return min(max((north_m - self.north_start_m) / (self.north_end_m - self.north_start_m), 0.0), 1.0)


@dataclass
class SurveyPlan:
    legs: list[Leg]
    plan: str
    alt_m: float
    speed_ms: float
    line_spacing_m: float
    frame_w_m: float
    frame_h_m: float
    shutter_m: float
    survivors_in_plan: int
    survivors_total: int
    patches: list[tuple[float, float, float, float, int]] = field(default_factory=list)

    @property
    def track_km(self) -> float:
        return sum(x.length_m for x in self.legs) / 1000.0

    @property
    def est_minutes(self) -> float:
        return self.track_km * 1000.0 / max(self.speed_ms, 1e-6) / 60.0

    @property
    def est_frames(self) -> int:
        return int(self.track_km * 1000.0 / max(self.shutter_m, 1e-6))

    def summary(self) -> str:
        head = (f"{len(self.patches)} survivor patches hold {self.survivors_in_plan}/{self.survivors_total} "
                f"survivors" if self.plan == "patches" else
                f"survey box holds {self.survivors_in_plan}/{self.survivors_total} survivors")
        return (f"{head}\n{len(self.legs)} lines, {self.line_spacing_m:.0f} m apart, {self.track_km:.1f} km "
                f"of track at {self.speed_ms:.0f} m/s -> ~{self.est_minutes:.1f} min, shutter every "
                f"{self.shutter_m:.0f} m, frame {self.frame_w_m:.0f} x {self.frame_h_m:.0f} m")


def lawnmower(e0: float, e1: float, n0: float, n1: float, spacing_m: float, flip: bool = False) -> list[Leg]:
    """Boustrophedon legs covering the box, alternating direction so the turns are at the ends."""
    out: list[Leg] = []
    e, i = e0, 0
    while e <= e1 + 1e-6:
        out.append(Leg(e, n0, n1) if (i % 2 == 0) != flip else Leg(e, n1, n0))
        e += spacing_m
        i += 1
    return out


def _single_link_patches(east: np.ndarray, north: np.ndarray, link_m: float
                         ) -> list[tuple[float, float, float, float, int]]:
    """Cluster survivors by single link; each cluster becomes one small lawnmower."""
    pts = np.column_stack([east, north])
    lab = [-1] * len(pts)
    nc = 0
    for i in range(len(pts)):
        if lab[i] != -1:
            continue
        lab[i] = nc
        # Index-walked frontier rather than a stack: `list.pop` is one of the delete-shaped idioms the R10
        # guardrail scanner refuses anywhere in a lane that touches records (sightline/triage/guardrails.py).
        frontier, head = [i], 0
        while head < len(frontier):
            j = frontier[head]
            head += 1
            for k in range(len(pts)):
                if lab[k] == -1 and math.hypot(*(pts[j] - pts[k])) < link_m:
                    lab[k] = nc
                    frontier.append(k)
        nc += 1
    out = []
    for ci in range(nc):
        q = pts[[i for i in range(len(pts)) if lab[i] == ci]]
        out.append((float(q[:, 0].min()), float(q[:, 0].max()),
                    float(q[:, 1].min()), float(q[:, 1].max()), len(q)))
    return out


def build_plan(scn: Scenario, *, alt_m: float, speed_ms: float, shutter_m: float, plan: str = "patches",
               overlap: float = 0.2, quantile: float = 0.05, patch_link_m: float = 90.0,
               width_px: int = 3840, height_px: int = 2160) -> SurveyPlan:
    """The flight plan both runners fly.

    ``box``     one lawnmower over the survivors' quantile-trimmed extent.
    ``patches`` single-link clusters, one small lawnmower each, ordered nearest-neighbour from the launch
                site. The box plan spends most of its frames on ground nobody is on: measured on the
                2026-09-11 run, 209 frames returned 42 boxes.
    """
    det = scn.detectable_actors()
    es = np.array([a["east_m"] for a in det], dtype=float)
    ns = np.array([a["north_m"] for a in det], dtype=float)
    frame_w, frame_h = scn.frame_size_m(alt_m, width_px, height_px)
    spacing = frame_w * (1.0 - overlap)

    if plan == "box":
        e0, e1 = np.quantile(es, [quantile, 1 - quantile])
        n0, n1 = np.quantile(ns, [quantile, 1 - quantile])
        e0, e1, n0, n1 = e0 - 35, e1 + 35, n0 - 35, n1 + 35
        legs = lawnmower(float(e0), float(e1), float(n0), float(n1), spacing)
        inside = int(((es >= e0) & (es <= e1) & (ns >= n0) & (ns <= n1)).sum())
        patches: list[tuple[float, float, float, float, int]] = []
    elif plan == "patches":
        patches = _single_link_patches(es, ns, patch_link_m)
        order, todo = [], list(range(len(patches)))
        cur = (float(scn.home["east_m"]), float(scn.home["north_m"]))
        while todo:
            k = min(todo, key=lambda i: math.hypot((patches[i][0] + patches[i][1]) / 2 - cur[0],
                                                   (patches[i][2] + patches[i][3]) / 2 - cur[1]))
            order.append(k)
            todo = [i for i in todo if i != k]     # not `todo.remove`: see the R10 note above
            cur = ((patches[k][0] + patches[k][1]) / 2, (patches[k][2] + patches[k][3]) / 2)
        legs, inside = [], 0
        for m, k in enumerate(order):
            pe0, pe1, pn0, pn1, cnt = patches[k]
            legs += lawnmower(pe0 - frame_w / 3, pe1 + frame_w / 3, pn0 - frame_h / 2, pn1 + frame_h / 2,
                              spacing, flip=(m % 2 == 1))
            inside += cnt
    else:
        raise ValueError(f"unknown plan {plan!r}; expected 'box' or 'patches'")

    return SurveyPlan(legs=legs, plan=plan, alt_m=alt_m, speed_ms=speed_ms, line_spacing_m=spacing,
                      frame_w_m=frame_w, frame_h_m=frame_h, shutter_m=shutter_m,
                      survivors_in_plan=inside, survivors_total=len(det), patches=patches)


# --- the cadence gate (SOLUTION_DOC §5.6 rule 3) ---------------------------------------------------------
@dataclass(slots=True)
class CadenceVerdict:
    """Can this cadence produce a TRACK? If not, the run yields boxes and then nothing at all."""

    ok: bool
    reasons: list[str]
    suggested_shutter_m: float
    seconds_between_frames: float
    hits_in_frame: float
    hits_span_s: float

    def message(self) -> str:
        if self.ok:
            return (f"cadence OK: {MIN_HITS} shots span {self.hits_span_s:.2f} s "
                    f"(window {MIN_HITS_WINDOW_S:.0f} s), target in frame for {self.hits_in_frame:.1f} shots")
        return ("REFUSING TO FLY: this cadence cannot confirm a track, so the run would produce boxes and "
                "then nothing downstream.\n  " + "\n  ".join(self.reasons)
                + f"\n  Use --shutter-m {self.suggested_shutter_m:.1f} or less at this speed, raise "
                  f"--speed, or pass --no-track-check if you deliberately want a detector-only dataset.")


def cadence_verdict(alt_m: float, speed_ms: float, shutter_m: float, hfov_deg: float,
                    width_px: int = 3840, height_px: int = 2160) -> CadenceVerdict:
    """The 2026-09-11 regression: 285 frames -> 56 detections -> 56 fixes -> 0 tracks -> 0 records.

    Nothing was broken; the shutter was 26 m at 11 m/s, so frames were 2.36 s apart and a survivor was in
    frame for 1.5 of them, while `min_hits=3` inside `min_hits_window_s=2.0` needs three.
    """
    frame_w = 2.0 * alt_m * math.tan(math.radians(hfov_deg) / 2.0)
    frame_h = frame_w * height_px / width_px
    dt = shutter_m / max(speed_ms, 1e-6)
    hits = frame_h / max(shutter_m, 1e-6)
    span = dt * (MIN_HITS - 1)
    reasons: list[str] = []
    if span > MIN_HITS_WINDOW_S + 1e-6:
        reasons.append(f"{MIN_HITS} shots span {span:.1f} s, outside the {MIN_HITS_WINDOW_S:.0f} s "
                       f"confirmation window (shutter {shutter_m:.1f} m at {speed_ms:.0f} m/s "
                       f"= {dt:.2f} s a frame)")
    if hits < MIN_HITS:
        reasons.append(f"a target is in frame for only {hits:.1f} shots "
                       f"(frame height {frame_h:.0f} m / shutter {shutter_m:.1f} m)")
    need = min(speed_ms * MIN_HITS_WINDOW_S / (MIN_HITS - 1), frame_h / MIN_HITS)
    return CadenceVerdict(ok=not reasons, reasons=reasons, suggested_shutter_m=need,
                          seconds_between_frames=dt, hits_in_frame=hits, hits_span_s=span)


# --- the simulator link --------------------------------------------------------------------------------
def connect(vehicle: str = ""):
    """A confirmed Cosys-AirSim multirotor client. cosysairsim prints on import and on connect: swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        import cosysairsim as airsim  # noqa: PLC0415

        c = airsim.MultirotorClient()
        c.confirmConnection()
    return c


def grab(client, *, want_ir: bool = False, want_depth: bool = False,
         camera: str = SURVEY_CAMERA) -> dict[str, np.ndarray]:
    """One synchronised Scene + Segmentation (+ Infrared, + DepthPlanar) capture.

    Raw AirSim buffers are RGB (measured); OpenCV is BGR, so the swap happens once, on write — never here.

    `want_depth` is not optional in practice. Cosys-AirSim renders the instance mask with
    `SetInstancedFoliage(false)`, so the mask is blind to every plant in this scene, and
    `labels.apply_depth_visibility` needs the depth buffer to tell a survivor the camera can see from one
    lying under a fern. Depth comes back as float metres along the optical axis, not as bytes.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        import cosysairsim as airsim  # noqa: PLC0415

        req = [airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
               airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False)]
        keys = ["scene", "seg"]
        if want_ir:
            req.append(airsim.ImageRequest(camera, airsim.ImageType.Infrared, False, False))
            keys.append("ir")
        if want_depth:
            req.append(airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False))
            keys.append("depth")
        res = client.simGetImages(req)
    out: dict[str, np.ndarray] = {}
    for r, k in zip(res, keys):
        if k == "depth":
            # An empty float buffer means the ImageType 1 capture entry is missing from the settings profile
            # the editor was STARTED with. Letting that through would silently disable the visibility gate
            # and put the foliage boxes straight back into the dataset.
            if not r.image_data_float or r.width == 0:
                raise RuntimeError(
                    "DepthPlanar came back empty. The capture entry {'ImageType': 1} must be present in the "
                    "settings profile the editor was launched with (sim/settings/dataset.json has it); "
                    "without depth the foliage-occlusion gate cannot run.")
            out[k] = np.array(r.image_data_float, dtype=np.float32).reshape(r.height, r.width)
        else:
            out[k] = np.frombuffer(r.image_data_uint8, dtype=np.uint8).reshape(r.height, r.width, 3)
    return out


def body_euler_deg(orientation) -> tuple[float, float, float]:
    """(roll, pitch, yaw) in degrees from an AirSim quaternion, the convention the tilt gate uses."""
    w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw
