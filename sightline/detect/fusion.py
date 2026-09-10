"""F9 late fusion at the box level: WBF geometry + the ProbEn score rule (SOLUTION_DOC §5.5, §5.5b).

Three decisions from §5.5 are implemented literally here, because each of them is the reason a design that looks
more sophisticated was rejected:

1. **Late, at the box level.** Boxes from two independently trained detectors are clustered with Weighted Boxes
   Fusion (`ensemble-boxes`, MIT). Box-level fusion at IoU 0.5 tolerates the 0-15 px registration error that every
   "aligned" RGB-thermal benchmark still contains; a 4-channel stem does not.
2. **The score rule is ProbEn, not WBF's weighted average.** WBF decides *where* the fused box is; ProbEn decides
   *how confident* it is. The difference matters at the edges: WBF rescales a box seen by only one model by
   ``n/N``, which halves a thermal-only hot spot at night and an RGB-only detection over midday water. ProbEn
   marginalises instead -- a single-modality box **keeps its own posterior unchanged**. That is exactly the
   behaviour §5.5 asks for. The rule itself is the vendored, Apache-2.0 `vendor/proben`.
3. **The RGB-only fallback is structural, not learned.** :func:`fuse_detections` returns *the RGB list itself* --
   the same objects, same order -- whenever thermal is absent, empty, or the registration residual is too large.
   There is no thermal-zeroed input, no out-of-distribution tensor, nothing to go wrong at 3 a.m. in a demo.

Registration. In the simulator both cameras share a pose, so the thermal->RGB map is not fitted at all: it is
derived in closed form from the two `Intrinsics` (:func:`homography_from_intrinsics`) and its residual is zero by
construction. The per-altitude fitted table of §5.5 (`AltitudeHomographyTable`) is kept for the real-footage replay
path and **refuses to run on a sim bundle** -- fitting a homography to a known-exact mapping is how a demo acquires
a mysterious 3 px bias.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from sightline.detect.tiler import iou_xyxy
from sightline.schemas import Detection, FrameBundle, Intrinsics

__all__ = [
    "DEFAULT_IOU_THR",
    "DEFAULT_SKIP_BOX_THR",
    "MAX_RESIDUAL_PX",
    "AltitudeHomographyTable",
    "FusionResult",
    "Registration",
    "fuse_detections",
    "homography_from_intrinsics",
    "register_thermal",
    "thermal_weight",
    "warp_boxes",
    "warp_points",
]

#: §5.5's inference sketch: `weighted_boxes_fusion(..., iou_thr=0.5, skip_box_thr=0.05)`.
DEFAULT_IOU_THR = 0.5
DEFAULT_SKIP_BOX_THR = 0.05
#: Registration residual above which fusion is skipped entirely. 8 px in RGB coordinates is ~2 thermal pixels at
#: the 3.5:1 ratio of §5.5's DJI payload analysis: beyond that, box-level fusion starts pairing the wrong targets.
MAX_RESIDUAL_PX = 8.0


# --- registration --------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Registration:
    """The thermal -> RGB pixel map and how much to trust it."""

    H: np.ndarray  # 3x3, homogeneous, maps thermal pixels to RGB pixels
    residual_px: float
    source: str  # "intrinsics" (sim, exact) | "altitude_table" (real, fitted) | "identity"
    valid: bool = True
    reject_reason: str = ""

    def __post_init__(self) -> None:
        if self.H.shape != (3, 3):
            raise ValueError(f"H must be 3x3, got {self.H.shape}")


def homography_from_intrinsics(thermal: Intrinsics, rgb: Intrinsics) -> np.ndarray:
    """Exact thermal->RGB map for two pinhole cameras that **share a pose** (the simulator's case).

    A thermal pixel back-projects to the ray ``((u - cx_t)/fx_t, (v - cy_t)/fy_t, 1)``; with a common optical
    centre and orientation that same ray projects into RGB at ``fx_r * X + cx_r``. Composing gives a pure
    scale-and-translate homography, with no parallax term -- which is the whole reason §5.5 says the simulator can
    switch the registration step off.
    """
    sx = rgb.fx / thermal.fx
    sy = rgb.fy / thermal.fy
    return np.array([
        [sx, 0.0, rgb.cx - sx * thermal.cx],
        [0.0, sy, rgb.cy - sy * thermal.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """(N,2) -> (N,2) through a 3x3 homography."""
    p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    hom = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1) @ np.asarray(H, dtype=np.float64).T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return hom[:, :2] / w


def warp_boxes(H: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
    """Warp xyxy boxes through H and re-axis-align (all four corners, not just two: H may rotate)."""
    out = np.zeros((len(boxes), 4), dtype=np.float64)
    for i, b in enumerate(boxes):
        corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], dtype=np.float64)
        w = warp_points(H, corners)
        out[i] = [w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()]
    return out


@dataclass(slots=True)
class AltitudeHomographyTable:
    """§5.5's real-payload recipe: one homography fitted per altitude band from >= 8 clicked correspondences.

    **Real-footage path only.** :func:`register_thermal` never consults this for a `sim` bundle, where the exact
    intrinsics map is available and fitting would only add error.
    """

    bands: list[tuple[float, np.ndarray, float]] = field(default_factory=list)  # (agl_m, H, fit_residual_px)

    def add(self, agl_m: float, H: np.ndarray, residual_px: float = 0.0) -> None:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError("H must be 3x3")
        self.bands.append((float(agl_m), H, float(residual_px)))
        self.bands.sort(key=lambda b: b[0])

    def lookup(self, agl_m: float) -> tuple[np.ndarray, float] | None:
        """Nearest band. No interpolation between homographies: a mid-way H is not a calibrated H."""
        if not self.bands:
            return None
        alts = [b[0] for b in self.bands]
        i = bisect.bisect_left(alts, agl_m)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self.bands)]
        j = min(cands, key=lambda k: abs(alts[k] - agl_m))
        return self.bands[j][1], self.bands[j][2]


def register_thermal(
    bundle: FrameBundle,
    *,
    table: AltitudeHomographyTable | None = None,
    domain: str = "sim",
    max_residual_px: float = MAX_RESIDUAL_PX,
) -> Registration | None:
    """Choose the thermal->RGB map for this frame, or None when there is nothing to fuse.

    Returns None (not an invalid `Registration`) when the bundle carries no thermal partner at all: "there is no
    thermal frame" and "the thermal frame is misregistered" are different states and the caller reports which.
    """
    if bundle.thermal is None:
        return None
    ti = bundle.thermal_intrinsics
    if domain == "sim":
        if ti is None:
            # Same pose, same camera model, no thermal intrinsics published: the frames are already in register.
            return Registration(np.eye(3), 0.0, "identity", True)
        return Registration(homography_from_intrinsics(ti, bundle.intrinsics), 0.0, "intrinsics", True)
    if table is None or ti is None:
        return Registration(np.eye(3), math.inf, "identity", False,
                            "real footage needs a fitted per-altitude homography (SOLUTION_DOC 5.5)")
    hit = table.lookup(bundle.telemetry.agl_m)
    if hit is None:
        return Registration(np.eye(3), math.inf, "altitude_table", False, "no calibrated band for this altitude")
    H, residual = hit
    ok = residual <= max_residual_px
    return Registration(H, residual, "altitude_table", ok,
                        "" if ok else f"registration residual {residual:.1f} px > {max_residual_px} px")


# --- the thermal fusion weight -------------------------------------------------------------------------------
#: §5.5: the thermal weight is lowered at midday and during the §2.3 crossover windows, and raised at night.
_TOD_WEIGHT = {"night": 1.6, "dawn": 0.7, "dusk": 0.7, "day": 0.8, "midday": 0.5, "": 1.0, "all": 1.0}


def thermal_weight(
    time_of_day: str = "",
    *,
    radiometric_contrast_c: float | None = None,
    base: float = 1.0,
) -> float:
    """`w_t` for WBF. §5.5b: when radiometry exists it **replaces** the time-of-day heuristic with a measurement.

    `radiometric_contrast_c` is the median target-minus-background temperature difference over the frame's
    candidates. Below ~1 degC the thermal channel carries no evidence either way (crossover, immersion) and its
    weight collapses; above ~6 degC it is the strongest channel available.
    """
    if radiometric_contrast_c is not None:
        c = float(radiometric_contrast_c)
        return float(base * min(1.8, max(0.2, 0.2 + 0.3 * c)))
    return float(base * _TOD_WEIGHT.get(str(time_of_day).lower(), 1.0))


# --- fusion ----------------------------------------------------------------------------------------------------
@dataclass(slots=True)
class FusionResult:
    """What fusion produced and, just as importantly, whether thermal was used and why not."""

    detections: list[Detection]
    used_thermal: bool
    reason: str
    w_rgb: float = 1.0
    w_thermal: float = 0.0
    n_rgb_in: int = 0
    n_thermal_in: int = 0
    registration: Registration | None = None

    @property
    def is_rgb_only(self) -> bool:
        return not self.used_thermal


_CLASS_IDS = {"human": 0, "animal": 1}
_ID_CLASS = {v: k for k, v in _CLASS_IDS.items()}


def _normalise(boxes: Sequence[Sequence[float]], w: int, h: int) -> list[list[float]]:
    return [[min(max(b[0] / w, 0.0), 1.0), min(max(b[1] / h, 0.0), 1.0),
             min(max(b[2] / w, 0.0), 1.0), min(max(b[3] / h, 0.0), 1.0)] for b in boxes]


def fuse_detections(
    rgb_dets: Sequence[Detection],
    thermal_dets: Sequence[Detection] | None,
    *,
    frame_w: int,
    frame_h: int,
    registration: Registration | None = None,
    w_rgb: float = 1.0,
    w_thermal: float = 1.0,
    iou_thr: float = DEFAULT_IOU_THR,
    skip_box_thr: float = DEFAULT_SKIP_BOX_THR,
) -> FusionResult:
    """Late fusion of RGB and thermal boxes. **Thermal boxes must already be in RGB coordinates** -- pass the
    registration so this function can say *why* it fell back, and warp with :func:`warp_boxes` beforehand or let
    :func:`fuse_frame` do both.

    The fallback is the first thing in the function on purpose: when it fires, `result.detections` **is** the list
    of RGB `Detection` objects that came in, not a copy of them.
    """
    rgb_list = list(rgb_dets)
    if thermal_dets is None or len(thermal_dets) == 0:
        why = "no thermal frame" if thermal_dets is None else "thermal produced no boxes"
        return FusionResult(rgb_list, False, why, w_rgb, 0.0, len(rgb_list), 0, registration)
    if registration is not None and not registration.valid:
        return FusionResult(rgb_list, False, f"registration rejected: {registration.reject_reason}",
                            w_rgb, 0.0, len(rgb_list), len(thermal_dets), registration)
    if not rgb_list and not thermal_dets:
        return FusionResult([], False, "nothing to fuse", w_rgb, 0.0, 0, 0, registration)

    from ensemble_boxes import weighted_boxes_fusion  # numpy-only; imported here to keep import cost off ingest

    from vendor.proben import bayesian_fusion, proben_single

    th_list = list(thermal_dets)
    groups = [rgb_list, th_list]
    boxes_list = [_normalise([d.bbox_px for d in g], frame_w, frame_h) for g in groups]
    scores_list = [[float(d.score) for d in g] for g in groups]
    labels_list = [[_CLASS_IDS.get(d.cls, 0) for d in g] for g in groups]

    fb, _fs, fl = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, weights=[w_rgb, w_thermal],
        iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type="avg", allows_overflow=False,
    )

    out: list[Detection] = []
    for box_n, label in zip(np.asarray(fb, dtype=np.float64), np.asarray(fl)):
        box = (float(box_n[0] * frame_w), float(box_n[1] * frame_h),
               float(box_n[2] * frame_w), float(box_n[3] * frame_h))
        cls = _ID_CLASS.get(int(label), "human")
        # Re-derive cluster membership: WBF clusters by "IoU with the running fused box > iou_thr", so testing the
        # inputs against the final fused box reproduces the membership it used.
        members: list[list[Detection]] = []
        for g in groups:
            hits = [d for d in g if d.cls == cls and iou_xyxy(d.bbox_px, box) >= iou_thr]
            members.append(sorted(hits, key=lambda d: d.score, reverse=True))
        rgb_best = members[0][0] if members[0] else None
        th_best = members[1][0] if members[1] else None

        if rgb_best is not None and th_best is not None:
            score = bayesian_fusion([rgb_best.score, th_best.score])
            modality = "fused"
        elif rgb_best is not None:
            score = proben_single(rgb_best.score)   # marginalisation: keep the RGB posterior as it stands
            modality = "rgb"
        elif th_best is not None:
            score = proben_single(th_best.score)    # a thermal-only hot spot survives the night
            modality = "thermal"
        else:  # WBF produced a cluster whose members no longer reach iou_thr against the fused box
            continue

        base = rgb_best or th_best
        det = replace(base, bbox_px=box, score=float(score), cls=cls, modality=modality)  # type: ignore[arg-type]
        if th_best is not None:
            det.thermal_c = th_best.thermal_c if th_best.thermal_c is not None else det.thermal_c
            det.thermal_hot = det.thermal_hot or th_best.thermal_hot
        if rgb_best is not None:
            det.tile_idx = rgb_best.tile_idx
        out.append(det)

    out.sort(key=lambda d: d.score, reverse=True)
    return FusionResult(out, True, "fused", w_rgb, w_thermal, len(rgb_list), len(th_list), registration)


def fuse_frame(
    bundle: FrameBundle,
    rgb_dets: Sequence[Detection],
    thermal_dets: Sequence[Detection] | None,
    *,
    table: AltitudeHomographyTable | None = None,
    domain: str = "sim",
    w_rgb: float = 1.0,
    w_thermal: float | None = None,
    iou_thr: float = DEFAULT_IOU_THR,
    skip_box_thr: float = DEFAULT_SKIP_BOX_THR,
    max_residual_px: float = MAX_RESIDUAL_PX,
) -> FusionResult:
    """Register, warp the thermal boxes into RGB pixels, then fuse. The one call the pipeline makes per frame."""
    reg = register_thermal(bundle, table=table, domain=domain, max_residual_px=max_residual_px)
    if thermal_dets is None or reg is None or not reg.valid:
        return fuse_detections(rgb_dets, None if reg is None else thermal_dets, frame_w=bundle.intrinsics.width_px,
                               frame_h=bundle.intrinsics.height_px, registration=reg, w_rgb=w_rgb,
                               w_thermal=0.0, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    warped = warp_boxes(reg.H, [d.bbox_px for d in thermal_dets])
    mapped = [replace(d, bbox_px=(float(b[0]), float(b[1]), float(b[2]), float(b[3])))
              for d, b in zip(thermal_dets, warped)]
    if w_thermal is None:
        w_thermal = thermal_weight(bundle.telemetry.time_of_day)
    return fuse_detections(rgb_dets, mapped, frame_w=bundle.intrinsics.width_px,
                           frame_h=bundle.intrinsics.height_px, registration=reg, w_rgb=w_rgb,
                           w_thermal=w_thermal, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
