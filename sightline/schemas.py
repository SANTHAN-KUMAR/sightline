"""FROZEN CONTRACT: the data types every pipeline module exchanges (SOLUTION_DOC §5.4-§5.12).

This module is owned by the orchestrator. **Do not change a field name or its meaning** — several modules are
written in parallel against it. If a field is genuinely missing, add an OPTIONAL one with a default and say so in
`docs/CONTRACTS.md`; never repurpose or rename an existing field.

Design rules:
  * stdlib + numpy only, so every module (and the editor-side scripts) can import it with no heavy dependency;
  * plain dataclasses, not pydantic: they are cheap to build per frame and trivial to serialise;
  * SI units everywhere, `_m` / `_s` / `_deg` / `_px` suffixes; angles in degrees unless the name says `_rad`;
  * every timestamp is a UTC POSIX float (`t_utc`), never a naive local datetime;
  * geographic coordinates are WGS-84, ordered lon/lat only inside GeoJSON, lat/lon everywhere else.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = "1.3.0"

# --- vocabularies (closed sets; the evaluation harness slices on these) ------------------------------------
ClassName = Literal["human", "animal"]
Modality = Literal["rgb", "thermal", "fused"]
Posture = Literal["standing", "sitting", "prone", "supine", "half_submerged", "trapped", "waving", "unknown"]
Submersion = Literal["dry", "wet", "partial", "half", "head_only", "unknown"]
UrgencyClass = Literal["immersed", "trapped", "stranded", "animal", "unknown"]
RecordStatus = Literal["candidate", "confirmed", "stale", "dismissed"]
FlightMode = Literal["AUTO", "MANUAL", "HOLD", "RTL"]
Zone = Literal["fan", "settlement", "channel", "hillslope", "unknown"]

POSTURES: tuple[str, ...] = ("standing", "sitting", "prone", "supine", "half_submerged", "trapped",
                            "waving", "unknown")
SUBMERSIONS: tuple[str, ...] = ("dry", "wet", "partial", "half", "head_only", "unknown")
OCCLUSION_BINS: tuple[int, ...] = (0, 1, 2)  # 0 = <25 %, 1 = 25-75 %, 2 = >75 % (§6.3)


# --- 1. camera and telemetry (§5.4, §5.7) -----------------------------------------------------------------
@dataclass(slots=True)
class Intrinsics:
    """Pinhole + Brown-Conrady. `dist` is (k1, k2, p1, p2, k3) in OpenCV order; empty means "already rectified"."""

    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()
    source: str = "fov"  # "fov" | "dronemodels" | "calibration" | "sim"

    @classmethod
    def from_hfov(cls, width_px: int, height_px: int, hfov_deg: float, **kw: Any) -> "Intrinsics":
        """§5.7 step 1. Cosys-AirSim publishes a horizontal FOV, so this is the simulator's path."""
        f = (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width_px, height_px, f, f, width_px / 2.0, height_px / 2.0, **kw)

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.width_px / 2.0) / self.fx))


@dataclass(slots=True)
class Telemetry:
    """One pose sample. The ingest module (F7) emits these; geolocation (F13) consumes them.

    Attitudes are quaternions (w, x, y, z). `q_body` rotates body(FRD) -> NED. `q_gimbal` rotates
    camera(optical) -> NED when `gimbal_is_earth_referenced`, else camera -> body. The simulator always sets
    `gimbal_is_earth_referenced = True` because the camera pose is exact.
    """

    t_utc: float
    lat: float
    lon: float
    alt_msl_m: float
    agl_m: float
    q_body: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    q_gimbal: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    gimbal_is_earth_referenced: bool = True
    # NED position relative to the scenario origin; the simulator's native frame, None for real footage.
    ned_m: tuple[float, float, float] | None = None
    vel_ned_ms: tuple[float, float, float] = (0.0, 0.0, 0.0)
    h_acc_m: float = 2.5  # GNSS-reported horizontal accuracy (1σ), consumer default from §5.7
    v_acc_m: float = 1.0
    mode: FlightMode = "AUTO"  # F3 logs every switch; the evaluation harness attributes coverage to it
    clip_id: str = ""
    frame_idx: int = -1
    # scene attributes the evaluation harness slices on (§5.12); simulator-only, empty for real footage
    weather: dict[str, float] = field(default_factory=dict)  # rain, fog, wind_ms, cloud
    time_of_day: str = ""  # ISO-8601 local scene time
    flood_level_asl_m: float | None = None
    noise_injected: bool = False  # True once the §5.7 noise model has been applied

    def gimbal_pitch_deg(self) -> float:
        """Convenience for the nadir policy: -90 = straight down, 0 = horizon."""
        w, x, y, z = self.q_gimbal
        return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))) - 90.0


@dataclass(slots=True)
class FrameBundle:
    """What ingest hands to detection: one processed frame, its optional thermal partner and its pose.

    `rgb` / `thermal` are HxWx3 uint8 BGR and HxW uint16 (centi-kelvin) or HxW uint8 (AGC) arrays. They may be
    None when only metadata is being replayed (the evaluation harness does this).
    """

    frame_idx: int
    t_utc: float
    telemetry: Telemetry
    intrinsics: Intrinsics
    rgb: np.ndarray | None = None
    thermal: np.ndarray | None = None
    thermal_intrinsics: Intrinsics | None = None
    thermal_is_radiometric: bool = False  # True => `thermal` is centi-kelvin uint16 (§5.5b)
    clip_id: str = ""
    source_path: str = ""

    def thermal_celsius(self) -> np.ndarray | None:
        if self.thermal is None or not self.thermal_is_radiometric:
            return None
        return self.thermal.astype(np.float32) / 100.0 - 273.15


# --- 2. detection (§5.5, §5.5a) ---------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """One box in FULL-FRAME RGB pixel coordinates, xyxy. Thermal boxes are mapped through H before fusion."""

    bbox_px: tuple[float, float, float, float]
    score: float
    cls: ClassName = "human"
    modality: Modality = "rgb"
    tile_idx: int = -1  # provenance: which tile produced it (-1 = whole frame)
    frame_idx: int = -1
    # filled by the crop verifier (F10, §5.5a); `is_real` gates precision, the rest feed triage
    is_real: float | None = None
    posture: Posture = "unknown"
    posture_conf: float = 0.0
    submersion: Submersion = "unknown"
    submersion_conf: float = 0.0
    occlusion: int | None = None  # 0/1/2 per OCCLUSION_BINS
    # amodal box when known (simulator ground truth, §6.3); visible extent stays in `bbox_px`
    bbox_amodal_px: tuple[float, float, float, float] | None = None
    visible_fraction: float | None = None
    thermal_c: float | None = None  # median radiometric temperature inside the box (§5.5b)
    thermal_hot: bool = False

    @property
    def width_px(self) -> float:
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height_px(self) -> float:
        return self.bbox_px[3] - self.bbox_px[1]

    @property
    def size_px(self) -> float:
        """Longest side — the axis the §5.12 pixel-size slices are binned on."""
        return max(self.width_px, self.height_px)

    def centre_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_px
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def foot_px(self) -> tuple[float, float]:
        """Bottom-centre: the pixel geolocation projects (a standing body's ground contact)."""
        x1, _, x2, y2 = self.bbox_px
        return (x1 + x2) / 2.0, y2


# --- 3. geolocation (§5.7) --------------------------------------------------------------------------------
@dataclass(slots=True)
class GeoFix:
    """A pixel projected to the ground, with the error radius for THIS pixel's geometry."""

    lat: float
    lon: float
    alt_msl_m: float
    h_acc_m: float  # 1σ horizontal; CE90 = CE90_FACTOR * h_acc_m
    off_nadir_deg: float
    method: Literal["flat_plane", "dem", "water_plane"] = "flat_plane"
    h_acc_basis: str = "budget_v1"  # which error budget produced h_acc_m
    dem_source: str = ""
    agl_m: float = 0.0
    slant_range_m: float = 0.0
    valid: bool = True
    reject_reason: str = ""  # e.g. "near_horizon" when d_z <= 0.1 (§5.7 step 5)


CE90_FACTOR = 2.1460  # §5.7 "multiply by ~2.1 for CE90"; exact 2D Rayleigh 90th percentile = sqrt(-2 ln 0.1)


def ce90_m(h_acc_m: float) -> float:
    return CE90_FACTOR * h_acc_m


# --- 4. tracking (§5.6) -----------------------------------------------------------------------------------
@dataclass(slots=True)
class Observation:
    """One confirmed track's sighting in one frame: detection + where it landed on the ground."""

    track_id: int
    frame_idx: int
    t_utc: float
    det: Detection
    fix: GeoFix
    clip_id: str = ""
    pass_id: int = 0  # incremented per coverage pass; dedup reports `seen_in_passes`


@dataclass(slots=True)
class Track:
    """A confirmed frame-to-frame track (>= 3 hits within 2 s, §5.6 rule 3)."""

    track_id: int
    cls: ClassName
    observations: list[Observation] = field(default_factory=list)
    confirmed: bool = False
    clip_id: str = ""

    def best(self) -> Observation | None:
        """Highest-scoring observation — the one whose crop becomes the evidence thumbnail."""
        return max(self.observations, key=lambda o: o.det.score, default=None)

    def weighted_median_position(self) -> tuple[float, float]:
        """§5.6: per-track position is the confidence-weighted median, not the mean (outlier-robust)."""
        if not self.observations:
            raise ValueError("track has no observations")
        lats = np.array([o.fix.lat for o in self.observations], dtype=float)
        lons = np.array([o.fix.lon for o in self.observations], dtype=float)
        w = np.array([max(o.det.score, 1e-6) for o in self.observations], dtype=float)
        return float(_weighted_median(lats, w)), float(_weighted_median(lons, w))


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, c[-1] / 2.0)])


# --- 5. records (§5.8) ------------------------------------------------------------------------------------
@dataclass(slots=True)
class ScoreComponents:
    """Every term of the §5.8 priority score, stored so the UI can show WHY a record ranks where it does.

    `score = p_living * w_class * urgency * (1 + 0.1 * count_estimate)` — never displayed without its parts.
    """

    p_living: float = 0.0
    w_class: float = 1.0
    urgency: float = 1.0
    count_bonus: float = 1.0
    urgency_class: UrgencyClass = "unknown"
    elapsed_h: float = 0.0
    thermal_boost: float = 1.0
    motion_boost: float = 1.0
    posture_promoted: bool = False  # §5.5a: posture may RAISE urgency, never lower it

    def total(self) -> float:
        return self.p_living * self.w_class * self.urgency * self.count_bonus


@dataclass(slots=True)
class Evidence:
    thumb_uri: str
    clip_id: str
    frame_idx: int
    frame_time_utc: float
    bbox_px: tuple[float, float, float, float]
    det_conf: float
    camera: str = "rgb"


@dataclass(slots=True)
class Record:
    """One living being. The GeoJSON Feature of §5.8; `to_feature()` is the wire format.

    GUARDRAIL R10: there is no delete. `status` may become "dismissed" WITH a reason; the record stays.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: int = -1
    status: RecordStatus = "candidate"
    cls: ClassName = "human"
    lat: float = 0.0
    lon: float = 0.0
    alt_msl_m: float = 0.0
    h_acc_m: float = 3.0
    h_acc_basis: str = "budget_v1"
    method: str = "flat_plane"
    dem_source: str = ""
    agl_m: float = 0.0
    off_nadir_deg: float = 0.0
    confidence: float = 0.0  # 1 - prod(1 - conf_track)
    confidence_max_det: float = 0.0
    score: float = 0.0
    priority_rank: int = -1
    components: ScoreComponents = field(default_factory=ScoreComponents)
    n_observations: int = 0
    n_tracks_merged: int = 0
    seen_in_passes: list[int] = field(default_factory=list)
    first_seen_utc: float = 0.0
    last_seen_utc: float = 0.0
    motion_state: Literal["still", "moving", "unknown"] = "unknown"
    motion_displacement_m: float = 0.0
    motion_window_s: float = 0.0
    count_estimate: int = 1
    count_min: int = 1
    count_max: int = 1
    count_basis: str = "max_simultaneous_tracks"
    posture: Posture = "unknown"
    posture_conf: float = 0.0
    submersion: Submersion = "unknown"
    submersion_conf: float = 0.0
    occlusion: int | None = None
    modality: Modality = "rgb"
    thermal_hot: bool = False
    thermal_c: float | None = None
    pixel_size_px: float = 0.0
    gsd_cm_px: float = 0.0
    zone: Zone = "unknown"
    evidence: list[Evidence] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)  # platform, telemetry, sim, aoi_id
    notes: str = ""
    dismissed_reason: str = ""  # R10: dismissal always carries a reason
    dismissed_by: str = ""
    dismissed_utc: float = 0.0
    schema_version: str = SCHEMA_VERSION
    version: int = 1  # bumped on every update; the outbox key is (clip_id, record_id, version)

    def to_feature(self) -> dict[str, Any]:
        """RFC 7946 Feature. Coordinates are [lon, lat, alt] at 6 decimals (§5.8)."""
        props = asdict(self)
        props.pop("lat"), props.pop("lon"), props.pop("alt_msl_m")
        props["score_components"] = props.pop("components")
        return {
            "type": "Feature",
            "id": self.record_id,
            "geometry": {
                "type": "Point",
                "coordinates": [round(self.lon, 6), round(self.lat, 6), round(self.alt_msl_m, 2)],
            },
            "properties": props,
        }


def feature_collection(records: list[Record]) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": time.time(),
        "features": [r.to_feature() for r in records],
    }


# --- 6. coverage / search quality (§5.3, §5.3b, §5.16) ----------------------------------------------------
#: The presentations a coverage layer can be computed for (§5.3b). "body" and "limb_only" are the MVP pair.
PRESENTATIONS: tuple[str, ...] = ("body", "prone", "cluster", "upright", "wading", "head_only", "limb_only", "buried")


@dataclass(slots=True)
class CoverageGrid:
    """Per-cell accumulated search effort and probability of detection, POD = 1 - exp(-k * C) (§5.3).

    Arrays are (n_north, n_east) float32 in the scenario's local ENU metres, origin at the grid's SW corner.
    `pod` is derived, never stored independently — `recompute_pod()` is the only writer.
    """

    origin_lat: float
    origin_lon: float
    cell_m: float
    n_north: int
    n_east: int
    coverage: np.ndarray  # C: dimensionless swept-area ratio, accumulated over passes
    pod: np.ndarray
    presentation: str = "body"
    k: float = 1.0  # Koopman random-search constant; calibrated per presentation (§5.3)
    cannot_clear: np.ndarray | None = None  # bool mask: burial polygons, "aerial search cannot clear" (R10)

    @classmethod
    def empty(cls, origin_lat: float, origin_lon: float, cell_m: float, n_north: int, n_east: int, **kw: Any):
        z = np.zeros((n_north, n_east), dtype=np.float32)
        return cls(origin_lat, origin_lon, cell_m, n_north, n_east, z.copy(), z.copy(), **kw)

    def recompute_pod(self) -> None:
        self.pod = (1.0 - np.exp(-self.k * self.coverage)).astype(np.float32)


# --- 7. evaluation (§5.12) --------------------------------------------------------------------------------
# `frozen=True` is load-bearing, not tidiness. `MetricRow` was frozen to stop `row.slice = SliceKey("real")`
# relabelling a simulation number as a real-world one in place (hard rule 5, the project's hardest reporting
# rule). That freeze alone did NOT close the hole: `row.slice.domain = "real"` reached straight through the
# frozen row into a mutable key and relabelled it anyway, in one character. Measured before this change:
#     row = metric_row("recall@iou0.5", 0.94, make_slice("sim"), 500)
#     row.slice.domain = "real"   ->  succeeded, str(row) == 'recall@iou0.5=0.94 [domain=real] n=500'
# A frozen row wrapping a mutable key is a lock on the door of an open window.
@dataclass(frozen=True, slots=True)
class SliceKey:
    """The slice grid every accuracy number must carry (§5.12 + hard rule 5).

    `domain` is NEVER optional: a sim number and a real number may not be averaged together.
    """

    domain: Literal["sim", "real"]
    zone: Zone = "unknown"
    altitude_band: str = "all"  # "30-45", "45-60", "60-90", "90+"
    time_of_day: str = "all"  # "dawn", "day", "dusk", "night"
    occlusion: str = "all"  # "0", "1", "2"
    posture: str = "all"
    pixel_size: str = "all"  # "<20", "20-40", "40-80", "80+"
    modality: str = "all"
    #: The terrain the box sits on, MEASURED from the placed scene by `sightline.eval.context` — never the
    #: survivor's zone label. 5.12 requires FP/min per terrain type, and a false positive has no ground-truth
    #: box to inherit a context from, so this has to live on the slice rather than on `GtBox`.
    context: str = "all"  # "open_ground", "water", "debris", "vegetation", "structure", "vehicle"

    def label(self) -> str:
        parts = [f"domain={self.domain}"]
        parts += [f"{k}={v}" for k, v in asdict(self).items() if k != "domain" and v not in ("all", "unknown")]
        return " ".join(parts)


#: Wording used when a row is an explicit "no sample, no value" report. Kept next to `MetricRow` so producers
#: in three different lanes phrase the same fact the same way and a reader can grep for it.
UNDEFINED_KEY = "undefined_reason"


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One measured number with its slice and sample size. Printing this without `slice.domain` is a bug.

    Two invariants are enforced in `__post_init__` — in the TYPE, not in a helper — because the helper was
    exactly what leaked.

    **1. `value` is finite.** `sightline.eval.slicing.metric_row()` has always refused a NaN, but it is only one
    of three constructors: `dedup/metrics.py` and `detect/threshold.py` build `MetricRow` straight from this
    module. AUDIT S8 measured the consequence on the "we found nothing" dedup case — the one you most need to
    report honestly — and got `dedup.mean_position_error_m = nan` and `dedup.ce90_m = nan` (plus
    `dedup.duplicate_rate = inf` when records exist but none match). `json.dumps` then wrote a bare `NaN`, which
    RFC 8259 does not allow, and a strict re-parse of the metrics blob raised. `sightline/export/geojson.py`
    already refuses non-finite coordinates for that reason; this is the same rule one layer up. Putting the check
    on the dataclass rather than in each producer is the point: the defect WAS a producer bypassing the shared
    helper, so a fourth helper would not have caught the fifth producer, and a lane written next month inherits
    the guard for free.

    **2. An empty result stays reportable.** A statistic with no samples has no value. The fix is neither to
    invent a number nor to drop the row — dropping it hides precisely the case `docs/QUALITY_GATE.md` most wants
    on the record. `MetricRow.undefined()` / `MetricRow.finite_or_undefined()` build an `n = 0` row carrying
    `undefined_reason`: `__str__` prints "undefined (reason)" instead of a number, and `value` is pinned to a
    neutral `0.0` so the row still survives `json.dumps(..., allow_nan=False)`. `undefined_reason` requires
    `n == 0`, which keeps that placeholder out of `slicing.combine_rows`' weighted mean (it weights by `n`); an
    undefined value over a NON-empty sample is a broken computation, and raises instead.

    `frozen=True` closes the other half of S8: a row is a *reported* number, and `row.slice = SliceKey("real")`
    relabelled a simulation number as a real-world one in place — the one thing hard rule 5 exists to prevent.
    Nothing in the tree ever mutated a row, so it cost nothing; build a variant with `dataclasses.replace`.
    `detail` is a dict and therefore still mutable: the freeze is shallow, by design, so the existing
    `dict(detail)` idiom at every call site keeps working.
    """

    name: str
    value: float
    slice: SliceKey
    n: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    #: Non-empty => this row is an explicit "no sample, so no value" report, NOT a measurement that came out 0.
    undefined_reason: str = ""

    def __post_init__(self) -> None:
        # frozen dataclass: a validator normalises its own fields through object.__setattr__.
        set_ = object.__setattr__
        n = int(self.n)
        set_(self, "n", n)
        # Copy, never adopt. `detail.setdefault(UNDEFINED_KEY, reason)` below otherwise writes into the dict
        # the CALLER still holds: `d = {"k": 1}; MetricRow(..., detail=d, undefined_reason="none")` left
        # `d == {"k": 1, "undefined_reason": "none"}`. Every in-tree call site happens to pass `dict(detail)`,
        # so this was latent rather than live — which is exactly the kind of thing that stops being latent.
        detail = dict(self.detail)
        reason = str(self.undefined_reason or detail.get(UNDEFINED_KEY) or "")
        where = f"{self.name!r} [domain={getattr(self.slice, 'domain', '?')}]"

        if reason:
            # `MetricSet.to_dicts()` (eval lane) carries `detail` but not this field, so mirror the reason both
            # ways: a row that survives to_dicts -> from_dicts stays flagged instead of coming back as a
            # measured 0.0. This also matches the eval lane's existing "undefined: ..." note idiom, so
            # `"undefined" in str(row.detail)` holds for rows from every lane.
            detail.setdefault(UNDEFINED_KEY, reason)
            set_(self, "detail", detail)
            set_(self, "undefined_reason", reason)
            v = float(self.value)
            if n != 0:
                raise ValueError(
                    f"metric {where} is marked undefined ({reason!r}) but reports n={n} with value={v!r}. A "
                    "value that is undefined over a NON-EMPTY sample is a broken computation, not an honest "
                    "empty result: fix the computation, or report the n it was actually measured over."
                )
            if math.isfinite(v) and v != 0.0:
                raise ValueError(
                    f"metric {where} carries a measured value {v!r} AND undefined_reason {reason!r}. "
                    "A row is one or the other."
                )
            set_(self, "value", 0.0)  # neutral placeholder; __str__ never prints it, JSON stays RFC 8259 valid
            return

        v = float(self.value)
        if not math.isfinite(v):
            raise ValueError(
                f"metric {where} is {v} (n={n}); report an explicit n=0 row with a note instead of a NaN or an "
                "infinity — MetricRow.undefined(name, slice, reason) or "
                "MetricRow.finite_or_undefined(name, value, slice, n, reason=...). RFC 8259 has no NaN, so this "
                "row would serialise to JSON that no strict parser will read back (AUDIT S8)."
            )
        set_(self, "value", v)

    @property
    def is_defined(self) -> bool:
        """False when this row is an honest "no sample" report. `value` is a placeholder, not a measurement."""
        return not self.undefined_reason

    @classmethod
    def undefined(cls, name: str, slice_key: SliceKey, reason: str,
                  detail: dict[str, Any] | None = None) -> MetricRow:
        """The honest n=0 row: this metric has no sample, and `reason` says why. `value` is a 0.0 placeholder."""
        if not reason:
            raise ValueError(f"metric {name!r}: an undefined row must say WHY the value does not exist")
        return cls(name=name, value=0.0, slice=slice_key, n=0, detail=dict(detail or {}), undefined_reason=reason)

    @classmethod
    def finite_or_undefined(cls, name: str, value: float, slice_key: SliceKey, n: int = 0, *, reason: str,
                            detail: dict[str, Any] | None = None) -> MetricRow:
        """Measured when `value` is finite; the honest n=0 row carrying `reason` when it is not.

        For producers whose metric is a ratio or a mean that genuinely has no value on an empty sample. If the
        value is non-finite while `n > 0` this raises, which is correct: that is arithmetic gone wrong, not an
        empty result, and it must not be laundered into a tidy "undefined" row.
        """
        v = float(value)
        if math.isfinite(v):
            return cls(name=name, value=v, slice=slice_key, n=int(n), detail=dict(detail or {}))
        return cls(name=name, value=0.0, slice=slice_key, n=int(n), detail=dict(detail or {}),
                   undefined_reason=reason)

    def __str__(self) -> str:
        if self.undefined_reason:
            return f"{self.name}=undefined ({self.undefined_reason}) [{self.slice.label()}] n={self.n}"
        return f"{self.name}={self.value:.4g} [{self.slice.label()}] n={self.n}"
