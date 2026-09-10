"""Synthetic detection sequences: a static survivor under a moving camera (SOLUTION_DOC §5.6).

This is **test scaffolding, not a stub of a shipped component**. It exists because the tracking lane must be
verifiable without the detector, without the simulator and without a GPU, and because the case §5.6 calls hard —
"survivors are mostly static and the camera moves" — is easy to generate exactly and impossible to find by
accident in a recorded clip. Everything it produces is labelled `domain = "sim"` by the evaluation harness.

What it gives you, all analytic and reproducible from a seed:

* a survivor pinned to a real lat/lon on the ground plane, never moving;
* a camera flying over it, whose `Telemetry` is the same pose the projection used, so the telemetry-predicted
  homography of `geometry.py` has something true to be checked against;
* boxes obtained by projecting the survivor's ground footprint through the camera — so pixel size, aspect and
  motion all follow from the geometry instead of being invented;
* the hard parts on demand: **dropouts** (frames with no detection at all), **confidence dips** (frames whose
  score falls into ByteTrack's second-stage band), **box jitter**, and **water** — either incoherent ripple that
  destroys optical flow, or a coherent surface drift that gives optical flow a confident wrong answer;
* `GeoFix` values with the §5.7 error structure: per-observation random noise **plus** a per-pass common-mode
  bias, because yaw and boresight biases do not average out over frames and dedup has to absorb them.

Default geometry is one native-resolution tile of a 4K frame (§5.5c "tile the 4K frame at native resolution"):
960 x 540 px at 30 deg HFOV, 45 m AGL. That puts a standing survivor at about 24 px and 40 px to the ground
metre, so 3 m/s of ground speed is 24 px of image motion per processed frame at 5 FPS — a target that moves its
own width every frame, which is precisely the case camera-motion compensation exists for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sightline.common import geodesy
from sightline.schemas import Detection, FrameBundle, GeoFix, Intrinsics, Telemetry
from sightline.track.geometry import camera_matrix, rotation_optical_to_ned


def tile_intrinsics(width_px: int = 960, height_px: int = 540, hfov_deg: float = 30.0) -> Intrinsics:
    """One native-resolution tile of a 4K frame, per §5.5c."""
    return Intrinsics.from_hfov(width_px, height_px, hfov_deg, source="sim")


@dataclass(slots=True)
class Survivor:
    """A living being that does not move. Position is metres north/east of the scenario origin."""

    name: str
    north_m: float
    east_m: float
    #: Ground footprint of the body as seen from above; 0.6 x 0.6 m is a standing person, 0.6 x 1.7 m prone.
    width_m: float = 0.6
    length_m: float = 0.6
    cls: str = "human"
    #: Base detection confidence when nothing else lowers it.
    score: float = 0.80

    def latlon(self, origin_lat: float, origin_lon: float) -> tuple[float, float]:
        return geodesy.offset_ne(origin_lat, origin_lon, self.north_m, self.east_m)


@dataclass(slots=True)
class WaterConfig:
    """How the water surface misbehaves under the camera (§5.6 rule 4).

    `mode`:
        "ripple" — the surface texture is regenerated every frame, so consecutive frames share no structure.
                   Lucas-Kanade either loses its correspondences or returns noise. The doc's literal case.
        "drift"  — the surface texture is coherent but translates with its own velocity. Optical flow locks onto
                   it and returns a confident, wrong camera motion. This is the failure an inlier count cannot
                   see, and the reason `SightlineCMC` cross-checks against telemetry instead.
    """

    mode: str = "ripple"
    drift_ne_ms: tuple[float, float] = (0.0, 0.0)
    #: Fraction of the frame covered by water, measured from the top edge. 1.0 = the whole frame.
    coverage: float = 1.0
    ripple_strength: float = 1.0


@dataclass(slots=True)
class SceneConfig:
    """The scenario: where it is, how the camera flies and what the sensor is."""

    origin_lat: float = 12.9000
    origin_lon: float = 77.6000
    #: Ground elevation above MSL. The camera's `alt_msl_m` is this plus `agl_m`.
    ground_alt_msl_m: float = 600.0
    agl_m: float = 45.0
    intrinsics: Intrinsics = field(default_factory=tile_intrinsics)
    fps: float = 5.0
    #: Where the camera starts, metres north/east of the origin.
    start_ne_m: tuple[float, float] = (-6.5, 0.0)
    #: Ground track direction (degrees from north) and speed.
    heading_deg: float = 0.0
    speed_ms: float = 3.0
    #: Gimbal pitch in the schema's convention: -90 is nadir, 0 the horizon.
    gimbal_pitch_deg: float = -90.0
    gimbal_yaw_deg: float = 0.0
    #: 1-sigma attitude wobble injected into the reported telemetry, degrees (§5.7 "vehicle pitch/roll 0.5").
    attitude_noise_deg: float = 0.0
    #: 1-sigma GNSS noise on the reported position, metres. Only affects telemetry, never the true projection,
    #: which is what makes the telemetry-predicted homography imperfect the way it is in the field.
    gnss_noise_m: float = 0.0
    t0_utc: float = 1_800_000_000.0

    def camera_ne(self, k: int) -> tuple[float, float]:
        d = self.speed_ms * k / self.fps
        return (
            self.start_ne_m[0] + d * math.cos(math.radians(self.heading_deg)),
            self.start_ne_m[1] + d * math.sin(math.radians(self.heading_deg)),
        )


@dataclass(slots=True)
class SequenceOptions:
    """Everything about one pass that is not the scene itself."""

    n_frames: int = 30
    #: Frame indices (0-based within the pass) with no detections at all.
    dropout_frames: tuple[int, ...] = ()
    #: Frame index -> detection score, for the frames whose confidence dips into the second-stage band.
    conf_dips: dict[int, float] = field(default_factory=dict)
    #: 1-sigma box-corner jitter in pixels (detector localisation noise; §5.7 assumes 5 px).
    box_jitter_px: float = 0.0
    #: 1-sigma per-observation geolocation error in metres (§5.7: 2.6 m at 60 m nadir, consumer GNSS).
    geo_sigma_m: float = 2.6
    #: Fixed per-pass offset in metres (north, east). Yaw and boresight biases do NOT average out over frames
    #: (§5.7), so two passes over the same survivor land at two slightly different places. Dedup must absorb it.
    geo_bias_ne_m: tuple[float, float] = (0.0, 0.0)
    #: `h_acc_m` published on every GeoFix. Defaults to `geo_sigma_m` so the stated 1-sigma is the true one.
    h_acc_m: float | None = None
    render: bool = False
    water: WaterConfig | None = None
    #: Extra spurious detections per frame (uniform in the frame). Exercises the confirmation gate.
    false_positives_per_frame: int = 0
    fp_score: float = 0.55
    pass_id: int = 0
    clip_id: str = "synth"
    seed: int = 0
    frame_idx_offset: int = 0
    t_offset_s: float = 0.0


@dataclass(slots=True)
class SyntheticFrame:
    """One processed frame plus everything an assertion could want to know about it."""

    bundle: FrameBundle
    detections: list[Detection]
    fixes: list[GeoFix]
    #: survivor name -> the true pixel centre this frame, for survivors inside the frame.
    truth_px: dict[str, tuple[float, float]]
    #: survivor name -> index into `detections`, for the detections that came from a real survivor.
    truth_det_index: dict[str, int]
    #: survivor name -> true (lat, lon).
    truth_latlon: dict[str, tuple[float, float]]


# --- telemetry ------------------------------------------------------------------------------------------
def _telemetry(scene: SceneConfig, k: int, rng: np.random.Generator, opts: SequenceOptions) -> tuple[Telemetry, Telemetry]:
    """Return (true pose used for projection, reported pose written into the bundle)."""
    north, east = scene.camera_ne(k)
    lat, lon = geodesy.offset_ne(scene.origin_lat, scene.origin_lon, north, east)
    q = geodesy.euler_to_quat(0.0, scene.gimbal_pitch_deg + 90.0, scene.gimbal_yaw_deg)
    t = scene.t0_utc + opts.t_offset_s + k / scene.fps
    truth = Telemetry(
        t_utc=t,
        lat=lat,
        lon=lon,
        alt_msl_m=scene.ground_alt_msl_m + scene.agl_m,
        agl_m=scene.agl_m,
        q_gimbal=q,
        gimbal_is_earth_referenced=True,
        ned_m=(north, east, -scene.agl_m),
        clip_id=opts.clip_id,
        frame_idx=opts.frame_idx_offset + k,
    )
    if scene.gnss_noise_m <= 0 and scene.attitude_noise_deg <= 0:
        return truth, truth

    dn, de = rng.normal(0.0, scene.gnss_noise_m, 2) if scene.gnss_noise_m > 0 else (0.0, 0.0)
    rlat, rlon = geodesy.offset_ne(lat, lon, float(dn), float(de))
    dr, dp, dy = (
        rng.normal(0.0, scene.attitude_noise_deg, 3) if scene.attitude_noise_deg > 0 else (0.0, 0.0, 0.0)
    )
    reported = Telemetry(
        t_utc=t,
        lat=rlat,
        lon=rlon,
        alt_msl_m=truth.alt_msl_m,
        agl_m=scene.agl_m,
        q_gimbal=geodesy.euler_to_quat(
            float(dr), scene.gimbal_pitch_deg + 90.0 + float(dp), scene.gimbal_yaw_deg + float(dy)
        ),
        gimbal_is_earth_referenced=True,
        ned_m=(north + float(dn), east + float(de), -scene.agl_m),
        h_acc_m=max(scene.gnss_noise_m, 0.5),
        clip_id=opts.clip_id,
        frame_idx=opts.frame_idx_offset + k,
        noise_injected=True,
    )
    return truth, reported


# --- projection -----------------------------------------------------------------------------------------
def _project_rel(tel: Telemetry, intr: Intrinsics, north_m: float, east_m: float) -> tuple[float, float] | None:
    """Project a ground point (metres north/east of the scenario origin) sitting on the ground plane.

    The scenario's NED origin is at ground level, so a ground point has down = 0 and the camera sits at
    `tel.ned_m` with a negative down component.
    """
    assert tel.ned_m is not None
    cn, ce, cd = tel.ned_m
    rel = np.array([north_m - cn, east_m - ce, 0.0 - cd], dtype=float)  # ground is down = 0 in scenario NED
    p = rotation_optical_to_ned(tel).T @ rel
    if p[2] <= 1e-9:
        return None
    return (intr.fx * p[0] / p[2] + intr.cx, intr.fy * p[1] / p[2] + intr.cy)


def _footprint_box(
    tel: Telemetry, intr: Intrinsics, s: Survivor
) -> tuple[float, float, float, float] | None:
    """Axis-aligned box around the projected ground footprint of a body. None if not fully in front."""
    hw, hl = s.width_m / 2.0, s.length_m / 2.0
    pts = []
    for dn in (-hl, hl):
        for de in (-hw, hw):
            p = _project_rel(tel, intr, s.north_m + dn, s.east_m + de)
            if p is None:
                return None
            pts.append(p)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _inside(box: tuple[float, float, float, float], intr: Intrinsics, margin: float = 0.0) -> bool:
    x1, y1, x2, y2 = box
    return x1 >= -margin and y1 >= -margin and x2 <= intr.width_px + margin and y2 <= intr.height_px + margin


# --- rendering ------------------------------------------------------------------------------------------
class GroundTexture:
    """A static, feature-rich ground patch in scenario NED metres, rendered per frame by exact homography.

    Optical flow needs corners, so the patch is blobs and rectangles over noise rather than smooth gradients.
    Because the warp uses the same pose the detections were projected with, any camera motion the CMC recovers
    from these frames is directly comparable with the analytic telemetry prediction.
    """

    def __init__(self, north_range: tuple[float, float], east_range: tuple[float, float], mpp: float, seed: int):
        import cv2

        self.mpp = mpp
        self.n0, self.n1 = north_range
        self.e0, self.e1 = east_range
        h = max(64, int((self.n1 - self.n0) / mpp))
        w = max(64, int((self.e1 - self.e0) / mpp))
        rng = np.random.default_rng(seed)
        img = (rng.normal(110, 18, (h, w))).clip(0, 255).astype(np.uint8)
        for _ in range(max(40, (h * w) // 4000)):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            r = int(rng.integers(3, max(5, min(h, w) // 25)))
            colour = int(rng.integers(25, 235))
            if rng.random() < 0.5:
                cv2.circle(img, (x, y), r, colour, -1)
            else:
                cv2.rectangle(img, (x, y), (x + r, y + int(r * float(rng.uniform(0.4, 2.0)))), colour, -1)
        self.img = img
        self.h, self.w = h, w

    def _tex_to_ned(self) -> np.ndarray:
        """(3, 3) mapping texture pixel (tx, ty, 1) -> (north, east, 1) in metres. North is up in the texture."""
        return np.array(
            [[0.0, -self.mpp, self.n1], [self.mpp, 0.0, self.e0], [0.0, 0.0, 1.0]], dtype=float
        )

    def render(self, tel: Telemetry, intr: Intrinsics, shift_ne_m: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Warp the patch into the camera. `shift_ne_m` moves the SURFACE (not the camera): the water drift."""
        import cv2

        assert tel.ned_m is not None
        cn, ce, cd = tel.ned_m
        t2n = self._tex_to_ned()
        # ground point relative to camera, as a linear map of (tx, ty, 1)
        a = np.array(
            [
                [t2n[0, 0], t2n[0, 1], t2n[0, 2] + shift_ne_m[0] - cn],
                [t2n[1, 0], t2n[1, 1], t2n[1, 2] + shift_ne_m[1] - ce],
                [0.0, 0.0, -cd],
            ],
            dtype=float,
        )
        h = camera_matrix(intr) @ rotation_optical_to_ned(tel).T @ a
        out = cv2.warpPerspective(
            self.img, h, (intr.width_px, intr.height_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
        )
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# --- the generator --------------------------------------------------------------------------------------
def iter_sequence(
    scene: SceneConfig,
    survivors: list[Survivor],
    opts: SequenceOptions | None = None,
):
    """Yield `SyntheticFrame`s one at a time. Use this when the frames are rendered: they are not small."""
    import cv2

    opts = opts or SequenceOptions()
    rng = np.random.default_rng(opts.seed)
    intr = scene.intrinsics
    h_acc = opts.h_acc_m if opts.h_acc_m is not None else max(opts.geo_sigma_m, 1e-3)

    texture = None
    water_texture = None
    if opts.render:
        ns = [scene.camera_ne(k)[0] for k in range(opts.n_frames)] + [s.north_m for s in survivors]
        es = [scene.camera_ne(k)[1] for k in range(opts.n_frames)] + [s.east_m for s in survivors]
        pad = 1.2 * scene.agl_m * math.tan(math.radians(intr.hfov_deg() / 2.0)) + 10.0
        mpp = scene.agl_m / max(intr.fx, intr.fy) / 2.0
        texture = GroundTexture((min(ns) - pad, max(ns) + pad), (min(es) - pad, max(es) + pad), mpp, opts.seed + 1)
        if opts.water is not None and opts.water.mode == "drift":
            water_texture = GroundTexture(
                (min(ns) - pad, max(ns) + pad), (min(es) - pad, max(es) + pad), mpp, opts.seed + 77
            )

    for k in range(opts.n_frames):
        truth_tel, reported_tel = _telemetry(scene, k, rng, opts)
        frame_idx = opts.frame_idx_offset + k
        t_utc = truth_tel.t_utc

        rgb = None
        if opts.render and texture is not None:
            rgb = texture.render(truth_tel, intr)
            water = opts.water
            if water is not None:
                if water.mode == "ripple":
                    speckle = rng.integers(0, 256, (intr.height_px, intr.width_px), dtype=np.uint8)
                    speckle = cv2.GaussianBlur(speckle, (5, 5), 0)
                    overlay = cv2.cvtColor(speckle, cv2.COLOR_GRAY2BGR)
                elif water.mode == "drift" and water_texture is not None:
                    dt = k / scene.fps
                    overlay = water_texture.render(
                        truth_tel, intr, shift_ne_m=(water.drift_ne_ms[0] * dt, water.drift_ne_ms[1] * dt)
                    )
                else:
                    overlay = None
                if overlay is not None:
                    rows = int(round(intr.height_px * min(1.0, max(0.0, water.coverage))))
                    if rows > 0:
                        alpha = min(1.0, max(0.0, water.ripple_strength))
                        rgb[:rows] = (alpha * overlay[:rows] + (1.0 - alpha) * rgb[:rows]).astype(np.uint8)

        bundle = FrameBundle(
            frame_idx=frame_idx,
            t_utc=t_utc,
            telemetry=reported_tel,
            intrinsics=intr,
            rgb=rgb,
            clip_id=opts.clip_id,
            source_path="synthetic://track",
        )

        detections: list[Detection] = []
        fixes: list[GeoFix] = []
        truth_px: dict[str, tuple[float, float]] = {}
        truth_det_index: dict[str, int] = {}
        truth_latlon: dict[str, tuple[float, float]] = {}

        dropped = k in opts.dropout_frames
        for s in survivors:
            lat, lon = s.latlon(scene.origin_lat, scene.origin_lon)
            truth_latlon[s.name] = (lat, lon)
            box = _footprint_box(truth_tel, intr, s)
            if box is None or not _inside(box, intr):
                continue
            truth_px[s.name] = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            if dropped:
                continue
            if opts.box_jitter_px > 0:
                j = rng.normal(0.0, opts.box_jitter_px, 4)
                box = (box[0] + j[0], box[1] + j[1], box[2] + j[2], box[3] + j[3])
            score = float(opts.conf_dips.get(k, s.score))
            truth_det_index[s.name] = len(detections)
            detections.append(
                Detection(
                    bbox_px=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=score,
                    cls=s.cls,  # type: ignore[arg-type]
                    modality="rgb",
                    frame_idx=frame_idx,
                )
            )
            dn, de = rng.normal(0.0, opts.geo_sigma_m, 2) if opts.geo_sigma_m > 0 else (0.0, 0.0)
            flat, flon = geodesy.offset_ne(
                lat, lon, float(dn) + opts.geo_bias_ne_m[0], float(de) + opts.geo_bias_ne_m[1]
            )
            fixes.append(
                GeoFix(
                    lat=flat,
                    lon=flon,
                    alt_msl_m=scene.ground_alt_msl_m,
                    h_acc_m=h_acc,
                    off_nadir_deg=abs(scene.gimbal_pitch_deg + 90.0),
                    method="flat_plane",
                    h_acc_basis="synthetic_budget",
                    agl_m=scene.agl_m,
                    slant_range_m=scene.agl_m,
                )
            )

        for _ in range(opts.false_positives_per_frame):
            cx = float(rng.uniform(20, intr.width_px - 20))
            cy = float(rng.uniform(20, intr.height_px - 20))
            half = float(rng.uniform(8, 18))
            detections.append(
                Detection(
                    bbox_px=(cx - half, cy - half, cx + half, cy + half),
                    score=opts.fp_score,
                    cls="human",
                    frame_idx=frame_idx,
                )
            )
            flat, flon = geodesy.offset_ne(reported_tel.lat, reported_tel.lon, float(rng.uniform(-8, 8)), float(rng.uniform(-8, 8)))
            fixes.append(
                GeoFix(
                    lat=flat,
                    lon=flon,
                    alt_msl_m=scene.ground_alt_msl_m,
                    h_acc_m=h_acc,
                    off_nadir_deg=abs(scene.gimbal_pitch_deg + 90.0),
                    agl_m=scene.agl_m,
                )
            )

        yield SyntheticFrame(bundle, detections, fixes, truth_px, truth_det_index, truth_latlon)


def make_sequence(
    scene: SceneConfig,
    survivors: list[Survivor],
    opts: SequenceOptions | None = None,
) -> list[SyntheticFrame]:
    """`iter_sequence` materialised. Fine without rendering; watch the memory when `opts.render` is on."""
    return list(iter_sequence(scene, survivors, opts))
