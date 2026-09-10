"""Simulator export readers (SOLUTION_DOC §5.4 row 1) — **the primary path; the demo runs on this**.

Two readers:

* `SimExportReader` — the project's own `sightline-sim-capture` v1.0 format (`spec.py`). Everything the
  pipeline needs is in the file: pose, gimbal, intrinsics, weather, flood level, flight mode, thermal partner.
* `AirSimRecReader` — Cosys-AirSim's built-in `airsim_rec.txt`, read straight from
  `Plugins/AirSim/Source/Recording/RecordingFile.cpp` (verified in this repo, not guessed)::

      VehicleName<TAB>TimeStamp<TAB>POS_X<TAB>POS_Y<TAB>POS_Z<TAB>Q_W<TAB>Q_X<TAB>Q_Y<TAB>Q_Z<TAB>ImageFile

  `TimeStamp` is `clock()->nowNanos()/1e6`, i.e. **milliseconds on the AirSim clock** — under `SteppableClock`
  that is not wall time. `ImageFile` holds one or more names separated by `;`, formatted
  `img_<vehicle>_<camera>_<imagetype>[_<annotation>]_<nanos>.<png|pfm|ppm>`. The file carries **no camera
  orientation, no GPS and no intrinsics**, so those must come from the AirSim settings profile; that gap is
  the reason the project defines its own export format rather than reusing this one.

Also here: `inject_noise()`, the §5.7 noise model. Ground-truth sim pose gives ~0 geolocation error, which
would tune the dedup radius wrong for real footage, so the exported telemetry is perturbed on the way in and
the clean truth is kept beside it for evaluation.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from sightline.common.geodesy import euler_to_quat, offset_ne, quat_to_euler
from sightline.ingest import spec
from sightline.ingest.align import TelemetrySeries
from sightline.ingest.decimate import FrameIndex, FrameRef
from sightline.ingest.spec import (
    AIRSIM_REC,
    FRAMES_JSONL,
    MANIFEST_NAME,
    CaptureFormatError,
    CaptureManifest,
    CameraSpec,
    SimFrameRow,
    gimbal_quat_from_frd_quat,
)
from sightline.schemas import Intrinsics, Telemetry

__all__ = [
    "SimExportReader",
    "AirSimRecReader",
    "AirSimRecRow",
    "NoiseModel",
    "inject_noise",
    "is_sim_export",
    "is_airsim_rec",
]

_AIRSIM_HEADER = ("VehicleName", "TimeStamp", "POS_X", "POS_Y", "POS_Z", "Q_W", "Q_X", "Q_Y", "Q_Z", "ImageFile")
#: `img_<vehicle>_<camera>_<imagetype>[_<annotation>]_<nanos>.<ext>` (RecordingFile.cpp)
_IMG_NAME = re.compile(
    r"^img_(?P<vehicle>.+?)_(?P<camera>.+?)_(?P<itype>\d+)(?:_(?P<annot>.+?))?_(?P<nanos>\d+)\.(?P<ext>\w+)$"
)
#: AirSim `ImageType` enum -> the role this project uses it for.
AIRSIM_IMAGE_TYPES = {0: "scene", 1: "depth_planar", 2: "depth_perspective", 3: "depth_vis", 4: "disparity",
                      5: "segmentation", 6: "surface_normals", 7: "infrared", 8: "annotation"}


def is_sim_export(path: str | os.PathLike[str]) -> bool:
    p = Path(path)
    return (p / MANIFEST_NAME).is_file() if p.is_dir() else p.name == MANIFEST_NAME


def is_airsim_rec(path: str | os.PathLike[str]) -> bool:
    p = Path(path)
    if p.is_dir():
        p = p / AIRSIM_REC
    if not p.is_file():
        return False
    with open(p, "r", encoding="utf-8-sig", errors="replace") as fh:
        return fh.readline().strip().split("\t")[:3] == list(_AIRSIM_HEADER[:3])


# --- 1. the project's own export ---------------------------------------------------------------------------
class SimExportReader:
    """Read a `sightline-sim-capture` v1.0 clip directory. Streams; the whole clip never lives in RAM."""

    def __init__(self, clip_dir: str | os.PathLike[str], *, validate: bool = True) -> None:
        self.root = Path(clip_dir)
        if self.root.is_file() and self.root.name == MANIFEST_NAME:
            self.root = self.root.parent
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise CaptureFormatError(f"{self.root}: no {MANIFEST_NAME} (not a sightline-sim-capture clip)")
        self.manifest = CaptureManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        self._rows: list[SimFrameRow] | None = None
        if validate:
            missing = spec.validate_header(self._header())
            if missing:
                raise CaptureFormatError(
                    f"{self.root}: frames table is missing required column(s): {', '.join(missing)}. "
                    f"See sightline/ingest/spec.py (format {spec.FORMAT_NAME} v{spec.FORMAT_VERSION})."
                )

    # -- raw rows ----------------------------------------------------------------------------------------
    def _frames_path(self) -> Path:
        csv_path = self.root / self.manifest.frames_csv
        if csv_path.is_file():
            return csv_path
        jsonl = self.root / FRAMES_JSONL
        if jsonl.is_file():
            return jsonl
        raise CaptureFormatError(f"{self.root}: neither {self.manifest.frames_csv} nor {FRAMES_JSONL} exists")

    def _iter_raw(self) -> Iterator[SimFrameRow]:
        path = self._frames_path()
        it = spec.iter_jsonl_rows(path) if path.suffix == ".jsonl" else spec.iter_csv_rows(path)
        yield from it

    def _header(self) -> list[str]:
        for row in self._iter_raw():
            return list(row.values) + list(row.extra)
        raise CaptureFormatError(f"{self.root}: the frames table has no data rows")

    @property
    def rows(self) -> list[SimFrameRow]:
        if self._rows is None:
            self._rows = list(self._iter_raw())
        return self._rows

    @property
    def clip_id(self) -> str:
        return self.manifest.clip_id or self.root.name

    # -- intrinsics --------------------------------------------------------------------------------------
    def intrinsics(self, row: SimFrameRow | None = None) -> Intrinsics:
        """Per-frame intrinsics: explicit fx/fy/cx/cy when the capture wrote them, else HFOV (§5.7 step 1)."""
        cam = self.manifest.camera
        w = int(row.get("img_w_px", cam.width_px)) if row else cam.width_px
        h = int(row.get("img_h_px", cam.height_px)) if row else cam.height_px
        dist = tuple(float(row.get(f"dist_{k}", 0.0)) for k in ("k1", "k2", "p1", "p2", "k3")) if row else ()
        if row is not None and not any(dist):
            dist = tuple(cam.dist)
        if row is not None and row.has("fx_px", "fy_px", "cx_px", "cy_px"):
            return Intrinsics(w, h, float(row["fx_px"]), float(row["fy_px"]), float(row["cx_px"]),
                              float(row["cy_px"]), dist=dist, source="calibration")
        hfov = float(row.get("hfov_deg", cam.hfov_deg)) if row else cam.hfov_deg
        return Intrinsics.from_hfov(w, h, hfov, dist=dist, source=cam.source or "sim")

    def thermal_intrinsics(self, row: SimFrameRow | None = None) -> Intrinsics | None:
        cam = self.manifest.thermal_camera
        if cam is None and (row is None or not row.has("thermal_w_px", "thermal_h_px", "thermal_hfov_deg")):
            return None
        base = cam or CameraSpec(name="thermal")
        w = int(row.get("thermal_w_px", base.width_px)) if row else base.width_px
        h = int(row.get("thermal_h_px", base.height_px)) if row else base.height_px
        hfov = float(row.get("thermal_hfov_deg", base.hfov_deg)) if row else base.hfov_deg
        if not (w and h and hfov):
            return None
        return Intrinsics.from_hfov(w, h, hfov, dist=tuple(base.dist), source="sim")

    def thermal_is_radiometric(self, row: SimFrameRow | None = None) -> bool:
        if row is not None and row.values.get("thermal_radiometric") is not None:
            return bool(row["thermal_radiometric"])
        return bool(self.manifest.thermal_camera and self.manifest.thermal_camera.radiometric)

    # -- telemetry ---------------------------------------------------------------------------------------
    def telemetry(self, row: SimFrameRow) -> Telemetry:
        """One row -> `Telemetry`, with every documented fallback applied."""
        q_body = self._quat(row, "q_body", ("body_roll_deg", "body_pitch_deg", "body_yaw_deg"), gimbal=False)
        q_gimbal = self._quat(row, "q_gimbal", ("gimbal_roll_deg", "gimbal_pitch_deg", "gimbal_yaw_deg"),
                              gimbal=True)
        weather = {k: float(row[k]) for k in spec.WEATHER_KEYS if row.values.get(k) is not None}
        flood = row.values.get("flood_level_asl_m")
        if flood is None:
            flood = self.manifest.flood_level_asl_m
        ned = None
        if row.has("ned_n_m", "ned_e_m", "ned_d_m"):
            ned = (float(row["ned_n_m"]), float(row["ned_e_m"]), float(row["ned_d_m"]))
        return Telemetry(
            t_utc=float(row["t_utc"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            alt_msl_m=float(row["alt_msl_m"]),
            agl_m=float(row["agl_m"]),
            q_body=q_body,
            q_gimbal=q_gimbal,
            gimbal_is_earth_referenced=True,   # the simulator's camera pose is exact (schema docstring)
            ned_m=ned,
            vel_ned_ms=(float(row.get("vel_n_ms", 0.0)), float(row.get("vel_e_ms", 0.0)),
                        float(row.get("vel_d_ms", 0.0))),
            h_acc_m=float(row.get("h_acc_m", 0.0)),
            v_acc_m=float(row.get("v_acc_m", 0.0)),
            mode=str(row.get("mode", "AUTO")),  # type: ignore[arg-type]
            clip_id=self.clip_id,
            frame_idx=int(row["frame_idx"]),
            weather=weather,
            time_of_day=str(row.get("time_of_day", "")),
            flood_level_asl_m=(None if flood is None else float(flood)),
            noise_injected=bool(row.get("noise_injected", False)),
        )

    @staticmethod
    def _quat(row: SimFrameRow, prefix: str, euler_cols: tuple[str, str, str],
              *, gimbal: bool) -> tuple[float, float, float, float]:
        cols = tuple(f"{prefix}_{a}" for a in "wxyz")
        if row.has(*cols):
            q = tuple(float(row[c]) for c in cols)
            n = math.sqrt(sum(v * v for v in q))
            return tuple(v / n for v in q)  # type: ignore[return-value]
        if row.has(*euler_cols):
            roll, pitch, yaw = (float(row[c]) for c in euler_cols)
            return spec.gimbal_quat_from_euler(roll, pitch, yaw) if gimbal else euler_to_quat(roll, pitch, yaw)
        if gimbal:  # a capture with no gimbal columns at all: nadir, the settings default (Pitch -90)
            return spec.gimbal_quat_from_euler(0.0, -90.0, 0.0)
        return (1.0, 0.0, 0.0, 0.0)

    def telemetry_series(self) -> TelemetrySeries:
        return TelemetrySeries.from_telemetry(
            [self.telemetry(r) for r in self.rows], clip_id=self.clip_id, t_offset_s=self.manifest.t_offset_s
        )

    # -- frames ------------------------------------------------------------------------------------------
    def frame_source(self, row: SimFrameRow) -> tuple[str, str, int]:
        """Resolve `rgb_path` into `(image_path, video_path, video_index)` (see `spec.REQUIRED_COLUMNS`)."""
        raw = str(row.get("rgb_path", "") or "")
        if "#" in raw:
            name, _, idx = raw.rpartition("#")
            return "", str(self.root / name), int(idx)
        if self.manifest.video and (raw == "" or raw.isdigit()):
            idx = int(raw) if raw.isdigit() else int(row["frame_idx"])
            return "", str(self.root / self.manifest.video), idx
        return str(self.root / raw), "", -1

    def frame_index(self) -> FrameIndex:
        """Index EVERY captured frame (§5.4: skipped frames stay reachable for evidence thumbnails)."""
        index = FrameIndex(self.clip_id)
        for row in self.rows:
            img, video, vidx = self.frame_source(row)
            thermal = row.get("thermal_path", "") or ""
            aux = {k[:-5]: str(self.root / row[k]) for k in ("seg_path", "depth_path", "annotation_path")
                   if row.values.get(k)}
            index.add(FrameRef(
                frame_idx=int(row["frame_idx"]),
                t_utc=float(row["t_utc"]),
                pts_s=float(row.get("t_sim_s", 0.0)),
                path=img,
                video_path=video,
                video_index=vidx,
                thermal_path=str(self.root / thermal) if thermal else "",
                aux=aux,
            ))
        return index


# --- 2. Cosys-AirSim's own recording -----------------------------------------------------------------------
@dataclass(slots=True)
class AirSimRecRow:
    """One `airsim_rec.txt` line. `t_sim_s` is on the AIRSIM clock, not wall time (SteppableClock!)."""

    vehicle: str
    t_sim_s: float
    pos_ned_m: tuple[float, float, float]
    q_body: tuple[float, float, float, float]
    images: dict[str, str] = field(default_factory=dict)   # role ("scene", "segmentation", ...) -> filename
    line_no: int = -1


class AirSimRecReader:
    """Read `airsim_rec.txt`. Needs an `OriginGeopoint` and an HFOV, because the file carries neither.

    The gimbal attitude is not recorded either: with `gimbal_pitch_deg` left at its default the camera is
    assumed to be at the settings' `-90` nadir with the vehicle's yaw, which is true for
    `sim/settings/capture_4k.json` (`Gimbal.Stabilization = 1.0`, `Pitch = -90`) but is an ASSUMPTION and is
    reported as one. Prefer the `sightline-sim-capture` format, which records the real camera pose.
    """

    def __init__(self, path: str | os.PathLike[str], *, origin_lat: float, origin_lon: float,
                 origin_alt_msl_m: float, hfov_deg: float, width_px: int, height_px: int,
                 clip_id: str = "", capture_start_utc: float = 0.0, vehicle: str = "",
                 gimbal_pitch_deg: float = -90.0, gimbal_stabilised: bool = True,
                 ground_alt_msl_m: float | None = None) -> None:
        p = Path(path)
        self.path = p / AIRSIM_REC if p.is_dir() else p
        self.image_dir = self.path.parent / "images"
        self.origin = (float(origin_lat), float(origin_lon), float(origin_alt_msl_m))
        self.intrinsics = Intrinsics.from_hfov(width_px, height_px, hfov_deg, source="sim")
        self.clip_id = clip_id or self.path.parent.name
        self.capture_start_utc = float(capture_start_utc)
        self.vehicle = vehicle
        self.gimbal_pitch_deg = gimbal_pitch_deg
        self.gimbal_stabilised = gimbal_stabilised
        self.ground_alt_msl_m = ground_alt_msl_m
        self.assumptions: list[str] = [
            "airsim_rec.txt carries no camera orientation: the gimbal is assumed to be at "
            f"{gimbal_pitch_deg:.1f} deg pitch"
            + (" (earth-stabilised, yaw follows the vehicle)" if gimbal_stabilised else " in the body frame"),
            "airsim_rec.txt carries no GPS: lat/lon are derived from NED and the settings OriginGeopoint",
            "airsim_rec.txt carries no intrinsics: they come from the settings CaptureSettings FOV_Degrees",
        ]
        if capture_start_utc == 0.0:
            self.assumptions.append(
                "capture_start_utc was not supplied: t_utc equals the raw AirSim clock, which under "
                "SteppableClock is NOT wall time. Alignment with any real clock will be wrong."
            )
        if ground_alt_msl_m is None:
            self.assumptions.append("no ground altitude supplied: agl_m is measured against the OriginGeopoint")

    def rows(self) -> list[AirSimRecRow]:
        out: list[AirSimRecRow] = []
        with open(self.path, "r", encoding="utf-8-sig", errors="replace") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            if header[:9] != list(_AIRSIM_HEADER[:9]):
                raise CaptureFormatError(f"{self.path}: unexpected header {header!r}")
            for line_no, line in enumerate(fh, start=2):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                images: dict[str, str] = {}
                if len(parts) > 9:
                    for name in parts[9].split(";"):
                        name = name.strip()
                        if not name:
                            continue
                        m = _IMG_NAME.match(name)
                        role = AIRSIM_IMAGE_TYPES.get(int(m.group("itype")), "unknown") if m else "unknown"
                        if m and m.group("annot"):
                            role = f"annotation:{m.group('annot')}"
                        images[role] = name
                out.append(AirSimRecRow(
                    vehicle=parts[0],
                    t_sim_s=float(parts[1]) / 1000.0,
                    pos_ned_m=(float(parts[2]), float(parts[3]), float(parts[4])),
                    q_body=(float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])),
                    images=images,
                    line_no=line_no,
                ))
        if self.vehicle:
            out = [r for r in out if r.vehicle == self.vehicle]
        return out

    def telemetry(self, row: AirSimRecRow) -> Telemetry:
        n, e, d = row.pos_ned_m
        lat, lon = offset_ne(self.origin[0], self.origin[1], n, e)
        alt = self.origin[2] - d
        ground = self.ground_alt_msl_m if self.ground_alt_msl_m is not None else self.origin[2]
        _, _, yaw = quat_to_euler(row.q_body)
        q_gimbal = spec.gimbal_quat_from_euler(0.0, self.gimbal_pitch_deg, yaw if self.gimbal_stabilised else 0.0)
        if not self.gimbal_stabilised:  # camera rigidly mounted: compose the body attitude in
            q_gimbal = gimbal_quat_from_frd_quat(
                spec.quat_mul(row.q_body, euler_to_quat(0.0, self.gimbal_pitch_deg, 0.0))
            )
        t0 = self.capture_start_utc
        return Telemetry(
            t_utc=(t0 + row.t_sim_s - self._t0_sim()) if t0 else row.t_sim_s,
            lat=lat, lon=lon, alt_msl_m=alt, agl_m=alt - ground,
            q_body=row.q_body, q_gimbal=q_gimbal, gimbal_is_earth_referenced=True,
            ned_m=row.pos_ned_m, h_acc_m=0.0, v_acc_m=0.0,
            clip_id=self.clip_id, frame_idx=row.line_no - 2,
        )

    def _t0_sim(self) -> float:
        rows = self.rows()
        return rows[0].t_sim_s if rows else 0.0

    def telemetry_series(self) -> TelemetrySeries:
        return TelemetrySeries.from_telemetry([self.telemetry(r) for r in self.rows()], clip_id=self.clip_id)

    def frame_index(self) -> FrameIndex:
        index = FrameIndex(self.clip_id)
        t0_sim = self._t0_sim()
        for row in self.rows():
            scene = row.images.get("scene", "")
            t_utc = (self.capture_start_utc + row.t_sim_s - t0_sim) if self.capture_start_utc else row.t_sim_s
            index.add(FrameRef(
                frame_idx=row.line_no - 2,
                t_utc=t_utc,
                pts_s=row.t_sim_s,
                path=str(self.image_dir / scene) if scene else "",
                thermal_path=str(self.image_dir / row.images["infrared"]) if "infrared" in row.images else "",
                aux={k: str(self.image_dir / v) for k, v in row.images.items() if k not in ("scene", "infrared")},
            ))
        return index


# --- 3. the §5.7 noise model -------------------------------------------------------------------------------
@dataclass(slots=True)
class NoiseModel:
    """The §5.7 sensor-error model, applied to clean simulator telemetry.

    Defaults are the doc's consumer-GNSS inputs: **GNSS 2.5 m random walk, 1.5 deg yaw bias, 0.5 deg
    pitch/roll, 1 m barometric**. Each term is split into a per-clip BIAS and a time-correlated WANDER, in
    equal variance, because the doc's own conclusion — "averaging N frames shrinks the random terms by
    sqrt(N) but not the biases" — is only reproduced if a bias actually exists in the model.

    `h_acc_m` / `v_acc_m` are what the perturbed telemetry then *reports*, so downstream error budgets and
    dedup radii see the same numbers a real receiver would publish.
    """

    gnss_sigma_m: float = 2.5          # radial 1-sigma; per-axis sigma is this / sqrt(2)
    baro_sigma_m: float = 1.0
    yaw_bias_sigma_deg: float = 1.5    # magnetometer: a BIAS, not noise (§5.7)
    attitude_sigma_deg: float = 0.5    # pitch / roll, per sample
    correlation_time_s: float = 30.0   # GNSS/baro wander correlation time
    bias_fraction: float = 0.5         # share of the variance carried by the per-clip constant bias
    seed: int | None = 0
    label: str = "budget_v1"


def inject_noise(series: TelemetrySeries, model: NoiseModel | None = None) -> TelemetrySeries:
    """Return a NOISED copy of a clean simulator series (§5.7 "In the simulator"). The input is not modified.

    The caller keeps the clean series: `Clip.telemetry_truth` is exactly that, and the evaluation harness
    compares against it. Deterministic for a given `NoiseModel.seed`.
    """
    m = model or NoiseModel()
    rng = np.random.default_rng(m.seed)
    n = len(series)
    t = series.t
    bias_w = math.sqrt(max(0.0, min(1.0, m.bias_fraction)))
    wander_w = math.sqrt(max(0.0, 1.0 - m.bias_fraction))

    axis_sigma = m.gnss_sigma_m / math.sqrt(2.0)
    d_north = _ou(rng, t, axis_sigma * wander_w, m.correlation_time_s) + rng.normal(0.0, axis_sigma * bias_w)
    d_east = _ou(rng, t, axis_sigma * wander_w, m.correlation_time_s) + rng.normal(0.0, axis_sigma * bias_w)
    d_alt = _ou(rng, t, m.baro_sigma_m * wander_w, m.correlation_time_s) + rng.normal(0.0, m.baro_sigma_m * bias_w)
    yaw_bias = float(rng.normal(0.0, m.yaw_bias_sigma_deg))
    d_pitch = rng.normal(0.0, m.attitude_sigma_deg, n)
    d_roll = rng.normal(0.0, m.attitude_sigma_deg, n)

    lat = np.empty(n)
    lon = np.empty(n)
    for i in range(n):
        lat[i], lon[i] = offset_ne(float(series.lat[i]), float(series.lon[i]), float(d_north[i]), float(d_east[i]))
    q_gimbal = np.empty_like(series.q_gimbal)
    q_body = np.empty_like(series.q_body)
    for i in range(n):
        gr, gp, gy = spec.gimbal_euler_from_quat(tuple(series.q_gimbal[i]))
        q_gimbal[i] = spec.gimbal_quat_from_euler(gr + d_roll[i], gp + d_pitch[i], gy + yaw_bias)
        br, bp, by = quat_to_euler(tuple(series.q_body[i]))
        q_body[i] = euler_to_quat(br + d_roll[i], bp + d_pitch[i], by + yaw_bias)

    return replace(
        series,
        lat=lat,
        lon=lon,
        alt_msl_m=series.alt_msl_m + d_alt,
        agl_m=series.agl_m + d_alt,
        q_body=q_body,
        q_gimbal=q_gimbal,
        h_acc_m=np.full(n, m.gnss_sigma_m),
        v_acc_m=np.full(n, m.baro_sigma_m),
        noise_injected=True,
    )


def _ou(rng: np.random.Generator, t: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    """Ornstein-Uhlenbeck series with steady-state sigma and correlation time tau, sampled at times `t`.

    A plain random walk would grow without bound over a long flight; GNSS error does not. OU is the standard
    stationary model and reproduces the doc's point that the error is correlated, not independent per frame.
    """
    n = t.size
    out = np.empty(n, dtype=np.float64)
    if n == 0 or sigma <= 0.0:
        return np.zeros(n, dtype=np.float64)
    out[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        dt = max(0.0, float(t[i] - t[i - 1]))
        a = math.exp(-dt / max(tau, 1e-6))
        out[i] = a * out[i - 1] + rng.normal(0.0, sigma * math.sqrt(max(0.0, 1.0 - a * a)))
    return out
