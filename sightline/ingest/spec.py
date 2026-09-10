"""THE SIMULATOR CAPTURE FORMAT (`sightline-sim-capture`), version 1.0 — SOLUTION_DOC §5.4 row 1.

This file is the *contract between the capture side (orchestrator / sim lane, F5) and ingest (F7)*. It is
deliberately dependency-free (stdlib only) so `tools/capture/*.py` can import it inside the editor or a
`-game` run without pulling numpy or opencv in.

Layout of one capture ("clip"), all paths inside the clip directory relative to it::

    <clip_dir>/
        capture.json        the manifest (schema below); the ONLY file ingest needs to auto-detect the format
        frames.csv          one row per captured frame, tab-or-comma separated, header row required
        frames/             rgb_0000000.png, ... (or clip.mp4 when `video` is set in the manifest)
        thermal/            thermal_0000000.png  (16-bit centi-kelvin, or 8-bit AGC; optional)
        seg/ depth/ annotation/                  (optional; auto-label sources, F5/F6 — carried, not decoded)

Why CSV and not JSON-per-frame: a capture run is killed by a crashed editor often enough that the telemetry
file must be *append-only and readable when truncated*. A half-written CSV loses one row; a half-written JSON
array loses everything. `frames.jsonl` (one JSON object per line) is accepted as an equivalent alternative.

**Clock rule (the trap).** Under `SteppableClock` — which `sim/settings/capture_4k.json` uses — AirSim's
`timestamp` does not follow wall time. The capture side must therefore write::

    t_utc = capture_start_utc + (timestamp_ns - timestamp_ns_at_start) / 1e9

`t_utc` is the authoritative clock for the whole pipeline; `t_sim_s` is kept alongside it for debugging only.

**Column stability.** Column ORDER is not significant — ingest reads by header name. Adding a column is a
minor version bump; removing or re-meaning one is a major bump. Unknown columns are preserved in
`SimFrameRow.extra` and never cause an error.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

FORMAT_NAME = "sightline-sim-capture"
FORMAT_VERSION = "1.0"

MANIFEST_NAME = "capture.json"
FRAMES_CSV = "frames.csv"
FRAMES_JSONL = "frames.jsonl"
AIRSIM_REC = "airsim_rec.txt"

# --- the per-frame columns ---------------------------------------------------------------------------------
#: Columns every capture MUST write. Ingest raises `CaptureFormatError` if one is missing.
REQUIRED_COLUMNS: tuple[str, ...] = (
    # identity + clock
    "frame_idx",  # int, 0-based, strictly increasing (gaps allowed: a dropped frame is a gap, not a renumber)
    "t_utc",      # float, UTC POSIX seconds, >= 6 decimals. See the clock rule above.
    # str, path relative to the clip dir, e.g. "frames/rgb_0000123.png". When the clip is stored as a video,
    # write "<video>#<index>" instead (e.g. "clip.mp4#123"): the column stays required and self-describing.
    "rgb_path",
    # vehicle pose: NED metres relative to the scenario origin (AirSim's native frame; +D is DOWN)
    "ned_n_m", "ned_e_m", "ned_d_m",
    # body(FRD) -> NED attitude quaternion, (w, x, y, z), normalised
    "q_body_w", "q_body_x", "q_body_y", "q_body_z",
    # camera(optical) -> NED attitude quaternion, (w, x, y, z). EARTH-REFERENCED: the sim camera pose is exact,
    # so `gimbal_is_earth_referenced` is True and the body attitude is never composed in (§5.7 step 4).
    "q_gimbal_w", "q_gimbal_x", "q_gimbal_y", "q_gimbal_z",
    # geodetic
    "lat", "lon", "alt_msl_m",
    # height above the surface the geolocation ray will be intersected with: the WATER surface where water
    # covers the ground, the terrain otherwise (§5.7 step 5). `ground_asl_m` below lets geo recompute it.
    "agl_m",
    # camera
    "img_w_px", "img_h_px", "hfov_deg",
)

#: Columns a capture SHOULD write. Missing ones fall back to the manifest, then to the documented default.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "t_sim_s",            # float, AirSim clock seconds since clip start. Debug only; never used for alignment.
    "cam_name",           # str, the AirSim camera name ("survey"); default: manifest camera.name
    "vel_n_ms", "vel_e_ms", "vel_d_ms",
    "h_acc_m", "v_acc_m",  # GNSS 1-sigma. Clean sim truth is 0.0; the noise model writes what it applied.
    "ground_asl_m",        # bare terrain height under the vehicle (before flooding)
    "flood_level_asl_m",   # water surface ASL; default: manifest flood_level_asl_m
    "mode",                # AUTO | MANUAL | HOLD | RTL (F3 logs every switch)
    "pass_id",             # int, coverage pass counter; dedup reports `seen_in_passes`
    "zone",                # fan | settlement | channel | hillslope | unknown
    "time_of_day",         # ISO-8601 LOCAL scene time, exactly what was passed to simSetTimeOfDay
    "rain", "fog", "wind_ms", "cloud",   # the `Telemetry.weather` dict; rain/fog/cloud in 0..1, wind in m/s
    "sun_elevation_deg", "sun_azimuth_deg",
    "noise_injected",      # 0/1; 1 means the §5.7 noise model was already applied by the capture side
    # intrinsics overrides — write these only when the camera is NOT an ideal pinhole from hfov_deg
    "fx_px", "fy_px", "cx_px", "cy_px",
    "dist_k1", "dist_k2", "dist_p1", "dist_p2", "dist_k3",
    # human-readable duplicates of the quaternions. Used ONLY when the q_* columns are absent.
    "body_roll_deg", "body_pitch_deg", "body_yaw_deg",
    "gimbal_pitch_deg", "gimbal_roll_deg", "gimbal_yaw_deg",
    # thermal partner frame (§5.5b)
    "thermal_path",          # 16-bit PNG of centi-kelvin, or 8-bit PNG of AGC grey
    "thermal_radiometric",   # 0/1; 1 => thermal_path is uint16 centi-kelvin
    "thermal_w_px", "thermal_h_px", "thermal_hfov_deg",
    # auto-label sources (F5/F6). Carried through as aux paths; ingest never decodes them.
    "seg_path", "depth_path", "annotation_path",
)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Column -> python type, for the reader's coercion table.
_INT_COLUMNS = frozenset({"frame_idx", "img_w_px", "img_h_px", "pass_id", "thermal_w_px", "thermal_h_px"})
_BOOL_COLUMNS = frozenset({"noise_injected", "thermal_radiometric"})
_STR_COLUMNS = frozenset(
    {"rgb_path", "cam_name", "mode", "zone", "time_of_day", "thermal_path", "seg_path", "depth_path",
     "annotation_path"}
)

#: Weather keys, in the order they are written into `Telemetry.weather`.
WEATHER_KEYS: tuple[str, ...] = ("rain", "fog", "wind_ms", "cloud")


class CaptureFormatError(ValueError):
    """Raised when a capture directory does not satisfy this specification."""


# --- the gimbal quaternion convention (ONE definition, shared by the capture side and every reader) ---------
#: Rotation from the schema's camera frame to the camera's FRD frame: Ry(+90) as (w, x, y, z).
#:
#: `Telemetry.q_gimbal` is documented as "rotates camera(optical) -> NED", and the frozen schema's own helper
#: `Telemetry.gimbal_pitch_deg()` subtracts 90 from the quaternion's Euler pitch and calls -90 "straight down".
#: The only camera frame consistent with BOTH statements has
#:     z = the viewing axis, y = image right, x = image up
#: (an OpenCV optical frame rolled by 90 deg), and it relates to the usual camera FRD frame
#: (x = forward/viewing, y = right, z = down) by the constant rotation Ry(+90). Hence:
#:     q_gimbal = q_camera_FRD_to_NED  (x)  Q_FRD_FROM_CAM
#: With roll = 0 this is exactly `euler_to_quat(roll, dji_pitch + 90, yaw)`, so a nadir camera (-90) gives the
#: identity-pitch quaternion and `gimbal_pitch_deg()` reports -90. **Every producer must use these helpers.**
Q_FRD_FROM_CAM: tuple[float, float, float, float] = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


def quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    """Hamilton product of two (w, x, y, z) quaternions. Pure stdlib so the capture side can import it."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Aerospace 3-2-1 (yaw, then pitch, then roll) FRD -> NED, as (w, x, y, z).

    Identical to `sightline.common.geodesy.euler_to_quat`; duplicated here only so `spec.py` stays stdlib-only
    and can be imported by editor-side capture scripts (editor Python has no numpy — HANDBOOK §6).
    """
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def gimbal_quat_from_frd_quat(q_cam_frd_to_ned: Sequence[float]) -> tuple[float, float, float, float]:
    """`q_gimbal` from a camera-FRD-to-NED quaternion — **the simulator path**.

    Cosys-AirSim's `ImageResponse.camera_orientation` (and `simGetCameraInfo().pose.orientation`) is exactly
    this: the camera's FRD frame in world NED, so a nadir camera reports Euler pitch -90. Feed it straight in.
    """
    return quat_mul(q_cam_frd_to_ned, Q_FRD_FROM_CAM)


def gimbal_quat_from_euler(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """`q_gimbal` from earth-referenced gimbal angles — **the DJI path** (-90 pitch = nadir, yaw 0 = north)."""
    return gimbal_quat_from_frd_quat(euler_to_quat(roll_deg, pitch_deg, yaw_deg))


def gimbal_euler_from_quat(q_gimbal: Sequence[float]) -> tuple[float, float, float]:
    """Inverse of `gimbal_quat_from_euler`: returns (roll_deg, pitch_deg, yaw_deg) with -90 = nadir."""
    w, x, y, z = quat_mul(q_gimbal, (Q_FRD_FROM_CAM[0], -Q_FRD_FROM_CAM[1], -Q_FRD_FROM_CAM[2],
                                     -Q_FRD_FROM_CAM[3]))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


# --- the manifest ------------------------------------------------------------------------------------------
@dataclass(slots=True)
class CameraSpec:
    """One camera as declared in the manifest. `hfov_deg` is Cosys-AirSim's `FOV_Degrees` (HORIZONTAL)."""

    name: str = "survey"
    width_px: int = 3840
    height_px: int = 2160
    hfov_deg: float = 75.5
    dist: tuple[float, ...] = ()
    source: str = "sim"
    radiometric: bool = False          # thermal only: True => uint16 centi-kelvin (§5.5b)
    units: str = ""                    # thermal only: "centi_kelvin" | "agc8"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "width_px": self.width_px, "height_px": self.height_px,
                             "hfov_deg": self.hfov_deg, "dist": list(self.dist), "source": self.source}
        if self.radiometric or self.units:
            d["radiometric"], d["units"] = self.radiometric, self.units
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraSpec":
        return cls(
            name=str(d.get("name", "survey")),
            width_px=int(d.get("width_px", 0)),
            height_px=int(d.get("height_px", 0)),
            hfov_deg=float(d.get("hfov_deg", 0.0)),
            dist=tuple(float(v) for v in d.get("dist", ()) or ()),
            source=str(d.get("source", "sim")),
            radiometric=bool(d.get("radiometric", False)),
            units=str(d.get("units", "")),
        )


@dataclass(slots=True)
class CaptureManifest:
    """`capture.json`. Everything constant for the clip lives here; per-frame values live in `frames.csv`."""

    clip_id: str
    format: str = FORMAT_NAME
    format_version: str = FORMAT_VERSION
    domain: str = "sim"                       # "sim" | "real" — becomes SliceKey.domain (hard rule 5)
    created_utc: float = 0.0
    #: t_utc of the first row; also the epoch the SteppableClock offset was taken against.
    capture_start_utc: float = 0.0
    frames_csv: str = FRAMES_CSV
    frames_dir: str = "frames"
    video: str | None = None                  # relative path to an MP4 when frames are in a container
    video_start_utc: float | None = None      # t_utc of container PTS 0.0 (required when `video` is set)
    t_offset_s: float = 0.0                   # R12 per-clip video/telemetry offset; ingest may override
    scenario: dict[str, Any] = field(default_factory=dict)   # {"name", "settings_profile", "level"}
    #: The AirSim `OriginGeopoint`; NED (0,0,0) is at this point (HANDBOOK §5: anchored at the UE world origin).
    origin_geopoint: dict[str, float] = field(default_factory=dict)   # {"lat","lon","alt_msl_m"}
    vehicle: str = "Drone"
    camera: CameraSpec = field(default_factory=CameraSpec)
    thermal_camera: CameraSpec | None = None
    flood_level_asl_m: float | None = None
    noise: dict[str, Any] = field(default_factory=lambda: {"injected": False, "model": "", "seed": None})
    counts: dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "format": self.format, "format_version": self.format_version, "clip_id": self.clip_id,
            "domain": self.domain, "created_utc": self.created_utc, "capture_start_utc": self.capture_start_utc,
            "frames_csv": self.frames_csv, "frames_dir": self.frames_dir, "video": self.video,
            "video_start_utc": self.video_start_utc, "t_offset_s": self.t_offset_s, "scenario": self.scenario,
            "origin_geopoint": self.origin_geopoint, "vehicle": self.vehicle, "camera": self.camera.to_dict(),
            "flood_level_asl_m": self.flood_level_asl_m, "noise": self.noise, "counts": self.counts,
            "notes": self.notes,
        }
        if self.thermal_camera is not None:
            d["thermal_camera"] = self.thermal_camera.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptureManifest":
        fmt = str(d.get("format", ""))
        if fmt and fmt != FORMAT_NAME:
            raise CaptureFormatError(f"manifest format is {fmt!r}, expected {FORMAT_NAME!r}")
        ver = str(d.get("format_version", FORMAT_VERSION))
        if ver.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise CaptureFormatError(f"capture format_version {ver} is not compatible with reader {FORMAT_VERSION}")
        thermal = d.get("thermal_camera")
        return cls(
            clip_id=str(d.get("clip_id", "")),
            format=fmt or FORMAT_NAME,
            format_version=ver,
            domain=str(d.get("domain", "sim")),
            created_utc=float(d.get("created_utc", 0.0) or 0.0),
            capture_start_utc=float(d.get("capture_start_utc", 0.0) or 0.0),
            frames_csv=str(d.get("frames_csv", FRAMES_CSV)),
            frames_dir=str(d.get("frames_dir", "frames")),
            video=d.get("video"),
            video_start_utc=(None if d.get("video_start_utc") is None else float(d["video_start_utc"])),
            t_offset_s=float(d.get("t_offset_s", 0.0) or 0.0),
            scenario=dict(d.get("scenario", {}) or {}),
            origin_geopoint=dict(d.get("origin_geopoint", {}) or {}),
            vehicle=str(d.get("vehicle", "Drone")),
            camera=CameraSpec.from_dict(d.get("camera", {}) or {}),
            thermal_camera=(CameraSpec.from_dict(thermal) if thermal else None),
            flood_level_asl_m=(None if d.get("flood_level_asl_m") is None else float(d["flood_level_asl_m"])),
            noise=dict(d.get("noise", {}) or {}),
            counts=dict(d.get("counts", {}) or {}),
            notes=str(d.get("notes", "")),
        )


# --- one parsed row ----------------------------------------------------------------------------------------
@dataclass(slots=True)
class SimFrameRow:
    """A `frames.csv` row after type coercion. `extra` holds columns this version does not know about."""

    values: dict[str, Any]
    extra: dict[str, str] = field(default_factory=dict)
    line_no: int = -1

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        v = self.values.get(key, default)
        return default if v is None else v

    def has(self, *keys: str) -> bool:
        return all(self.values.get(k) is not None for k in keys)


def coerce_row(raw: dict[str, str], line_no: int = -1) -> SimFrameRow:
    """Type-coerce one raw CSV/JSONL row. Blank strings become None (= "not written"), never 0.0."""
    values: dict[str, Any] = {}
    extra: dict[str, str] = {}
    for key, text in raw.items():
        if key is None:
            continue
        key = key.strip()
        if key not in ALL_COLUMNS:
            extra[key] = "" if text is None else str(text)
            continue
        if text is None or (isinstance(text, str) and text.strip() == ""):
            values[key] = None
            continue
        if key in _STR_COLUMNS:
            values[key] = str(text).strip()
        elif key in _BOOL_COLUMNS:
            values[key] = str(text).strip().lower() in ("1", "true", "yes", "y", "t")
        elif key in _INT_COLUMNS:
            values[key] = int(float(str(text).strip()))
        else:
            try:
                values[key] = float(str(text).strip())
            except ValueError as exc:  # a malformed number is a hard error: silent 0.0 would poison geolocation
                raise CaptureFormatError(f"line {line_no}: column {key!r} is not a number: {text!r}") from exc
    return SimFrameRow(values=values, extra=extra, line_no=line_no)


def validate_header(header: Sequence[str]) -> list[str]:
    """Return the list of REQUIRED columns missing from `header` (empty list = valid)."""
    present = {h.strip() for h in header}
    return [c for c in REQUIRED_COLUMNS if c not in present]


def validate_row(row: SimFrameRow) -> list[str]:
    """Return a list of human-readable problems with one row (empty list = valid)."""
    problems: list[str] = []
    for col in REQUIRED_COLUMNS:
        if row.values.get(col) is None:
            problems.append(f"required column {col!r} is empty")
    for prefix in ("q_body", "q_gimbal"):
        q = [row.values.get(f"{prefix}_{a}") for a in "wxyz"]
        if all(v is not None for v in q):
            n = math.sqrt(sum(float(v) * float(v) for v in q))
            if not (0.98 <= n <= 1.02):
                problems.append(f"{prefix} quaternion norm {n:.4f} is not 1 (write normalised quaternions)")
    lat, lon = row.values.get("lat"), row.values.get("lon")
    if lat is not None and not (-90.0 <= float(lat) <= 90.0):
        problems.append(f"lat {lat} out of range")
    if lon is not None and not (-180.0 <= float(lon) <= 180.0):
        problems.append(f"lon {lon} out of range")
    hfov = row.values.get("hfov_deg")
    if hfov is not None and not (1.0 < float(hfov) < 179.0):
        problems.append(f"hfov_deg {hfov} out of range")
    return problems


def validate_capture(clip_dir: str | os.PathLike[str], check_files: bool = True, max_rows: int = 0) -> list[str]:
    """Validate a capture directory against this spec. Returns a list of problems ([] = valid).

    `check_files=True` also asserts that every referenced frame file exists. `max_rows > 0` stops early.
    """
    root = Path(clip_dir)
    problems: list[str] = []
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing manifest {manifest_path}"]
    try:
        manifest = CaptureManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (CaptureFormatError, json.JSONDecodeError) as exc:
        return [f"{MANIFEST_NAME}: {exc}"]
    if not manifest.clip_id:
        problems.append("manifest.clip_id is empty (every record is keyed on it)")
    if manifest.domain not in ("sim", "real"):
        problems.append(f"manifest.domain must be 'sim' or 'real', got {manifest.domain!r}")
    if manifest.video and manifest.video_start_utc is None:
        problems.append("manifest.video is set but video_start_utc is null (PTS 0.0 has no UTC anchor)")
    if not manifest.origin_geopoint:
        problems.append("manifest.origin_geopoint is empty (NED cannot be converted to lat/lon)")

    frames_path = root / manifest.frames_csv
    jsonl_path = root / FRAMES_JSONL
    if frames_path.is_file():
        rows = list(iter_csv_rows(frames_path))
    elif jsonl_path.is_file():
        rows = list(iter_jsonl_rows(jsonl_path))
    else:
        return problems + [f"missing {frames_path} (and no {FRAMES_JSONL})"]
    if not rows:
        return problems + [f"{manifest.frames_csv} has no data rows"]

    missing = validate_header(list(rows[0].values) + list(rows[0].extra))
    if missing:
        problems.append(f"missing required column(s): {', '.join(missing)}")
    last_idx, last_t = None, None
    for i, row in enumerate(rows):
        if max_rows and i >= max_rows:
            break
        problems += [f"line {row.line_no}: {p}" for p in validate_row(row)]
        idx, t = row.values.get("frame_idx"), row.values.get("t_utc")
        if last_idx is not None and idx is not None and idx <= last_idx:
            problems.append(f"line {row.line_no}: frame_idx {idx} is not increasing (previous {last_idx})")
        if last_t is not None and t is not None and t < last_t:
            problems.append(f"line {row.line_no}: t_utc {t} goes backwards (previous {last_t})")
        last_idx, last_t = idx if idx is not None else last_idx, t if t is not None else last_t
        if check_files:
            rgb = row.values.get("rgb_path")
            if rgb and not (root / rgb).is_file():
                problems.append(f"line {row.line_no}: rgb_path {rgb!r} does not exist")
            th = row.values.get("thermal_path")
            if th and not (root / th).is_file():
                problems.append(f"line {row.line_no}: thermal_path {th!r} does not exist")
    return problems


def iter_csv_rows(path: str | os.PathLike[str]) -> Iterable[SimFrameRow]:
    """Stream `frames.csv`. The delimiter is sniffed, so tab- and comma-separated files both work."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(fh, delimiter=delim)
        for line_no, raw in enumerate(reader, start=2):
            if not any((v or "").strip() for v in raw.values()):
                continue  # blank line: a killed capture can leave one
            yield coerce_row(raw, line_no)


def iter_jsonl_rows(path: str | os.PathLike[str]) -> Iterable[SimFrameRow]:
    """Stream `frames.jsonl` (one JSON object per line). A truncated last line is skipped, not fatal."""
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if line_no > 1:  # a killed capture truncates the final line; earlier lines must be valid
                    continue
                raise
            yield coerce_row({k: ("" if v is None else str(v)) for k, v in obj.items()}, line_no)


# --- the writer the capture side should use ----------------------------------------------------------------
class SimCaptureWriter:
    """Write a spec-compliant capture. **The sim lane should import this rather than format rows by hand.**

    stdlib only, append-only, flushed every row, so a killed editor costs at most the row in flight::

        w = SimCaptureWriter(clip_dir, manifest)
        w.open()
        for frame in flight:
            w.append(frame_idx=i, t_utc=t, rgb_path=f"frames/rgb_{i:07d}.png", ...)
        w.close()          # writes counts.frames back into capture.json
    """

    def __init__(self, clip_dir: str | os.PathLike[str], manifest: CaptureManifest,
                 columns: Sequence[str] | None = None) -> None:
        self.root = Path(clip_dir)
        self.manifest = manifest
        self.columns: list[str] = list(columns) if columns else list(ALL_COLUMNS)
        missing = validate_header(self.columns)
        if missing:
            raise CaptureFormatError(f"column set is missing required column(s): {', '.join(missing)}")
        self._fh = None
        self._writer: csv.DictWriter | None = None
        self._n = 0

    def open(self) -> "SimCaptureWriter":
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / self.manifest.frames_dir).mkdir(parents=True, exist_ok=True)
        if not self.manifest.created_utc:
            self.manifest.created_utc = time.time()
        self._write_manifest()
        self._fh = open(self.root / self.manifest.frames_csv, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns, delimiter=",", extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()
        return self

    def append(self, **row: Any) -> None:
        """Append one frame. Unknown keys are rejected loudly — a typo must not become a silent blank column."""
        if self._writer is None or self._fh is None:
            raise RuntimeError("SimCaptureWriter.open() has not been called")
        unknown = [k for k in row if k not in ALL_COLUMNS]
        if unknown:
            raise CaptureFormatError(f"unknown column(s) {unknown}; see spec.ALL_COLUMNS")
        out = {k: ("" if v is None else (int(v) if k in _BOOL_COLUMNS else v)) for k, v in row.items()}
        problems = validate_row(coerce_row({k: str(v) for k, v in out.items()}))
        if problems:
            raise CaptureFormatError(f"frame {row.get('frame_idx')}: " + "; ".join(problems))
        if self._n == 0 and not self.manifest.capture_start_utc:
            self.manifest.capture_start_utc = float(row["t_utc"])
        self._writer.writerow(out)
        self._fh.flush()
        self._n += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh, self._writer = None, None
        self.manifest.counts["frames"] = self._n
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.root / MANIFEST_NAME).write_text(
            json.dumps(self.manifest.to_dict(), indent=2, sort_keys=False), encoding="utf-8"
        )

    def __enter__(self) -> "SimCaptureWriter":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()
