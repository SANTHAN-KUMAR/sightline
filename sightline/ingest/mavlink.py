"""MAVLink `.tlog` and ArduPilot DataFlash `.bin` readers (SOLUTION_DOC §5.4 row 4). **Stretch path.**

Status: the message extraction below is complete and written against the MAVLink common-message-set
definitions shipped in `pymavlink` 2.4.49 (installed and verified). It is a **stretch path** for this project
in the sense of §5.4 / the F7 row of §9 — the demo runs on the simulator export — and it has only been
exercised against synthetic messages, not a real flight log, because none is available offline in this repo.

Messages read, and why (the doc names exactly these):

* `GLOBAL_POSITION_INT` — `time_boot_ms`, `lat`/`lon` (1e7 deg), `alt` (mm AMSL), `relative_alt` (mm),
  `vx/vy/vz` (cm/s NED).
* `ATTITUDE` — `time_boot_ms`, `roll`/`pitch`/`yaw` (rad) -> `q_body`.
* `GPS_RAW_INT` — `h_acc`/`v_acc` (mm, MAVLink 2 extension) -> `Telemetry.h_acc_m` / `v_acc_m`; `fix_type`.
* `GIMBAL_DEVICE_ATTITUDE_STATUS` — `q` (w,x,y,z) + `flags`. **The flags decide the frame** (§5.7 step 4):
  with `YAW_IN_EARTH_FRAME` the quaternion is already earth-referenced; otherwise it is vehicle-relative and
  must be composed with `ATTITUDE`.
* `SYSTEM_TIME` — `time_boot_ms` + `time_unix_usec` -> the boot-clock-to-UTC fit (`align.BootClock`).
* `CAMERA_INFORMATION` — `focal_length` mm, `sensor_size_h/v` mm, `resolution_h/v` px -> real `Intrinsics`.
* `CAMERA_IMAGE_CAPTURED` — `time_utc`, `image_index`, `file_url`: the still index for the replay harness.

ArduPilot `.bin` logs come through pymavlink's `DFReader` with DataFlash names instead (`GPS`, `ATT`, `POS`,
`MNT`); the equivalent mapping is in `_DF_MAP` and is applied automatically.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sightline.common.geodesy import euler_to_quat
from sightline.ingest.align import BootClock, TelemetrySeries
from sightline.ingest.spec import gimbal_quat_from_frd_quat, quat_mul
from sightline.schemas import Intrinsics, Telemetry

__all__ = ["LogReport", "CapturedImage", "read_mavlink", "MavlinkLog"]

#: Fallbacks for the two gimbal frame flags, used only if the installed pymavlink lacks the constants.
_YAW_IN_EARTH_FRAME_DEFAULT = 64
_YAW_IN_VEHICLE_FRAME_DEFAULT = 32


@dataclass(slots=True)
class LogReport:
    """What a log reader found and what it had to assume. Shared by `mavlink.py` and `ulog.py`."""

    source: str = ""
    n_messages: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    boot_clock: BootClock | None = None
    gimbal_frame: str = "none"        # "earth" | "vehicle" | "none"
    intrinsics: Intrinsics | None = None
    warnings: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)


@dataclass(slots=True)
class CapturedImage:
    """One `CAMERA_IMAGE_CAPTURED` — a still, with the pose the autopilot recorded for it."""

    index: int
    t_utc: float
    lat: float
    lon: float
    alt_msl_m: float
    rel_alt_m: float
    q: tuple[float, float, float, float]
    file_url: str = ""


@dataclass(slots=True)
class MavlinkLog:
    """The raw per-topic series pulled out of a log, before resampling onto frame times."""

    pos_t_boot_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    alt_msl_m: np.ndarray
    rel_alt_m: np.ndarray
    vel_ned_ms: np.ndarray
    att_t_boot_s: np.ndarray
    q_body: np.ndarray
    gimbal_t_boot_s: np.ndarray
    q_gimbal: np.ndarray
    gimbal_earth_frame: bool
    acc_t_boot_s: np.ndarray
    h_acc_m: np.ndarray
    v_acc_m: np.ndarray
    images: list[CapturedImage] = field(default_factory=list)
    report: LogReport = field(default_factory=LogReport)


# --- DataFlash (.bin) name mapping ---------------------------------------------------------------------------
#: ArduPilot DataFlash message -> the MAVLink message it stands in for, plus the field renames applied.
_DF_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "GPS": ("GPS_RAW_INT", {"TimeUS": "time_boot_us", "Lat": "lat_deg", "Lng": "lon_deg", "Alt": "alt_m",
                            "HDop": "hdop", "NSats": "satellites_visible", "GWk": "gps_week", "GMS": "gps_ms"}),
    "POS": ("GLOBAL_POSITION_INT", {"TimeUS": "time_boot_us", "Lat": "lat_deg", "Lng": "lon_deg",
                                    "Alt": "alt_m", "RelHomeAlt": "rel_alt_m"}),
    "ATT": ("ATTITUDE", {"TimeUS": "time_boot_us", "Roll": "roll_deg", "Pitch": "pitch_deg", "Yaw": "yaw_deg"}),
    "MNT": ("GIMBAL_DEVICE_ATTITUDE_STATUS", {"TimeUS": "time_boot_us", "DPitch": "pitch_deg",
                                              "DRoll": "roll_deg", "DYaw": "yaw_deg"}),
}


def read_mavlink(path: str | os.PathLike[str], *, clip_id: str = "", vehicle_agl_from: str = "relative_alt",
                 max_messages: int = 0) -> tuple[TelemetrySeries, LogReport]:
    """Read a `.tlog` / `.bin` into a `TelemetrySeries` on the UTC clock.

    `vehicle_agl_from` selects what becomes `Telemetry.agl_m`: "relative_alt" (height above the arming point,
    the only thing MAVLink reliably carries) or "none" (0.0, when the geo lane will supply a DEM instead).
    """
    log = read_mavlink_raw(path, max_messages=max_messages)
    return _to_series(log, clip_id=clip_id, vehicle_agl_from=vehicle_agl_from), log.report


def read_mavlink_raw(path: str | os.PathLike[str], *, max_messages: int = 0) -> MavlinkLog:
    """Pull every message the doc names out of a log. Never resamples — that is `align.TelemetrySeries`' job."""
    from pymavlink import mavutil

    p = Path(path)
    report = LogReport(source=str(p))
    conn = mavutil.mavlink_connection(str(p), robust_parsing=True, dialect="common")

    pos: list[tuple[float, float, float, float, float, float, float, float]] = []
    att: list[tuple[float, float, float, float]] = []
    gim: list[tuple[float, float, float, float, float, int]] = []
    acc: list[tuple[float, float, float]] = []
    boot_pairs: list[tuple[float, float]] = []
    images: list[CapturedImage] = []
    n = 0
    while True:
        msg = conn.recv_match(blocking=False)
        if msg is None:
            break
        n += 1
        if max_messages and n > max_messages:
            report.note(f"stopped after {max_messages} messages (max_messages)")
            break
        kind = msg.get_type()
        if kind in ("BAD_DATA", "UNKNOWN"):
            continue
        mapped = _DF_MAP.get(kind)
        name = mapped[0] if mapped else kind
        report.counts[name] = report.counts.get(name, 0) + 1
        d = msg.to_dict()
        if name == "GLOBAL_POSITION_INT":
            t = _boot_s(d)
            if mapped:
                pos.append((t, float(d["Lat"]), float(d["Lng"]), float(d["Alt"]),
                            float(d.get("RelHomeAlt", 0.0)), 0.0, 0.0, 0.0))
            else:
                pos.append((t, d["lat"] / 1e7, d["lon"] / 1e7, d["alt"] / 1e3, d["relative_alt"] / 1e3,
                            d["vx"] / 100.0, d["vy"] / 100.0, d["vz"] / 100.0))
        elif name == "ATTITUDE":
            t = _boot_s(d)
            if mapped:
                att.append((t, float(d["Roll"]), float(d["Pitch"]), float(d["Yaw"])))
            else:
                att.append((t, math.degrees(d["roll"]), math.degrees(d["pitch"]), math.degrees(d["yaw"])))
        elif name == "GPS_RAW_INT":
            t = _boot_s(d)
            if mapped:
                acc.append((t, float(d.get("HDop", 0.0)) * 2.5, float(d.get("VDop", 0.0)) * 2.5))
            else:
                h = d.get("h_acc")
                v = d.get("v_acc")
                eph, epv = d.get("eph", 65535), d.get("epv", 65535)
                h_m = (h / 1e3) if h not in (None, 0) else (eph / 100.0 if eph not in (0, 65535) else 2.5)
                v_m = (v / 1e3) if v not in (None, 0) else (epv / 100.0 if epv not in (0, 65535) else 1.0)
                acc.append((t, float(h_m), float(v_m)))
        elif name == "GIMBAL_DEVICE_ATTITUDE_STATUS":
            t = _boot_s(d)
            if mapped:
                q = euler_to_quat(float(d.get("DRoll", 0.0)), float(d.get("DPitch", -90.0)),
                                  float(d.get("DYaw", 0.0)))
                gim.append((t, *q, _YAW_IN_EARTH_FRAME_DEFAULT))
            else:
                q = tuple(float(v) for v in d["q"])
                gim.append((t, q[0], q[1], q[2], q[3], int(d.get("flags", 0))))
        elif name == "SYSTEM_TIME":
            boot_pairs.append((float(d["time_boot_ms"]) / 1e3, float(d["time_unix_usec"]) / 1e6))
        elif name == "CAMERA_INFORMATION":
            report.intrinsics = _intrinsics_from_camera_information(d)
        elif name == "CAMERA_IMAGE_CAPTURED":
            images.append(CapturedImage(
                index=int(d.get("image_index", -1)),
                t_utc=float(d.get("time_utc", 0)) / 1e9,
                lat=float(d.get("lat", 0)) / 1e7,
                lon=float(d.get("lon", 0)) / 1e7,
                alt_msl_m=float(d.get("alt", 0)) / 1e3,
                rel_alt_m=float(d.get("relative_alt", 0)) / 1e3,
                q=tuple(float(v) for v in d.get("q", (1.0, 0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                file_url=str(d.get("file_url", "")),
            ))
    report.n_messages = n
    if not pos:
        raise ValueError(f"{p}: no GLOBAL_POSITION_INT / POS messages — nothing to geolocate from")
    if not att:
        report.note("no ATTITUDE messages: the vehicle attitude is left level (identity quaternion)")

    if boot_pairs:
        report.boot_clock = BootClock.from_pairs([b for b, _ in boot_pairs], [u for _, u in boot_pairs])
    else:
        report.note(
            "no SYSTEM_TIME message: the boot clock cannot be mapped to UTC. Times are seconds since boot; "
            "set a per-clip t_offset_s, or supply the video's start UTC by hand (SOLUTION_DOC 5.4)."
        )
        report.boot_clock = BootClock(offset_s=0.0, skew=1.0, n_pairs=0)

    pos_a = np.asarray(pos, dtype=np.float64)
    att_a = np.asarray(att, dtype=np.float64) if att else np.zeros((0, 4))
    gim_a = np.asarray(gim, dtype=np.float64) if gim else np.zeros((0, 6))
    acc_a = np.asarray(acc, dtype=np.float64) if acc else np.zeros((0, 3))
    earth = False
    if gim_a.size:
        from pymavlink import mavutil as _mu

        earth_flag = getattr(_mu.mavlink, "GIMBAL_DEVICE_FLAGS_YAW_IN_EARTH_FRAME", _YAW_IN_EARTH_FRAME_DEFAULT)
        veh_flag = getattr(_mu.mavlink, "GIMBAL_DEVICE_FLAGS_YAW_IN_VEHICLE_FRAME", _YAW_IN_VEHICLE_FRAME_DEFAULT)
        flags = int(gim_a[0, 5])
        earth = bool(flags & earth_flag) or not bool(flags & veh_flag)
        report.gimbal_frame = "earth" if earth else "vehicle"
        if not earth:
            report.note("GIMBAL_DEVICE_ATTITUDE_STATUS is vehicle-framed: composed with ATTITUDE (5.7 step 4)")
    else:
        report.note("no GIMBAL_DEVICE_ATTITUDE_STATUS: assuming a -90 deg earth-referenced nadir gimbal")

    return MavlinkLog(
        pos_t_boot_s=pos_a[:, 0], lat=pos_a[:, 1], lon=pos_a[:, 2], alt_msl_m=pos_a[:, 3],
        rel_alt_m=pos_a[:, 4], vel_ned_ms=pos_a[:, 5:8],
        att_t_boot_s=att_a[:, 0] if att_a.size else np.zeros(0),
        q_body=(np.asarray([euler_to_quat(r, p_, y) for _, r, p_, y in att_a], dtype=np.float64)
                if att_a.size else np.zeros((0, 4))),
        gimbal_t_boot_s=gim_a[:, 0] if gim_a.size else np.zeros(0),
        q_gimbal=gim_a[:, 1:5] if gim_a.size else np.zeros((0, 4)),
        gimbal_earth_frame=earth,
        acc_t_boot_s=acc_a[:, 0] if acc_a.size else np.zeros(0),
        h_acc_m=acc_a[:, 1] if acc_a.size else np.zeros(0),
        v_acc_m=acc_a[:, 2] if acc_a.size else np.zeros(0),
        images=images,
        report=report,
    )


def _boot_s(d: dict[str, Any]) -> float:
    """Seconds on the autopilot's boot clock, from whichever stamp the message carries."""
    if "time_boot_ms" in d:
        return float(d["time_boot_ms"]) / 1e3
    if "time_boot_us" in d:
        return float(d["time_boot_us"]) / 1e6
    if "TimeUS" in d:
        return float(d["TimeUS"]) / 1e6
    if "time_usec" in d:
        return float(d["time_usec"]) / 1e6
    return 0.0


def _intrinsics_from_camera_information(d: dict[str, Any]) -> Intrinsics | None:
    """`CAMERA_INFORMATION` -> real intrinsics: `f_px = f_mm * resolution_px / sensor_size_mm`."""
    try:
        f_mm = float(d["focal_length"])
        sw, sh = float(d["sensor_size_h"]), float(d["sensor_size_v"])
        rw, rh = int(d["resolution_h"]), int(d["resolution_v"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(f_mm, sw, sh) <= 0 or min(rw, rh) <= 0:
        return None
    return Intrinsics(rw, rh, f_mm * rw / sw, f_mm * rh / sh, rw / 2.0, rh / 2.0, source="calibration")


def _to_series(log: MavlinkLog, *, clip_id: str, vehicle_agl_from: str) -> TelemetrySeries:
    """Resample attitude/gimbal/accuracy onto the position message times and build the pose series."""
    clock = log.report.boot_clock or BootClock()
    t_boot = log.pos_t_boot_s
    t_utc = np.asarray(clock.to_utc(t_boot), dtype=np.float64)
    samples: list[Telemetry] = []
    default_gimbal = gimbal_quat_from_frd_quat(euler_to_quat(0.0, -90.0, 0.0))
    for i, tb in enumerate(t_boot):
        q_body = _nearest_quat(log.att_t_boot_s, log.q_body, tb, (1.0, 0.0, 0.0, 0.0))
        if log.q_gimbal.size:
            q_g = _nearest_quat(log.gimbal_t_boot_s, log.q_gimbal, tb, (1.0, 0.0, 0.0, 0.0))
            q_g = q_g if log.gimbal_earth_frame else quat_mul(q_body, q_g)
            q_gimbal = gimbal_quat_from_frd_quat(q_g)
        else:
            q_gimbal = default_gimbal
        h_acc = float(np.interp(tb, log.acc_t_boot_s, log.h_acc_m)) if log.acc_t_boot_s.size else 2.5
        v_acc = float(np.interp(tb, log.acc_t_boot_s, log.v_acc_m)) if log.acc_t_boot_s.size else 1.0
        samples.append(Telemetry(
            t_utc=float(t_utc[i]),
            lat=float(log.lat[i]),
            lon=float(log.lon[i]),
            alt_msl_m=float(log.alt_msl_m[i]),
            agl_m=float(log.rel_alt_m[i]) if vehicle_agl_from == "relative_alt" else 0.0,
            q_body=q_body,
            q_gimbal=q_gimbal,
            gimbal_is_earth_referenced=True,
            vel_ned_ms=tuple(float(v) for v in log.vel_ned_ms[i]),  # type: ignore[arg-type]
            h_acc_m=h_acc,
            v_acc_m=v_acc,
            clip_id=clip_id,
            frame_idx=i,
        ))
    return TelemetrySeries.from_telemetry(samples, clip_id=clip_id)


def _nearest_quat(times: np.ndarray, quats: np.ndarray, t: float,
                  default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Nearest-sample attitude lookup on the raw log clock. Frame-time SLERP happens later, in `align`."""
    if times.size == 0:
        return default
    i = int(np.argmin(np.abs(times - t)))
    q = quats[i]
    n = float(np.linalg.norm(q))
    return tuple(float(v) / (n or 1.0) for v in q)  # type: ignore[return-value]
