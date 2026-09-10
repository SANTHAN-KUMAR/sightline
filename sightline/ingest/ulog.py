"""PX4 `.ulg` reader (SOLUTION_DOC §5.4 row 5). **Stretch path**, but a real one — it is exercised offline
against `data/samples/px4_sample_log_small.ulg` in `tests/test_ingest.py`.

Topics, in the priority order the reader tries them:

* `vehicle_global_position` — `lat`, `lon` (deg), `alt` (m AMSL), `eph`, `epv`, `terrain_alt(_valid)`.
  The preferred source.
* `vehicle_local_position` — `x/y/z` NED, `vx/vy/vz`, `ref_lat/ref_lon/ref_alt`, `dist_bottom(_valid)`.
  The only topic that gives a true AGL, and only when a range finder is fitted.
* `vehicle_gps_position` / `sensor_gps` — `lat`/`lon` (1e7 int **or** float deg), `alt` (mm **or** m),
  `eph`, `epv`, `time_utc_usec`, `vel_[ned]_m_s`. The fallback, and the ONLY source of UTC.
* `vehicle_attitude` — `q[0..3]` = (w, x, y, z), body FRD -> NED: `Telemetry.q_body` with no conversion.
* `gimbal_device_attitude_status` — `q[0..3]`, `device_flags`; frame flags read as in MAVLink (§5.7 step 4).
* `vehicle_air_data` — `baro_alt_meter`, the altitude fallback and the take-off signal for `estimate_t_offset`.

**Timestamps are microseconds since boot**, so every series is mapped to UTC through `align.BootClock`, fitted
on (`timestamp`, `time_utc_usec`) pairs from the GPS topic. PX4 field naming changed across releases
(`lat` as scaled int vs degrees, `alt` in mm vs metres); both are detected from the magnitude and the choice
is recorded in the report rather than guessed silently.
"""

from __future__ import annotations

import os

import numpy as np

from sightline.ingest.align import BootClock, TelemetrySeries
from sightline.ingest.mavlink import LogReport
from sightline.ingest.spec import gimbal_quat_from_frd_quat, quat_mul
from sightline.common.geodesy import euler_to_quat, offset_ne
from sightline.schemas import Telemetry

__all__ = ["read_ulog", "read_ulog_topics"]

_GPS_TOPICS = ("vehicle_gps_position", "sensor_gps")


def read_ulog_topics(path: str | os.PathLike[str]) -> dict[str, dict[str, np.ndarray]]:
    """Every logged topic as `{topic: {field: array}}`. Duplicated multi-instance topics keep instance 0."""
    import pyulog

    log = pyulog.ULog(str(path))
    out: dict[str, dict[str, np.ndarray]] = {}
    for dataset in log.data_list:
        if dataset.name in out:
            continue  # multi_id > 0: a second IMU/baro; instance 0 is the one the estimator used
        out[dataset.name] = {k: np.asarray(v) for k, v in dataset.data.items()}
    return out


def read_ulog(path: str | os.PathLike[str], *, clip_id: str = "",
              agl_from: str = "auto") -> tuple[TelemetrySeries, LogReport]:
    """Read a `.ulg` into a `TelemetrySeries` on the UTC clock.

    `agl_from` is "auto" (range finder `dist_bottom` if valid, else `terrain_alt`, else altitude above the
    first sample), "dist_bottom", "terrain" or "takeoff".
    """
    topics = read_ulog_topics(path)
    report = LogReport(source=str(path), counts={k: int(v["timestamp"].size) for k, v in topics.items()
                                                 if "timestamp" in v})
    report.n_messages = sum(report.counts.values())

    gps = next((topics[t] for t in _GPS_TOPICS if t in topics), None)
    report.boot_clock = _fit_clock(gps, report)

    pos = _position_series(topics, gps, report)
    if pos is None:
        wanted = ("vehicle_global_position", "vehicle_local_position") + _GPS_TOPICS
        raise ValueError(f"{path}: no position topic (looked for {', '.join(wanted)})")
    t_boot_s, lat, lon, alt_msl, vel_ned, h_acc, v_acc = pos
    agl = _agl_series(topics, t_boot_s, alt_msl, agl_from, report)

    att = topics.get("vehicle_attitude")
    if att is None:
        report.note("no vehicle_attitude topic: the body attitude is left level (identity quaternion)")
    q_body = _quat_at(att, t_boot_s)

    gimbal = topics.get("gimbal_device_attitude_status")
    if gimbal is None:
        report.note("no gimbal_device_attitude_status: assuming a -90 deg earth-referenced nadir gimbal")
        report.gimbal_frame = "none"
        default = gimbal_quat_from_frd_quat(euler_to_quat(0.0, -90.0, 0.0))
        q_gimbal = np.tile(np.asarray(default, dtype=np.float64), (t_boot_s.size, 1))
    else:
        raw = _quat_at(gimbal, t_boot_s)
        flags = int(gimbal.get("device_flags", np.zeros(1))[0])
        earth = bool(flags & 64) or not bool(flags & 32)   # YAW_IN_EARTH_FRAME / YAW_IN_VEHICLE_FRAME
        report.gimbal_frame = "earth" if earth else "vehicle"
        q_gimbal = np.asarray([
            gimbal_quat_from_frd_quat(tuple(raw[i]) if earth else quat_mul(tuple(q_body[i]), tuple(raw[i])))
            for i in range(t_boot_s.size)
        ], dtype=np.float64)

    clock = report.boot_clock or BootClock()
    t_utc = np.asarray(clock.to_utc(t_boot_s), dtype=np.float64)
    samples = [
        Telemetry(
            t_utc=float(t_utc[i]), lat=float(lat[i]), lon=float(lon[i]), alt_msl_m=float(alt_msl[i]),
            agl_m=float(agl[i]), q_body=tuple(float(v) for v in q_body[i]),  # type: ignore[arg-type]
            q_gimbal=tuple(float(v) for v in q_gimbal[i]),  # type: ignore[arg-type]
            gimbal_is_earth_referenced=True,
            vel_ned_ms=tuple(float(v) for v in vel_ned[i]),  # type: ignore[arg-type]
            h_acc_m=float(h_acc[i]), v_acc_m=float(v_acc[i]), clip_id=clip_id, frame_idx=i,
        )
        for i in range(t_boot_s.size)
    ]
    return TelemetrySeries.from_telemetry(samples, clip_id=clip_id), report


# --- helpers -----------------------------------------------------------------------------------------------
def _fit_clock(gps: dict[str, np.ndarray] | None, report: LogReport) -> BootClock:
    if gps is None or "time_utc_usec" not in gps:
        report.note("no GPS UTC in this log: timestamps stay on the boot clock (seconds since boot)")
        return BootClock(offset_s=0.0, skew=1.0, n_pairs=0)
    boot_s = np.asarray(gps["timestamp"], dtype=np.float64) / 1e6
    unix_s = np.asarray(gps["time_utc_usec"], dtype=np.float64) / 1e6
    good = unix_s > 1.0e9
    if not good.any():
        report.note("GPS time_utc_usec is zero throughout (no GNSS time fix): staying on the boot clock")
        return BootClock(offset_s=0.0, skew=1.0, n_pairs=0)
    return BootClock.from_pairs(boot_s[good], unix_s[good])


def _scale_lat_lon(lat: np.ndarray, lon: np.ndarray, report: LogReport, topic: str) -> tuple[np.ndarray, np.ndarray]:
    """PX4 wrote lat/lon as int32 * 1e7 for years and as float degrees later. Detect, do not guess."""
    if np.nanmax(np.abs(lat)) > 180.0 or np.nanmax(np.abs(lon)) > 360.0:
        report.note(f"{topic}: lat/lon are scaled integers (1e-7 deg)")
        return lat / 1e7, lon / 1e7
    return lat.astype(np.float64), lon.astype(np.float64)


def _scale_alt(alt: np.ndarray, report: LogReport, topic: str) -> np.ndarray:
    """`alt` is millimetres in the GPS topics of older PX4 and metres in `vehicle_global_position`."""
    if np.nanmax(np.abs(alt)) > 20000.0:
        report.note(f"{topic}: alt is in millimetres")
        return alt / 1e3
    return alt.astype(np.float64)


def _position_series(topics: dict[str, dict[str, np.ndarray]], gps: dict[str, np.ndarray] | None,
                     report: LogReport) -> tuple[np.ndarray, ...] | None:
    g = topics.get("vehicle_global_position")
    if g is not None and "lat" in g:
        t = np.asarray(g["timestamp"], dtype=np.float64) / 1e6
        lat, lon = _scale_lat_lon(np.asarray(g["lat"], float), np.asarray(g["lon"], float), report,
                                  "vehicle_global_position")
        alt = _scale_alt(np.asarray(g["alt"], float), report, "vehicle_global_position")
        vel = _velocity_on(topics, t)
        h = np.asarray(g.get("eph", np.full(t.size, 2.5)), dtype=np.float64)
        v = np.asarray(g.get("epv", np.full(t.size, 1.0)), dtype=np.float64)
        return t, lat, lon, alt, vel, h, v

    lp = topics.get("vehicle_local_position")
    if lp is not None and "ref_lat" in lp and np.any(np.asarray(lp.get("xy_global", np.ones(1)))):
        t = np.asarray(lp["timestamp"], dtype=np.float64) / 1e6
        ref_lat = float(np.asarray(lp["ref_lat"], float)[0])
        ref_lon = float(np.asarray(lp["ref_lon"], float)[0])
        ref_alt = float(np.asarray(lp.get("ref_alt", np.zeros(1)), float)[0])
        report.note("position taken from vehicle_local_position + its NED reference point")
        lat = np.empty(t.size)
        lon = np.empty(t.size)
        x, y, z = (np.asarray(lp[k], float) for k in ("x", "y", "z"))
        for i in range(t.size):
            lat[i], lon[i] = offset_ne(ref_lat, ref_lon, float(x[i]), float(y[i]))
        vel = np.column_stack([np.asarray(lp.get(k, np.zeros(t.size)), float) for k in ("vx", "vy", "vz")])
        return (t, lat, lon, ref_alt - z, vel,
                np.asarray(lp.get("eph", np.full(t.size, 2.5)), float),
                np.asarray(lp.get("epv", np.full(t.size, 1.0)), float))

    if gps is not None and "lat" in gps:
        t = np.asarray(gps["timestamp"], dtype=np.float64) / 1e6
        lat, lon = _scale_lat_lon(np.asarray(gps["lat"], float), np.asarray(gps["lon"], float), report, "gps")
        alt = _scale_alt(np.asarray(gps["alt"], float), report, "gps")
        report.note("no vehicle_global_position: position taken from the raw GPS topic (unfiltered)")
        vel = np.column_stack([np.asarray(gps.get(k, np.zeros(t.size)), float)
                               for k in ("vel_n_m_s", "vel_e_m_s", "vel_d_m_s")])
        return (t, lat, lon, alt, vel,
                np.asarray(gps.get("eph", np.full(t.size, 2.5)), float),
                np.asarray(gps.get("epv", np.full(t.size, 1.0)), float))
    return None


def _velocity_on(topics: dict[str, dict[str, np.ndarray]], t: np.ndarray) -> np.ndarray:
    lp = topics.get("vehicle_local_position")
    if lp is None or "vx" not in lp:
        return np.zeros((t.size, 3), dtype=np.float64)
    tl = np.asarray(lp["timestamp"], dtype=np.float64) / 1e6
    return np.column_stack([np.interp(t, tl, np.asarray(lp[k], float)) for k in ("vx", "vy", "vz")])


def _agl_series(topics: dict[str, dict[str, np.ndarray]], t: np.ndarray, alt_msl: np.ndarray,
                agl_from: str, report: LogReport) -> np.ndarray:
    lp = topics.get("vehicle_local_position")
    if agl_from in ("auto", "dist_bottom") and lp is not None and "dist_bottom" in lp:
        tl = np.asarray(lp["timestamp"], dtype=np.float64) / 1e6
        db = np.asarray(lp["dist_bottom"], dtype=np.float64)
        valid = np.asarray(lp.get("dist_bottom_valid", np.ones(db.size)), dtype=bool)
        if valid.any():
            report.note("agl_m from the range finder (vehicle_local_position.dist_bottom)")
            return np.interp(t, tl[valid], db[valid])
    g = topics.get("vehicle_global_position")
    if agl_from in ("auto", "terrain") and g is not None and "terrain_alt" in g:
        tv = np.asarray(g.get("terrain_alt_valid", np.zeros(g["terrain_alt"].size)), dtype=bool)
        if tv.any():
            tg = np.asarray(g["timestamp"], dtype=np.float64) / 1e6
            report.note("agl_m from the estimator's terrain altitude (vehicle_global_position.terrain_alt)")
            return alt_msl - np.interp(t, tg[tv], np.asarray(g["terrain_alt"], float)[tv])
    report.note("agl_m is height above the FIRST logged altitude (no range finder, no terrain estimate)")
    return alt_msl - float(alt_msl[0]) if alt_msl.size else np.zeros(t.size)


def _quat_at(topic: dict[str, np.ndarray] | None, t: np.ndarray) -> np.ndarray:
    """Nearest-sample quaternion lookup. Handles both `q[0]`-style and `q_0`-style field names."""
    if topic is None:
        return np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (t.size, 1))
    keys = [f"q[{i}]" for i in range(4)]
    if keys[0] not in topic:
        keys = [f"q_{i}" for i in range(4)]
    if keys[0] not in topic:
        return np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (t.size, 1))
    tq = np.asarray(topic["timestamp"], dtype=np.float64) / 1e6
    q = np.column_stack([np.asarray(topic[k], dtype=np.float64) for k in keys])
    idx = np.clip(np.searchsorted(tq, t), 0, tq.size - 1)
    out = q[idx]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return out / n
