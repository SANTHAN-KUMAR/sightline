"""Time alignment (SOLUTION_DOC §5.4 "Time alignment"): one monotonic pose series, sampled at frame times.

Three jobs:

1. **`TelemetrySeries`** — a sorted, de-duplicated, monotonic pose series with `at(t_utc) -> Telemetry`.
   Positions and scalars are `np.interp`; attitude and gimbal quaternions are **SLERP**
   (`sightline.common.geodesy.slerp`), never linear-interpolated Euler angles. Discrete fields (flight mode,
   time of day, zone) are held with nearest-previous (step) lookup, because interpolating a mode is nonsense.

2. **`time_boot_ms` -> UTC** (`BootClock`) — MAVLink and ULog stamp messages on a monotonic boot clock. One or
   more `SYSTEM_TIME` messages carry both `time_boot_ms` and `time_unix_usec`; a least-squares fit over all of
   them gives the offset (and, when several are present, a drift/skew term).

3. **`estimate_t_offset`** — the per-clip video/telemetry offset (R12). Cross-correlate barometric altitude
   against an altitude proxy read from the video (the visible take-off) and return the lag. The convention,
   used everywhere in this package, is::

       t_telemetry = t_frame + t_offset_s

   so a POSITIVE `t_offset_s` means the video clock runs BEHIND the telemetry clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from sightline.common.geodesy import slerp
from sightline.schemas import FlightMode, Telemetry

__all__ = [
    "TelemetrySeries",
    "BootClock",
    "estimate_t_offset",
    "cross_correlate_lag",
    "video_log_height_series",
]


# --- 1. the pose series ------------------------------------------------------------------------------------
@dataclass(slots=True)
class TelemetrySeries:
    """A monotonic pose series that can be sampled at any time.

    Build it with `from_telemetry()`; sample it with `at()` or `sample()`. The arrays are public so the
    evaluation harness can plot them, but they must stay sorted by `t` — use `from_telemetry`, do not append.
    """

    t: np.ndarray                      # (N,) float64 UTC POSIX seconds, strictly increasing
    lat: np.ndarray                    # (N,)
    lon: np.ndarray                    # (N,)
    alt_msl_m: np.ndarray              # (N,)
    agl_m: np.ndarray                  # (N,)
    q_body: np.ndarray                 # (N, 4) (w, x, y, z)
    q_gimbal: np.ndarray               # (N, 4)
    vel_ned_ms: np.ndarray             # (N, 3)
    h_acc_m: np.ndarray                # (N,)
    v_acc_m: np.ndarray                # (N,)
    ned_m: np.ndarray | None = None    # (N, 3) or None when the source has no local frame
    mode: list[str] = field(default_factory=list)
    time_of_day: list[str] = field(default_factory=list)
    flood_level_asl_m: np.ndarray | None = None
    weather: list[dict[str, float]] = field(default_factory=list)
    gimbal_is_earth_referenced: bool = True
    clip_id: str = ""
    #: R12: added to a FRAME time before the series is sampled. `t_telemetry = t_frame + t_offset_s`.
    t_offset_s: float = 0.0
    noise_injected: bool = False

    # -- construction ------------------------------------------------------------------------------------
    @classmethod
    def from_telemetry(cls, samples: Iterable[Telemetry], *, clip_id: str = "", t_offset_s: float = 0.0,
                       gimbal_is_earth_referenced: bool | None = None) -> "TelemetrySeries":
        """Sort by `t_utc`, drop exact-duplicate timestamps (keeping the last), and stack into arrays."""
        rows = list(samples)
        if not rows:
            raise ValueError("cannot build a TelemetrySeries from zero samples")
        order = np.argsort(np.array([s.t_utc for s in rows], dtype=np.float64), kind="stable")
        rows = [rows[i] for i in order]
        keep: list[Telemetry] = []
        for s in rows:  # de-duplicate: a later sample at the same instant supersedes the earlier one
            if keep and s.t_utc == keep[-1].t_utc:
                keep[-1] = s
            else:
                keep.append(s)
        n = len(keep)
        ned = np.array([s.ned_m if s.ned_m is not None else (np.nan,) * 3 for s in keep], dtype=np.float64)
        flood = np.array([np.nan if s.flood_level_asl_m is None else s.flood_level_asl_m for s in keep])
        return cls(
            t=np.array([s.t_utc for s in keep], dtype=np.float64),
            lat=np.array([s.lat for s in keep], dtype=np.float64),
            lon=np.array([s.lon for s in keep], dtype=np.float64),
            alt_msl_m=np.array([s.alt_msl_m for s in keep], dtype=np.float64),
            agl_m=np.array([s.agl_m for s in keep], dtype=np.float64),
            q_body=_unit_quats(np.array([s.q_body for s in keep], dtype=np.float64)),
            q_gimbal=_unit_quats(np.array([s.q_gimbal for s in keep], dtype=np.float64)),
            vel_ned_ms=np.array([s.vel_ned_ms for s in keep], dtype=np.float64).reshape(n, 3),
            h_acc_m=np.array([s.h_acc_m for s in keep], dtype=np.float64),
            v_acc_m=np.array([s.v_acc_m for s in keep], dtype=np.float64),
            ned_m=(None if np.all(np.isnan(ned)) else ned),
            mode=[str(s.mode) for s in keep],
            time_of_day=[s.time_of_day for s in keep],
            flood_level_asl_m=(None if np.all(np.isnan(flood)) else flood),
            weather=[dict(s.weather) for s in keep],
            gimbal_is_earth_referenced=(keep[0].gimbal_is_earth_referenced
                                        if gimbal_is_earth_referenced is None else gimbal_is_earth_referenced),
            clip_id=clip_id or keep[0].clip_id,
            t_offset_s=t_offset_s,
            noise_injected=any(s.noise_injected for s in keep),
        )

    # -- sampling ----------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.t.size)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.t.size else 0.0

    def covers(self, t_frame: float) -> bool:
        """True when `t_frame` (a FRAME time) falls inside the series after the offset is applied."""
        tt = t_frame + self.t_offset_s
        return bool(self.t.size and self.t[0] <= tt <= self.t[-1])

    def at(self, t_frame: float, *, frame_idx: int = -1, clamp: bool = True) -> Telemetry:
        """Interpolate one `Telemetry` at a FRAME time (the `t_offset_s` convention above is applied here).

        `clamp=False` raises when `t_frame` is outside the series instead of holding the end sample.
        """
        tt = t_frame + self.t_offset_s
        if not clamp and self.t.size and not (self.t[0] <= tt <= self.t[-1]):
            raise ValueError(f"t={tt} is outside the telemetry series [{self.t[0]}, {self.t[-1]}]")
        i, frac = self._locate(tt)
        j = min(i + 1, self.t.size - 1)
        weather = dict(self.weather[i]) if self.weather else {}
        return Telemetry(
            t_utc=float(tt),
            lat=float(np.interp(tt, self.t, self.lat)),
            lon=float(np.interp(tt, self.t, self.lon)),
            alt_msl_m=float(np.interp(tt, self.t, self.alt_msl_m)),
            agl_m=float(np.interp(tt, self.t, self.agl_m)),
            q_body=slerp(tuple(self.q_body[i]), tuple(self.q_body[j]), frac),
            q_gimbal=slerp(tuple(self.q_gimbal[i]), tuple(self.q_gimbal[j]), frac),
            gimbal_is_earth_referenced=self.gimbal_is_earth_referenced,
            ned_m=(None if self.ned_m is None else tuple(
                float(np.interp(tt, self.t, self.ned_m[:, k])) for k in range(3))),  # type: ignore[arg-type]
            vel_ned_ms=tuple(float(np.interp(tt, self.t, self.vel_ned_ms[:, k])) for k in range(3)),  # type: ignore[arg-type]
            h_acc_m=float(np.interp(tt, self.t, self.h_acc_m)),
            v_acc_m=float(np.interp(tt, self.t, self.v_acc_m)),
            mode=_as_mode(self.mode[i] if self.mode else "AUTO"),
            clip_id=self.clip_id,
            frame_idx=frame_idx,
            weather=weather,
            time_of_day=(self.time_of_day[i] if self.time_of_day else ""),
            flood_level_asl_m=(None if self.flood_level_asl_m is None
                               else float(np.interp(tt, self.t, self.flood_level_asl_m))),
            noise_injected=self.noise_injected,
        )

    def sample(self, t_frames: Sequence[float], *, clamp: bool = True) -> list[Telemetry]:
        return [self.at(float(t), frame_idx=i, clamp=clamp) for i, t in enumerate(t_frames)]

    def _locate(self, tt: float) -> tuple[int, float]:
        """Return (left index, fraction in [0, 1]) for a time on the series' own clock."""
        n = self.t.size
        if n == 1:
            return 0, 0.0
        i = int(np.searchsorted(self.t, tt, side="right") - 1)
        i = max(0, min(i, n - 2))
        span = float(self.t[i + 1] - self.t[i])
        frac = 0.0 if span <= 0.0 else (tt - float(self.t[i])) / span
        return i, float(min(1.0, max(0.0, frac)))


def _unit_quats(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return q / n


def _as_mode(value: str) -> FlightMode:
    v = (value or "AUTO").upper()
    return v if v in ("AUTO", "MANUAL", "HOLD", "RTL") else "AUTO"  # type: ignore[return-value]


# --- 2. boot clock -> UTC ----------------------------------------------------------------------------------
@dataclass(slots=True)
class BootClock:
    """Maps a monotonic autopilot clock to UTC: `t_utc = boot_s * skew + offset` (§5.4).

    MAVLink stamps every message with `time_boot_ms`; only `SYSTEM_TIME` carries `time_unix_usec`. With a
    single pair the fit is a pure offset; with several, a least-squares line also absorbs clock skew (the
    boot clock and the GNSS clock genuinely drift apart over a long flight).
    """

    offset_s: float = 0.0
    skew: float = 1.0
    n_pairs: int = 0
    residual_s: float = 0.0

    @classmethod
    def from_pairs(cls, boot_s: Sequence[float], unix_s: Sequence[float]) -> "BootClock":
        b = np.asarray(boot_s, dtype=np.float64)
        u = np.asarray(unix_s, dtype=np.float64)
        good = np.isfinite(b) & np.isfinite(u) & (u > 1.0e9)  # a zero time_unix_usec means "no GNSS time yet"
        b, u = b[good], u[good]
        if b.size == 0:
            raise ValueError("no usable SYSTEM_TIME pairs: cannot map the boot clock to UTC")
        if b.size == 1 or float(np.ptp(b)) < 1.0:
            return cls(offset_s=float(u[0] - b[0]), skew=1.0, n_pairs=int(b.size), residual_s=0.0)
        skew, offset = np.polyfit(b, u, 1)
        resid = float(np.sqrt(np.mean((u - (skew * b + offset)) ** 2)))
        return cls(offset_s=float(offset), skew=float(skew), n_pairs=int(b.size), residual_s=resid)

    def to_utc(self, boot_s: float | np.ndarray) -> Any:
        return np.asarray(boot_s, dtype=np.float64) * self.skew + self.offset_s

    def utc_of_ms(self, time_boot_ms: float | np.ndarray) -> Any:
        return self.to_utc(np.asarray(time_boot_ms, dtype=np.float64) / 1.0e3)


# --- 3. per-clip time offset (R12) -------------------------------------------------------------------------
def cross_correlate_lag(t_a: Sequence[float], y_a: Sequence[float], t_b: Sequence[float], y_b: Sequence[float],
                        *, max_lag_s: float = 10.0, step_s: float = 0.02,
                        differentiate: bool = True) -> tuple[float, float]:
    """Return `(lag_s, peak_correlation)` such that `y_a(t + lag_s)` best matches `y_b(t)`.

    Both series are resampled onto a common uniform grid, optionally differentiated (which removes the
    arbitrary datum difference between "height above take-off" and "height proxy from video"), z-normalised,
    and correlated. The peak is refined to sub-grid accuracy by fitting a parabola to its two neighbours.
    """
    ta, ya = np.asarray(t_a, dtype=np.float64), np.asarray(y_a, dtype=np.float64)
    tb, yb = np.asarray(t_b, dtype=np.float64), np.asarray(y_b, dtype=np.float64)
    if ta.size < 4 or tb.size < 4:
        raise ValueError("need at least 4 samples in each series to estimate a lag")
    t0 = max(float(ta[0]), float(tb[0]))
    t1 = min(float(ta[-1]), float(tb[-1]))
    if t1 - t0 <= 4.0 * step_s:
        raise ValueError("the two series barely overlap; widen the window or reduce step_s")
    grid = np.arange(t0, t1, step_s)
    ga = np.interp(grid, ta, ya)
    gb = np.interp(grid, tb, yb)
    if differentiate:
        ga, gb = np.diff(ga), np.diff(gb)
    ga, gb = _znorm(ga), _znorm(gb)
    max_k = int(round(max_lag_s / step_s))
    max_k = max(1, min(max_k, ga.size - 2))
    lags = np.arange(-max_k, max_k + 1)
    corr = np.empty(lags.size, dtype=np.float64)
    n = ga.size
    for i, k in enumerate(lags):
        # y_a(t + lag) ~= y_b(t)  =>  shifting A left by k samples must line it up with B
        if k >= 0:
            a, b = ga[k:], gb[: n - k]
        else:
            a, b = ga[: n + k], gb[-k:]
        corr[i] = 0.0 if a.size < 4 else float(np.dot(a, b) / a.size)
    p = int(np.argmax(corr))
    lag = float(lags[p]) * step_s
    if 0 < p < corr.size - 1:  # parabolic sub-sample refinement
        y0, y1, y2 = corr[p - 1], corr[p], corr[p + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            lag += float(0.5 * (y0 - y2) / denom) * step_s
    return lag, float(corr[p])


def estimate_t_offset(frame_times: Sequence[float], frame_height_proxy: Sequence[float],
                      telem_times: Sequence[float], telem_alt_m: Sequence[float], *,
                      max_lag_s: float = 10.0, step_s: float = 0.02,
                      min_correlation: float = 0.3) -> tuple[float, float]:
    """Estimate the per-clip `t_offset_s` from the visible take-off (§5.4, R12).

    `frame_height_proxy` is any monotone-in-height signal read from the video — `video_log_height_series()`
    produces one from optical-flow scale. `telem_alt_m` is the barometric/GNSS altitude on the telemetry
    clock. Returns `(t_offset_s, peak_correlation)` with the package convention
    `t_telemetry = t_frame + t_offset_s`.

    Raises `ValueError` when the peak correlation is below `min_correlation`: a bad alignment silently
    applied is worse than no alignment, because every geolocation downstream inherits it.
    """
    lag, peak = cross_correlate_lag(frame_times, frame_height_proxy, telem_times, telem_alt_m,
                                    max_lag_s=max_lag_s, step_s=step_s, differentiate=True)
    if peak < min_correlation:
        raise ValueError(
            f"take-off cross-correlation peaked at {peak:.3f} (< {min_correlation}); refusing to guess "
            f"t_offset_s. Set it by hand on the clip, or widen max_lag_s."
        )
    # cross_correlate_lag returns the shift that maps the FRAME series onto the TELEMETRY series, which is
    # exactly the quantity added to a frame time to land on the telemetry clock -- but with the opposite sign,
    # because it shifts the frame SIGNAL rather than the frame CLOCK.
    return -lag, peak


def _znorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = float(np.std(x))
    return (x - float(np.mean(x))) / (s if s > 1e-12 else 1.0)


def video_log_height_series(frames: Iterable[np.ndarray], times_s: Sequence[float], *,
                            downscale_to: int = 320, max_features: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Read a relative log-height signal out of a nadir video: `log h(t)` up to an additive constant.

    For a nadir camera at height `h`, the whole ground plane scales by `h_k / h_{k+1}` between consecutive
    frames, so the similarity-transform scale `s_k` estimated between them satisfies
    `log h_{k+1} = log h_k - log s_k`. Accumulating `-log s_k` gives `log h` up to the unknown initial height,
    which is exactly what `estimate_t_offset` needs (it differentiates the signal anyway).

    Uses Lucas-Kanade tracking + a partial-affine RANSAC fit (OpenCV, CPU). Frames may be BGR or grey.
    Returns `(times, log_h_rel)`; both have one entry per input frame.
    """
    import cv2  # local import: ingest must stay importable without opencv for metadata-only replay

    times = np.asarray(times_s, dtype=np.float64)
    log_h = [0.0]
    prev_grey = None
    prev_pts = None
    for frame in frames:
        grey = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if grey.shape[1] > downscale_to:
            scale = downscale_to / grey.shape[1]
            grey = cv2.resize(grey, (downscale_to, max(1, int(round(grey.shape[0] * scale)))))
        if prev_grey is not None:
            if prev_pts is None or len(prev_pts) < 12:
                log_h.append(log_h[-1])
            else:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_grey, grey, prev_pts, None)
                ok = (status.reshape(-1) == 1)
                if ok.sum() < 12:
                    log_h.append(log_h[-1])
                else:
                    m, _ = cv2.estimateAffinePartial2D(prev_pts[ok], nxt[ok], method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.0)
                    if m is None:
                        log_h.append(log_h[-1])
                    else:
                        s = float(np.hypot(m[0, 0], m[0, 1]))
                        log_h.append(log_h[-1] - (np.log(s) if s > 1e-6 else 0.0))
        prev_grey = grey
        prev_pts = cv2.goodFeaturesToTrack(grey, maxCorners=max_features, qualityLevel=0.01, minDistance=8)
    out = np.asarray(log_h[: times.size], dtype=np.float64)
    if out.size < times.size:  # fewer frames than timestamps: trim the clock to match
        times = times[: out.size]
    return times, out
