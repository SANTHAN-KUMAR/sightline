"""Simulator telemetry noise injector (SOLUTION_DOC §5.7 "In the simulator").

    "Ground-truth pose makes sim geolocation error ~0, which would tune the dedup radius wrong for real
    footage. Inject the noise model above into the exported telemetry (GNSS 2.5 m random walk, 1.5 deg yaw
    bias, 0.5 deg pitch/roll, 1 m barometric), and keep a switch to turn it off for debugging."

Design decisions, and why
-------------------------
* **GNSS is a first-order Gauss-Markov walk, not a free random walk.** A free walk's variance grows without
  bound, so a five-minute sortie would end tens of metres out and the measured geolocation error would depend
  on clip length. A GM process (τ = 30 s) wanders like a real receiver but has a *stationary* 1σ of exactly
  2.5 m per horizontal axis — which is the same quantity `budget.gnss_h_m` and `Telemetry.h_acc_m` mean, so
  the injected noise and the published `h_acc_m` describe the same thing. Initialised from the stationary
  distribution, so sample 0 is already correctly distributed.
* **Yaw is a per-run BIAS, drawn once.** §5.7: "Averaging N frames shrinks the random terms by ~sqrt(N) but not
  the biases (yaw, boresight), which is why single-pass records still carry the full yaw term." A per-sample
  yaw jitter would average away and would make the sim look better than reality.
* **Pitch/roll is per-sample white noise**, and the SAME draw is applied to `q_body` and `q_gimbal`: a real
  gimbal's attitude estimate comes from the vehicle IMU, so the errors are correlated, and the doc's budget
  lumps "vehicle pitch/roll" into the *pointing* term of the camera ray. Perturbing only `q_body` would leave
  the chain's ray untouched, because the simulator sets `gimbal_is_earth_referenced = True`.
* **Barometric altitude is a slow GM walk** (τ = 60 s, σ = 1 m): baro drifts with pressure, it does not jitter.
  `alt_msl_m`, `agl_m` and the down component of `ned_m` all move together, so the telemetry stays consistent.
* **Ground truth is preserved.** `apply()` returns a *new* `Telemetry`; the original object is never mutated,
  and every pair is retained on `self.samples` so `sightline.eval` can measure true geolocation error.

Reproducibility: everything comes from one `numpy.random.default_rng(seed)`. The same seed and the same input
sequence give bit-identical output.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass

import numpy as np

from sightline.common.geodesy import euler_to_quat, ne_between, offset_ne_geodesic
from sightline.schemas import Telemetry

__all__ = ["NoiseConfig", "NoisySample", "TelemetryNoise", "inject"]


@dataclass(frozen=True, slots=True)
class NoiseConfig:
    """§5.7's simulator noise model. `enabled = False` is the debugging switch (a pass-through)."""

    enabled: bool = True
    seed: int = 0
    # GNSS: stationary 1σ per horizontal axis, and its correlation time
    gnss_sigma_m: float = 2.5
    gnss_tau_s: float = 30.0
    reported_h_acc_m: float = 2.5  # what the receiver claims; the chain reads this for the budget's δp
    # barometric altitude
    baro_sigma_m: float = 1.0
    baro_tau_s: float = 60.0
    reported_v_acc_m: float = 1.0
    # attitude
    yaw_bias_sigma_deg: float = 1.5  # magnetometer heading bias, drawn once per run
    attitude_sigma_deg: float = 0.5  # vehicle pitch/roll, per sample
    perturb_gimbal: bool = True  # the ray the chain actually uses; leave True for realistic geolocation error
    perturb_body: bool = True
    keep_samples: bool = True  # retain (truth, noisy) pairs for the evaluation harness
    max_dt_s: float = 5.0  # a gap longer than this restarts the walk from the stationary distribution

    @classmethod
    def off(cls, **kw) -> "NoiseConfig":
        return cls(enabled=False, **kw)

    @classmethod
    def from_env(cls, **kw) -> "NoiseConfig":
        """`SIGHTLINE_GEO_NOISE=0` turns injection off; `SIGHTLINE_GEO_NOISE_SEED` sets the seed."""
        env = os.environ.get("SIGHTLINE_GEO_NOISE", "1").strip().lower()
        enabled = env not in ("0", "false", "off", "no")
        seed = int(os.environ.get("SIGHTLINE_GEO_NOISE_SEED", kw.pop("seed", 0)))
        return cls(enabled=enabled, seed=seed, **kw)


@dataclass(slots=True)
class NoisySample:
    """One (truth, noisy) pair plus the error that was injected, in the units the eval harness reports."""

    truth: Telemetry
    noisy: Telemetry
    d_north_m: float
    d_east_m: float
    d_alt_m: float
    d_yaw_deg: float
    d_pitch_deg: float
    d_roll_deg: float

    @property
    def d_horizontal_m(self) -> float:
        return math.hypot(self.d_north_m, self.d_east_m)


class TelemetryNoise:
    """Seeded, reproducible injector. One instance per clip (the yaw bias is per-instance, i.e. per run).

    ``noise = TelemetryNoise(NoiseConfig(seed=7)); noisy = noise.apply(truth_telemetry)``
    """

    def __init__(self, cfg: NoiseConfig | None = None) -> None:
        self.cfg = cfg or NoiseConfig()
        self.samples: list[NoisySample] = []
        self.reset()

    # -- state -----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Redraw every per-run quantity from the configured seed. Called by `__init__`; makes reruns identical."""
        c = self.cfg
        self._rng = np.random.default_rng(c.seed)
        # per-run biases
        self._yaw_bias_deg = float(self._rng.normal(0.0, c.yaw_bias_sigma_deg))
        # stationary initial conditions for the Gauss-Markov walks
        self._e_n = float(self._rng.normal(0.0, c.gnss_sigma_m))
        self._e_e = float(self._rng.normal(0.0, c.gnss_sigma_m))
        self._e_h = float(self._rng.normal(0.0, c.baro_sigma_m))
        self._t_prev: float | None = None
        self.samples.clear()

    @property
    def yaw_bias_deg(self) -> float:
        """The heading bias this run drew. Constant for the whole clip, by design."""
        return self._yaw_bias_deg

    def _advance(self, t_utc: float) -> None:
        c = self.cfg
        if self._t_prev is None:
            self._t_prev = float(t_utc)
            return  # the first sample keeps its stationary draw from reset()
        dt = float(t_utc) - self._t_prev
        self._t_prev = float(t_utc)
        if dt > c.max_dt_s or dt <= 0.0:
            # long gap (or a non-monotonic timestamp): re-draw from the stationary distribution
            self._e_n = float(self._rng.normal(0.0, c.gnss_sigma_m))
            self._e_e = float(self._rng.normal(0.0, c.gnss_sigma_m))
            self._e_h = float(self._rng.normal(0.0, c.baro_sigma_m))
            return
        phi_g = math.exp(-dt / c.gnss_tau_s)
        q_g = c.gnss_sigma_m * math.sqrt(max(0.0, 1.0 - phi_g * phi_g))
        self._e_n = phi_g * self._e_n + float(self._rng.normal(0.0, q_g))
        self._e_e = phi_g * self._e_e + float(self._rng.normal(0.0, q_g))
        phi_b = math.exp(-dt / c.baro_tau_s)
        q_b = c.baro_sigma_m * math.sqrt(max(0.0, 1.0 - phi_b * phi_b))
        self._e_h = phi_b * self._e_h + float(self._rng.normal(0.0, q_b))

    # -- application -----------------------------------------------------------------------------------------
    def apply(self, tel: Telemetry) -> Telemetry:
        """Return a noisy copy of `tel`. The input is never mutated; the truth is kept on `self.samples`."""
        return self.apply_pair(tel).noisy

    def apply_pair(self, tel: Telemetry) -> NoisySample:
        c = self.cfg
        if not c.enabled:
            return NoisySample(tel, tel, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        self._advance(tel.t_utc)
        d_pitch = float(self._rng.normal(0.0, c.attitude_sigma_deg))
        d_roll = float(self._rng.normal(0.0, c.attitude_sigma_deg))
        d_yaw = self._yaw_bias_deg

        lat, lon = offset_ne_geodesic(float(tel.lat), float(tel.lon), self._e_n, self._e_e)
        noisy = dataclasses.replace(
            tel,
            lat=lat,
            lon=lon,
            alt_msl_m=float(tel.alt_msl_m) + self._e_h,
            agl_m=float(tel.agl_m) + self._e_h,
            h_acc_m=c.reported_h_acc_m,
            v_acc_m=c.reported_v_acc_m,
            noise_injected=True,
        )
        if tel.ned_m is not None:
            n, e, d = tel.ned_m
            noisy.ned_m = (n + self._e_n, e + self._e_e, d - self._e_h)  # down decreases as altitude rises
        if c.perturb_body:
            noisy.q_body = _perturb(tel.q_body, d_roll, d_pitch, d_yaw)
        if c.perturb_gimbal:
            noisy.q_gimbal = _perturb(tel.q_gimbal, d_roll, d_pitch, d_yaw)

        s = NoisySample(tel, noisy, self._e_n, self._e_e, self._e_h, d_yaw, d_pitch, d_roll)
        if c.keep_samples:
            self.samples.append(s)
        return s

    def apply_stream(self, tels: list[Telemetry]) -> list[Telemetry]:
        return [self.apply(t) for t in tels]

    # -- what the evaluation harness reads --------------------------------------------------------------------
    def truth_log(self) -> list[Telemetry]:
        """The ground-truth poses, in the order they were injected (guardrail: nothing is discarded)."""
        return [s.truth for s in self.samples]

    def error_table(self) -> dict[str, np.ndarray]:
        """Arrays of the injected errors, for the eval harness's sim-vs-truth geolocation report."""
        return {
            "d_north_m": np.array([s.d_north_m for s in self.samples]),
            "d_east_m": np.array([s.d_east_m for s in self.samples]),
            "d_horizontal_m": np.array([s.d_horizontal_m for s in self.samples]),
            "d_alt_m": np.array([s.d_alt_m for s in self.samples]),
            "d_yaw_deg": np.array([s.d_yaw_deg for s in self.samples]),
            "d_pitch_deg": np.array([s.d_pitch_deg for s in self.samples]),
            "d_roll_deg": np.array([s.d_roll_deg for s in self.samples]),
        }

    def horizontal_offset_m(self, sample: NoisySample) -> float:
        """The true horizontal displacement between a truth pose and its noisy copy, on the geodesic."""
        n, e = ne_between(sample.truth.lat, sample.truth.lon, sample.noisy.lat, sample.noisy.lon)
        return math.hypot(n, e)


def _quat_mul(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
    """Hamilton product (w, x, y, z). `common.geodesy` has no quaternion product, so it lives here."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _perturb(q: tuple[float, float, float, float], d_roll: float, d_pitch: float, d_yaw: float):
    """Perturb an attitude by LEFT-multiplying a small rotation of the NED frame.

    ``q_noisy = R(d_yaw about Down) · R(d_pitch about East) · R(d_roll about North) · q``

    Left multiplication is the physically right model — a magnetometer bias rotates the estimated north, and an
    IMU levelling error tilts the estimated local-level frame — and it reproduces the doc's sensitivities
    exactly: a yaw error leaves a nadir ray untouched (heading term ``h·tanθ·δψ`` -> 0 at θ = 0) while a
    pitch/roll error deflects it by the full angle (pointing term ``h·sec²θ·δθ`` -> h·δθ at θ = 0).

    It is also singularity-free, which the obvious implementation is not: `quat_to_euler` cannot separate roll
    from yaw at pitch = ±90, so adding euler errors and rebuilding would silently rotate a **nadir** camera's
    azimuth by its whole heading (measured: a q_gimbal of (roll 0, pitch -90, yaw 30) round-trips through
    `quat_to_euler` as (180, -90, 180), i.e. an azimuth of 0 instead of 30). See tests/test_geo.py.
    """
    return _quat_mul(euler_to_quat(d_roll, d_pitch, d_yaw), q)


def inject(tels: list[Telemetry], cfg: NoiseConfig | None = None) -> tuple[list[Telemetry], TelemetryNoise]:
    """One-shot helper: `noisy, injector = inject(truth_poses, NoiseConfig(seed=7))`.

    The returned injector holds the ground truth (`injector.truth_log()`) and the injected errors
    (`injector.error_table()`), so nothing is lost.
    """
    n = TelemetryNoise(cfg)
    return n.apply_stream(tels), n
