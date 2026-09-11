"""The pipeline spine: one clip in, ranked records out (SOLUTION_DOC section 3.2).

    uv run python -m sightline.pipeline _artifacts/dataset/seed23_alt45 --out _artifacts/run1

This is the module that makes the separate lanes a system. It owns no algorithms of its own - every stage is
the lane that owns it - and its whole job is to hold the contract between them and to fail loudly when a stage
is missing rather than quietly skipping it.

    ingest  -> FrameBundle          (F7,  sightline.ingest)
    detect  -> list[Detection]      (F8,  sightline.detect, or ground truth via --detector truth)
    geo     -> GeoFix per detection (F13, sightline.geo)
    track   -> Track                (F11, sightline.track)
    dedup   -> Record               (F12, sightline.dedup)
    triage  -> ranked Record        (F14, sightline.triage)
    store   -> SQLite + outbox      (F18, sightline.store)
    export  -> GeoJSON/KML/CoT      (F14/F17, sightline.export)
    coverage-> POD raster           (F16, sightline.coverage)

`--detector truth` replays the simulator's own labels instead of a model. That is not a shortcut: it is how the
rest of the chain is exercised and debugged before a model exists, and it gives the ceiling every real detector
is measured against. Any run made that way is stamped `detector: "truth"` in the manifest so a number from it
can never be mistaken for a detection result.

**`FramePipeline` is the per-frame spine and it is shared.** The offline replay below and the real-time
mission runner (`sightline.mission.live`) both call `FramePipeline.process()` and nothing else, so the live
demo and the replay harness cannot drift apart - the only difference between them is where the frames come
from. `tests/test_live_mission.py::test_live_and_offline_agree` feeds the same frames to both and compares the
records; if anyone reimplements a stage in one path, that test fails.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sightline.ingest.spec import Q_FRD_FROM_CAM, gimbal_quat_from_euler, quat_mul
from sightline.schemas import (SCHEMA_VERSION, Detection, GeoFix, Intrinsics, Record, Telemetry,
                               feature_collection)

REPO = Path(__file__).resolve().parents[1]


# --- the gimbal frame, and the one place two lanes are reconciled -------------------------------------------
# `Telemetry.q_gimbal` is FROZEN as "rotates camera(optical) -> NED", so a nadir camera is the IDENTITY
# quaternion and `Telemetry.gimbal_pitch_deg()` reports -90 for it. That is what `sightline.ingest.spec`
# builds, what every ingest reader emits, and what `sightline.track` and `sightline.coverage` read.
#
# `sightline.geo` reads the SAME field as a camera-FRD -> NED quaternion (`ChainConfig.gimbal_frame="frd"`,
# its default), for which a nadir camera is (0.7071, 0, -0.7071, 0) instead. That configuration is the one
# validated against the simulator (48 observations, 5.6 px RMS), so it is not being second-guessed here.
#
# The two differ by exactly `spec.Q_FRD_FROM_CAM`, which is why the conversion below is spec's own constant
# rather than a matrix typed out by hand. Telemetry is stored, exported and handed to every other lane in the
# CONTRACTUAL form; only the call into `sightline.geo` is converted, in one named place.
#
# This is a WORKAROUND for a real cross-lane defect, not a design: see docs/TRACKER.md. Measured
# 2026-09-11 at gimbal yaw 0 and 90: `sightline.geo` (through this conversion) and `sightline.track` agree on
# where every pixel lands, at both yaws. `sightline.coverage.footprint.ground_footprint` is rotated 90 deg
# from BOTH of them at both yaws - at yaw 0 it puts the frame's wide axis north-south, while geo and track put
# image-right due east (wide axis east-west, which is what `survey.py`'s east line spacing assumes). Which of
# the two is "right" depends on the gimbal yaw the survey actually flies, and the telemetry CSV has no
# `gimbal_yaw_deg` column to say - so the live map's footprint comes from `sightline.geo.footprint_ned`, the
# same chain that geolocates the pins inside it, rather than from coverage.
def to_geo_gimbal(tel: Telemetry) -> Telemetry:
    """Re-express the contractual optical->NED `q_gimbal` as the camera-FRD->NED quaternion `sightline.geo`
    expects. Exact inverse of `sightline.ingest.spec.gimbal_quat_from_frd_quat`."""
    import dataclasses                                     # noqa: PLC0415

    w, x, y, z = Q_FRD_FROM_CAM
    return dataclasses.replace(tel, q_gimbal=quat_mul(tel.q_gimbal, (w, -x, -y, -z)))


def nadir_gimbal_quat(yaw_deg: float = 0.0) -> tuple[float, float, float, float]:
    """The contractual `q_gimbal` for a straight-down camera. Never hand-roll this (spec.py's own rule)."""
    return gimbal_quat_from_euler(0.0, -90.0, yaw_deg)

#: `truth` replays the simulator's labels; `rgb` runs the trained detector. `model` is the pre-2026-09-11
#: spelling of `rgb` and is accepted so old commands keep working; both resolve to `rgb` in the manifest.
DETECTORS: tuple[str, ...] = ("truth", "rgb", "model")


def resolve_detector(name: str) -> str:
    if name not in DETECTORS:
        raise ValueError(f"unknown detector {name!r}; expected one of {DETECTORS}")
    return "rgb" if name == "model" else name


# --- stage 1: frames ---------------------------------------------------------------------------------------
def iter_frames(clip_dir: Path, *, every: int = 1):
    """Yield (frame_idx, telemetry, intrinsics, label_json_path) for a capture run.

    Reads the capture format written by `tools/capture/run.py` directly rather than going through the video
    path: a simulator run is a directory of PNGs plus telemetry.csv, and the ingest lane's `CaptureRunReader`
    understands exactly that.
    """
    import csv

    cal = json.loads((REPO / "data/scene/camera_survey.json").read_text())
    tele = clip_dir / "telemetry.csv"
    if not tele.exists():
        raise SystemExit(f"{tele} not found - is {clip_dir} a capture run?")
    with tele.open(newline="", encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i % every:
                continue
            k = int(row["frame_idx"])
            intr = Intrinsics(width_px=int(row["width_px"]), height_px=int(row["height_px"]),
                              fx=cal["f_px"], fy=cal["f_px"], cx=cal["cx"], cy=cal["cy"], source="calibration")
            t = Telemetry(
                t_utc=float(row["t_utc"]), lat=float(row["lat"]), lon=float(row["lon"]),
                alt_msl_m=float(row["alt_msl_m"]), agl_m=float(row["agl_m"]),
                q_body=(float(row["q_w"]), float(row["q_x"]), float(row["q_y"]), float(row["q_z"])),
                # The CONTRACTUAL gimbal quaternion, built by the ingest lane's own helper rather than
                # hand-rolled. `survey.py` records gimbal pitch as a scalar column, not a quaternion.
                q_gimbal=nadir_gimbal_quat(),
                gimbal_is_earth_referenced=True, mode=row.get("mode", "AUTO") or "AUTO",
                clip_id=row["clip_id"], frame_idx=k,
                flood_level_asl_m=float(row["flood_level_asl_m"]) if row.get("flood_level_asl_m") else None,
            )
            stem = f"{row['clip_id']}_{k:05d}"
            yield k, t, intr, clip_dir / "labels" / f"{stem}.json"


# --- stage 2: detections -----------------------------------------------------------------------------------
def detections_from_truth(label_json: Path) -> list[Detection]:
    """Replay the simulator's exact labels as if a perfect detector had produced them."""
    if not label_json.exists():
        return []
    return detections_from_label_records(json.loads(label_json.read_text()))


def detections_from_label_records(labels: list[dict[str, Any]]) -> list[Detection]:
    """The label dicts `tools/capture/labels.py` writes -> `Detection`. Used by the replay AND by the live
    runner, which has the same dicts in memory and must not re-implement the mapping."""
    out = []
    for m in labels:
        x1, y1, x2, y2 = m["bbox_px"]
        out.append(Detection(bbox_px=(float(x1), float(y1), float(x2), float(y2)), score=1.0,
                             cls=m.get("cls", "human"), modality="rgb",
                             posture=m.get("pose", "unknown"), posture_conf=1.0,
                             submersion=m.get("submersion", "unknown"), submersion_conf=1.0,
                             occlusion=m.get("occlusion")))
    return out


#: One loaded detector per (weights, device). Constructing `RgbDetector` loads the weights off disk and onto
#: the GPU, so building one per frame would make the live loop unusable - it is the difference between loading
#: a model once and loading it every ~1.2 s for a 20-minute flight.
_DETECTOR_CACHE: dict[tuple[str, str], Any] = {}


def _detector(weights: str, device: str = "cuda:0"):
    """The trained detector, built once per (weights, device). Imports torch lazily."""
    key = (str(weights), device)
    got = _DETECTOR_CACHE.get(key)
    if got is None:
        from sightline.detect.rgb import RAW_CONF, DetectorConfig, RgbDetector  # noqa: PLC0415

        # `raw_conf` is the LOW per-tile threshold SOLUTION_DOC 5.5c step 4 asks for ("run a low confidence
        # threshold and recover precision downstream"). The caller's `conf` is the frozen OPERATING point and
        # is applied once, below, after tile merging - filtering inside the tiles instead would throw away
        # boxes the seam-union merge is there to recover.
        got = RgbDetector(DetectorConfig(weights=str(weights), raw_conf=RAW_CONF, device=device))
        _DETECTOR_CACHE[key] = got
    return got


def detections_from_model_array(rgb: np.ndarray, weights: str, conf: float) -> list[Detection]:
    """Run the trained detector on an in-memory **BGR** frame - the live path never writes the PNG first."""
    dets = _detector(weights).detect(rgb)
    return [d for d in dets if d.score >= conf]


def detections_from_model(frame_png: Path, weights: str, conf: float) -> list[Detection]:
    """Run the trained detector on a frame file. Imported lazily: torch must not load for a truth replay."""
    import cv2  # noqa: PLC0415

    frame = cv2.imread(str(frame_png))                       # BGR, which is what the detector expects
    if frame is None:
        raise FileNotFoundError(f"could not read frame for detection: {frame_png}")
    return detections_from_model_array(frame, weights, conf)


# --- latency (§5.11: every stage measured, every number carrying its slice) ---------------------------------
@dataclass(slots=True)
class StageLatency:
    """Wall clock for ONE frame, in milliseconds. `domain` is stamped by whoever aggregates these."""

    capture_ms: float = 0.0     # simulator -> numpy buffers (live only; 0 on replay)
    detect_ms: float = 0.0
    geo_ms: float = 0.0
    track_ms: float = 0.0
    dedup_ms: float = 0.0
    triage_ms: float = 0.0
    publish_ms: float = 0.0     # handing the changed records to the C2 (live only)

    STAGES = ("capture", "detect", "geo", "track", "dedup", "triage", "publish")

    def pipeline_ms(self) -> float:
        """Detection through triage: the part the replay harness and the live runner share exactly."""
        return self.detect_ms + self.geo_ms + self.track_ms + self.dedup_ms + self.triage_ms

    def total_ms(self) -> float:
        return self.capture_ms + self.pipeline_ms() + self.publish_ms

    def as_dict(self) -> dict[str, float]:
        return {f"{s}_ms": round(getattr(self, f"{s}_ms"), 3) for s in self.STAGES}

    def dominant_stage(self) -> str:
        return max(self.STAGES, key=lambda s: getattr(self, f"{s}_ms"))


def latency_report(samples: list[StageLatency], *, domain: str = "sim", label: str = "") -> dict[str, Any]:
    """Median / p90 / max per stage plus the dominant one. Never averaged across domains (hard rule 5)."""
    if not samples:
        return {"domain": domain, "label": label, "n": 0, "stages": {}, "dominant_stage": None}
    stages: dict[str, dict[str, float]] = {}
    for s in StageLatency.STAGES:
        v = np.array([getattr(x, f"{s}_ms") for x in samples], dtype=float)
        stages[s] = {"median_ms": round(float(np.median(v)), 3),
                     "p90_ms": round(float(np.percentile(v, 90)), 3),
                     "max_ms": round(float(v.max()), 3),
                     "mean_ms": round(float(v.mean()), 3)}
    tot = np.array([x.total_ms() for x in samples], dtype=float)
    pipe = np.array([x.pipeline_ms() for x in samples], dtype=float)
    dominant = max(StageLatency.STAGES, key=lambda s: stages[s]["median_ms"])
    return {
        "domain": domain, "label": label, "n": len(samples), "stages": stages,
        "end_to_end_ms": {"median": round(float(np.median(tot)), 3),
                          "p90": round(float(np.percentile(tot, 90)), 3),
                          "max": round(float(tot.max()), 3)},
        "pipeline_ms": {"median": round(float(np.median(pipe)), 3),
                        "p90": round(float(np.percentile(pipe, 90)), 3)},
        "dominant_stage": dominant,
        "dominant_share": round(float(stages[dominant]["median_ms"] / max(np.median(tot), 1e-9)), 4),
    }


# --- what one frame produced -------------------------------------------------------------------------------
#: Fields that identify a record's CONTENT - what the drone has LEARNED about it. "Did this record change?"
#: must not fire on a version bump alone, and it must not fire on the clock either.
#:
#: `score` is deliberately NOT here. The triage score decays continuously with time since the incident, so
#: with `now_utc` pinned per frame it changes on EVERY frame for EVERY record. Including it made the live
#: runner push all 32 records on all 320 frames - 1019 uploads for 32 records - and, worse, re-push the SAME
#: version with a different score on any frame that confirmed no new track (dedup bumps `version` only when
#: it re-clusters). The C2's idempotent upsert correctly rejected that as a stale version, one 409 jammed the
#: head of the outbox queue, and 1015 of the 1019 records never reached the map. Measured 2026-09-11.
#:
#: `priority_rank` IS here, because the map sizes, labels and sorts its triage list by it, and a stale rank
#: puts three records numbered "0" on a commander's screen. It is safe to include because a rank only shifts
#: when the record set changes, which only happens on a frame where dedup re-clustered - and re-clustering
#: bumps every record's version, so the push carries a fresh version. `C2Client.push_records` refuses a
#: same-version re-push regardless, so a future change here cannot resurrect the 409.
_FINGERPRINT_FIELDS = ("status", "cls", "lat", "lon", "h_acc_m", "confidence", "priority_rank",
                       "n_observations", "n_tracks_merged", "posture", "submersion", "occlusion",
                       "motion_state", "count_estimate", "zone", "thermal_hot", "dismissed_reason")


def record_fingerprint(rec: Record) -> tuple:
    """A record's content, rounded to what a map can show. `version` is deliberately excluded."""
    out: list[Any] = []
    for f in _FINGERPRINT_FIELDS:
        v = getattr(rec, f)
        out.append(round(v, 7) if isinstance(v, float) else v)
    return tuple(out)


@dataclass
class FrameResult:
    """Everything one frame produced, including what the map should be told about."""

    frame_idx: int
    t_utc: float
    detections: list[Detection] = field(default_factory=list)
    fixes: list[GeoFix | None] = field(default_factory=list)
    n_located: int = 0
    tracks_touched: int = 0
    records: list[Record] = field(default_factory=list)   # every active record, ranked
    changed: list[Record] = field(default_factory=list)   # only those whose content moved this frame
    latency: StageLatency = field(default_factory=StageLatency)


class FramePipeline:
    """detect -> geo -> track -> dedup -> triage, for one frame at a time.

    Shared by the offline replay (`run()` below) and the live mission runner. Construct it once per pass,
    call :meth:`process` per frame, then :meth:`close`.

    **Camera-motion compensation is ON by default, and a survey is unusable without it.** A shutter step of
    9.8 m at 45 m AGL with f = 2548.7 px moves the whole image 555 px, against a survivor box about 60 px
    wide - consecutive frames have literally zero overlap, so IoU association cannot link anything and the
    run yields detections, geolocations and then no tracks at all. Measured on the fixture: `cmc_enabled=
    False` gives 15 detections and 0 tracks; `True` gives 15 detections and 4 tracks, one per survivor.
    `tests/test_live_mission.py::test_camera_motion_compensation_is_not_optional_for_a_survey` pins it.

    Dedup and triage run EVERY frame, not once at the end: that is what makes a record appear on the map
    while the drone is still flying, and running the replay the same way is what keeps the two paths
    comparable. `Deduplicator` is incremental by design - it re-clusters every track it has ever seen and
    re-attaches clusters to their existing record ids - so the final state after the last frame is the same
    clustering a single terminal call would have produced.
    """

    def __init__(self, *, detector: str = "truth", weights: str = "", conf: float = 0.25,
                 tracker_fps: float = 1.0, track_window_s: float = 0.0,
                 cmc_enabled: bool = True, clip_id: str = "",
                 pass_id: int = 0, dedup_cfg: Any = None, triage_ctx: Any = None,
                 geo_cfg: Any = None, noise_seed: int | None = None,
                 incident_t0_utc: float | None = None, incident_hours_ago: float = 0.0,
                 water_temp_c: float | None = None, domain: str = "sim"):
        from sightline.dedup import DedupConfig, Deduplicator      # noqa: PLC0415
        from sightline.geo import ChainConfig                      # noqa: PLC0415
        from sightline.triage import DEFAULT_WATER_TEMP_C          # noqa: PLC0415

        self.detector = resolve_detector(detector)
        self.weights = weights
        self.conf = conf
        self.tracker_fps = tracker_fps
        # SOLUTION_DOC 5.6 rule 3 is "3 hits within 2 seconds", which silently assumes the ~5 FPS loop
        # 5.11 sizes. This loop is latency-bound: MEASURED median 1.368 s per frame (4K capture + tiled
        # inference on a 4060 sharing VRAM with the editor), i.e. 0.73 FPS. At that rate three hits span
        # 2.74 s and the gate can NEVER close - which is exactly why a correct chain produced 0 records.
        # The rule's intent is "seen repeatedly in quick succession", so the window is scaled to the
        # frame rate rather than the rule being abandoned: it stays >= the doc's 2 s, and never shrinks.
        self._explicit_window = float(track_window_s)
        self._frame_dts: list[float] = []
        self._last_frame_t: float = 0.0
        self.track_window_s = float(track_window_s) if track_window_s > 0 else max(
            2.0, (3 - 1) / max(1e-6, tracker_fps) * 1.25)
        self.cmc_enabled = cmc_enabled
        self.clip_id = clip_id
        self.pass_id = pass_id
        self.geo_cfg = geo_cfg if geo_cfg is not None else ChainConfig()
        self.dedup = Deduplicator(dedup_cfg if dedup_cfg is not None else DedupConfig())
        # TRIAGE IS NOT OPTIONAL. `rank_records(records, None)` sorts without scoring, so every record keeps
        # score 0.0, every component keeps its default, and the "triage list" is an arbitrary order with
        # `P(living) 0.00` on every card - which is what the first live run put on the map. The context is
        # therefore always built; `triage_ctx` only lets a caller override it.
        self.triage_ctx = triage_ctx
        self.incident_t0_utc = incident_t0_utc
        self.incident_hours_ago = float(incident_hours_ago)
        self.water_temp_c = DEFAULT_WATER_TEMP_C if water_temp_c is None else float(water_temp_c)
        self.domain = domain
        self._tracker = None
        self._fingerprints: dict[str, tuple] = {}
        self._noise = None
        if noise_seed is not None:
            from sightline.geo import NoiseConfig, TelemetryNoise   # noqa: PLC0415

            self._noise = TelemetryNoise(NoiseConfig(seed=int(noise_seed)))
        self.noise_seed = noise_seed
        self.frames = 0
        self._last_t_utc = 0.0
        self.detections = 0
        self.located = 0
        self.latencies: list[StageLatency] = []

    # -- stage 2 -----------------------------------------------------------------------------------
    def _detect(self, *, labels: list[dict[str, Any]] | None, label_json: Path | None,
                rgb: np.ndarray | None, frame_png: Path | None) -> list[Detection]:
        if self.detector == "truth":
            if labels is not None:
                return detections_from_label_records(labels)
            if label_json is not None:
                return detections_from_truth(label_json)
            raise ValueError("detector=truth needs either `labels` or `label_json`")
        if rgb is not None:
            return detections_from_model_array(rgb, self.weights, self.conf)
        if frame_png is not None:
            return detections_from_model(frame_png, self.weights, self.conf)
        raise ValueError(f"detector={self.detector} needs either `rgb` or `frame_png`")

    # -- the frame ---------------------------------------------------------------------------------
    def process(self, *, frame_idx: int, telemetry: Telemetry, intrinsics: Intrinsics,
                labels: list[dict[str, Any]] | None = None, label_json: Path | None = None,
                rgb: np.ndarray | None = None, frame_png: Path | None = None,
                capture_ms: float = 0.0, detect_ms_extra: float = 0.0) -> FrameResult:
        """One frame through every stage.

        ``detect_ms_extra`` is detection work the CALLER already did on this frame's behalf - the live
        runner turns the instance mask into truth labels itself, because it has the mask in memory and the
        replay does not. Charging it to `detect_ms` keeps the latency table honest: that time is detection,
        not capture.
        """
        from sightline.geo import project_detection                 # noqa: PLC0415
        from sightline.schemas import FrameBundle                   # noqa: PLC0415
        from sightline.track import Tracker, TrackerConfig          # noqa: PLC0415
        from sightline.triage import rank_records                   # noqa: PLC0415

        lat = StageLatency(capture_ms=capture_ms)
        tel = self._noise.apply(telemetry) if self._noise is not None else telemetry

        # Learn the loop rate from the frames themselves and widen the confirmation gate to match. The
        # tracker is built before any timing exists, so a window chosen up front is always a guess: on this
        # machine the guess was 0.73 FPS and the truth was 0.58, which left the gate 40 ms too narrow and
        # produced 0 records from a chain that was otherwise working end to end. The window only ever grows,
        # so a fast machine keeps the doc's 2 s and a slow one stops silently failing.
        t_now = float(getattr(tel, "t_utc", 0.0) or 0.0)
        if t_now and self._last_frame_t:
            dt = t_now - self._last_frame_t
            if 0.0 < dt < 60.0:
                self._frame_dts.append(dt)
        if t_now:
            self._last_frame_t = t_now
        if self._tracker is not None and len(self._frame_dts) >= 3:
            want = self._measured_window()
            if want > self._tracker.cfg.min_hits_window_s + 1e-6:
                self._tracker.cfg.min_hits_window_s = want
                self.track_window_s = want

        t = time.perf_counter()
        dets = self._detect(labels=labels, label_json=label_json, rgb=rgb, frame_png=frame_png)
        for d in dets:
            d.frame_idx = frame_idx
        lat.detect_ms = (time.perf_counter() - t) * 1e3 + detect_ms_extra

        t = time.perf_counter()
        fixes: list[GeoFix | None] = []
        n_located = 0
        tel_geo = to_geo_gimbal(tel)
        for d in dets:
            f = project_detection(d, tel_geo, intrinsics, self.geo_cfg)
            fixes.append(f)
            if f is not None and f.valid:
                n_located += 1
        lat.geo_ms = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        if self._tracker is None:
            # Frames are captured one per shutter point, so consecutive frames are a full grid step apart:
            # this is a still-survey clip, not video. `fps` only sets the buffer length in PROCESSED-frame
            # units.
            self._tracker = Tracker(
                TrackerConfig(fps=self._measured_fps(), min_hits_window_s=self._measured_window(),
                              cmc_enabled=self.cmc_enabled, pass_id=self.pass_id),
                intrinsics=intrinsics, clip_id=self.clip_id or tel.clip_id)
        bundle = FrameBundle(frame_idx=frame_idx, t_utc=tel.t_utc, telemetry=tel, intrinsics=intrinsics,
                             rgb=rgb, clip_id=self.clip_id or tel.clip_id)
        touched = self._tracker.update(dets, bundle, fixes)
        lat.track_ms = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        # Only re-cluster when a confirmed track actually moved: `Deduplicator.ingest` bumps every record's
        # version on every call, and a no-op frame must not manufacture a new version of every record.
        records = (self.dedup.ingest(self._tracker.confirmed_tracks()) if touched
                   else self.dedup.active_records())
        lat.dedup_ms = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        ranked = rank_records(records, self._ctx_at(tel.t_utc)) if records else []
        lat.triage_ms = (time.perf_counter() - t) * 1e3

        changed: list[Record] = []
        for r in ranked:
            fp = record_fingerprint(r)
            if self._fingerprints.get(r.record_id) != fp:
                self._fingerprints[r.record_id] = fp
                changed.append(r)

        self.frames += 1
        self._last_t_utc = tel.t_utc
        self.detections += len(dets)
        self.located += n_located
        self.latencies.append(lat)
        return FrameResult(frame_idx=frame_idx, t_utc=tel.t_utc, detections=dets, fixes=fixes,
                           n_located=n_located, tracks_touched=len(touched), records=ranked,
                           changed=changed, latency=lat)

    # -- the end -----------------------------------------------------------------------------------
    def close(self) -> list[Record]:
        """Finish the pass and return the ranked records. Never deletes anything (R10)."""
        from sightline.triage import rank_records                   # noqa: PLC0415

        if self._tracker is None:
            return []
        records = self.dedup.ingest(self._tracker.close())
        return rank_records(records, self._ctx_at(self._last_t_utc))

    def _ctx_at(self, t_utc: float):
        """The triage context for a frame. `now_utc` is pinned to the FRAME's clock, not wall-clock, so the
        live run and a later replay of the same frames score identically instead of drifting apart by
        however long sat between them."""
        from sightline.triage import TriageContext                 # noqa: PLC0415

        import dataclasses                                         # noqa: PLC0415

        if self.triage_ctx is not None:
            return dataclasses.replace(self.triage_ctx, now_utc=t_utc)
        if self.incident_t0_utc is None:
            # Default: the incident began when the survey did. Elapsed time is then 0 at the first frame,
            # which is the LEAST alarming assumption the survival curves can be given - it never inflates
            # urgency. Pass --incident-hours-ago to say how long ago it really started.
            self.incident_t0_utc = t_utc - self.incident_hours_ago * 3600.0
        return TriageContext(incident_t0_utc=self.incident_t0_utc, now_utc=t_utc,
                             water_temp_c=self.water_temp_c, domain=self.domain)

    @property
    def tracker(self):
        return self._tracker

    def _measured_fps(self) -> float:
        """The rate this machine ACTUALLY delivers, not the rate someone typed on the command line.

        Measured 2026-09-11: a 4K frame tiled to 15 crops through YOLO26s on an RTX 4060 that is sharing its
        8 GB with the Unreal editor costs 1.2-3.2 s of inference, so the loop runs at 0.58 FPS. SOLUTION_DOC
        5.11 budgets ~113 ms for detect and 5.6's confirmation gate is written for the ~5 FPS that implies.
        Every timing constant downstream inherits that assumption, and a CLI flag guessing the rate is one
        more thing to get wrong - I passed 0.73 when the truth was 0.58, and the gate missed by 40 ms.
        """
        dts = getattr(self, "_frame_dts", None)
        if not dts or len(dts) < 3:
            return float(self.tracker_fps)
        med = sorted(dts)[len(dts) // 2]
        return 1.0 / med if med > 0 else float(self.tracker_fps)

    def _measured_window(self) -> float:
        """Confirmation window sized to the rate actually being achieved, never below the doc's 2 s."""
        if self._explicit_window > 0:
            return self._explicit_window
        return max(2.0, (3 - 1) / max(1e-6, self._measured_fps()) * 1.35)

    def tracker_describe(self) -> dict:
        """The tracker's own bookkeeping, or why it never ran. Never raises: this is diagnostics."""
        t = getattr(self, "_tracker", None)
        if t is None:
            return {"built": False, "why": "no frame reached the tracker"}
        try:
            d = dict(t.describe())
            d["built"] = True
            return d
        except Exception as exc:                              # noqa: BLE001
            return {"built": True, "describe_failed": f"{type(exc).__name__}: {exc}"}

    def stats(self) -> dict[str, Any]:
        tracks = self._tracker.confirmed_tracks() if self._tracker is not None else []
        return {"frames": self.frames, "detections": self.detections, "located": self.located,
                "tracks": len(tracks), "records": len(self.dedup.all_records())}


# --- the run -----------------------------------------------------------------------------------------------
def run(clip_dir: Path, out_dir: Path, *, detector: str = "truth", weights: str = "", conf: float = 0.25,
        every: int = 1, noise_seed: int | None = None, cmc_enabled: bool = True,
        tracker_fps: float = 1.0, track_window_s: float = 0.0, incident_hours_ago: float = 0.0,
        water_temp_c: float | None = None) -> dict:
    """Replay one capture run through the shared spine. Same stages, same order as the live mission."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    fp = FramePipeline(detector=detector, weights=weights, conf=conf, tracker_fps=tracker_fps,
                       track_window_s=track_window_s,
                       cmc_enabled=cmc_enabled, noise_seed=noise_seed,
                       incident_hours_ago=incident_hours_ago, water_temp_c=water_temp_c)
    for k, tel, intr, labels in iter_frames(clip_dir, every=every):
        fp.process(frame_idx=k, telemetry=tel, intrinsics=intr, label_json=labels,
                   frame_png=clip_dir / "images" / f"{labels.stem}.png")

    ranked = fp.close()
    stages = fp.stats()
    stages["ranked"] = len(ranked)

    (out_dir / "records.geojson").write_text(json.dumps(feature_collection(ranked), indent=1), encoding="utf-8")
    top = [{"rank": r.priority_rank, "id": r.record_id[:8], "cls": r.cls, "score": round(r.score, 4),
            "lat": round(r.lat, 6), "lon": round(r.lon, 6), "h_acc_m": round(r.h_acc_m, 2),
            "posture": r.posture, "submersion": r.submersion, "n_obs": r.n_observations,
            "count": r.count_estimate, "status": r.status} for r in ranked[:15]]
    (out_dir / "top_records.json").write_text(json.dumps(top, indent=1), encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION, "clip": str(clip_dir), "detector": fp.detector,
        "weights": weights, "conf": conf, "frame_stride": every,
        # Honest: nothing injects telemetry noise unless --noise-seed is given. This field used to default to
        # true while no noise model ran at all.
        "noise_injected": noise_seed is not None, "noise_seed": noise_seed,
        "domain": "sim", "path": "offline_replay", "stages": stages,
        # OBSERVABILITY, added after a real debugging session cost hours. "tracks: 0" told us the chain was
        # broken but not WHERE: detections that never associate, tracklets that never confirm, and tracklets
        # pruned before they could confirm are three different faults with three different fixes. The
        # tracker counts all of them and nothing was reading them out. `cmc_sources` is the other half - an
        # "unavailable" majority means the inter-frame camera shift is never being compensated.
        "tracker": fp.tracker_describe(),
        "triage": {"incident_t0_utc": fp.incident_t0_utc, "incident_hours_ago": incident_hours_ago,
                   "water_temp_c": fp.water_temp_c,
                   "note": "elapsed time runs from incident_t0_utc; with --incident-hours-ago 0 the "
                           "incident is assumed to start with the survey, which is the least alarming "
                           "assumption the survival curves can be given"},
        "latency": latency_report(fp.latencies, domain="sim", label="offline replay (no capture, no publish)"),
        "seconds": round(time.time() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--out", default="_artifacts/pipeline_run")
    ap.add_argument("--detector", choices=DETECTORS, default="truth")
    ap.add_argument("--weights", default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--incident-hours-ago", type=float, default=0.0,
                    help="how long before the first frame the incident began; drives the survival curves")
    ap.add_argument("--water-temp-c", type=float, default=None)
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="inject the §5.7 telemetry noise model with this seed (default: no noise, and the "
                         "manifest says so)")
    a = ap.parse_args()
    clip = Path(a.clip) if Path(a.clip).is_absolute() else REPO / a.clip
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    m = run(clip, out, detector=a.detector, weights=a.weights, conf=a.conf, every=a.every,
            noise_seed=a.noise_seed, incident_hours_ago=a.incident_hours_ago,
            water_temp_c=a.water_temp_c)
    print(json.dumps(m, indent=1))
    if m["detector"] == "truth":
        print("\nNOTE: detector=truth replays the simulator's own labels. These are ceiling numbers for the "
              "chain after detection, NOT detection results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
