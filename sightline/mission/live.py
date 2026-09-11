"""The live demo loop: fly the survey, run the pipeline ON EACH FRAME AS IT IS CAPTURED, push to the map.

    # terminal 1 - the C2 server
    uv run python -m sightline.api.serve --port 8781
    # terminal 2 - the flight
    uv run python -m sightline.mission.live --alt 45 --speed 12 --detector truth

    # no simulator: replay a recorded flight through the identical live loop, at its own cadence
    uv run python -m sightline.mission.live --replay _artifacts/dataset/train_seed23 --pace 4

SOLUTION_DOC §3.3 "Live simulation" and §7 steps 1-3. This is the mode that makes the system a system: the
offline replay (`python -m sightline.pipeline <clip>`) proves the chain is correct, and this proves it runs
while the drone is still in the air.

What it is NOT allowed to be
----------------------------
A second implementation. Two things would have drifted the moment either was touched, so neither is copied:

* the **flight plan and terrain follower** come from `sightline.mission.pattern`, which
  `sightline.mission.survey` also flies - one lawnmower, one heightfield sampler, one cadence gate;
* the **pipeline stages** come from `sightline.pipeline.FramePipeline`, which the offline replay also calls -
  one detect, one geolocate, one track, one dedup, one triage, in one order.

`--verify` closes the loop on that claim: the run writes its frames out in the capture-run format as it flies,
then replays its own output through `sightline.pipeline.run` and compares the records. A mismatch is a
non-zero exit, not a warning. `tests/test_live_mission.py` runs the same comparison on a synthetic clip.

Detector provenance (the rule that stops a truth number being read as a detection number)
-----------------------------------------------------------------------------------------
``--detector truth`` replays the simulator's instance mask as if a perfect detector had produced it, so the
whole loop is demonstrable before a model exists. ``--detector rgb --weights <path>`` is the real detector.
Whichever ran is stamped into the run manifest, into every record's ``source.detector``, into the telemetry
CSV header comment and into the data card. A number from ``truth`` can never be mistaken for a detection
result.

Guardrail R10: nothing here deletes a record or marks anything done. Records only ever go INTO the store.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sightline.mission.pattern import (Scenario, SurveyPlan, body_euler_deg, build_plan,  # noqa: E402
                                       cadence_verdict, connect, grab)
from sightline.mission.takeover import (ControlSource, ModeTransition, NullSource,  # noqa: E402
                                        TakeoverMachine, VehicleAuthority, open_control_source)
from sightline.pipeline import (FramePipeline, StageLatency, latency_report,  # noqa: E402
                                nadir_gimbal_quat, resolve_detector, to_geo_gimbal)
from sightline.schemas import Intrinsics, Record, Telemetry  # noqa: E402

__all__ = ["LiveFrame", "C2Client", "MapArrivalProbe", "LiveMission", "main"]

#: Nadir gimbal, the orientation every survey frame is shot at. Built by `sightline.ingest.spec`'s own helper
#: via `pipeline.nadir_gimbal_quat`, NOT hand-rolled: `Telemetry.q_gimbal` is frozen as optical->NED, so a
#: straight-down camera is the IDENTITY quaternion and `Telemetry.gimbal_pitch_deg()` reports -90 for it. The
#: obvious "rotate -90 about Y" literal reads back as -180 and throws the map's camera footprint 90 deg out -
#: a trap `sightline/api/demo_seed.py` documents having already been caught by once.
#: Identical to `sightline.pipeline.iter_frames`, which is what makes the live and replayed telemetry the same.
NADIR_Q = nadir_gimbal_quat()

#: The telemetry CSV `sightline.mission.survey` writes and `sightline.pipeline.iter_frames` reads. Live runs
#: write the identical header so a live run IS a replayable capture run.
TELEMETRY_COLUMNS = ["frame_idx", "t_utc", "clip_id", "east_m", "north_m", "alt_msl_m", "agl_m", "lat", "lon",
                     "q_w", "q_x", "q_y", "q_z", "gimbal_pitch_deg", "hfov_deg", "width_px", "height_px",
                     "gsd_cm_px", "mode", "flood_level_asl_m", "n_labels", "speed_ms"]


# --- one frame, as the live loop sees it -------------------------------------------------------------------
@dataclass
class LiveFrame:
    frame_idx: int
    t_capture: float                       # perf_counter at the START of the capture - the latency origin
    telemetry: Telemetry
    intrinsics: Intrinsics
    labels: list[dict[str, Any]] = field(default_factory=list)
    rgb: np.ndarray | None = None
    seg: np.ndarray | None = None
    capture_ms: float = 0.0
    label_ms: float = 0.0
    east_m: float = 0.0
    north_m: float = 0.0
    speed_ms: float = 0.0
    leg_idx: int = -1


# --- the C2 link -------------------------------------------------------------------------------------------
class C2Client:
    """Pushes records and drone pose into the RUNNING `sightline.api.serve` process.

    Records travel the real §5.10 route - `Outbox` (SQLite, persist-queue) -> `Uploader` thread ->
    ``POST /api/upload`` -> the server's idempotent upsert -> `RecordStore.put` -> the store's change
    callback -> the WebSocket. Nothing here invents a transport: the queue survives a link drop, the status
    bar reads its depth (§7 step 6), and the flight loop never waits on the network because enqueueing is a
    local SQLite write.

    Pose goes straight over HTTP to ``/api/mission/pose`` instead: it is small, it is superseded by the next
    one within a second, and a durable queue full of stale drone positions helps nobody.
    """

    def __init__(self, base_url: str, *, outbox_dir: Path, timeout_s: float = 5.0):
        import httpx                                        # noqa: PLC0415

        from sightline.store import Outbox, Uploader        # noqa: PLC0415
        from sightline.store.outbox import HttpTransport    # noqa: PLC0415

        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)
        self.outbox = Outbox(outbox_dir)
        self.uploader = Uploader(self.outbox, HttpTransport(self.base_url), base_backoff_s=0.25,
                                 max_backoff_s=10.0)
        self.uploader.start()
        self.records_pushed = 0
        self.records_skipped = 0
        self.poses_pushed = 0
        self.pose_errors = 0
        self._pushed_version: dict[str, int] = {}

    @classmethod
    def probe(cls, base_url: str) -> dict[str, Any]:
        """Is the C2 actually up? A live run against a dead server is the failure this catches at second 0."""
        import httpx                                        # noqa: PLC0415

        r = httpx.get(base_url.rstrip("/") + "/health", timeout=5.0)
        r.raise_for_status()
        return r.json()

    def push_records(self, records: list[Record]) -> float:
        """Enqueue changed records. Returns the milliseconds the FLIGHT LOOP paid (not the round trip).

        Refuses to enqueue the same ``(record_id, version)`` twice. The cloud upsert is idempotent only for
        an IDENTICAL replay; the same version with different content is a stale write and comes back 409,
        which the uploader retries for ever and which therefore blocks every record behind it.
        """
        t0 = time.perf_counter()
        for rec in records:
            if self._pushed_version.get(rec.record_id) == rec.version:
                self.records_skipped += 1
                continue
            self._pushed_version[rec.record_id] = rec.version
            self.outbox.enqueue_record(rec)
            self.records_pushed += 1
        return (time.perf_counter() - t0) * 1e3

    def push_pose(self, tel: Telemetry, *, footprint: list[list[float]] | None = None,
                  extra: dict[str, Any] | None = None) -> float:
        t0 = time.perf_counter()
        body: dict[str, Any] = {k: getattr(tel, k) for k in Telemetry.__slots__}
        if footprint is not None:
            body["footprint"] = footprint
        if extra:
            body.update(extra)
        try:
            self._http.post(self.base_url + "/api/mission/pose", json=body).raise_for_status()
            self.poses_pushed += 1
        except Exception:
            self.pose_errors += 1
        return (time.perf_counter() - t0) * 1e3

    def push_route(self, route: dict[str, Any]) -> bool:
        try:
            self._http.post(self.base_url + "/api/plan/route", json=route).raise_for_status()
            return True
        except Exception:
            return False

    def stats(self) -> dict[str, Any]:
        s = self.uploader.stats()
        s.update(records_pushed=self.records_pushed, records_skipped=self.records_skipped,
                 poses_pushed=self.poses_pushed, pose_errors=self.pose_errors)
        return s

    def drain(self, timeout_s: float = 20.0) -> dict[str, Any]:
        """Wait for the outbox to empty so the end-of-run numbers describe a delivered state."""
        t0 = time.time()
        while self.outbox.depth() > 0 and time.time() - t0 < timeout_s:
            time.sleep(0.1)
        return self.stats()

    def close(self) -> None:
        try:
            self.uploader.stop()
        finally:
            self.outbox.close()
            self._http.close()


class MapArrivalProbe:
    """A WebSocket client on the same feed the map uses, so "on the map" is MEASURED, not assumed.

    `publish_ms` in the latency table is only what the flight loop paid to enqueue a record. The number the
    demo is actually judged on is capture -> the pin appearing, and the only honest way to get it is to be a
    map client and timestamp the `record` frame when it lands. That is what this does.
    """

    def __init__(self, base_url: str):
        self.url = base_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + "/ws"
        self.arrivals: dict[str, float] = {}          # f"{record_id}:{version}" -> perf_counter at arrival
        self.first_arrival: dict[str, float] = {}     # record_id -> perf_counter of its FIRST appearance
        self.messages = 0
        self.records_seen = 0
        self.error = ""
        self.connected = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="map-arrival-probe", daemon=True)

    def start(self) -> "MapArrivalProbe":
        self._thread.start()
        self.connected.wait(timeout=10.0)
        return self

    def _run(self) -> None:
        import asyncio                                      # noqa: PLC0415

        async def go() -> None:
            import websockets                               # noqa: PLC0415

            async with websockets.connect(self.url, max_size=None) as ws:
                self.connected.set()
                while not self._stop_event.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    now = time.perf_counter()
                    self.messages += 1
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("type") != "record":
                        continue
                    feat = msg.get("feature") or {}
                    rid = str(feat.get("id") or "")
                    ver = int((feat.get("properties") or {}).get("version", 0))
                    if not rid:
                        continue
                    with self._lock:
                        self.records_seen += 1
                        self.arrivals[f"{rid}:{ver}"] = now
                        self.first_arrival.setdefault(rid, now)

        try:
            asyncio.new_event_loop().run_until_complete(go())
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            self.connected.set()

    def arrival_of(self, record_id: str, version: int, *, not_before: float | None = None) -> float | None:
        """When this exact record version reached a map client.

        ``not_before`` guards against the trap this measurement fell into on its first run: a record's
        (id, version) pair can already be in the table from an EARLIER push, and returning that timestamp
        against a later frame's capture clock produced a **negative** latency, reported as
        "median -1010.0 ms" with a straight face. An arrival older than the capture cannot be that
        capture's arrival, so it is not one.
        """
        with self._lock:
            t = self.arrivals.get(f"{record_id}:{version}")
        if t is None or (not_before is not None and t < not_before):
            return None
        return t

    def first_arrival_of(self, record_id: str) -> float | None:
        with self._lock:
            return self.first_arrival.get(record_id)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)


# --- the mission -------------------------------------------------------------------------------------------
class LiveMission:
    """Fly, detect, geolocate, track, dedup, triage and publish - one frame at a time, in the air."""

    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.out = Path(args.out) if Path(args.out).is_absolute() else REPO / args.out
        for d in ("images", "labels", "masks"):
            (self.out / d).mkdir(parents=True, exist_ok=True)

        self.scn = Scenario.load()
        self.detector = resolve_detector(args.detector)
        self.replay_dir = Path(args.replay) if args.replay else None
        if self.replay_dir is not None and not self.replay_dir.is_absolute():
            self.replay_dir = REPO / self.replay_dir
        self.source_kind = "replay" if self.replay_dir else "sim_live"

        self.shutter_m = (args.shutter_m if args.shutter_m > 0
                          else _derived_shutter(args.alt, args.speed, self.scn.hfov_deg))
        self.plan: SurveyPlan = build_plan(self.scn, alt_m=args.alt, speed_ms=args.speed,
                                           shutter_m=self.shutter_m, plan=args.plan,
                                           overlap=args.overlap, patch_link_m=args.patch_link_m)
        self.cadence = cadence_verdict(args.alt, args.speed, self.shutter_m, self.scn.hfov_deg)
        self.safety = self.check_safety()

        self.clip_id = args.clip_id or f"live_seed{self.scn.seed}_alt{int(args.alt)}_{self.detector}"
        self.pipe = FramePipeline(detector=self.detector, weights=args.weights, conf=args.conf,
                                  tracker_fps=args.tracker_fps, track_window_s=args.track_window_s, cmc_enabled=args.cmc,
                                  clip_id=self.clip_id, noise_seed=args.noise_seed,
                                  incident_hours_ago=args.incident_hours_ago,
                                  water_temp_c=args.water_temp_c, domain="sim")
        self.takeover = TakeoverMachine(deadband=args.deadband, centred_s=args.centred_s)
        self.control: ControlSource = NullSource()
        #: Set by a harness to drive the takeover machine from a script instead of a device. The runner uses
        #: it verbatim; everything downstream (state machine, authority, telemetry, map) is untouched.
        self.control_override: ControlSource | None = None
        self.authority: VehicleAuthority | None = None
        self.c2: C2Client | None = None
        self.probe: MapArrivalProbe | None = None

        self.client = None
        self.names: list[str] = []
        self.cmap = None
        self.intrinsics = Intrinsics(width_px=3840, height_px=2160, fx=self.scn.f_px, fy=self.scn.f_px,
                                     cx=float(self.scn.cal["cx"]), cy=float(self.scn.cal["cy"]),
                                     source="calibration")

        self.frames = 0
        self.boxes = 0
        self.actors_seen: set[int] = set()
        self.agls: list[float] = []
        self.map_latency_ms: list[float] = []
        #: (record_id, version, t_capture, frame_idx, t_utc, lat, lon) awaiting their WebSocket arrival.
        self._pending_latency: list[tuple] = []
        self.map_latency_unresolved = 0
        self.first_pin: dict[str, Any] | None = None
        self.mode_rows: list[dict[str, Any]] = []
        self.t_start = 0.0
        self._tw = None
        self._tf = None

    def check_safety(self):
        """F2 constraints - geofence, 120 m ceiling, minimum AGL, battery RTL reserve - against THIS plan.

        `sightline/mission/safety.py` exists precisely because the two planners in this project were not
        connected: `sightline/plan/` knows about `Constraints` and `sightline/mission/pattern.py` is the one
        that actually flies. This is the join, on the flight side. It REPORTS and never mutates: silently
        shortening a route would hide the fact that an area was never searched, which R10 forbids.
        """
        try:
            from sightline.mission.safety import check_plan       # noqa: PLC0415
            from sightline.plan.constraints import Constraints    # noqa: PLC0415
        except Exception as e:
            print(f"safety check unavailable ({type(e).__name__}: {e}); flying without one")
            return None
        home = self.scn.home
        c = Constraints(home_ne=(float(home["north_m"]), float(home["east_m"])))
        try:
            return check_plan(self.plan, c, transit_speed_ms=self.a.speed)
        except Exception as e:
            print(f"safety check FAILED to run ({type(e).__name__}: {e}); flying without one")
            return None

    # -- wiring ------------------------------------------------------------------------------------
    def open_c2(self) -> None:
        if self.a.no_c2:
            print("C2: disabled (--no-c2); records go to the run directory only")
            return
        try:
            health = C2Client.probe(self.a.c2)
        except Exception as e:
            raise SystemExit(
                f"no C2 server at {self.a.c2} ({type(e).__name__}: {e}).\n"
                f"  Start it first:  uv run python -m sightline.api.serve --port "
                f"{self.a.c2.rsplit(':', 1)[-1]}\n"
                f"  or pass --no-c2 to fly without a map (which is not the demo).") from None
        print(f"C2 up at {self.a.c2}: {health['store']['records']} records already in the log, "
              f"{health['live_clients']} map client(s) connected")
        self.c2 = C2Client(self.a.c2, outbox_dir=self.out / "outbox")
        if not self.a.no_probe:
            self.probe = MapArrivalProbe(self.a.c2).start()
            if self.probe.error:
                print(f"  map-arrival probe FAILED to connect ({self.probe.error}); the end-to-end "
                      f"capture->map number will be missing from this run")
                self.probe = None
            else:
                print("  map-arrival probe connected: capture->map latency will be MEASURED on the "
                      "same WebSocket feed the map uses")
        self.c2.push_route(self.route_json())

    def route_json(self) -> dict[str, Any]:
        """The plan, in the plan lane's own route format, so the map draws the pattern the drone is flying."""
        from sightline.common.geodesy import offset_ne      # noqa: PLC0415

        home = self.scn.home
        geo = home["geopoint"]        # the launch pad's WGS-84 fix; `home` itself is local ENU metres
        wps = []
        for i, leg in enumerate(self.plan.legs):
            for j, north in enumerate((leg.north_start_m, leg.north_end_m)):
                asl = self.scn.terrain.surface_asl(leg.east_m, north) + self.a.alt
                lat, lon = offset_ne(float(geo["lat"]), float(geo["lon"]),
                                     north - float(home["north_m"]), leg.east_m - float(home["east_m"]))
                wps.append({"seq": len(wps), "action": "waypoint", "lat": lat, "lon": lon,
                            "alt_asl_m": round(asl, 2), "agl_m": self.a.alt, "speed_ms": self.a.speed,
                            "gimbal_pitch_deg": -90.0, "segment_id": f"leg{i}", "pass_id": 0,
                            "reason": "boustrophedon leg start" if j == 0 else "boustrophedon leg end"})
        return {"product": "sightline.plan.route", "pattern": f"boustrophedon/{self.plan.plan}",
                "domain": "sim", "aborted": False, "abort_reason": "",
                "params": {"agl_m": self.a.alt, "presentation": "body", "speed_ms": self.a.speed,
                           "line_spacing_m": round(self.plan.line_spacing_m, 2),
                           "shutter_m": self.shutter_m, "detector": self.detector},
                "totals": {"legs": len(self.plan.legs), "track_km": round(self.plan.track_km, 3),
                           "est_minutes": round(self.plan.est_minutes, 1)},
                "notes": [f"flown live by sightline.mission.live, detector={self.detector}"],
                "waypoints": wps}

    def open_telemetry(self) -> None:
        self._tf = (self.out / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self._tw = csv.writer(self._tf)
        self._tw.writerow(TELEMETRY_COLUMNS)

    # -- the pipeline, per frame -------------------------------------------------------------------
    def handle_frame(self, fr: LiveFrame) -> None:
        """detect -> geo -> track -> dedup -> triage -> map, for ONE frame, while the drone keeps flying."""
        res = self.pipe.process(frame_idx=fr.frame_idx, telemetry=fr.telemetry, intrinsics=fr.intrinsics,
                                labels=fr.labels if self.detector == "truth" else None,
                                rgb=fr.rgb, capture_ms=fr.capture_ms, detect_ms_extra=fr.label_ms)
        lat: StageLatency = res.latency

        if self.c2 is not None:
            lat.publish_ms = self.c2.push_records(res.changed)
            fp = self.footprint_ring(fr) if self.a.footprint else None
            lat.publish_ms += self.c2.push_pose(
                fr.telemetry, footprint=fp,
                extra={"clip_id": self.clip_id, "frame_idx": fr.frame_idx})

        # capture -> record actually on the map, measured on the map's own feed.
        # Recorded as a PENDING pair and resolved later, never waited for: blocking the flight loop until a
        # WebSocket frame comes back makes the aircraft's shutter cadence depend on the map's network, and
        # the first version of this did exactly that - up to 1.5 s of stall per changed record.
        if self.probe is not None:
            for rec in res.changed:
                self._pending_latency.append((rec.record_id, rec.version, fr.t_capture, fr.frame_idx,
                                              fr.telemetry.t_utc, rec.lat, rec.lon))
            self.resolve_map_latency()

        self.write_frame(fr, res)
        self.frames += 1
        self.boxes += len(fr.labels)
        self.agls.append(fr.telemetry.agl_m)
        self.actors_seen.update(int(m["actor_id"]) for m in fr.labels if m.get("actor_id") is not None)

        if self.frames % self.a.log_every == 0 or res.changed:
            st = self.pipe.stats()
            print(f"  f{fr.frame_idx:05d} {self.takeover.mode:6s} "
                  f"agl {fr.telemetry.agl_m:5.1f} m  det {len(res.detections):2d}  "
                  f"trk {st['tracks']:2d}  rec {st['records']:2d}  "
                  f"{lat.total_ms():6.1f} ms ({lat.dominant_stage()})", flush=True)

    def footprint_ring(self, fr: LiveFrame) -> list[list[float]] | None:
        """The ground quadrilateral the camera actually sees, as a closed [lon, lat] ring.

        Taken from `sightline.geo.footprint_ned` - the SAME chain that geolocates the detections, so the
        drawn rectangle and the pins inside it cannot disagree. Measured here: 67.8 m east x 38.1 m north
        at 45 m AGL, which is the across-track x along-track orientation `survey.py`'s line spacing
        assumes. `sightline.coverage.footprint.ground_footprint`, which `api.mission_feed.camera_footprint`
        would otherwise use, comes out rotated 90 deg from both the geo and the track lanes on the same
        telemetry (measured at gimbal yaw 0 and 90, 2026-09-11; see docs/TRACKER.md), so using it would draw
        a footprint at right angles to the flight lines and nobody would notice until they looked at the map.
        """
        from sightline.common.geodesy import offset_ne     # noqa: PLC0415
        from sightline.geo import footprint_ned            # noqa: PLC0415

        try:
            poly = footprint_ned(fr.intrinsics, to_geo_gimbal(fr.telemetry), self.pipe.geo_cfg)
        except Exception:
            return None
        if not poly:
            return None
        ring = [list(reversed(offset_ne(fr.telemetry.lat, fr.telemetry.lon, float(n), float(e))))
                for n, e in poly]
        ring.append(ring[0])
        return ring

    def resolve_map_latency(self, *, final: bool = False) -> None:
        """Match pending pushes against what the map client has actually received. Never blocks."""
        if self.probe is None:
            return
        still: list[tuple] = []
        for rid, ver, t_cap, k, t_utc, lat, lon in self._pending_latency:
            arrived = self.probe.arrival_of(rid, ver, not_before=t_cap)
            if arrived is None:
                if final:
                    self.map_latency_unresolved += 1
                else:
                    still.append((rid, ver, t_cap, k, t_utc, lat, lon))
                continue
            ms = (arrived - t_cap) * 1e3
            if ms < 0:            # impossible with not_before, and a bug in this file if it ever happens
                self.map_latency_unresolved += 1
                continue
            self.map_latency_ms.append(ms)
            if self.first_pin is None:
                self.first_pin = {"record_id": rid, "frame_idx": k, "t_utc": t_utc,
                                  "capture_to_map_ms": round(ms, 1), "lat": round(lat, 6),
                                  "lon": round(lon, 6), "detector": self.detector, "domain": "sim"}
                print(f"\n  *** FIRST PIN ON THE MAP: record {rid[:8]} at frame {k}, "
                      f"{ms:.0f} ms after the shutter, while still flying ***\n")
        self._pending_latency = still

    def write_frame(self, fr: LiveFrame, res: Any) -> None:
        stem = f"{self.clip_id}_{fr.frame_idx:05d}"
        if fr.rgb is not None and not self.a.no_images:
            import cv2                                      # noqa: PLC0415

            if self.a.jpeg:
                cv2.imwrite(str(self.out / "images" / f"{stem}.jpg"), fr.rgb[:, :, ::-1],
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.a.jpeg)])
            else:
                cv2.imwrite(str(self.out / "images" / f"{stem}.png"), fr.rgb[:, :, ::-1])
            if fr.seg is not None:
                cv2.imwrite(str(self.out / "masks" / f"{stem}.png"), fr.seg[:, :, ::-1])
        (self.out / "labels" / f"{stem}.json").write_text(json.dumps(fr.labels, indent=1), encoding="utf-8")
        t = fr.telemetry
        self._tw.writerow([fr.frame_idx, t.t_utc, self.clip_id, round(fr.east_m, 2), round(fr.north_m, 2),
                           round(t.alt_msl_m, 2), round(t.agl_m, 2), t.lat, t.lon,
                           round(t.q_body[0], 6), round(t.q_body[1], 6), round(t.q_body[2], 6),
                           round(t.q_body[3], 6), -90.0, round(self.scn.hfov_deg, 3),
                           fr.intrinsics.width_px, fr.intrinsics.height_px,
                           round(self.scn.gsd_cm_px(t.agl_m), 4), t.mode,
                           round(self.scn.water_level_m, 3), len(fr.labels), round(fr.speed_ms, 2)])
        self._tf.flush()

    # -- takeover ----------------------------------------------------------------------------------
    def poll_takeover(self, frame_idx: int = -1) -> ModeTransition | None:
        self.takeover.frame_idx = frame_idx
        tr = self.takeover.poll(self.control.read(time.time()))
        if tr is None:
            return None
        applied = self.authority.apply(tr) if self.authority is not None else {"did": ["(no vehicle)"]}
        if tr.to_mode == "MANUAL" and self.client is not None:
            try:
                self.client.cancelLastTask()
            except Exception:
                pass
        row = {**tr.as_dict(), "applied": applied.get("did", []), "apply_ms": applied.get("ms", 0.0)}
        self.mode_rows.append(row)
        print(f"\n  >>> MODE {tr}  [{', '.join(applied.get('did', [])) or 'no vehicle action'}] "
              f"in {applied.get('ms', 0.0):.1f} ms\n", flush=True)
        return tr

    # -- frame sources -----------------------------------------------------------------------------
    def capture_live(self, frame_idx: int, leg_idx: int) -> LiveFrame | None:
        """One real capture from the simulator: Scene + instance mask + the pose that produced them."""
        from tools.capture.labels import attach_truth, labels_from_mask  # noqa: PLC0415

        t_capture = time.perf_counter()
        st = self.client.simGetGroundTruthKinematics()
        g = grab(self.client)
        gp = self.client.getMultirotorState().gps_location
        capture_ms = (time.perf_counter() - t_capture) * 1e3

        t = time.perf_counter()
        labs = attach_truth(labels_from_mask(g["seg"], self.names, self.cmap), self.scn.truth)
        label_ms = (time.perf_counter() - t) * 1e3

        home = self.scn.home
        north = float(home["north_m"]) + st.position.x_val
        east = float(home["east_m"]) + st.position.y_val
        asl = float(home["ground_asl_m"]) - st.position.z_val
        agl = asl - self.scn.terrain.surface_asl(east, north)
        o, v = st.orientation, st.linear_velocity
        tel = Telemetry(
            t_utc=time.time(), lat=gp.latitude, lon=gp.longitude, alt_msl_m=asl, agl_m=agl,
            q_body=(o.w_val, o.x_val, o.y_val, o.z_val), q_gimbal=NADIR_Q,
            gimbal_is_earth_referenced=True, ned_m=(st.position.x_val, st.position.y_val, st.position.z_val),
            vel_ned_ms=(v.x_val, v.y_val, v.z_val), mode=self.takeover.mode, clip_id=self.clip_id,
            frame_idx=frame_idx, flood_level_asl_m=self.scn.water_level_m)
        intr = Intrinsics(width_px=g["seg"].shape[1], height_px=g["seg"].shape[0], fx=self.scn.f_px,
                          fy=self.scn.f_px, cx=float(self.scn.cal["cx"]), cy=float(self.scn.cal["cy"]),
                          source="calibration")
        return LiveFrame(frame_idx=frame_idx, t_capture=t_capture, telemetry=tel, intrinsics=intr,
                         labels=[_label_dict(m) for m in labs], rgb=g["scene"], seg=g["seg"],
                         capture_ms=capture_ms, label_ms=label_ms, east_m=east, north_m=north,
                         speed_ms=math.sqrt(v.x_val ** 2 + v.y_val ** 2 + v.z_val ** 2), leg_idx=leg_idx)

    def replay_frames(self) -> Iterator[LiveFrame]:
        """Drive the identical live loop from a recorded capture run - no simulator, real timing.

        This is NOT a shortcut around the flight: it is the §3.3 replay harness feeding the LIVE path, so
        the map, the WebSocket, the outbox and the latency instrumentation can be exercised (and shown to a
        reviewer) with the editor shut. Runs made this way are stamped `source: "replay"` everywhere.
        """
        from sightline.pipeline import iter_frames          # noqa: PLC0415

        prev_t = None
        for k, tel, intr, label_json in iter_frames(self.replay_dir, every=self.a.every):
            # Pace FIRST, then start the latency clock. Timing the sleep as if it were capture work makes
            # `capture_ms` the dominant stage in every replay run and buries the real pipeline cost - which
            # is exactly what the first run of this reported (505 ms median, 99 % "capture", at --pace 2).
            if prev_t is not None and self.a.pace > 0:
                dt = (tel.t_utc - prev_t) / self.a.pace
                if 0 < dt < 10.0:
                    time.sleep(dt)
            prev_t = tel.t_utc
            t_capture = time.perf_counter()
            labels = json.loads(label_json.read_text()) if label_json.exists() else []
            tel.mode = self.takeover.mode
            tel.clip_id = self.clip_id
            rgb = None
            if self.detector != "truth":
                import cv2                                  # noqa: PLC0415

                for ext in (".png", ".jpg"):
                    p = self.replay_dir / "images" / f"{label_json.stem}{ext}"
                    if p.exists():
                        rgb = cv2.imread(str(p))[:, :, ::-1]
                        break
            yield LiveFrame(frame_idx=k, t_capture=t_capture, telemetry=tel, intrinsics=intr,
                            labels=labels, rgb=rgb, capture_ms=(time.perf_counter() - t_capture) * 1e3,
                            east_m=0.0, north_m=0.0,
                            speed_ms=math.sqrt(sum(x * x for x in tel.vel_ned_ms)))

    # -- the flight --------------------------------------------------------------------------------
    def fly(self) -> None:
        """The boustrophedon, flown for real, capturing on the move, with takeover live throughout."""
        import cosysairsim as airsim                        # noqa: PLC0415

        a, home = self.a, self.scn.home
        c = self.client
        c.enableApiControl(True)
        c.armDisarm(True)
        print("\ntaking off...")
        c.takeoffAsync(timeout_sec=20).join()

        frame_idx = 0
        for leg_idx, leg in enumerate(self.plan.legs):
            if (time.time() - self.t_start) / 60.0 > a.max_minutes:
                self.takeover.force("RTL", "time budget reached", detail=f"{a.max_minutes} min")
                print(f"  stopping: {a.max_minutes} min budget reached at leg {leg_idx}/{len(self.plan.legs)}")
                break
            print(f"  leg {leg_idx + 1}/{len(self.plan.legs)} east {leg.east_m:7.1f}  "
                  f"north {leg.north_start_m:.0f} -> {leg.north_end_m:.0f}", flush=True)

            for phase in ("transit", "line"):
                target_n = leg.north_start_m if phase == "transit" else leg.north_end_m
                asl = self.scn.terrain.surface_asl(leg.east_m, target_n) + a.alt
                issued = False
                shots, last_n, t_phase, last_shot_t = 0, None, time.time(), 0.0
                while True:
                    if time.time() - t_phase > a.leg_timeout_s:
                        print(f"    {phase}: {a.leg_timeout_s:.0f} s timeout, moving on")
                        break
                    tr = self.poll_takeover(frame_idx)
                    if tr is not None and tr.to_mode == "RTL":
                        return
                    # Re-issue the command from wherever the pilot left the vehicle. This is §5.2's
                    # AUTO-RESUME: continue the pattern from the current pose, never restart it.
                    if self.takeover.resume_pending:
                        self.takeover.resume_pending = False
                        issued = False
                        print(f"    AUTO-RESUME: re-planning {phase} of leg {leg_idx + 1} from the "
                              f"current pose (leg {leg_idx + 1}/{len(self.plan.legs)} kept, "
                              f"pattern NOT restarted)")
                    if self.takeover.flying_itself and not issued:
                        c.moveToPositionAsync(
                            float(target_n - home["north_m"]), float(leg.east_m - home["east_m"]),
                            float(-(asl - home["ground_asl_m"])), a.speed, timeout_sec=a.leg_timeout_s,
                            drivetrain=airsim.DrivetrainType.ForwardOnly,
                            yaw_mode=airsim.YawMode(False, 0))
                        issued = True

                    st = c.simGetGroundTruthKinematics()
                    cur_n = float(home["north_m"]) + st.position.x_val
                    cur_e = float(home["east_m"]) + st.position.y_val
                    roll, pitch, _ = body_euler_deg(st.orientation)
                    agl = (float(home["ground_asl_m"]) - st.position.z_val) \
                        - self.scn.terrain.surface_asl(cur_e, cur_n)

                    if phase == "line" and self._should_shoot(shots, cur_n, last_n, last_shot_t,
                                                              roll, pitch, agl):
                        fr = self.capture_live(frame_idx, leg_idx)
                        if fr is not None:
                            self.handle_frame(fr)
                            frame_idx += 1
                            shots += 1
                            last_n = cur_n
                            last_shot_t = time.time()

                    if self.takeover.flying_itself and abs(cur_n - target_n) <= 2.0:
                        break
                    time.sleep(a.poll_s)

        print("\nreturning to launch...")
        self.takeover.force("RTL", "pattern complete")
        home_asl = self.scn.terrain.surface_asl(float(home["east_m"]), float(home["north_m"])) + a.alt
        c.moveToPositionAsync(0.0, 0.0, -(home_asl - float(home["ground_asl_m"])), a.speed,
                              timeout_sec=180).join()

    def _should_shoot(self, shots: int, cur_n: float, last_n: float | None, last_shot_t: float,
                      roll: float, pitch: float, agl: float) -> bool:
        """The shutter rule. Identical gates to `survey.py`, plus a time cadence for MANUAL flight.

        §7 step 3: "Every frame in MANUAL still runs detection, geolocation and coverage." A pilot who is
        hovering to look under an eave covers no along-track distance, so a distance-only shutter would
        stop capturing exactly when the operator is looking hardest.
        """
        if max(abs(roll), abs(pitch)) > self.a.max_tilt_deg:
            return False
        if abs(agl - self.a.alt) > self.a.alt_tol_m and self.takeover.flying_itself:
            return False
        if shots == 0 or last_n is None:
            return True
        if abs(cur_n - last_n) >= self.shutter_m:
            return True
        return (self.takeover.manual
                and (time.time() - last_shot_t) >= self.a.manual_shutter_s)

    # -- run ---------------------------------------------------------------------------------------
    def run(self) -> int:
        a = self.a
        print(self.plan.summary())
        print(self.cadence.message())
        if not self.cadence.ok and not a.no_track_check:
            return 2
        if self.safety is not None:
            print(self.safety.summary())
            if not self.safety.ok and not a.ignore_safety:
                print("REFUSING TO FLY: the plan breaks an F2 constraint above. Fix the plan, or pass "
                      "--ignore-safety to fly it anyway (the violation is stamped into the data card "
                      "either way).")
                return 4
        if a.dry_run:
            print(f"dry run: {self.plan.est_frames} frames, ~{self.plan.est_minutes:.1f} min of flying. "
                  f"Nothing was armed or flown.")
            return 0

        self.open_c2()
        self.open_telemetry()
        self.t_start = time.time()
        rc = 0
        try:
            if self.replay_dir is not None:
                print(f"\nREPLAY source: {self.replay_dir} (no simulator; the LIVE loop, recorded frames)")
                self.control = self.control_override or open_control_source(a.control)
                print(f"control source: {json.dumps(self.control.describe())}")
                for fr in self.replay_frames():
                    self.poll_takeover(fr.frame_idx)
                    self.handle_frame(fr)
                    if (time.time() - self.t_start) / 60.0 > a.max_minutes:
                        print(f"  stopping: {a.max_minutes} min budget reached")
                        break
            else:
                self.client = connect()
                self.names = self.client.simListInstanceSegmentationObjects()
                self.cmap = self.client.simGetSegmentationColorMap()
                grab(self.client)          # warm-up frame, discarded (first buffers are unconverted)
                _assert_segmentation_healthy(self.names, self.cmap)
                self.control = self.control_override or open_control_source(a.control, client=self.client)
                self.authority = VehicleAuthority(
                    self.client,
                    hover_fn=lambda: self.client.hoverAsync(),
                    rtl_fn=lambda: self.client.moveToPositionAsync(
                        0.0, 0.0, -(self.scn.terrain.surface_asl(
                            float(self.scn.home["east_m"]), float(self.scn.home["north_m"]))
                            + a.alt - float(self.scn.home["ground_asl_m"])), a.speed, timeout_sec=180))
                print(f"control source: {json.dumps(self.control.describe())}")
                self.fly()
        except KeyboardInterrupt:
            print("\ninterrupted by the operator")
            rc = 130
        finally:
            if self._tf is not None:
                self._tf.close()
            if self.client is not None:
                for fn in (lambda: self.client.landAsync(timeout_sec=30),
                           lambda: self.client.armDisarm(False),
                           lambda: self.client.enableApiControl(False)):
                    try:
                        fn()
                    except Exception:
                        pass
            self.control.close()

        ranked = self.pipe.close()
        if self.c2 is not None:
            self.c2.push_records(ranked)
            self.c2.drain()
        if self.probe is not None:
            for _ in range(20):          # give the last WebSocket frames a moment to land
                self.resolve_map_latency()
                if not self._pending_latency:
                    break
                time.sleep(0.1)
            self.resolve_map_latency(final=True)
        self.write_outputs(ranked)
        if self.probe is not None:
            self.probe.stop()
        if self.c2 is not None:
            self.c2.close()

        if a.verify:
            rc = max(rc, self.verify_against_offline(ranked))
        return rc

    # -- outputs -----------------------------------------------------------------------------------
    def write_outputs(self, ranked: list[Record]) -> dict[str, Any]:
        from sightline.schemas import SCHEMA_VERSION, feature_collection   # noqa: PLC0415

        lat = latency_report(self.pipe.latencies, domain="sim",
                             label=f"live {self.source_kind}, detector={self.detector}")
        e2e = None
        if self.map_latency_ms:
            v = np.array(self.map_latency_ms, dtype=float)
            e2e = {"n": int(v.size), "median_ms": round(float(np.median(v)), 1),
                   "p90_ms": round(float(np.percentile(v, 90)), 1), "max_ms": round(float(v.max()), 1),
                   "min_ms": round(float(v.min()), 1), "unresolved_pushes": self.map_latency_unresolved,
                   "measured_on": "the map's own WebSocket feed (capture shutter -> record frame received)"}
        card = {
            "product": "sightline.mission.live", "schema_version": SCHEMA_VERSION,
            "clip_id": self.clip_id, "domain": "sim", "source": self.source_kind,
            "detector": self.detector, "weights": self.a.weights,
            "detector_note": ("truth = the simulator's own instance mask replayed as a perfect detector. "
                              "These are ceiling numbers for the chain AFTER detection, not detection "
                              "results." if self.detector == "truth"
                              else f"trained RGB detector, weights={self.a.weights!r}"),
            "scenario_seed": self.scn.seed, "altitude_m_agl": self.a.alt, "speed_ms": self.a.speed,
            "shutter_m": self.shutter_m, "plan": self.plan.plan, "legs": len(self.plan.legs),
            "frames": self.frames, "total_boxes": self.boxes, "unique_actors_seen": len(self.actors_seen),
            "detectable_in_plan": self.plan.survivors_in_plan,
            "agl_m_measured": ({"min": round(min(self.agls), 1),
                                "median": round(float(np.median(self.agls)), 1),
                                "max": round(max(self.agls), 1)} if self.agls else None),
            "stages": self.pipe.stats(), "ranked": len(ranked),
            "safety": (None if self.safety is None else {
                "ok": self.safety.ok, "total_legs": self.safety.total_legs,
                "flyable_legs": self.safety.flyable_legs,
                "battery_truncates_at": self.safety.battery_truncates_at,
                "est_flight_s": round(self.safety.est_flight_s, 1),
                "est_usable_s": round(self.safety.est_usable_s, 1),
                "ignored": bool(self.a.ignore_safety),
                "violations": [{"kind": v.kind, "leg_index": v.leg_index, "detail": v.detail}
                               for v in self.safety.violations]}),
            "latency": lat, "capture_to_map_ms": e2e, "first_pin": self.first_pin,
            "noise_injected": self.a.noise_seed is not None, "noise_seed": self.a.noise_seed,
            "triage": {"incident_t0_utc": self.pipe.incident_t0_utc,
                       "incident_hours_ago": self.a.incident_hours_ago,
                       "water_temp_c": self.pipe.water_temp_c, "domain": "sim"},
            "randomisation": "off (5.5c)",
            "takeover": {**self.takeover.status(), "control": self.control.describe(),
                         "transitions": [r for r in self.mode_rows],
                         "refusals": [{"t_utc": t, "why": w} for t, w in self.takeover.refusals],
                         "seconds_by_mode": {m: round(self.takeover.seconds_in(m), 1)
                                             for m in ("AUTO", "MANUAL", "HOLD", "RTL")}},
            "c2": self.c2.stats() if self.c2 is not None else None,
            "map_probe": ({"messages": self.probe.messages, "record_frames": self.probe.records_seen}
                          if self.probe is not None else None),
            "minutes": round((time.time() - self.t_start) / 60.0, 2),
        }
        (self.out / "data_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
        (self.out / "records.geojson").write_text(json.dumps(feature_collection(ranked), indent=1),
                                                  encoding="utf-8")
        (self.out / "mode_log.json").write_text(json.dumps(self.mode_rows, indent=1), encoding="utf-8")
        print(f"\n{self.frames} frames, {self.boxes} boxes, {len(ranked)} records, "
              f"{card['minutes']} min -> {self.out}")
        print(f"latency (domain=sim, {self.source_kind}, detector={self.detector}): "
              f"end-to-end median {lat['end_to_end_ms']['median']} ms, "
              f"dominant stage {lat['dominant_stage']} "
              f"({lat['dominant_share'] * 100:.0f} % of it)")
        if e2e:
            print(f"capture -> record ON THE MAP: median {e2e['median_ms']} ms over {e2e['n']} updates "
                  f"(measured on the map's WebSocket)")
        if self.detector == "truth":
            print("\nNOTE: detector=truth replays the simulator's own labels. These are ceiling numbers "
                  "for the chain after detection, NOT detection results.")
        return card

    def verify_against_offline(self, live_ranked: list[Record]) -> int:
        """Replay THIS RUN'S OWN frames through `sightline.pipeline.run` and compare the records.

        The two paths share `FramePipeline`, so a mismatch means the frame sources disagree - the live
        telemetry is not what the CSV recorded, or the labels written differ from the labels processed.
        Either way it is a defect, and it exits non-zero.
        """
        from sightline.pipeline import run as offline_run   # noqa: PLC0415

        print("\n--- verify: replaying this run's own frames offline and comparing records ---")
        m = offline_run(self.out, self.out / "offline_replay", detector=self.detector,
                        weights=self.a.weights, conf=self.a.conf, noise_seed=self.a.noise_seed,
                        cmc_enabled=self.a.cmc, tracker_fps=self.a.tracker_fps,
                        track_window_s=self.a.track_window_s,
                        incident_hours_ago=self.a.incident_hours_ago,
                        water_temp_c=self.a.water_temp_c)
        off = json.loads((self.out / "offline_replay" / "records.geojson").read_text())
        diff = compare_record_sets(live_ranked, off["features"])
        (self.out / "verify.json").write_text(json.dumps(
            {"live_records": len(live_ranked), "offline_records": len(off["features"]),
             "offline_stages": m["stages"], "match": diff["match"], "differences": diff["differences"]},
            indent=1), encoding="utf-8")
        if diff["match"]:
            print(f"verify OK: {len(live_ranked)} live records == {len(off['features'])} replayed records "
                  f"on every compared field")
            return 0
        print(f"verify FAILED: {len(diff['differences'])} difference(s) between the live and replayed "
              f"records; see {self.out / 'verify.json'}")
        for d in diff["differences"][:8]:
            print(f"   {d}")
        return 3


# --- comparing the two paths -------------------------------------------------------------------------------
#: What "the same record" means. `record_id` is a uuid4 and `version` counts re-clusterings, so neither can be
#: compared across two runs; everything a commander would act on is compared.
#:
#: `priority_rank` is NOT in this list, and that is a measured limitation rather than a convenience:
#: `sightline.triage.rank.sort_key` breaks a full tie (same score, same confidence, same observation count)
#: on `record_id`, which is a uuid4. Two runs over identical frames therefore hand tied records different
#: ranks. Instead of comparing it per record, :func:`compare_record_sets` compares the whole RANKED SCORE
#: SEQUENCE, which is what actually orders the commander's list, and additionally pins the rank of every
#: record whose score is unique. A genuine ranking change still fails; a coin toss between equals does not.
COMPARED_FIELDS = ("cls", "status", "posture", "submersion", "occlusion", "motion_state", "count_estimate",
                   "n_observations", "n_tracks_merged", "zone")
COMPARED_FLOATS = {"lat": 1e-7, "lon": 1e-7, "h_acc_m": 1e-4, "score": 1e-6, "confidence": 1e-6,
                   "agl_m": 1e-3, "off_nadir_deg": 1e-3}


def compare_record_sets(live: list[Record], offline_features: list[dict[str, Any]]) -> dict[str, Any]:
    """Pair records by position and compare content. Ids and versions are deliberately not compared."""
    diffs: list[str] = []
    live_f = [r.to_feature() for r in live]
    if len(live_f) != len(offline_features):
        diffs.append(f"record COUNT differs: live {len(live_f)} vs offline {len(offline_features)}")

    def key(f: dict[str, Any]) -> tuple:
        c = f["geometry"]["coordinates"]
        return (round(float(c[1]), 6), round(float(c[0]), 6), f["properties"]["cls"])

    def ranked_scores(feats: list[dict[str, Any]]) -> list[tuple]:
        s = sorted(feats, key=lambda f: int(f["properties"]["priority_rank"]))
        return [(round(float(f["properties"]["score"]), 9), f["properties"]["cls"],
                 f["properties"]["status"]) for f in s]

    live_seq, off_seq = ranked_scores(live_f), ranked_scores(offline_features)
    if live_seq != off_seq:
        diffs.append(f"the RANKED ORDER differs: live {live_seq} vs offline {off_seq}")

    lo = {key(f): f for f in sorted(live_f, key=key)}
    of = {key(f): f for f in sorted(offline_features, key=key)}
    # A record whose score is unique in its own run has no tie to break, so its rank IS comparable.
    def unique_scores(feats: list[dict[str, Any]]) -> set:
        seen: dict[float, int] = {}
        for f in feats:
            k = round(float(f["properties"]["score"]), 9)
            seen[k] = seen.get(k, 0) + 1
        return {k for k, n in seen.items() if n == 1}

    unique_both = unique_scores(live_f) & unique_scores(offline_features)
    for k in sorted(set(lo) | set(of)):
        if k not in lo:
            diffs.append(f"record at {k} exists offline but NOT live")
            continue
        if k not in of:
            diffs.append(f"record at {k} exists live but NOT offline")
            continue
        a, b = lo[k]["properties"], of[k]["properties"]
        for f in COMPARED_FIELDS:
            if a.get(f) != b.get(f):
                diffs.append(f"{k} field {f}: live {a.get(f)!r} vs offline {b.get(f)!r}")
        if round(float(a.get("score", 0.0)), 9) in unique_both and \
                a.get("priority_rank") != b.get("priority_rank"):
            diffs.append(f"{k} field priority_rank (score is unique, so this is not a tie): "
                         f"live {a.get('priority_rank')!r} vs offline {b.get('priority_rank')!r}")
        for f, tol in COMPARED_FLOATS.items():
            x, y = a.get(f), b.get(f)
            if x is None or y is None:
                if x != y:
                    diffs.append(f"{k} field {f}: live {x!r} vs offline {y!r}")
                continue
            if not (math.isclose(float(x), float(y), abs_tol=tol) or
                    (math.isinf(float(x)) and math.isinf(float(y)))):
                diffs.append(f"{k} field {f}: live {x} vs offline {y} (tol {tol})")
    return {"match": not diffs, "differences": diffs}


# --- helpers -----------------------------------------------------------------------------------------------
def _label_dict(m: Any) -> dict[str, Any]:
    """A `tools.capture.labels.MaskLabel` in the exact JSON shape `survey.py` writes and the replay reads."""
    return {"actor_id": m.actor_id, "name": m.name, "cls": m.cls, "bbox_px": list(m.bbox_px),
            "visible_px": m.visible_px, "size_px": m.size_px, "pose": m.pose,
            "submersion": m.submersion, "occlusion": m.occlusion, "zone": m.zone,
            "group": m.group, "aerially_detectable": m.aerially_detectable}


def _derived_shutter(alt: float, speed: float, hfov_deg: float) -> float:
    from tools.capture.campaign import shutter_m            # noqa: PLC0415

    return float(shutter_m(alt, speed, hfov_deg))


def _assert_segmentation_healthy(names: list[str], cmap: Any) -> None:
    """A capture is only as good as its instance colours (the `thermal_ids.py` trap, docs/TRACKER.md)."""
    pal: dict[tuple, list[str]] = {}
    for i, n in enumerate(names):
        pal.setdefault(tuple(int(v) for v in cmap[i]), []).append(n)
    dupes = {c: ns for c, ns in pal.items()
             if len(ns) > 1 and any(x.startswith(("Human_", "Animal_")) for x in ns)}
    if dupes:
        print("\nFATAL: instance segmentation is degenerate - actors share a colour with other objects:")
        for c, ns in list(dupes.items())[:5]:
            print(f"   colour {c} used by {len(ns)} objects: {ns[:4]}")
        raise SystemExit(2)
    print(f"pre-flight: {len(names)} instances, every actor colour unique")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alt", type=float, default=45.0, help="metres AGL")
    p.add_argument("--speed", type=float, default=12.0, help="m/s ground speed")
    p.add_argument("--plan", choices=("box", "patches"), default="patches")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--patch-link-m", type=float, default=90.0)
    p.add_argument("--shutter-m", type=float, default=0.0,
                   help="0 = derive it from the tracker's confirmation gate (recommended)")
    p.add_argument("--out", default="")
    p.add_argument("--clip-id", default="")
    p.add_argument("--detector", choices=("truth", "rgb", "model"), default="truth",
                   help="truth replays the simulator's instance mask; rgb runs the trained detector")
    p.add_argument("--weights", default="")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--incident-hours-ago", type=float, default=0.0,
                   help="how long before the flight the incident began. The survival curves of SS2.5 run on "
                        "time since the INCIDENT, so 0 means every survivor is scored as freshly stranded.")
    p.add_argument("--water-temp-c", type=float, default=None)
    p.add_argument("--noise-seed", type=int, default=None,
                   help="inject the §5.7 telemetry noise model with this seed (default: none, and every "
                        "output says so)")
    p.add_argument("--tracker-fps", type=float, default=1.0)
    p.add_argument("--track-window-s", type=float, default=0.0,
                   help="confirmation window for the 5.6 rule-3 gate. 0 = derive it from --tracker-fps (never below the doc's 2 s). The doc's 2 s assumes ~5 FPS; a latency-bound 4K "
                        "loop runs near 0.7 FPS, where 3 hits cannot fit in 2 s and the gate never closes.")
    p.add_argument("--cmc", action="store_true", default=True,
                   help="camera-motion compensation in the tracker. ON by default and effectively "
                        "mandatory: a survey moves the image ~555 px between frames, so without it "
                        "nothing associates and the run produces zero tracks.")
    p.add_argument("--no-cmc", dest="cmc", action="store_false")
    p.add_argument("--c2", default="http://127.0.0.1:8781")
    p.add_argument("--no-c2", action="store_true", help="fly without a map (not the demo)")
    p.add_argument("--no-probe", action="store_true",
                   help="do not open a WebSocket client; the capture->map latency will not be measured")
    p.add_argument("--footprint", action="store_true", default=True,
                   help="push the real projected camera footprint with every pose")
    p.add_argument("--no-footprint", dest="footprint", action="store_false")
    p.add_argument("--control", choices=("auto", "pygame", "airsim", "keyboard", "none"), default="auto")
    p.add_argument("--deadband", type=float, default=TakeoverMachine.DEADBAND)
    p.add_argument("--centred-s", type=float, default=TakeoverMachine.CENTRED_S)
    p.add_argument("--max-minutes", type=float, default=25.0)
    p.add_argument("--max-tilt-deg", type=float, default=8.0)
    p.add_argument("--alt-tol-m", type=float, default=8.0)
    p.add_argument("--manual-shutter-s", type=float, default=1.0,
                   help="in MANUAL, also shoot at least this often even if the pilot is hovering")
    p.add_argument("--poll-s", type=float, default=0.02, help="flight/takeover poll period (50 Hz)")
    p.add_argument("--leg-timeout-s", type=float, default=400.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--jpeg", type=int, default=0, metavar="Q")
    p.add_argument("--no-images", action="store_true", help="labels and telemetry only (fast demo)")
    p.add_argument("--no-track-check", action="store_true")
    p.add_argument("--ignore-safety", action="store_true",
                   help="fly a plan that breaks a geofence / ceiling / min-AGL / battery-reserve "
                        "constraint. The violation is recorded in the data card regardless.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="after the run, replay its own frames offline and fail if the records differ")
    p.add_argument("--replay", default="",
                   help="a capture-run directory: drive the live loop from recorded frames, no simulator")
    p.add_argument("--pace", type=float, default=1.0,
                   help="replay speed multiplier (1 = the original cadence, 0 = as fast as possible)")
    p.add_argument("--every", type=int, default=1, help="replay: frame stride")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not a.out:
        a.out = f"_artifacts/live/{time.strftime('%Y%m%d-%H%M%S')}"
    return LiveMission(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
