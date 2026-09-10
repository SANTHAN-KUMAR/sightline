# Lane B1/B3 — ingest (F7), tracking (F11), deduplication (F12)

Verification pass, 2026-09-10. The three modules already existed and had **no tests**; this pass read them,
wrote `tests/test_ingest.py`, `tests/test_track.py` and `tests/test_dedup.py`, ran them, and fixed the module
whenever a test exposed a real defect. No test was weakened to make it pass (hard rule 2).

```
D:\Tools\uv\uv.exe run pytest tests/test_ingest.py tests/test_track.py tests/test_dedup.py -q
    228 passed   tests/test_ingest.py   (1.7 s)
     38 passed   tests/test_track.py    (2.3 s)
     32 passed   tests/test_dedup.py    (3.1 s)
    298 passed in 6.3 s
```

Everything runs offline on synthetic data plus the two sample logs already in `data/samples/`. No Unreal, no
AirSim, no GPU, no torch, no ultralytics, no network. `ruff check --select E4,E7,E9,F` is clean on all six
files. Peak RAM during the suite is a few hundred MB (the rendered CMC sequences are 480x270 x 10 frames).

---

## 1. Bugs found and fixed

### 1.1 `spec.gimbal_euler_from_quat` was gimbal-locked at exactly nadir — **the simulator's default pose**

`sightline/ingest/spec.py`. The 3-2-1 readback used the naive formulas, whose `atan2` arguments both carry a
factor `cos(pitch)`. At pitch = -90 both collapse to `atan2(0, 0)` and floating-point signs decide the answer:

```
gimbal_euler_from_quat(gimbal_quat_from_euler(0, -90, yaw))  ->  (180, -90, 180)   for EVERY yaw
```

so `gimbal_quat_from_euler(gimbal_euler_from_quat(q))` rotated the camera's azimuth by up to 180 deg. That
round trip is exactly what `sim.inject_noise` performs on **every frame**, and `sim/settings/capture_4k.json`
plus `tools/capture/run.py` both fly a -90 nadir camera. A nadir camera's yaw is "which way north points in
the image", so the corruption silently mislocates every off-centre detection.

Fixed by handling lock explicitly: `roll = 0`, `yaw = 2*atan2(z, w)`, which reproduces the rotation exactly and
keeps "yaw" meaning the camera's azimuth. The lock band is `|sin(pitch)| >= 1 - 1e-12` (pitch within 8e-5 deg
of straight down) — deliberately tight so genuine near-nadir angles are not quantised onto -90.

Worst-case rotation error of `q -> euler -> q` over a grid of 126 (roll, pitch, yaw) triples: **2.4e-6 deg**,
was **120 deg**. Regression tests: `test_gimbal_euler_quaternion_round_trip`,
`test_gimbal_lock_reports_azimuth_as_yaw_with_zero_roll`,
`test_noise_injection_preserves_a_non_zero_nadir_heading`.

> **`sightline/common/geodesy.py::quat_to_euler` has the identical flaw.** That file is orchestrator-owned so
> it was not touched. It bites only at |pitch| = 90, which a multirotor airframe never reaches, so the risk is
> low — but `AirSimRecReader` and `inject_noise` both call it on `q_body`, and any future gimbal-frame use
> would hit it. Recommend porting the same three-line fix.

### 1.2 A perfectly stationary survivor reported `motion_state = "unknown"` instead of `"still"`

`sightline/dedup/cluster.py`. `_max_displacement` returned the time window *of the maximal displacement*, and
the update guard was `if d > best_d`. For a subject that never moved, every displacement is 0.0, the guard
never fires, the window comes back 0 s, and `_motion` concludes it never observed a long enough window. The
inverted answer landed on precisely the case §5.6 says is normal ("survivors are mostly static"). `_motion`
had the same degenerate guard for the reported `motion_window_s`.

Fixed in both places: when nothing moved, report the longest separation that was actually examined. A "still"
verdict now states the window it was measured over. Regression: `test_motion_state_moving_still_and_unknown`.

### 1.3 The DJI SRT frame index shifted by one after any entry without a position

`sightline/ingest/__init__.py::_open_dji_srt`. `srt_to_telemetry` drops entries with no lat/lon or no time, so
`zip(entries, samples)` paired sample *i* with entry *i+1* from the first gap onwards and stamped every later
frame with the wrong `frame_idx` and PTS. Now the index is built from the samples, with PTS looked up by
frame index. Regression: `test_srt_frame_index_stays_aligned_when_an_entry_has_no_position`.

### 1.4 `AirSimRecReader.rows()` re-parsed the whole file per row

`telemetry()` calls `_t0_sim()`, which called `rows()`, which re-read the file — O(n^2) file reads
(a 10 000-frame recording parsed the file 10 000 times). Rows are now cached. Pinned by
`test_airsim_rec_is_read_with_its_assumptions_declared`.

### 1.5 `SrtParseReport.tz_offset_h` was hardcoded to 0.0

The provenance report claimed UTC even when the caller parsed with `tz_offset_h=5.5`. `srt_to_telemetry` now
takes and echoes the offset, and `_open_dji_srt` passes it through.

---

## 2. New: `sim.py` reads the format `tools/capture/run.py` actually writes

`sim.py` only knew the `sightline-sim-capture` v1.0 format of `spec.py` (`capture.json` + `frames.csv`). The F5
dataset generator writes something different, so `detect_source()` refused its output entirely. Rather than ask
the sim lane to re-write, `sim.py` now reads the run.py layout directly (`sim.py` is this lane's file).

Added: `CaptureRunReader`, `CaptureRunRow`, `is_capture_run()`, `CAPTURE_RUN_COLUMNS`, `CAPTURE_RUN_REQUIRED`,
and a `"capture_run"` source type in `open_clip` / `detect_source`. Verified by hand against the real clip
`_artifacts/dataset/smoke` (4 frames, 1920x1080, hfov 89.904) as well as by synthetic fixtures.

### 2.1 The exact format expected — **the capture writer must keep matching this**

```
<clip_dir>/
    telemetry.csv                       one row per WRITTEN frame; header below, comma separated
    data_card.json                      optional; clip_id / domain / scenario_seed / altitude_m read from it
    images/<clip_id>_<frame_idx:05d>.png    RGB (BGR on disk, OpenCV-written)
    masks/<clip_id>_<frame_idx:05d>.png     instance-segmentation mask   -> FrameRef.aux["seg"]
    labels/<clip_id>_<frame_idx:05d>.json   ground-truth boxes           -> FrameRef.aux["labels"]
    labels/<clip_id>_<frame_idx:05d>.txt    the same boxes in YOLO form  (carried, not read)
```

`telemetry.csv` header, in this order (order is not significant — the reader reads by name — but the *names*
are the contract):

```
frame_idx, t_utc, clip_id, east_m, north_m, alt_msl_m, agl_m, lat, lon, q_w, q_x, q_y, q_z,
gimbal_pitch_deg, hfov_deg, width_px, height_px, gsd_cm_px, mode, flood_level_asl_m, n_labels
```

Required (a row without one of these cannot become a `Telemetry` + `Intrinsics`): `frame_idx`, `t_utc`,
`east_m`, `north_m`, `alt_msl_m`, `agl_m`, `lat`, `lon`, `q_w..q_z`, `gimbal_pitch_deg`, `hfov_deg`,
`width_px`, `height_px`. The rest are optional.

Semantics the reader relies on:

| column | meaning as read |
|---|---|
| `frame_idx` | strictly increasing; **gaps are fine** (a skipped empty frame is a gap, never a renumber) |
| `t_utc` | wall-clock UTC POSIX seconds, non-decreasing. NOT the AirSim `SteppableClock` |
| `east_m`, `north_m` | **scenario-origin** ENU metres (not launch-site relative) |
| `alt_msl_m`, `agl_m` | camera altitude ASL and height above the surface the geolocation ray hits |
| `q_w..q_z` | **vehicle body** FRD -> NED quaternion (`simGetGroundTruthKinematics().orientation`) |
| `gimbal_pitch_deg` | earth-referenced, **-90 = nadir** |
| `hfov_deg` | HORIZONTAL FOV, read from the sim; `f_px = (W/2)/tan(HFOV/2)` |
| `flood_level_asl_m` | the water surface; also the down datum for `Telemetry.ned_m` |
| `mode` | `AUTO` \| `MANUAL` \| `HOLD` \| `RTL` |
| `gsd_cm_px`, `n_labels` | carried into `Clip.reports`; not used for geometry |

Three assumptions the format forces, all listed in `CaptureRunReader.assumptions` and echoed in
`Clip.reports["assumptions"]`:

1. **The gimbal is a pitch angle, not a quaternion.** Yaw is taken from the vehicle quaternion (the settings
   stabilise the gimbal and let its yaw follow the airframe). `CaptureRunReader(..., gimbal_yaw_follows_vehicle=False)`
   locks it to north instead. If the sim lane ever writes `gimbal_yaw_deg` / `gimbal_roll_deg` columns, say so
   and this becomes exact rather than assumed.
2. **`ned_m` down datum.** `ned_m = (north_m, east_m, -(alt_msl_m - origin_alt))` with `origin_alt` defaulting
   to the first row's `flood_level_asl_m`, so `ned_d ~= -agl_m`. Only differences of `ned_m` are used
   downstream, so the datum is free — but it has to be stated, because a reader assuming MSL would disagree by
   ~1 km in FloodValley.
3. **No GNSS accuracy is written**, so `h_acc_m = v_acc_m = 0` (clean sim truth). Anything quoting a
   geolocation error must run `inject_noise()` first.

`test_capture_run_columns_match_the_capture_writer_source` reads the header literal **out of
`tools/capture/run.py` itself** and asserts it equals `CAPTURE_RUN_COLUMNS`. If the sim lane adds, renames or
reorders a column, that test fails and names the difference. Nothing in this lane writes to `tools/`.

### 2.2 One cross-lane mismatch found in the labels

`data/scene/actors.json` emits `pose = "waving"` (12 of 71 actors), which is **not** in the frozen
`schemas.POSTURES` vocabulary. `CaptureRunReader.truth_detections()` maps anything outside the contract to
`"unknown"` with `posture_conf = 0.0` rather than coercing it to a neighbour; the raw string stays reachable
through `truth_labels()`. **Orchestrator decision needed**: either add `"waving"` as an optional posture in
`schemas.py` (a CONTRACTS.md §5 amendment) or change the scene generator to emit `"standing"`.

Also note `labels_from_mask` writes an **inclusive** integer box (`width = x2 - x1 + 1`) while
`Detection.bbox_px` is half-open. `truth_detections()` adds one pixel to the far corners so `Detection.size_px`
equals `MaskLabel.size_px`. Pinned by `test_capture_run_truth_labels_become_schema_detections`.

---

## 3. What is verified, with the numbers

### Ingest (F7) — 228 tests

* **DJI SRT, both families.** The old `[key: value]` form (format1) and the newer `<font ...>SrtCnt/FrameCnt`
  form (format3 / format3b) parse to exact values, plus the Matrice function-call form (format2c). The §5.4
  unit trap is table-driven: `fnum 170 -> 1.7` (`legacy_x100`), `280 -> 2.8`, `1.8 -> 1.8` (`literal_decimal`),
  `22 -> 22.0`; `focal_len 240 -> 24.0 mm`, `24.00 -> 24.0 mm`; `SS 400 -> 1/400 s`. `GPS(77.6, 12.9)` is
  auto-resolved lon-first. The date line `2026-09-10 12:00:00,500,000` -> `t_utc` with the millis/micros tail.
  The no-gimbal nadir substitution is applied *and reported* (`assumed_nadir`, a warning, `alt_basis`).
* **Alignment.** `TelemetrySeries.at()` recovers a known pose: position to **1e-3 m**, altitude to **1e-9 m**,
  gimbal pitch/yaw to **1e-6 deg** at five interpolation fractions; SLERP of a constant yaw rate is exact,
  and normalised lerp is shown to be measurably off-arc (>1 deg at t=0.25 over a 120 deg arc). Discrete fields
  (flight mode) are held nearest-previous; `clamp=False` refuses to extrapolate. `BootClock` maps boot -> UTC
  to **1e-3 s** and refuses pairs with `time_unix_usec == 0`.
* **`t_offset_s` recovery.** The barometric cross-correlation recovers an injected offset of
  -2.5 / -0.4 / 0.0 / 1.37 / 3.2 s to within **0.05 s** with peak correlation > 0.9, on a 6 m/s take-off
  profile with 0.05 m noise at 10 Hz telemetry vs 30 FPS frames. It **raises** rather than guessing on an
  uncorrelated signal. The `y_a(t + lag) ~= y_b(t)` sign convention is pinned separately.
* **Decimation.** At k = 1, 3 and 6 the processed set is exactly `range(0, n, k)`, `len(clip)` matches, the
  frame index still holds all *n* frames, and `load_frame(i)` returns **the right frame** for every i
  (each synthetic frame is filled with its own index, so "not None" is not enough to pass). `target_fps=10`
  and `5` resolve to k = 3 and 6 from the source rate.
* **The §5.7 noise model.** Deterministic per seed; the clean truth series is untouched (`np.array_equal`);
  the noised series reports `h_acc_m = 2.5` / `v_acc_m = 1.0` as a real receiver would. Over 25 seeds the
  pooled RMS horizontal error is **2.5 m** and the vertical **1.0 m** (rel 0.35). Attitude, measured on the
  quaternion over 120 seeds: **sqrt(1.5^2 + 2*0.5^2) = 1.66 deg** pooled, and **sqrt(2)*0.5 = 0.71 deg** with
  the yaw bias switched off (both rel 0.25). `bias_fraction=1.0` produces a *constant* per-clip offset
  (ptp < 1e-6 m) — §5.7's "biases do not average out" is a structural property of the model, not a hope.
* **The `sightline-sim-capture` spec.** Writer -> reader round trip is exact (NED, weather dict, flood level,
  intrinsics from HFOV, gimbal pitch). A removed required column raises `CaptureFormatError` naming it; an
  unknown column and a non-unit quaternion are rejected by the writer; `frames.jsonl` is an equivalent table
  and a truncated final line costs one row.
* **Thermal (§5.5b).** A 16-bit centi-kelvin PNG reads back as `uint16` and converts to **36.5 C**; an 8-bit
  file declared radiometric raises rather than returning garbage.
* **Real logs.** `data/samples/px4_sample_log_small.ulg` reads into a UTC series with unit quaternions and
  every fallback declared. A **synthetic but real** `.tlog` (8-byte big-endian stamp + MAVLink 2 frames, built
  with pymavlink) reads back with correct unit conversions: `alt` mm -> 645.0 m, `relative_alt` -> 45.0 m AGL,
  `eph 150` -> 1.5 m, `vx 300` -> 3.0 m/s, and the nadir-gimbal substitution declared.
* **The source router.** All seven source types, including the actionable refusal for a bare video.

### Track (F11) — 38 tests

* **Static target, moving camera, one id.** 30 frames at 5 FPS, camera at 3 m/s across a survivor pinned to a
  lat/lon: **1 track**, one observation per visible frame, one track id, zero pending tracks. A companion test
  proves the target really moves >20 px per frame, so the headline test cannot pass vacuously.
* **Buffer units.** `track_buffer_frames == 150` at 5 FPS x 30 s. The `roboflow/trackers` trap is pinned from
  both ends: `BoTSORTTracker(lost_track_buffer=150, frame_rate=5)` resolves to **25 frames / 5 s**, while
  `roboflow_lost_track_buffer() == 900` resolves to **150 frames / 30 s**. Asserted on what the *library*
  resolved, not on the config.
* **Dropouts.** A 3-frame dropout keeps one id and the track continues past it. The contrast case — the same
  gap against a 0.4 s buffer — produces **two** tracks, so the first test is measuring the buffer.
* **Confirmation gate.** Confirms on the third hit inside 0.4 s and not before; refuses three hits 1.2 s apart
  (span 2.4 s) while still recording all four observations; accepts a span of exactly 2.0 s. 90 scattered
  single-frame false positives over 30 frames yield **zero** extra confirmed tracks, and the ghosts are pruned.
  An unreachable gate (3 hits in 1 s at 1 FPS) is rejected at construction.
* **CMC, all three sources observable.** Over rendered textured ground: **optical flow** on >=7 of 9 frames,
  agreeing with telemetry to <2 px while both report >5 px of real motion. Over **rippling** water: telemetry
  homography on **every** frame, minimum disagreement above the 12 px threshold, survivor still on one track.
  Over **coherent drift** (the case an inlier count cannot see): telemetry on every frame, and the log confirms
  optical flow *succeeded* (`optflow_failed == False`) and was simply wrong. With neither frame nor telemetry
  the source is `identity` and says so.
* **The telemetry homography.** `ground_plane_homography` agrees with an independent projection of nine ground
  points to **1e-6 px**. The 2x3 affine reduction is exact at nadir (residual < 1e-6 px) and degrades when
  oblique, which is what lets a caller refuse it. Impossible geometry returns `None`. The optical-frame
  convention (view axis -> down, image right -> east, image down -> south) is pinned.
* **Ultralytics stays unexercised.** The backend is reachable for the §5.6 A/B, its `update` raises
  `NotImplementedError` when poked without `__init__`, and the suite asserts `torch` is never in `sys.modules`.

### Dedup (F12) — 32 tests

* **Two passes, one record, same id.** Two full synthetic fly-overs (different seeds, a 1.5 m / -1.0 m
  per-pass geolocation bias) produce **one** record whose `record_id` and `cluster_id` are unchanged,
  `seen_in_passes == [0, 1]`, `n_tracks_merged == 2`, and whose `version` was bumped. The merged position is
  within 2 x CE90 of truth and `h_acc_m` is **unchanged** — merging tracks does not invent accuracy.
* **The radius.** `eps = 2 x CE90 = 2 x 2.146 x 2.6 = 11.16 m`; tracks 8 m apart merge, tracks 25 m apart do
  not. `CE90_FACTOR` is checked against `sqrt(-2 ln 0.1)`.
* **Confidence.** `1 - prod(1 - conf_track)`: 0.8 and 0.5 -> **0.90**; three 0.5s -> **0.875**; a single 0.7
  track -> 0.7.
* **Count.** Two tracks sharing frames -> `count_estimate = 2` (a group on a roof). Two tracks that never
  share a frame -> **1** (an ID switch is not a second person). Two clips over the same roof -> estimate 1,
  `count_max` 2.
* **Motion.** 3 m/s for 10 s -> `moving` (displacement > 3 x CE90 = 16.7 m over >= 5 s); static for 10 s ->
  `still` with a stated window; 1 m/s drift (10 m) -> `still`; 0.4 s of observation -> `unknown`, and the
  doc's literal behaviour is still available behind `still_when_window_short`.
* **Position.** The confidence-weighted median puts the record at **0.4 m** north when one of four
  observations is 50 m out; the arithmetic mean would be 12.65 m.
* **R10, behaviourally.** A record not re-observed goes `stale`, keeps its id, stays in `all_records()` and
  stays *visible* in `active_records()`; seeing it again restores `confirmed`. A record outside the re-imaged
  area is **not** marked stale. When a later track bridges two records the **older** survives and the newer is
  `dismissed` with `merged_into:<id>`, still in the registry, cross-linked both ways.
* **R10, statically.** `test_no_lane_source_deletes_a_record_or_a_file` greps all of
  `sightline/ingest`, `sightline/track` and `sightline/dedup` for delete-shaped operations
  (`del`, `.pop`, `.remove`, `.clear`, `.discard`) on anything whose line mentions a record, and for
  filesystem/SQL removals (`os.remove`, `shutil.rmtree`, `.unlink`, `DELETE FROM`, `TRUNCATE`) anywhere. A
  companion test proves the matcher is not vacuous — it fires on `self._records.pop(rid)` and on
  `shutil.rmtree(...)` — and asserts that the removals these lanes *do* contain are all in `tracker.py` and
  are tracker working state, not records.
* **Metrics, hand-checked.** 3 ground-truth survivors, 3 records, r = 6 m, one record a genuine duplicate:
  **precision 2/3, recall 2/3, duplicate rate 0.5, count MAE 1.0, count bias -1.0, mean error 0.75 m,
  CE90 0.95 m, record FP/min 0.5** — each computable on paper from the doc's definitions. Dismissed records
  are excluded by default (and scorable with `include_dismissed=True`). `MOT` metrics: `idf1 = 1.0`,
  `hota_alpha = 1.0`, `num_switches = 0` on a clean track; `num_switches = 1` after one relabelling; the same
  in geo space. Every `MetricRow` carries `SliceKey.domain`, and `SliceKey()` without a domain is a `TypeError`.

---

## 4. Public API other lanes call

```python
# --- ingest (F7) -----------------------------------------------------------------------------------
from sightline.ingest import open_clip, detect_source, Clip
clip = open_clip(path, decimate_k=6, noise=True)      # or target_fps=5.0
for bundle in clip:                                   # Iterator[FrameBundle]
    ...
clip.load_frame(i)          # pixels of ANY decoded frame, decimated or not (evidence thumbnails)
clip.telemetry              # TelemetrySeries; .at(t_frame) -> Telemetry (interp + SLERP)
clip.telemetry_truth        # the clean sim pose, kept beside the noised one
clip.t_offset_s             # R12, settable; clip.estimate_t_offset() measures it from video
clip.frame_index            # FrameIndex: every decoded frame, never shrinks
clip.reports                # manifest / data_card / assumptions / srt report / noise model

# source-specific readers, when the Clip abstraction is not wanted
from sightline.ingest.sim import (SimExportReader, CaptureRunReader, AirSimRecReader,
                                  NoiseModel, inject_noise, CAPTURE_RUN_COLUMNS)
from sightline.ingest.spec import (SimCaptureWriter, CaptureManifest, CameraSpec, validate_capture,
                                   gimbal_quat_from_euler, gimbal_quat_from_frd_quat,
                                   gimbal_euler_from_quat)   # the ONE gimbal convention; do not hand-roll
from sightline.ingest.dji_srt import parse_srt, srt_to_telemetry
from sightline.ingest.align import TelemetrySeries, BootClock, estimate_t_offset
from sightline.ingest.decimate import FrameIndex, FrameRef, decimate, decimation_for_fps

# the F5 dataset run, including its ground truth
reader = CaptureRunReader(clip_dir)
reader.telemetry_series(); reader.intrinsics(); reader.frame_index()
reader.truth_labels(frame_idx)        # raw label dicts, every field the capture wrote
reader.truth_detections(frame_idx)    # list[Detection], score 1.0 — the eval lane's reference boxes
reader.validate()                     # list[str] of problems; [] = valid

# --- track (F11) -----------------------------------------------------------------------------------
from sightline.track.config import TrackerConfig
from sightline.track.tracker import Tracker
tracker = Tracker(TrackerConfig(fps=5.0, pass_id=0, track_id_offset=0), intrinsics=bundle.intrinsics,
                  geolocate=None)                     # geolocate: (Detection, FrameBundle) -> GeoFix
live = tracker.update(detections, bundle, fixes)      # confirmed tracks touched by THIS frame
tracks = tracker.close()                              # list[Track] for dedup
tracker.cmc_source_counts()                           # {"optical_flow": n, "telemetry_homography": n, ...}
tracker.cmc_log                                       # list[CmcResult], one per compensated frame
tracker.describe()                                    # everything needed to reproduce the run (§5.12)

from sightline.track import synth                     # test scaffolding, NOT a shipped component
from sightline.track.geometry import ground_plane_homography, telemetry_affine, rotation_optical_to_ned

# --- dedup (F12) -----------------------------------------------------------------------------------
from sightline.dedup import (Deduplicator, DedupConfig, records_from_tracks,
                             dedup_accuracy, GroundTruthSurvivor, track_metrics,
                             frames_from_tracks, fp_per_min)
dedup = Deduplicator()
records = dedup.ingest(tracks)                        # per pass; ids are stable across passes
dedup.mark_stale(pass_id=1, was_reimaged=Deduplicator.radius_predicate(lat, lon, 100.0))
dedup.all_records()      # everything ever created, including dismissed (only grows — R10)
dedup.active_records()   # what a commander sees: everything except dismissed duplicates
acc = dedup_accuracy(dedup.active_records(), truth, match_radius_m=6.0, duration_s=t)
rows = acc.rows(SliceKey(domain="sim", altitude_band="30-45"))     # list[MetricRow]
```

**Two calling requirements worth repeating.** (1) `Tracker.update(detections, bundle, fixes)` needs `fixes`
parallel to `detections` (same order, same length) or a `geolocate` callback — without either, every
observation carries an invalid `GeoFix` and dedup will refuse the track rather than cluster on the aircraft's
own position. (2) Give each coverage pass its own `pass_id` **and** `track_id_offset` block, so track ids never
collide inside the deduplicator.

---

## 5. Still stubbed, unverified or open

Labelled honestly, per hard rule 1.

| Item | State |
|---|---|
| `decode.NvdecReader` (PyNvVideoCodec) | **Written from the documented API, never executed.** Already labelled a stub in the module. `open_video()` will not select it unless asked (`backend="nvdec"` or `SIGHTLINE_ALLOW_NVDEC=1`) and any failure degrades to PyAV. Needs one run against a real MP4 on a free GPU. |
| `backends.UltralyticsBoTSORT` | **Deliberately not exercised** (AGPL + pulls torch, forbidden in this lane by CONTRACTS.md §3 rule 2). `update()` raises `NotImplementedError` with the reason. Finish it in the ML lane before quoting any A/B number from it. |
| Video-backed clips (`PyAVReader`, `Clip.estimate_t_offset` from real frames, `frame_source` `"clip.mp4#123"`) | **Untested here** — no MP4 ships with the repo and generating one needs an encoder. The metadata-only and frame-file paths are fully covered. `align.video_log_height_series` (the optical-flow height proxy) is likewise unexercised; `estimate_t_offset` itself is verified against an analytic proxy. |
| `mavlink.py` `.bin` / DataFlash path (`_DF_MAP`) | **Untested.** The `.tlog` path is verified against a synthetic log. No ArduPilot `.bin` is available offline. |
| `mavlink.py` `GIMBAL_DEVICE_ATTITUDE_STATUS` frame-flag composition | **Untested.** The no-gimbal fallback is verified; the vehicle-framed composition path needs a log that contains the message. |
| `dji-log-parser` (encrypted DJI `.txt` flight logs) | **Not implemented.** HANDBOOK §6 already records that it has no Python bindings and needs the Rust CLI plus a DJI developer key. |
| `align.BootClock` skew fitting on short logs | **Works, but note the limitation.** On the 6-second PX4 sample it fits a 500 ppm skew (residual 3.7 ms) — an overfit that is harmless inside the fitted range and would be wrong if extrapolated. The existing guard only refuses a boot span below 1 s. Consider raising it to ~30 s; not changed here because it is a judgement call about behaviour. |
| `cluster._max_displacement` is O(n^2) in observations | A 150-observation track is ~11 k haversine calls (fine); a 1 500-observation track would be ~1.1 M and take seconds. It runs once per track, not per recluster. Not a correctness problem; flagging it before someone feeds it a five-minute continuous track. |
| `SrtParseReport.assumed_gps_order` | Never populated. The GPS-order resolution happens per block in `_parse_block` and is not reported. Cosmetic provenance gap. |
| `geodesy.quat_to_euler` gimbal lock | See §1.1. **Orchestrator-owned file, not touched.** |
| `pose = "waving"` vs `schemas.POSTURES` | See §2.2. Needs an orchestrator decision. |

## 6. What this lane needs from others

1. **Sim lane (F5).** Keep `tools/capture/run.py`'s `telemetry.csv` header as it is, or expect
   `test_capture_run_columns_match_the_capture_writer_source` to fail and name the change. If you can write
   `gimbal_yaw_deg` and `gimbal_roll_deg`, the gimbal assumption in §2.1 disappears; if you can write
   `h_acc_m` / `v_acc_m` (even as 0), the noise provenance becomes explicit rather than implied.
2. **Orchestrator.** The two contract questions above: `"waving"` in `POSTURES`, and the `quat_to_euler`
   gimbal-lock fix in `sightline/common/geodesy.py`.
3. **Geo lane (F13).** `sightline/track/geometry.py` carries a local `rotation_optical_to_ned`, duplicated
   only because F11 and F13 were built in parallel. If F13 publishes an equivalent in `sightline/common/`,
   delete the local one and import theirs — there must not be two conventions in the repo.
