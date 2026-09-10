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
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from sightline.schemas import (SCHEMA_VERSION, Detection, GeoFix, Intrinsics, Record, Telemetry,
                               feature_collection)

REPO = Path(__file__).resolve().parents[1]


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
                q_gimbal=(0.70710678, 0.0, -0.70710678, 0.0),        # nadir, verified at capture time
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
    out = []
    for m in json.loads(label_json.read_text()):
        x1, y1, x2, y2 = m["bbox_px"]
        out.append(Detection(bbox_px=(float(x1), float(y1), float(x2), float(y2)), score=1.0,
                             cls=m.get("cls", "human"), modality="rgb",
                             posture=m.get("pose", "unknown"), posture_conf=1.0,
                             submersion=m.get("submersion", "unknown"), submersion_conf=1.0,
                             occlusion=m.get("occlusion")))
    return out


def detections_from_model(frame_png: Path, weights: str, conf: float) -> list[Detection]:
    """Run the trained detector. Imported lazily: torch must not load for a truth-replay run."""
    from sightline.detect.rgb import detect_frame            # noqa: PLC0415

    return detect_frame(frame_png, weights=weights, conf=conf)


# --- the run -----------------------------------------------------------------------------------------------
def run(clip_dir: Path, out_dir: Path, *, detector: str = "truth", weights: str = "", conf: float = 0.25,
        every: int = 1, noise: bool = True) -> dict:
    from sightline.dedup import records_from_tracks
    from sightline.geo import project_detection
    from sightline.schemas import FrameBundle
    from sightline.track import Tracker, TrackerConfig
    from sightline.triage import rank_records

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stages: dict[str, int] = {}

    tracker: Tracker | None = None
    n_frames = n_det = n_fix = 0
    for k, tel, intr, labels in iter_frames(clip_dir, every=every):
        n_frames += 1
        dets = (detections_from_truth(labels) if detector == "truth"
                else detections_from_model(clip_dir / "images" / f"{labels.stem}.png", weights, conf))
        for d in dets:
            d.frame_idx = k
        n_det += len(dets)
        fixes = []
        for d in dets:
            f = project_detection(d, tel, intr)
            fixes.append(f)
            if f is not None and f.valid:
                n_fix += 1
        if tracker is None:
            # Frames are captured one per waypoint, so consecutive frames are a full grid step apart: this is
            # a still-survey clip, not video. `fps` only sets the buffer length in PROCESSED-frame units.
            tracker = Tracker(TrackerConfig(fps=1.0, cmc_enabled=False), intrinsics=intr,
                              clip_id=tel.clip_id)
        bundle = FrameBundle(frame_idx=k, t_utc=tel.t_utc, telemetry=tel, intrinsics=intr,
                             clip_id=tel.clip_id)
        tracker.update(dets, bundle, fixes)

    tracks = tracker.close() if tracker is not None else []
    stages.update(frames=n_frames, detections=n_det, located=n_fix, tracks=len(tracks))

    records = records_from_tracks(tracks)
    stages["records"] = len(records)
    ranked = rank_records(records)
    stages["ranked"] = len(ranked)

    (out_dir / "records.geojson").write_text(json.dumps(feature_collection(ranked), indent=1), encoding="utf-8")
    top = [{"rank": r.priority_rank, "id": r.record_id[:8], "cls": r.cls, "score": round(r.score, 4),
            "lat": round(r.lat, 6), "lon": round(r.lon, 6), "h_acc_m": round(r.h_acc_m, 2),
            "posture": r.posture, "submersion": r.submersion, "n_obs": r.n_observations,
            "count": r.count_estimate, "status": r.status} for r in ranked[:15]]
    (out_dir / "top_records.json").write_text(json.dumps(top, indent=1), encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION, "clip": str(clip_dir), "detector": detector,
        "weights": weights, "conf": conf, "frame_stride": every, "noise_injected": noise,
        "domain": "sim", "stages": stages, "seconds": round(time.time() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("clip")
    ap.add_argument("--out", default="_artifacts/pipeline_run")
    ap.add_argument("--detector", choices=("truth", "model"), default="truth")
    ap.add_argument("--weights", default="")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--every", type=int, default=1)
    a = ap.parse_args()
    clip = Path(a.clip) if Path(a.clip).is_absolute() else REPO / a.clip
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    m = run(clip, out, detector=a.detector, weights=a.weights, conf=a.conf, every=a.every)
    print(json.dumps(m, indent=1))
    if a.detector == "truth":
        print("\nNOTE: detector=truth replays the simulator's own labels. These are ceiling numbers for the "
              "chain after detection, NOT detection results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
