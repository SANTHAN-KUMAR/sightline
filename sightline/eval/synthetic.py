"""A synthetic clip with known ground truth and known predictions, for driving the harness with no model.

This exists for two reasons:

1. The evaluation lane must be provable **before** a detector exists. Every metric in this package is driven
   here by predictions whose correct score is known by construction, which is how `tests/test_eval.py` can
   assert exact numbers rather than "it ran".
2. `python -m sightline.eval --demo` renders a full report from it, so the report generator is exercised end to
   end without touching the GPU, the editor or AirSim.

Nothing here is a stand-in for real evaluation data. Every number it produces is labelled `domain="sim"` by the
same rule as everything else, and a report built from it says `demo-synthetic` in the clip column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sightline.common.geodesy import offset_ne
from sightline.eval.groundtruth import EvalDataset, GtBox, GtFrame, GtSurvivor
from sightline.eval.posture import AttributeSample, RankingComparison
from sightline.schemas import Detection, Evidence, GeoFix, Observation, Record, ScoreComponents, Track

ORIGIN_LAT, ORIGIN_LON = 11.4870, 76.1450  # FloodValley scenario origin (docs/HANDBOOK.md section 2)

ZONES = ("settlement", "channel", "fan")
POSTURE_MIX = ("standing", "sitting", "prone", "supine", "half_submerged", "trapped")
SUBMERSION_MIX = ("dry", "dry", "wet", "partial", "half", "head_only")
URGENCY_OF_SUBMERSION = {"head_only": "immersed", "half": "immersed", "partial": "immersed"}


@dataclass(slots=True)
class Scenario:
    """One synthetic clip plus everything the harness can be asked to score."""

    dataset: EvalDataset
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    records: list[Record] = field(default_factory=list)
    pod_predicted: list[float] = field(default_factory=list)
    pod_found: list[bool] = field(default_factory=list)
    ranking: RankingComparison = field(default_factory=RankingComparison)
    posture_samples: list[AttributeSample] = field(default_factory=list)
    submersion_samples: list[AttributeSample] = field(default_factory=list)
    truth: dict = field(default_factory=dict)  # what the construction guarantees, for the tests


def _box_for(rng: np.random.Generator, agl_m: float, posture: str) -> tuple[float, float]:
    """(width, height) in pixels for a 4K frame: ~30 px upright at 45 m, scaled by 1/AGL, prone is wider."""
    base = 30.0 * (45.0 / max(agl_m, 1.0))
    if posture in ("prone", "supine", "half_submerged"):
        w, h = base * 1.9, base * 0.55
    elif posture in ("sitting", "trapped"):
        w, h = base * 0.8, base * 0.7
    else:
        w, h = base * 0.45, base
    j = rng.uniform(0.9, 1.15)
    return max(6.0, w * j), max(6.0, h * j)


def make_scenario(
    *,
    domain: str = "sim",
    seed: int = 7,
    n_survivors: int = 10,
    n_frames: int = 60,
    fps_processed: float = 5.0,
    p_detect: float = 0.94,
    fp_per_frame: float = 0.35,
    n_buried: int = 2,
    duplicate_records: int = 1,
    spurious_records: int = 1,
    clip_id: str = "demo-synthetic",
    split: str = "test",
) -> Scenario:
    """Build a clip whose ground truth and predictions are both known, so the metrics can be checked by hand."""
    rng = np.random.default_rng(seed)
    ds = EvalDataset(domain=domain, fps_processed=fps_processed, clip_id=clip_id, split=split,  # type: ignore[arg-type]
                     seed_group=f"seed-{seed}", randomisation=False,
                     notes="synthetic ground truth and synthetic predictions; no model was run")

    survivors: list[GtSurvivor] = []
    for i in range(n_survivors + n_buried):
        buried = i >= n_survivors
        sub = SUBMERSION_MIX[i % len(SUBMERSION_MIX)]
        posture = POSTURE_MIX[i % len(POSTURE_MIX)]
        lat, lon = offset_ne(ORIGIN_LAT, ORIGIN_LON, float(rng.uniform(-250, 250)), float(rng.uniform(-250, 250)))
        survivors.append(GtSurvivor(
            gt_id=i, lat=lat, lon=lon, alt_msl_m=1058.0,
            count=int(rng.choice([1, 1, 1, 2, 3], p=[0.6, 0.1, 0.1, 0.1, 0.1])),
            urgency_class="trapped" if posture == "trapped" else URGENCY_OF_SUBMERSION.get(sub, "stranded"),
            posture=posture, submersion=sub, zone=ZONES[i % len(ZONES)], buried=buried))
    ds.survivors = survivors

    altitudes = [45.0, 45.0, 55.0, 70.0, 95.0]
    tods = ["day", "day", "day", "dusk", "night"]
    detections: list[Detection] = []
    obs_by_id: dict[int, list[tuple[int, Detection]]] = {}
    n_gt_boxes = 0
    for fi in range(n_frames):
        seg = fi // max(1, n_frames // len(altitudes))
        seg = min(seg, len(altitudes) - 1)
        agl, tod = altitudes[seg], tods[seg]
        frame = GtFrame(frame_idx=fi, t_utc=1_780_000_000.0 + fi / fps_processed, clip_id=clip_id,
                        zone=ZONES[seg % len(ZONES)], agl_m=agl, time_of_day=tod, modality="rgb",
                        gimbal_pitch_deg=-90.0, gsd_cm_px=100.0 * agl / 3840.0 * 1.2)
        visible = [s for s in survivors if s.findable and (s.gt_id + fi) % 4 != 3]
        for s in visible:
            w, h = _box_for(rng, agl, str(s.posture))
            cx = 300 + (s.gt_id * 317) % 3200 + fi * 3.0
            cy = 200 + (s.gt_id * 191) % 1700 + fi * 1.5
            occl = int(min(2, max(0, round(rng.normal(0.7, 0.7)))))
            box = GtBox(bbox_px=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), gt_id=s.gt_id, cls="human",
                        frame_idx=fi, occlusion=occl, posture=s.posture, submersion=s.submersion,
                        visible_fraction=float(np.clip(1.0 - 0.3 * occl - rng.uniform(0, 0.15), 0.05, 1.0)),
                        context="water" if s.submersion in ("half", "head_only") else "structure")
            frame.boxes.append(box)
            n_gt_boxes += 1
            # a detection, most of the time; harder when small, occluded or submerged
            p = p_detect - 0.25 * (occl == 2) - 0.20 * (s.submersion == "head_only") - 0.25 * (agl >= 90)
            if rng.random() < max(p, 0.05):
                jx, jy = rng.normal(0, w * 0.06), rng.normal(0, h * 0.06)
                score = float(np.clip(rng.beta(6.0, 2.0) * 0.75 + 0.22, 0.05, 0.999))
                # SOLUTION_DOC 5.5: on a limb-only or head-only target the box extent is ill-defined, so the
                # model draws something body-sized around a head-sized visible extent. IoU lands near 0.35:
                # counted at IoU 0.25 ("found the person"), missed at IoU 0.5. That is the whole reason the
                # relaxed threshold is reported.
                grow = 1.7 if s.submersion == "head_only" else 1.0
                gx, gy = w * (grow - 1) / 2.0, h * (grow - 1) / 2.0
                det = Detection(bbox_px=(box.bbox_px[0] + jx - gx, box.bbox_px[1] + jy - gy,
                                         box.bbox_px[2] + jx + gx, box.bbox_px[3] + jy + gy),
                                score=score, cls="human", modality="rgb", frame_idx=fi,
                                posture=s.posture if rng.random() < 0.7 else "unknown",
                                posture_conf=float(rng.uniform(0.4, 0.9)),
                                submersion=s.submersion if rng.random() < 0.65 else "unknown",
                                submersion_conf=float(rng.uniform(0.4, 0.9)),
                                occlusion=occl, visible_fraction=box.visible_fraction)
                detections.append(det)
                obs_by_id.setdefault(s.gt_id, []).append((fi, det))
        for _ in range(int(rng.poisson(fp_per_frame))):
            w = float(rng.uniform(10, 45))
            h = w * float(rng.uniform(0.5, 1.6))
            x = float(rng.uniform(0, 3840 - w))
            y = float(rng.uniform(0, 2160 - h))
            detections.append(Detection(bbox_px=(x, y, x + w, y + h),
                                        score=float(np.clip(rng.beta(1.6, 6.0) * 0.8 + 0.05, 0.02, 0.95)),
                                        cls="human", modality="rgb", frame_idx=fi))
        ds.frames.append(frame)

    # tracks: one per survivor that was seen at least three times (SOLUTION_DOC 5.6 rule 3)
    tracks: list[Track] = []
    for gid, seen in sorted(obs_by_id.items()):
        if len(seen) < 3:
            continue
        s = survivors[gid]
        t = Track(track_id=1000 + gid, cls="human", confirmed=True, clip_id=clip_id)
        for fi, det in seen:
            lat, lon = offset_ne(s.lat, s.lon, float(rng.normal(0, 1.8)), float(rng.normal(0, 1.8)))
            fix = GeoFix(lat=lat, lon=lon, alt_msl_m=s.alt_msl_m, h_acc_m=2.8, off_nadir_deg=float(rng.uniform(0, 20)),
                         agl_m=55.0, slant_range_m=58.0)
            t.observations.append(Observation(track_id=t.track_id, frame_idx=fi,
                                              t_utc=1_780_000_000.0 + fi / fps_processed, det=det, fix=fix,
                                              clip_id=clip_id))
        tracks.append(t)

    # records: one per confirmed track, plus a deliberate duplicate and a deliberate spurious record
    records: list[Record] = []
    for t in tracks:
        gid = t.track_id - 1000
        s = survivors[gid]
        lat, lon = t.weighted_median_position()
        best = t.best()
        promoted = str(s.submersion) in ("half", "head_only")
        comp = ScoreComponents(p_living=float(best.det.score if best else 0.5), w_class=1.0,
                               urgency=3.0 if promoted else 1.0,
                               count_bonus=1.0 + 0.1 * s.count,
                               urgency_class=s.urgency_class, posture_promoted=promoted)
        records.append(Record(
            cluster_id=gid, status="confirmed", cls="human", lat=lat, lon=lon, alt_msl_m=s.alt_msl_m,
            h_acc_m=2.8, agl_m=55.0, off_nadir_deg=8.0,
            confidence=float(1.0 - np.prod([1 - o.det.score for o in t.observations])),
            confidence_max_det=float(max(o.det.score for o in t.observations)),
            score=comp.total(), components=comp, n_observations=len(t.observations), n_tracks_merged=1,
            seen_in_passes=[0], first_seen_utc=t.observations[0].t_utc, last_seen_utc=t.observations[-1].t_utc,
            count_estimate=s.count + (1 if gid % 7 == 0 else 0), count_min=1, count_max=s.count + 1,
            posture=s.posture, submersion=s.submersion, zone=s.zone,
            evidence=[Evidence(thumb_uri=f"thumb/{gid}.jpg", clip_id=clip_id,
                               frame_idx=best.frame_idx if best else 0,
                               frame_time_utc=best.t_utc if best else 0.0,
                               bbox_px=best.det.bbox_px if best else (0, 0, 1, 1),
                               det_conf=best.det.score if best else 0.0)]))
    for k in range(duplicate_records):
        src = records[k % len(records)]
        lat, lon = offset_ne(src.lat, src.lon, 3.0, 2.0)
        dup = Record(cluster_id=900 + k, status="confirmed", cls="human", lat=lat, lon=lon,
                     alt_msl_m=src.alt_msl_m, h_acc_m=src.h_acc_m, confidence=src.confidence * 0.8,
                     score=src.score * 0.8, count_estimate=1, zone=src.zone,
                     notes="deliberate duplicate: the dedup step failed to merge this cluster")
        records.append(dup)
    for k in range(spurious_records):
        lat, lon = offset_ne(ORIGIN_LAT, ORIGIN_LON, 900.0 + 40 * k, -900.0)
        records.append(Record(cluster_id=800 + k, status="candidate", cls="human", lat=lat, lon=lon,
                              h_acc_m=3.4, confidence=0.42, score=0.42,
                              notes="deliberate false record: a roofing sheet at midday"))

    # POD calibration inputs: a coverage map that is honest in the middle and over-promises at the top
    pod_pred, pod_found = [], []
    for s in survivors:
        if not s.findable:
            continue
        seen = len(obs_by_id.get(s.gt_id, ()))
        base = float(np.clip(0.15 + 0.028 * seen, 0.05, 0.97))
        pod_pred.append(base)
        pod_found.append(seen >= 3)
    for _ in range(30):  # extra cells so the reliability bins have enough samples to be reportable
        p = float(rng.uniform(0.05, 0.95))
        pod_pred.append(p)
        pod_found.append(bool(rng.random() < p * 0.9))

    # ranking: the posture head promotes immersed records; without it every record uses the base urgency
    matched = [r for r in records if 0 <= r.cluster_id < len(survivors)]
    ranking = RankingComparison.from_urgency_classes(
        [r.score for r in matched],
        [r.components.p_living * r.components.count_bonus for r in matched],
        [str(survivors[r.cluster_id].urgency_class) for r in matched],
        [r.record_id for r in matched])

    posture_samples, submersion_samples = [], []
    for gid, seen in obs_by_id.items():
        s = survivors[gid]
        for _, det in seen:
            posture_samples.append(AttributeSample(gt=str(s.posture), pred=str(det.posture),
                                                   size_px=det.size_px, conf=det.posture_conf))
            submersion_samples.append(AttributeSample(gt=str(s.submersion), pred=str(det.submersion),
                                                      size_px=det.size_px, conf=det.submersion_conf))

    return Scenario(dataset=ds, detections=detections, tracks=tracks, records=records,
                    pod_predicted=pod_pred, pod_found=pod_found, ranking=ranking,
                    posture_samples=posture_samples, submersion_samples=submersion_samples,
                    truth={"n_gt_boxes": n_gt_boxes, "n_survivors_findable": n_survivors,
                           "n_buried": n_buried, "n_duplicate_records": duplicate_records,
                           "n_spurious_records": spurious_records, "seed": seed})


# --- tiny hand-built fixtures (the tests use these; the answers are computable on paper) ---------------------
def perfect_clip(n_frames: int = 4, n_boxes: int = 3, domain: str = "sim",
                 score: float = 0.9, split: str = "test") -> tuple[EvalDataset, list[Detection]]:
    """A clip where every ground-truth box has an exactly-coincident prediction. Recall must be 1.0."""
    ds = EvalDataset(domain=domain, fps_processed=5.0, clip_id="perfect", split=split)  # type: ignore[arg-type]
    dets: list[Detection] = []
    for fi in range(n_frames):
        frame = GtFrame(frame_idx=fi, clip_id="perfect", zone="settlement", agl_m=50.0, time_of_day="day")
        for b in range(n_boxes):
            x, y = 100.0 + 200 * b, 100.0 + 150 * fi
            box = (x, y, x + 40, y + 60)
            frame.boxes.append(GtBox(bbox_px=box, gt_id=b, frame_idx=fi, occlusion=0, posture="standing",
                                     submersion="dry"))
            dets.append(Detection(bbox_px=box, score=score, cls="human", frame_idx=fi))
        ds.frames.append(frame)
    return ds, dets


def shifted_boxes(box: tuple[float, float, float, float], iou: float) -> tuple[float, float, float, float]:
    """Translate a box along x so the IoU with the original is (close to) `iou`.

    For an axis-aligned translation by `d` of a w x h box, IoU = (w - d) / (w + d), so d = w (1 - iou)/(1 + iou).
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    d = w * (1.0 - iou) / (1.0 + iou)
    return (x1 + d, y1, x2 + d, y2)


__all__ = ["ORIGIN_LAT", "ORIGIN_LON", "Scenario", "make_scenario", "perfect_clip", "shifted_boxes"]
