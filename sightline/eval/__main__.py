r"""CLI for the evaluation harness (F19).

    D:\Tools\uv\uv.exe run python -m sightline.eval --demo
        Build a synthetic clip (known ground truth, known predictions — no model, no GPU, no editor), run every
        SOLUTION_DOC 5.12 metric on it, export the FiftyOne manifest and write the report to _artifacts/eval/.

    D:\Tools\uv\uv.exe run python -m sightline.eval --gt <truth.json> --pred <predictions.json>
        Same, on a real export. `--gt` is an `EvalDataset` written by `sightline.eval.save_dataset`;
        `--pred` is {"detections": [...], "records": [...], "tracks": [...]} of `schemas.py` dictionaries.

The operating threshold is frozen on the validation clip (`--val-gt` / `--val-pred`) when one is given, per
SOLUTION_DOC 5.5; otherwise `--conf` must be supplied explicitly, or `--sweep-on-test` acknowledged in writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path

from sightline.eval.calibration import evaluate_pod_calibration
from sightline.eval.detection import freeze_operating_threshold, match_dataset, sweep_confidence
from sightline.eval.fiftyone_export import write_manifest
from sightline.eval.geoloc import evaluate_geolocation, samples_from_records
from sightline.eval.groundtruth import load_dataset
from sightline.eval.harness import EvalConfig, add_record_eval, freeze_on_validation, run_detection_eval
from sightline.eval.posture import attribute_accuracy_rows, promotion_audit_rows, rank_correlation_rows
from sightline.eval.report import DEFAULT_REPORT_DIR, write_report
from sightline.eval.tracking import evaluate_tracking, frames_from_tracks
from sightline.schemas import Detection, Record


def _from_dicts(cls, items):
    names = {f.name for f in dataclass_fields(cls)}
    out = []
    for d in items:
        kw = {k: v for k, v in d.items() if k in names}
        if "bbox_px" in kw and kw["bbox_px"] is not None:
            kw["bbox_px"] = tuple(kw["bbox_px"])
        out.append(cls(**kw))
    return out


def run_demo(out_dir: Path, seed: int = 7) -> Path:
    from sightline.eval.synthetic import make_scenario

    cfg = EvalConfig()
    val = make_scenario(seed=seed + 1, clip_id="demo-synthetic-val", split="val", n_frames=40)
    op = freeze_on_validation(val.dataset, val.detections, cfg)

    sc = make_scenario(seed=seed, clip_id="demo-synthetic", split="test")
    res = run_detection_eval(sc.dataset, sc.detections, op.conf, cfg, operating=op)
    res = add_record_eval(res, sc.records)
    key = res.base_key()
    res.metrics.extend(evaluate_tracking(frames_from_tracks(sc.dataset, sc.tracks), key))
    res.metrics.extend(evaluate_geolocation(samples_from_records(sc.records), sc.dataset.survivors, key,
                                            noise_injected=True))
    res.metrics.extend(evaluate_pod_calibration(sc.pod_predicted, sc.pod_found, key))
    res.metrics.extend(attribute_accuracy_rows(sc.posture_samples, key, attribute="posture"))
    res.metrics.extend(attribute_accuracy_rows(sc.submersion_samples, key, attribute="submersion",
                                               slice_on_posture_axis=False))
    res.metrics.extend(rank_correlation_rows(sc.ranking, key))
    res.metrics.extend(promotion_audit_rows(sc.ranking, key))

    stats = write_manifest(sc.dataset, res.matches[f"{cfg.iou_primary:g}"], op.conf)
    path = write_report([res], out_dir=out_dir, stem="demo_synthetic",
                        title="Sightline evaluation report (synthetic self-check)")
    print(json.dumps({
        "report": str(path),
        "rows": len(res.metrics),
        "operating_conf": op.conf,
        "operating_recall": op.recall,
        "target_met": op.achieved,
        "fiftyone_manifest": str(stats.path),
        "fiftyone_launch": stats.launch_command,
    }, indent=1))
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sightline.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run on a synthetic clip and write a report")
    ap.add_argument("--gt", type=Path, help="ground-truth EvalDataset JSON (test split)")
    ap.add_argument("--pred", type=Path, help="predictions JSON: detections / records / tracks")
    ap.add_argument("--val-gt", type=Path, help="validation EvalDataset JSON, for freezing the threshold")
    ap.add_argument("--val-pred", type=Path, help="validation predictions JSON")
    ap.add_argument("--conf", type=float, help="a threshold that was already frozen elsewhere")
    ap.add_argument("--sweep-on-test", action="store_true",
                    help="acknowledge picking the threshold on the test clip (it will be stated in the report)")
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if args.demo or not args.gt:
        run_demo(args.out, args.seed)
        return 0

    ds = load_dataset(args.gt)
    payload = json.loads(args.pred.read_text(encoding="utf-8")) if args.pred else {}
    detections = _from_dicts(Detection, payload.get("detections", []))
    records = _from_dicts(Record, payload.get("records", []))
    cfg = EvalConfig()

    if args.val_gt and args.val_pred:
        val = load_dataset(args.val_gt)
        val_pred = _from_dicts(Detection, json.loads(args.val_pred.read_text(encoding="utf-8"))
                               .get("detections", []))
        val_res = match_dataset(val, val_pred, iou_thr=cfg.iou_primary)
        op = freeze_operating_threshold(sweep_confidence(val_res, val.domain), cfg.target_recall)
        conf = op.conf
    elif args.conf is not None:
        op, conf = None, args.conf
    elif args.sweep_on_test:
        res0 = match_dataset(ds, detections, iou_thr=cfg.iou_primary)
        op = freeze_operating_threshold(sweep_confidence(res0, ds.domain), cfg.target_recall)
        op.note = ("THRESHOLD SWEPT ON THE TEST CLIP at the operator's request; this is not a clean "
                   "held-out number (SOLUTION_DOC 5.5).")
        conf = op.conf
    else:
        print("error: give --val-gt/--val-pred, or --conf, or --sweep-on-test", file=sys.stderr)
        return 2

    res = run_detection_eval(ds, detections, conf, cfg, operating=op)
    if records:
        res = add_record_eval(res, records)
        res.metrics.extend(evaluate_geolocation(samples_from_records(records), ds.survivors, res.base_key()))
    stats = write_manifest(ds, res.matches[f"{cfg.iou_primary:g}"], conf)
    path = write_report([res], out_dir=args.out, stem=f"eval_{ds.clip_id or 'clip'}")
    print(json.dumps({"report": str(path), "rows": len(res.metrics), "operating_conf": conf,
                      "fiftyone_manifest": str(stats.path), "fiftyone_launch": stats.launch_command}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
