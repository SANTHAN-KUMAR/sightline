"""CLI for the detection + evaluation lane.

    uv run python -m sightline.detect dataset  _artifacts/dataset/<run> [...] --out _artifacts/yolo/sim
    uv run python -m sightline.detect predict  _artifacts/dataset/<run> --weights models/detect/.../best.pt
    uv run python -m sightline.detect overlay  _artifacts/dataset/<run> --preds <preds.json> --conf 0.25
    uv run python -m sightline.detect evaluate --val <run> --test <run> --preds <preds.json>

`dataset`, `overlay` and `evaluate` are CPU-only and safe while the Unreal editor is running. `predict` loads
torch and refuses to run beside the editor unless told to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sightline.detect.dataset import (
    build_yolo_dataset,
    implausible_boxes,
    load_run,
    load_runs,
    plausible_labels,
    split_by_seed,
    summarise_boxes,
)
from sightline.detect.threshold import GroundTruthBox

REPO = Path(__file__).resolve().parents[2]


def _gt_for(frame, gsd_cm_px: float = 0.0) -> list[GroundTruthBox]:
    """Capture labels -> the threshold module's ground truth.

    Non-detectable actors become `ignore` (§2.7), and a box too big to be a person is dropped entirely rather
    than scored: it is a mask defect, not a survivor (see `dataset.plausible_labels`).
    """
    return [
        GroundTruthBox(bbox_px=m.bbox_px, cls=("animal" if m.cls == "animal" else "human"),
                       occlusion=m.occlusion, posture=m.pose, submersion=m.submersion,
                       ignore=not m.aerially_detectable, track_id=m.actor_id)
        for m in plausible_labels(frame, gsd_cm_px=gsd_cm_px)
    ]


# --- subcommands -----------------------------------------------------------------------------------------------
def cmd_dataset(a: argparse.Namespace) -> int:
    runs = load_runs(a.runs)
    for r in runs:
        print(r.describe())
    print(json.dumps(summarise_boxes(runs), indent=2))
    bad = implausible_boxes(runs)
    if bad:
        print(f"WARNING: {len(bad)} labelled box(es) are too large to be people; the build will refuse "
              f"unless --allow-implausible-boxes is given. Worst: "
              f"{max(bad, key=lambda b: b['ground_m'])}", file=sys.stderr)
    splits = split_by_seed(runs, val_seeds=a.val_seed, test_seeds=a.test_seed, val_frac=a.val_frac)
    for k, v in splits.items():
        print(f"{k}: {[r.clip_id for r in v]}")
    man = build_yolo_dataset(splits, a.out, negative_frac=a.negative_frac, dry_run=a.dry_run,
                             write_images=not a.labels_only,
                             allow_implausible_boxes=a.allow_implausible_boxes)
    man.pop("data_yaml_text", None)
    print(json.dumps(man, indent=2))
    return 0


def cmd_predict(a: argparse.Namespace) -> int:
    import cv2

    from sightline.detect.overlay import detections_to_json
    from sightline.detect.rgb import DetectorConfig, RgbDetector
    from sightline.detect.train import editor_is_running

    editors = editor_is_running()
    if editors and not a.allow_editor_running:
        print(f"REFUSING: the Unreal editor is running ({', '.join(editors)}) and this machine has 8 GB of "
              "VRAM. Close it, or pass --allow-editor-running.", file=sys.stderr)
        return 2

    run = load_run(a.run)
    cfg = DetectorConfig(weights=a.weights, imgsz=a.imgsz, raw_conf=a.raw_conf, device=a.device,
                         max_batch=a.max_batch)
    det = RgbDetector(cfg)
    by_frame = {}
    frames = [f for f in run.frames if _gt_for(f, run.median_gsd_cm_px)] if a.only_labelled else run.frames
    if a.limit > 0:
        frames = frames[:: max(1, len(frames) // a.limit)][: a.limit]
    for i, fr in enumerate(frames):
        img = cv2.imread(str(fr.image_path))
        if img is None:
            raise FileNotFoundError(fr.image_path)
        by_frame[fr.stem] = det.detect(img, frame_idx=fr.frame_idx)
        del img
        print(f"[{i + 1}/{len(frames)}] {fr.stem}: {len(by_frame[fr.stem])} raw boxes", flush=True)
    out = Path(a.out or (REPO / "_artifacts" / "detect" / f"preds_{run.clip_id}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detections_to_json(by_frame, {"config": cfg.as_dict(), "run": str(run.root),
                                                            "clip_id": run.clip_id}), indent=1), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_overlay(a: argparse.Namespace) -> int:
    import cv2

    from sightline.detect.overlay import contact_sheet, detections_from_json, overlay_frame

    run = load_run(a.run)
    gsd = run.median_gsd_cm_px
    preds, meta = detections_from_json(a.preds)
    frames = [f for f in run.frames if f.stem in preds]
    if a.only_labelled:
        frames = [f for f in frames if f.scored_labels] or frames
    if a.limit > 0:
        step = max(1, len(frames) // a.limit)
        frames = frames[::step][: a.limit]
    panels, stats = [], []
    for fr in frames:
        img = cv2.imread(str(fr.image_path))
        if img is None:
            continue
        pan, st = overlay_frame(img, preds[fr.stem], _gt_for(fr, gsd), conf=a.conf, frame_name=fr.stem)
        panels.append(pan)
        stats.append(st)
        del img
        print(st.caption(), flush=True)
    if not panels:
        print("no frames drawn", file=sys.stderr)
        return 1
    out = Path(a.out or (REPO / "_artifacts" / "detect" / f"overlay_{run.clip_id}.png"))
    p = contact_sheet(panels, out)
    tot = {"tp": sum(s.tp for s in stats), "fp": sum(s.fp for s in stats), "fn": sum(s.fn for s in stats),
           "frames": len(stats), "conf": a.conf, "domain": "sim", "meta": meta}
    print(json.dumps(tot, indent=2))
    print(f"wrote {p}\nOPEN IT. A model can report mAP 0.9 while boxing shadows.")
    return 0


def cmd_threshold(a: argparse.Namespace) -> int:
    from sightline.detect.overlay import detections_from_json
    from sightline.detect.threshold import (
        FrameEval,
        choose_operating_threshold,
        freeze_operating_point,
    )

    preds, _ = detections_from_json(a.preds)
    frames = []
    for rp in a.runs:
        run = load_run(rp)
        gsd = run.median_gsd_cm_px
        for fr in run.frames:
            if fr.stem not in preds:
                continue  # the detector was never run on this frame; its truth is not a miss, it is unmeasured
            frames.append(FrameEval(predictions=preds[fr.stem], ground_truth=_gt_for(fr, gsd),
                                    frame_idx=fr.frame_idx, clip_id=fr.clip_id, seed=run.seed_group,
                                    altitude_band=f"{fr.agl_m:.0f}", time_of_day=fr.time_of_day))
    op, points, _t = choose_operating_threshold(frames, fps_processed=a.fps)
    print(op.label())
    if a.out:
        print(f"wrote {freeze_operating_point(op, a.out, points=points)}")
    return 0 if op.achieved else 3


def cmd_evaluate(a: argparse.Namespace) -> int:
    from sightline.detect.evaluate import bundle_from_runs, run_evaluation
    from sightline.detect.overlay import detections_from_json

    preds, _ = detections_from_json(a.preds)
    bundle = bundle_from_runs(load_runs(a.val), load_runs(a.test), preds, fps_processed=a.fps)
    summary = run_evaluation(bundle, out_dir=a.out or (REPO / "_artifacts" / "eval" / "detect"))
    summary.pop("_result", None)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("target_met") else 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sightline.detect")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="capture runs -> tiled Ultralytics dataset (CPU)")
    d.add_argument("runs", nargs="+")
    d.add_argument("--out", default=str(REPO / "_artifacts" / "yolo" / "sim"))
    d.add_argument("--val-seed", type=int, action="append", default=None)
    d.add_argument("--test-seed", type=int, action="append", default=None)
    d.add_argument("--val-frac", type=float, default=0.25)
    d.add_argument("--negative-frac", type=float, default=0.05)
    d.add_argument("--labels-only", action="store_true")
    d.add_argument("--dry-run", action="store_true")
    d.add_argument("--allow-implausible-boxes", action="store_true",
                   help="build anyway when a label is too large to be a person; say so in the report")
    d.set_defaults(fn=cmd_dataset)

    p = sub.add_parser("predict", help="run the detector over a capture run (GPU)")
    p.add_argument("run")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--raw-conf", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-batch", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-labelled", action="store_true",
                   help="only frames that carry at least one person-sized ground-truth box")
    p.add_argument("--allow-editor-running", action="store_true")
    p.set_defaults(fn=cmd_predict)

    o = sub.add_parser("overlay", help="draw predicted boxes on frames and LOOK at them (CPU)")
    o.add_argument("run")
    o.add_argument("--preds", required=True)
    o.add_argument("--conf", type=float, default=0.25)
    o.add_argument("--limit", type=int, default=8)
    o.add_argument("--out", default="")
    o.add_argument("--only-labelled", action="store_true", default=True)
    o.set_defaults(fn=cmd_overlay)

    t = sub.add_parser("threshold", help="sweep and freeze the operating threshold (CPU)")
    t.add_argument("runs", nargs="+")
    t.add_argument("--preds", required=True)
    t.add_argument("--fps", type=float, default=5.0)
    t.add_argument("--out", default="")
    t.set_defaults(fn=cmd_threshold)

    e = sub.add_parser("evaluate", help="freeze on val, measure on test, write the slice table (CPU)")
    e.add_argument("--val", nargs="+", required=True)
    e.add_argument("--test", nargs="+", required=True)
    e.add_argument("--preds", required=True)
    e.add_argument("--fps", type=float, default=5.0)
    e.add_argument("--out", default="")
    e.set_defaults(fn=cmd_evaluate)

    a = ap.parse_args(argv)
    try:
        return int(a.fn(a))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        # These are the lane's own refusals (leaked split, implausible boxes, editor running, missing data).
        # A traceback hides the sentence that says what to do, so print the sentence.
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
