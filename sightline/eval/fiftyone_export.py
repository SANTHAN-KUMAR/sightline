"""Export an evaluated clip for the FiftyOne error browser (§5.12: "browse false positives on glint and false
negatives under canopy").

**FiftyOne is not importable from the main environment on purpose.** It hard-requires
`opencv-python-headless` (the main env owns `cv2` through `opencv-python`) and pins starlette/pymongo/
strawberry against the main env's FastAPI stack, so it lives in its own uv project at `envs/fiftyone`
(see `envs/fiftyone/pyproject.toml` and `docs/verification/python_stack.md`).

The bridge is therefore a **file**, not an import: this module writes a plain-JSON manifest that
`sightline/eval/fiftyone_browse.py` — a script that imports only fiftyone, numpy and the stdlib — loads with
the FiftyOne interpreter. Neither side imports the other's dependencies.

Launch (both lines are printed by `write_manifest()` and repeated in `docs/lanes/eval.md`):

    D:\\Tools\\uv\\uv.exe run --project D:\\Sightline\\envs\\fiftyone python ^
        D:\\Sightline\\sightline\\eval\\fiftyone_browse.py D:\\Sightline\\_artifacts\\eval\\fiftyone\\<clip>.json

    # or straight from the env's interpreter, no uv:
    D:\\Sightline\\envs\\fiftyone\\.venv\\Scripts\\python.exe ^
        D:\\Sightline\\sightline\\eval\\fiftyone_browse.py <manifest.json>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sightline.eval.detection import MatchResult
from sightline.eval.groundtruth import EvalDataset, GtBox, GtFrame

REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPO / "_artifacts" / "eval" / "fiftyone"
FIFTYONE_PYTHON = REPO / "envs" / "fiftyone" / ".venv" / "Scripts" / "python.exe"
BROWSE_SCRIPT = Path(__file__).resolve().parent / "fiftyone_browse.py"


def _rel_bbox(bbox: Sequence[float], w: int, h: int) -> list[float]:
    """FiftyOne stores [x, y, width, height] as fractions of the image size."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


def _gt_detection(b: GtBox, f: GtFrame, found: bool) -> dict[str, Any]:
    return {
        "label": str(b.cls),
        "bounding_box": _rel_bbox(b.bbox_px, f.width_px, f.height_px),
        "gt_id": int(b.gt_id),
        "occlusion": b.occlusion,
        "posture": str(b.posture),
        "submersion": str(b.submersion),
        "visible_fraction": b.visible_fraction,
        "size_px": b.size_px,
        "uncertain": bool(b.uncertain),
        "iscrowd": bool(b.is_group),
        "eval": "tp" if found else "fn",
    }


@dataclass(slots=True)
class ExportStats:
    path: Path
    n_samples: int
    n_gt: int
    n_pred: int
    n_fp: int
    n_fn: int
    launch_command: str


def build_manifest(
    ds: EvalDataset,
    res: MatchResult,
    conf: float,
    *,
    name: str = "",
    extra_sample_fields: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One JSON document: samples with `ground_truth` and `predictions` fields, tagged fp / fn at `conf`."""
    found = res.gt_found(conf)
    gt_by_frame: dict[int, list[dict[str, Any]]] = {}
    for i, (b, f) in enumerate(zip(res.gt_boxes, res.gt_frames)):
        gt_by_frame.setdefault(f.frame_idx, []).append(_gt_detection(b, f, bool(found[i])))
    for b_i, (b, f) in enumerate(zip(res.group_boxes, res.group_frames)):
        gt_by_frame.setdefault(f.frame_idx, []).append(
            _gt_detection(b, f, bool(res.group_hit_score[b_i] >= conf))
        )

    frame_meta = {f.frame_idx: f for f in ds.frames}
    pred_by_frame: dict[int, list[dict[str, Any]]] = {}
    keep = res.pred_score >= conf
    for i, d in enumerate(res.detections):
        if not keep[i]:
            continue
        fi = int(res.pred_frame[i])
        f = frame_meta[fi]
        pred_by_frame.setdefault(fi, []).append({
            "label": str(d.cls),
            "bounding_box": _rel_bbox(d.bbox_px, f.width_px, f.height_px),
            "confidence": float(d.score),
            "modality": str(d.modality),
            "tile_idx": int(d.tile_idx),
            "posture": str(d.posture),
            "submersion": str(d.submersion),
            "thermal_c": d.thermal_c,
            "eval": str(res.pred_outcome[i]),
        })

    samples = []
    for f in ds.frames:
        gts = gt_by_frame.get(f.frame_idx, [])
        preds = pred_by_frame.get(f.frame_idx, [])
        n_fp = sum(1 for p in preds if p["eval"] == "fp")
        n_fn = sum(1 for g in gts if g["eval"] == "fn")
        tags = []
        if n_fp:
            tags.append("has_fp")
        if n_fn:
            tags.append("has_fn")
        if not n_fp and not n_fn and gts:
            tags.append("clean")
        sample = {
            "filepath": f.image_path,
            "frame_idx": f.frame_idx,
            "clip_id": f.clip_id or ds.clip_id,
            "tags": tags,
            "domain": ds.domain,
            "zone": str(f.zone),
            "agl_m": f.agl_m,
            "time_of_day": f.time_of_day,
            "modality": str(f.modality),
            "gimbal_pitch_deg": f.gimbal_pitch_deg,
            "gsd_cm_px": f.gsd_cm_px,
            "weather": dict(f.weather),
            "n_fp": n_fp,
            "n_fn": n_fn,
            "ground_truth": gts,
            "predictions": preds,
        }
        if extra_sample_fields and f.frame_idx in extra_sample_fields:
            sample.update(extra_sample_fields[f.frame_idx])
        samples.append(sample)

    return {
        "schema": "sightline-fiftyone-manifest/1",
        "dataset_name": name or f"sightline-{ds.domain}-{ds.clip_id or 'clip'}",
        "domain": ds.domain,
        "clip_id": ds.clip_id,
        "split": ds.split,
        "seed_group": ds.seed_group,
        "operating_conf": float(conf),
        "iou_thr": float(res.iou_thr),
        "fps_processed": ds.fps_processed,
        "randomisation": ds.randomisation,
        "samples": samples,
    }


def launch_command(manifest_path: Path) -> str:
    return f'"{FIFTYONE_PYTHON}" "{BROWSE_SCRIPT}" "{manifest_path}"'


def write_manifest(
    ds: EvalDataset,
    res: MatchResult,
    conf: float,
    *,
    out_dir: Path | str = DEFAULT_EXPORT_DIR,
    name: str = "",
) -> ExportStats:
    """Write the manifest and return the launch command for the error browser."""
    manifest = build_manifest(ds, res, conf, name=name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{manifest['dataset_name']}.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n_fp = sum(s["n_fp"] for s in manifest["samples"])
    n_fn = sum(s["n_fn"] for s in manifest["samples"])
    return ExportStats(
        path=path,
        n_samples=len(manifest["samples"]),
        n_gt=int(len(res.gt_boxes) + len(res.group_boxes)),
        n_pred=int((res.pred_score >= conf).sum()),
        n_fp=n_fp,
        n_fn=n_fn,
        launch_command=launch_command(path),
    )


def missing_images(manifest: dict[str, Any]) -> list[int]:
    """Frames whose `image_path` is empty or absent — FiftyOne can list them but cannot render them."""
    out = []
    for s in manifest["samples"]:
        p = s.get("filepath") or ""
        if not p or not Path(p).exists():
            out.append(int(s["frame_idx"]))
    return out


def failure_clusters(res: MatchResult, conf: float, top: int = 5) -> list[dict[str, Any]]:
    """The §5.12 "top failure clusters": which slice of the ground truth is being missed, most-missed first.

    Expected, per the doc: head-only in turbid water above 60 m; midday roofing-sheet false positives;
    crossover-window thermal misses. This groups the actual misses so the report names the real ones.
    """
    missed = ~res.gt_found(conf)
    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, m in enumerate(missed):
        if not m:
            continue
        b, f = res.gt_boxes[i], res.gt_frames[i]
        band = "<=60m" if f.agl_m <= 60 else ">60m"
        key = (str(b.submersion), str(f.zone), band)
        groups.setdefault(key, []).append(i)
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]
    return [
        {"submersion": k[0], "zone": k[1], "altitude": k[2], "n_missed": len(v),
         "median_size_px": float(np.median([res.gt_boxes[i].size_px for i in v]))}
        for k, v in ranked
    ]


__all__ = [
    "BROWSE_SCRIPT", "DEFAULT_EXPORT_DIR", "FIFTYONE_PYTHON", "ExportStats", "build_manifest",
    "failure_clusters", "launch_command", "missing_images", "write_manifest",
]
