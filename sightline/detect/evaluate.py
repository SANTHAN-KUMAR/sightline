"""F19 wiring: capture run + predictions -> frozen threshold -> the §5.12 slice table, every row `domain="sim"`.

`sightline/eval/` is a complete, tested harness that has never seen a real detector (`docs/lanes/
coverage_plan_eval.md` §6 item 10). This module is the bridge that changes that: it takes the capture runs this
lane trains on, the predictions a model produced on them, and drives the harness in the order §5.5 requires.

The order is the whole point and is enforced, not documented:

1. ``freeze_on_validation(val, val_dets)`` -- the highest confidence at which human recall >= 0.92, chosen on
   the **val** split. `harness.freeze_on_validation` refuses a non-val split unless the caller opts in out loud.
2. ``run_detection_eval(test, test_dets, op.conf, operating=op)`` -- measured at that frozen number on the test
   split, never re-chosen there.
3. The slice table (zone x altitude x posture x submersion x pixel size) and the report, with a domain column.

If the threshold cannot reach 0.92 the harness returns ``achieved=False`` and the honest recall. That is a
finding to report, not a target to move (hard rule 2).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sightline.detect.dataset import CaptureRun, to_eval_dataset
from sightline.schemas import Detection, MetricRow

__all__ = ["EvalBundle", "bundle_from_runs", "run_evaluation", "submersion_rows", "write_slice_table"]


@dataclass(slots=True)
class EvalBundle:
    """One evaluation: a val split to freeze on, a test split to measure on, and the predictions for both."""

    val_runs: list[CaptureRun] = field(default_factory=list)
    test_runs: list[CaptureRun] = field(default_factory=list)
    predictions: dict[str, list[Detection]] = field(default_factory=dict)  # keyed by frame stem
    fps_processed: float = 5.0

    def seeds(self) -> dict[str, list[int]]:
        return {"val": sorted({r.scenario_seed for r in self.val_runs}),
                "test": sorted({r.scenario_seed for r in self.test_runs})}


def _dataset_for(runs: Sequence[CaptureRun], split: str, fps: float):
    """Merge several runs of one split into a single `EvalDataset`, re-indexing frames so they stay unique."""
    from sightline.eval.groundtruth import EvalDataset

    frames: list[Any] = []
    survivors: list[Any] = []
    seen: set[int] = set()
    offset = 0
    clips: list[str] = []
    seeds: list[str] = []
    for run in runs:
        ds = to_eval_dataset(run, split=split, fps_processed=fps)
        for f in ds.frames:
            f.frame_idx = f.frame_idx + offset
            for b in f.boxes:
                b.frame_idx = f.frame_idx
            frames.append(f)
        offset += 100000
        for s in ds.survivors:
            if s.gt_id not in seen:
                seen.add(s.gt_id)
                survivors.append(s)
        clips.append(run.clip_id)
        seeds.append(run.seed_group)
    return EvalDataset(domain="sim", frames=frames, survivors=survivors, fps_processed=fps,
                       clip_id="+".join(clips), split=split,  # type: ignore[arg-type]
                       seed_group="+".join(sorted(set(seeds))), randomisation=False,
                       notes="built by sightline.detect.evaluate from capture runs")


def _detections_for(runs: Sequence[CaptureRun], preds: dict[str, list[Detection]]) -> list[Detection]:
    """Re-stamp `frame_idx` to match the merged dataset's re-indexing above (same offset scheme)."""
    from dataclasses import replace

    out: list[Detection] = []
    offset = 0
    for run in runs:
        for fr in run.frames:
            for d in preds.get(fr.stem, ()):
                out.append(replace(d, frame_idx=fr.frame_idx + offset))
        offset += 100000
    return out


def bundle_from_runs(val_runs: Sequence[CaptureRun], test_runs: Sequence[CaptureRun],
                     predictions: dict[str, list[Detection]], *, fps_processed: float = 5.0) -> EvalBundle:
    val_seeds = {r.scenario_seed for r in val_runs}
    test_seeds = {r.scenario_seed for r in test_runs}
    if val_seeds & test_seeds:
        raise ValueError(f"val and test share scenario seed(s) {sorted(val_seeds & test_seeds)}: the threshold "
                         "would be frozen on the clip it is measured on (SOLUTION_DOC 5.5)")
    return EvalBundle(list(val_runs), list(test_runs), dict(predictions), fps_processed)


def submersion_rows(ds, res, conf: float, key) -> list[MetricRow]:
    """Recall per submersion class. `slicing.BOX_AXES` covers occlusion/posture/pixel size but not submersion,
    and §5.12's slice grid names it, so it is added here rather than by editing the frozen axis list."""
    import numpy as np

    from sightline.eval.slicing import metric_row, narrow

    rows: list[MetricRow] = []
    subs = sorted({b.submersion for b in res.gt_boxes if b.scored})
    scores = np.asarray(res.gt_match_score, dtype=float)
    for s in subs:
        mask = np.asarray([b.submersion == s and b.scored for b in res.gt_boxes], dtype=bool)
        n = int(mask.sum())
        if not n:
            continue
        recall = float(np.count_nonzero(scores[mask] >= conf) / n)
        rows.append(metric_row(f"recall@submersion={s}", recall, narrow(key, posture="all"), n,
                               at_conf=conf, axis="submersion", bin=s))
    return rows


def run_evaluation(
    bundle: EvalBundle,
    *,
    out_dir: str | Path = "",
    target_recall: float = 0.92,
    allow_non_val_split: bool = False,
) -> dict[str, Any]:
    """Freeze on val, measure on test, slice everything, write the report. Returns a JSON-safe summary."""
    from sightline.eval.harness import (
        EvalConfig,
        freeze_on_validation,
        run_detection_eval,
    )
    from sightline.eval.report import write_report

    cfg = EvalConfig(target_recall=target_recall)
    if not bundle.val_runs:
        raise ValueError("no validation runs: the operating threshold must be frozen on a val split (5.5)")
    if not bundle.test_runs:
        raise ValueError("no test runs: a threshold frozen and measured on the same clip is not a measurement")

    val = _dataset_for(bundle.val_runs, "val", bundle.fps_processed)
    test = _dataset_for(bundle.test_runs, "test", bundle.fps_processed)
    val_dets = _detections_for(bundle.val_runs, bundle.predictions)
    test_dets = _detections_for(bundle.test_runs, bundle.predictions)

    op = freeze_on_validation(val, val_dets, cfg, allow_non_val_split=allow_non_val_split)
    res = run_detection_eval(test, test_dets, op.conf, cfg, operating=op)
    res.metrics.extend(submersion_rows(test, res.matches[f"{cfg.iou_primary:g}"], op.conf, res.base_key()))

    summary: dict[str, Any] = {
        "product": "sightline.detect.evaluate",
        "domain": "sim",
        "operating_conf": float(op.conf),
        "operating_recall": float(op.recall),
        "operating_precision": float(op.precision),
        "target_recall": float(op.target_recall),
        "target_met": bool(op.achieved),
        "frozen_on": {"split": op.frozen_on_split, "clip": op.frozen_on_clip},
        "measured_on": {"split": test.split, "clip": test.clip_id, "seed_group": test.seed_group,
                        "frames": test.n_frames, "gt_boxes": len(test.scored_boxes())},
        "seeds": bundle.seeds(),
        "n_rows": len(res.metrics.rows),
        "note": "every number here is domain=sim; it is a claim about the renderer, not about real footage "
                "(SOLUTION_DOC 5.5c).",
    }
    if not op.achieved:
        summary["finding"] = (
            f"no confidence reaches recall {op.target_recall} on the validation split; the best is "
            f"{op.recall:.4f} in simulation. SOLUTION_DOC 5.5c: fly lower or capture more before touching "
            "the target."
        )

    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        summary["report"] = str(write_report([res], d / "detect_eval.md"))
        summary["slice_table"] = str(write_slice_table(res, d / "slice_table.md"))
        (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["_result"] = res  # not JSON-safe; callers that want the rows take this and drop it before dumping
    return summary


def write_slice_table(res, out_path: str | Path) -> Path:
    """The §5.12 slice table as markdown: every row starts with its domain, no exceptions."""
    rows = [r for r in res.metrics.rows if r.name.startswith("recall") or r.name.startswith("precision")
            or r.name.startswith("fp_per_min")]
    lines = [
        "# Detection slice table (F8b / F19)",
        "",
        (f"Model operating point: conf {res.operating.conf:.4f}, frozen on "
         f"`{res.operating.frozen_on_split}` split of `{res.operating.frozen_on_clip}`; measured on "
         f"`{res.dataset.split}` split of `{res.dataset.clip_id}` (seed group `{res.dataset.seed_group}`)."),
        "",
        ("Every number below is **in simulation** (`domain=sim`), on the renderer this system deploys into "
         "(SOLUTION_DOC 5.5c). Do not average any of them with a real-footage figure."),
        "",
        "| domain | metric | slice | value | n |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda x: (x.name, x.slice.label())):
        lines.append(f"| {r.slice.domain} | {r.name} | {r.slice.label()} | {r.value:.4f} | {r.n} |")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
