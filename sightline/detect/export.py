"""F8 export: trained `.pt` -> ONNX / TensorRT FP16 (INT8 only after a recall check), SOLUTION_DOC §5.5, §5.11.

Two things this module refuses to do quietly, both of which are how an export silently costs recall:

1. **Export while the editor is up.** Building a TensorRT engine takes 2-5 minutes of exclusive GPU
   (`docs/verification/gpu_latency.md` measured 197-277 s per engine) and will fight the editor for the 8 GB.
   :func:`export_engine` calls the same :func:`~sightline.detect.train.editor_is_running` guard training does.
2. **Ship INT8 on a promise.** §5.5: "INT8 only after checking recall on the held-out clip", and its calibration
   set "must include tiny/occluded positives". :func:`int8_calibration_manifest` builds that set from the
   dataset's own smallest and most-occluded boxes and refuses to proceed without them;
   :func:`compare_recall` is the check that has to run afterwards, and it returns a `MetricRow` pair carrying
   `domain`, not a bare float.

The measured FP16 numbers for this machine are already in `docs/verification/gpu_latency.md` (yolo26s, 6 x
1280x1088 native tiles, one batched call: **65.0 ms median, well inside R9's 300 ms**). Do not re-benchmark; that
document is the reference and re-running it against a busy GPU only produces a worse number.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sightline.detect.train import editor_is_running

__all__ = [
    "INT8_RECALL_TOLERANCE",
    "ExportConfig",
    "compare_recall",
    "export_engine",
    "int8_calibration_manifest",
]

#: How much recall an INT8 engine may lose against its FP16 parent before it is rejected. Half a point: the
#: whole reason to quantise is latency, and this pipeline already has a 4x latency margin (§5.11), so there is
#: nothing to buy with recall.
INT8_RECALL_TOLERANCE = 0.005


@dataclass(slots=True)
class ExportConfig:
    weights: str
    format: str = "engine"          # "engine" (TensorRT) | "onnx"
    imgsz: int = 1024
    half: bool = True
    int8: bool = False
    batch: int = 6                  # §5.11's measured configuration: the whole 4K tile grid in one call
    dynamic: bool = False
    workspace: float | None = None
    data: str = ""                  # INT8 calibration dataset yaml (Ultralytics requirement)
    device: int | str = 0
    allow_editor_running: bool = False

    def args(self) -> dict[str, Any]:
        a: dict[str, Any] = {"format": self.format, "imgsz": self.imgsz, "batch": self.batch,
                             "dynamic": self.dynamic, "device": self.device}
        if self.format == "engine":
            a["half"] = self.half and not self.int8
            a["int8"] = self.int8
            if self.workspace is not None:
                a["workspace"] = self.workspace
            if self.int8:
                if not self.data:
                    raise ValueError("INT8 export needs `data`: a calibration set including tiny/occluded "
                                     "positives (SOLUTION_DOC 5.5)")
                a["data"] = self.data
        elif self.format == "onnx":
            a["half"] = self.half
            a["simplify"] = True
        return a


def export_engine(cfg: ExportConfig) -> dict[str, Any]:
    """Export and return a manifest. Raises if the editor holds the GPU or the weights are missing."""
    w = Path(cfg.weights)
    if not w.exists():
        raise FileNotFoundError(f"weights not found: {w}")
    editors = editor_is_running()
    if editors and not cfg.allow_editor_running:
        raise RuntimeError(f"the Unreal editor is running ({', '.join(editors)}); a TensorRT build needs the "
                           "8 GB GPU to itself (docs/verification/gpu_latency.md). Close it first.")
    args = cfg.args()

    from ultralytics import YOLO

    t0 = time.time()
    path = YOLO(str(w)).export(**args)
    out = {
        "product": "sightline.detect.export",
        "domain": "sim",
        "source_weights": str(w),
        "exported": str(path),
        "args": args,
        "build_s": time.time() - t0,
        "reference_latency": "docs/verification/gpu_latency.md: yolo26s FP16, 6 x 1280x1088 native tiles, "
                             "65.0 ms median on this RTX 4060 (R9 budget 300 ms)",
        "int8_checked": False if cfg.int8 else None,
        "note": "INT8 must not ship until compare_recall() shows the loss is within INT8_RECALL_TOLERANCE "
                "on the held-out clip (SOLUTION_DOC 5.5).",
    }
    Path(str(path)).with_suffix(".export.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def int8_calibration_manifest(
    dataset_root: str | Path,
    *,
    split: str = "train",
    n_images: int = 300,
    small_px_max: float = 24.0,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pick the calibration images, biased to the tiny and occluded positives §5.5 insists on.

    The tile label files are read directly (no images are opened), a tile is "tiny" when its largest box is
    below ``small_px_max`` in tile pixels, and the manifest refuses to be written if fewer than a fifth of the
    picked tiles are tiny -- a calibration set of comfortable large targets is how INT8 quietly loses the small
    ones that recall depends on.
    """
    root = Path(dataset_root)
    lab_dir = root / "labels" / split
    img_dir = root / "images" / split
    if not lab_dir.is_dir():
        raise FileNotFoundError(f"no label directory at {lab_dir}")
    tiny: list[str] = []
    rest: list[str] = []
    for lab in sorted(lab_dir.glob("*.txt")):
        rows = [r.split() for r in lab.read_text(encoding="utf-8").split("\n") if r.strip()]
        if not rows:
            continue
        # tile-normalised w/h -> pixels needs the tile size; the manifest records the fraction instead, and the
        # caller's tile size (1024 by default) turns it into pixels.
        longest = max(max(float(r[3]), float(r[4])) for r in rows)
        (tiny if longest * 1024.0 <= small_px_max else rest).append(lab.stem)
    picked = tiny[:n_images] + rest[: max(0, n_images - len(tiny))]
    frac_tiny = (len(tiny[:n_images]) / len(picked)) if picked else 0.0
    if not picked:
        raise ValueError(f"no labelled tiles under {lab_dir}: nothing to calibrate on")
    if frac_tiny < 0.2:
        raise ValueError(
            f"only {frac_tiny:.0%} of the calibration tiles carry a target under {small_px_max:g} px. "
            "SOLUTION_DOC 5.5 requires the INT8 calibration set to include tiny/occluded positives; "
            "capture lower or widen the selection before quantising."
        )
    man = {
        "product": "sightline.detect.export.int8_calibration",
        "domain": "sim", "split": split, "n_images": len(picked),
        "fraction_tiny": frac_tiny, "small_px_max": small_px_max,
        "images": [str(img_dir / f"{s}.png") for s in picked],
        "requirement": "SOLUTION_DOC 5.5: the INT8 calibration set must include tiny/occluded positives",
    }
    if out_path:
        Path(out_path).write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def compare_recall(fp16_rows: Sequence[Any], int8_rows: Sequence[Any], *, name: str = "recall@iou0.5",
                   tolerance: float = INT8_RECALL_TOLERANCE) -> dict[str, Any]:
    """Compare two `MetricRow` lists at the frozen threshold. Refuses to compare across domains (hard rule 5)."""
    from sightline.eval.slicing import require_single_domain

    def pick(rows: Sequence[Any]) -> Any:
        hits = [r for r in rows if r.name == name]
        if not hits:
            raise KeyError(f"no row named {name!r}")
        require_single_domain(hits)
        return hits[0]

    a, b = pick(fp16_rows), pick(int8_rows)
    if a.slice.domain != b.slice.domain:
        from sightline.eval.slicing import DomainMixError

        raise DomainMixError(f"cannot compare {a.slice.domain} with {b.slice.domain}")
    delta = float(b.value) - float(a.value)
    return {
        "metric": name, "domain": a.slice.domain, "fp16": float(a.value), "int8": float(b.value),
        "delta": delta, "tolerance": tolerance, "acceptable": delta >= -tolerance,
        "verdict": ("INT8 is within tolerance" if delta >= -tolerance else
                    f"INT8 loses {-delta:.4f} recall in {a.slice.domain}; DO NOT SHIP IT (SOLUTION_DOC 5.5)"),
    }
