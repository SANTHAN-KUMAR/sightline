"""F8 RGB detection pass: tile -> batched inference -> inverse map -> cross-tile merge (SOLUTION_DOC §5.5, §5.11).

The module is split so that **everything except the forward pass is testable without a GPU**:

* :func:`detect_frame` takes an ``infer`` callable (``list[np.ndarray] -> list[list[Detection]]``) and does the
  tiling, the batching, the inverse mapping with tile provenance and the merge. `tests/test_detect.py` runs it
  with a fake inferencer that returns boxes worked out by hand, so the geometry is checked exactly.
* :class:`RgbDetector` is the one implementation of ``infer`` that touches torch. It imports Ultralytics lazily,
  inside the constructor, so ``import sightline.detect.rgb`` costs nothing and works on a machine where the
  Unreal editor owns the GPU.

Batching. §5.11's measured configuration is 6 native tiles in **one** batched call (65.0 ms median for yolo26s
FP16 on this RTX 4060, `docs/verification/gpu_latency.md`). :func:`tile_batches` splits the tile list to whatever
batch a TensorRT engine was built for; the default ``max_batch`` is the whole grid, i.e. one call.

Confidence. The detector runs at ``conf=RAW_CONF`` (0.05, §5.5's ``skip_box_thr``) and the **operating threshold
is applied afterwards**, by `threshold.py`. Running the model at the operating threshold instead would make the
threshold sweep impossible, and §5.5c step 4 explicitly wants a low raw threshold with precision recovered
downstream.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sightline.detect.tiler import (
    DEFAULT_OVERLAP,
    DEFAULT_TILE_PX,
    TileGrid,
    merge_tile_detections,
    tile_batches,
    to_frame,
)
from sightline.schemas import ClassName, Detection, FrameBundle

__all__ = [
    "CLASS_NAMES",
    "DEFAULT_MAX_DET",
    "RAW_CONF",
    "DetectorConfig",
    "RgbDetector",
    "TileInferencer",
    "boxes_to_detections",
    "detect_bundle",
    "detect_frame",
]

#: §5.5 inference sketch: `skip_box_thr=0.05`. Everything above this is kept and filtered later.
RAW_CONF = 0.05
#: A 1024 tile of flood scene never legitimately holds 300 people; the cap only stops a broken model flooding RAM.
DEFAULT_MAX_DET = 300
#: Index -> class, matching `dataset.CLASSES`.
CLASS_NAMES: tuple[str, ...] = ("human", "animal")

#: ``infer(crops) -> per-crop detections in TILE-LOCAL pixel coordinates``.
TileInferencer = Callable[[Sequence[np.ndarray]], list[list[Detection]]]


@dataclass(slots=True)
class DetectorConfig:
    """Everything that changes a detection result. Written into the run manifest so a number is reproducible."""

    weights: str = ""
    tile_px: tuple[int, int] = DEFAULT_TILE_PX
    overlap: float = DEFAULT_OVERLAP
    imgsz: int = 1024
    raw_conf: float = RAW_CONF
    iou_nms: float = 0.7
    max_det: int = DEFAULT_MAX_DET
    max_batch: int = 0  # 0 = one batched call for the whole grid (the §5.11 measured configuration)
    half: bool = True
    device: str = "cuda:0"
    merge_iou: float = 0.55
    merge_containment: float = 0.8
    classes: tuple[str, ...] = CLASS_NAMES

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights, "tile_px": list(self.tile_px), "overlap": self.overlap,
            "imgsz": self.imgsz, "raw_conf": self.raw_conf, "iou_nms": self.iou_nms, "max_det": self.max_det,
            "max_batch": self.max_batch, "half": self.half, "device": self.device,
            "merge_iou": self.merge_iou, "merge_containment": self.merge_containment,
            "classes": list(self.classes),
        }


def boxes_to_detections(
    xyxy: np.ndarray,
    conf: np.ndarray,
    cls: np.ndarray,
    *,
    classes: Sequence[str] = CLASS_NAMES,
    modality: str = "rgb",
) -> list[Detection]:
    """Raw model output (N,4)+(N,)+(N,) -> `Detection`s in the coordinates the arrays are already in."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    cls = np.asarray(cls).reshape(-1).astype(int)
    if not (len(xyxy) == len(conf) == len(cls)):
        raise ValueError(f"ragged model output: {len(xyxy)} boxes, {len(conf)} scores, {len(cls)} classes")
    out: list[Detection] = []
    for b, s, c in zip(xyxy, conf, cls):
        name: ClassName = classes[c] if 0 <= c < len(classes) else "human"  # type: ignore[assignment]
        out.append(Detection(bbox_px=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                             score=float(s), cls=name, modality=modality))  # type: ignore[arg-type]
    return out


def detect_frame(
    frame: np.ndarray,
    infer: TileInferencer,
    *,
    cfg: DetectorConfig | None = None,
    grid: TileGrid | None = None,
    frame_idx: int = -1,
    tile_indices: Sequence[int] | None = None,
) -> list[Detection]:
    """One frame -> full-frame `Detection`s, score-ordered, with `tile_idx` provenance kept.

    ``tile_indices`` is the §5.11 dynamic-tiling hook: run only some tiles (e.g. the ones a previous frame's
    tracks predict into) and the inverse mapping is identical, because :func:`to_frame` maps through the grid.
    """
    cfg = cfg or DetectorConfig()
    grid = grid or TileGrid.for_frame(frame, cfg.tile_px, cfg.overlap)
    idxs = list(range(len(grid))) if tile_indices is None else list(tile_indices)
    batch = cfg.max_batch if cfg.max_batch > 0 else len(idxs)

    per_tile: list[list[Detection]] = []
    for group in tile_batches(len(idxs), max(1, batch)):
        chosen = [idxs[k] for k in group]
        crops = grid.crops(frame, chosen)
        results = infer(crops)
        if len(results) != len(crops):
            raise ValueError(f"inferencer returned {len(results)} results for {len(crops)} crops")
        per_tile.extend(results)

    mapped = to_frame(per_tile, grid, idxs, frame_idx=frame_idx)
    merged = merge_tile_detections(mapped, grid, iou_thr=cfg.merge_iou, containment_thr=cfg.merge_containment)
    merged.sort(key=lambda d: d.score, reverse=True)
    return merged


def detect_bundle(bundle: FrameBundle, infer: TileInferencer, *, cfg: DetectorConfig | None = None,
                  grid: TileGrid | None = None) -> list[Detection]:
    """`FrameBundle` in, `list[Detection]` out — the contract `docs/CONTRACTS.md` gives this lane."""
    if bundle.rgb is None:
        return []
    return detect_frame(bundle.rgb, infer, cfg=cfg, grid=grid, frame_idx=bundle.frame_idx)


class RgbDetector:
    """Ultralytics YOLO26 behind the :data:`TileInferencer` protocol. **Imports torch; needs a free GPU.**

    Constructing it loads weights, so nothing imports it at module scope. `detect(frame)` is the whole pass.
    """

    def __init__(self, cfg: DetectorConfig):
        if not cfg.weights:
            raise ValueError("DetectorConfig.weights must point at a .pt / .engine / .onnx file")
        w = Path(cfg.weights)
        if not w.exists():
            raise FileNotFoundError(f"detector weights not found: {w}")
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(w))
        self._is_engine = w.suffix == ".engine"

    # -- the inferencer -------------------------------------------------------------------------------------
    def __call__(self, crops: Sequence[np.ndarray]) -> list[list[Detection]]:
        kw: dict[str, Any] = {"conf": self.cfg.raw_conf, "iou": self.cfg.iou_nms, "max_det": self.cfg.max_det,
                              "imgsz": self.cfg.imgsz, "verbose": False, "device": self.cfg.device}
        if not self._is_engine:  # a TensorRT engine has its precision baked in; passing half= then errors
            kw["half"] = self.cfg.half
        results = self.model.predict(list(crops), **kw)
        out: list[list[Detection]] = []
        for r in results:
            b = r.boxes
            if b is None or len(b) == 0:
                out.append([])
                continue
            out.append(boxes_to_detections(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                           b.cls.cpu().numpy(), classes=self.cfg.classes))
        return out

    def detect(self, frame: np.ndarray, *, frame_idx: int = -1,
               grid: TileGrid | None = None) -> list[Detection]:
        return detect_frame(frame, self, cfg=self.cfg, grid=grid, frame_idx=frame_idx)
