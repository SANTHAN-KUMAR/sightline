"""Draw PREDICTED boxes on real frames so a human can look at them (`docs/QUALITY_GATE.md` item 1).

`tools/capture/contact_sheet.py` draws ground truth; this is its counterpart for predictions, and it exists
because a model reporting mAP 0.9 while boxing shadows passes every programmatic check ever written. The gate:

    "A model reporting mAP 0.9 while boxing shadows is exactly what this gate exists to catch."

Two things make the picture actually readable at this scale, and both are the point:

* **Zoom insets.** A survivor is 20-95 px in a 3840x2160 frame. Downscaled to a contact-sheet tile it is two
  pixels and every box looks correct. :func:`zoom_panel` cuts a magnified window around each box so the pixels
  that were classified are the pixels on screen.
* **Truth and prediction in one panel, colour-coded**, with the IoU printed. Green = a matched ground-truth box,
  red = an unmatched prediction (a false positive at this threshold), yellow = a missed ground truth. If the
  reds are all on roof edges and glint, the reader knows immediately.

Matching uses the lane's own `threshold.match_frame`, so what the picture shows and what the metric counts are
the same assignment, not two implementations that can disagree.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sightline.detect.threshold import PRIMARY_IOU, GroundTruthBox, match_frame
from sightline.detect.tiler import iou_xyxy
from sightline.schemas import Detection

__all__ = [
    "OverlayStats",
    "contact_sheet",
    "detections_from_json",
    "detections_to_json",
    "draw_boxes",
    "overlay_frame",
    "zoom_panel",
]

GREEN = (0, 220, 0)      # a ground-truth box a prediction matched
RED = (0, 0, 235)        # a prediction that matched nothing: a false positive at this threshold
YELLOW = (0, 215, 235)   # a ground-truth box nothing matched: a miss
GREY = (150, 150, 150)   # a prediction below the operating threshold, drawn thin for context
CYAN = (220, 220, 0)


@dataclass(slots=True)
class OverlayStats:
    """What the picture is showing, so the caption cannot drift from the pixels."""

    frame: str
    conf: float
    n_gt: int = 0
    n_pred_total: int = 0
    n_pred_kept: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ious: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ious is None:
            self.ious = []

    def caption(self) -> str:
        return (f"{self.frame}  conf>={self.conf:.2f}  gt={self.n_gt} pred={self.n_pred_kept}"
                f"/{self.n_pred_total}  tp={self.tp} fp={self.fp} fn={self.fn}")


def _cv2():
    import cv2

    return cv2


def draw_boxes(img: np.ndarray, boxes: Sequence[Sequence[float]], colour, labels: Sequence[str] | None = None,
               *, thickness: int = 2, pad: float = 0.0, font_scale: float = 0.5) -> np.ndarray:
    """Draw xyxy boxes in place-safe fashion (on a copy). ``pad`` grows the drawn rect so a 20 px box is visible."""
    cv2 = _cv2()
    out = img
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        p = pad * max(x2 - x1, y2 - y1)
        cv2.rectangle(out, (round(x1 - p), round(y1 - p)), (round(x2 + p), round(y2 + p)),
                      colour, thickness)
        if labels is not None and i < len(labels) and labels[i]:
            cv2.putText(out, labels[i], (round(x1 - p), max(12, round(y1 - p) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, max(1, thickness - 1), cv2.LINE_AA)
    return out


def zoom_panel(img: np.ndarray, boxes: Sequence[tuple[Sequence[float], Any, str]], *, out_px: int = 220,
               context_px: int = 120) -> np.ndarray | None:
    """A strip of magnified windows, one per box, so the actual classified pixels are visible.

    ``boxes`` is ``[(xyxy, colour, label), ...]``. Returns None when there is nothing to show.
    """
    if not boxes:
        return None
    cv2 = _cv2()
    h, w = img.shape[:2]
    panels: list[np.ndarray] = []
    for b, colour, label in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(context_px / 2.0, max(x2 - x1, y2 - y1) * 1.6)
        rx1, ry1 = int(max(0, cx - half)), int(max(0, cy - half))
        rx2, ry2 = int(min(w, cx + half)), int(min(h, cy + half))
        if rx2 - rx1 < 4 or ry2 - ry1 < 4:
            continue
        crop = img[ry1:ry2, rx1:rx2].copy()
        s = out_px / max(crop.shape[0], crop.shape[1])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                          interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (int((x1 - rx1) * s), int((y1 - ry1) * s)),
                      (int((x2 - rx1) * s), int((y2 - ry1) * s)), colour, 2)
        pan = np.full((out_px + 22, out_px, 3), 24, np.uint8)
        pan[22:22 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(pan, label[:34], (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
        panels.append(pan)
    if not panels:
        return None
    strip = np.full((panels[0].shape[0], out_px * len(panels), 3), 24, np.uint8)
    for i, p in enumerate(panels):
        strip[:, i * out_px:i * out_px + p.shape[1]] = p
    return strip


def overlay_frame(
    img: np.ndarray,
    preds: Sequence[Detection],
    gts: Sequence[GroundTruthBox] = (),
    *,
    conf: float = 0.0,
    frame_name: str = "",
    iou_thr: float = PRIMARY_IOU,
    show_below_threshold: bool = True,
    zoom: bool = True,
) -> tuple[np.ndarray, OverlayStats]:
    """One frame with truth and predictions drawn, plus a magnified strip. Returns (image, stats)."""
    cv2 = _cv2()
    out = img.copy()
    kept = [p for p in preds if p.score >= conf]
    st = OverlayStats(frame=frame_name, conf=conf, n_gt=len(gts), n_pred_total=len(preds), n_pred_kept=len(kept))

    m = match_frame(kept, gts, iou_thr, cls=None) if gts else None
    gt_matched = (m.gt_scores >= 0.0) if m is not None else np.zeros(len(gts), dtype=bool)
    pred_tp = m.pred_is_tp if m is not None else np.zeros(len(kept), dtype=bool)
    # match_frame visits predictions in descending score; recover that order to line the flags back up.
    order = sorted(range(len(kept)), key=lambda i: kept[i].score, reverse=True)
    is_tp = np.zeros(len(kept), dtype=bool)
    for slot, i in enumerate(order):
        if slot < len(pred_tp):
            is_tp[i] = bool(pred_tp[slot])

    zoom_items: list[tuple[Sequence[float], Any, str]] = []
    if show_below_threshold:
        below = [p for p in preds if p.score < conf]
        draw_boxes(out, [p.bbox_px for p in below], GREY, None, thickness=1, pad=0.35)

    for i, p in enumerate(kept):
        colour = GREEN if is_tp[i] else RED
        best_iou = max((iou_xyxy(p.bbox_px, g.bbox_px) for g in gts), default=0.0)
        label = f"{p.score:.2f} iou{best_iou:.2f} {p.size_px:.0f}px"
        draw_boxes(out, [p.bbox_px], colour, [label], thickness=2, pad=0.35)
        zoom_items.append((p.bbox_px, colour, ("TP " if is_tp[i] else "FP ") + label))
        st.ious.append(best_iou)
    st.tp = int(is_tp.sum())
    st.fp = len(kept) - st.tp

    for j, g in enumerate(gts):
        if not gt_matched[j]:
            lab = f"MISS {g.posture}/{g.submersion} {max(g.bbox_px[2] - g.bbox_px[0], g.bbox_px[3] - g.bbox_px[1]):.0f}px"
            draw_boxes(out, [g.bbox_px], YELLOW, [lab], thickness=2, pad=0.5)
            zoom_items.append((g.bbox_px, YELLOW, lab))
    st.fn = int((~gt_matched).sum()) if len(gts) else 0

    if not zoom:
        return out, st
    strip = zoom_panel(out, zoom_items)
    if strip is None:
        return out, st
    scale = strip.shape[1] / out.shape[1]
    small = cv2.resize(out, (strip.shape[1], max(1, int(out.shape[0] * scale))))
    cv2.putText(small, st.caption(), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)
    return np.vstack([small, strip]), st


def contact_sheet(panels: Sequence[np.ndarray], out_path: str | Path, *, cols: int = 1) -> Path:
    """Stack per-frame panels into one PNG. Column layout is 1 by default: the insets are already wide."""
    cv2 = _cv2()
    if not panels:
        raise ValueError("nothing to draw")
    w = max(p.shape[1] for p in panels)
    rows = []
    for p in panels:
        if p.shape[1] < w:
            pad = np.full((p.shape[0], w - p.shape[1], 3), 24, np.uint8)
            p = np.hstack([p, pad])
        rows.append(p)
        rows.append(np.full((6, w, 3), 60, np.uint8))
    sheet = np.vstack(rows)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), sheet)
    return p


# --- prediction serialisation (so a GPU run and a CPU inspection can be different processes) -------------------
def detections_to_json(by_frame: dict[str, list[Detection]], meta: dict[str, Any] | None = None) -> dict:
    return {
        "product": "sightline.detect.predictions",
        "domain": "sim",
        "meta": meta or {},
        "frames": {
            k: [{"bbox_px": list(d.bbox_px), "score": d.score, "cls": d.cls, "modality": d.modality,
                 "tile_idx": d.tile_idx, "frame_idx": d.frame_idx} for d in v]
            for k, v in by_frame.items()
        },
    }


def detections_from_json(path: str | Path) -> tuple[dict[str, list[Detection]], dict[str, Any]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {
        k: [Detection(bbox_px=tuple(x["bbox_px"]), score=float(x["score"]), cls=x.get("cls", "human"),
                      modality=x.get("modality", "rgb"), tile_idx=int(x.get("tile_idx", -1)),
                      frame_idx=int(x.get("frame_idx", -1)))
            for x in v]
        for k, v in d.get("frames", {}).items()
    }
    return out, d.get("meta", {})
