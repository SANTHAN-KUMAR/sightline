"""Cloud-fallback detection route contract (SOLUTION_DOC §5.10).

    "A FastAPI service with the *same* Ultralytics TensorRT/ONNX engine exposes ``POST /detect`` for JPEG
     tiles. If the edge misses its per-frame deadline it ships the downscaled frame (~200 KB JPEG) and merges
     results by frame index; when the link is down it keeps running the cheaper local configuration."

**STUB (labelled per project rule 1).** :class:`StubDetector` loads no model and returns no boxes. The ML lane
(C, ``sightline/detect/``) owns the real detector; this module owns only the *route*: the request/response
shape, the pluggable handler, and the merge-by-frame-index logic. Swap it in with::

    app = create_app(store, detector=MyDetector())     # anything satisfying DetectorPlugin

so no code in this lane ever imports torch / ultralytics / tensorrt.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DetectorPlugin",
    "StubDetector",
    "FrameMerger",
    "iou_xyxy",
    "merge_by_frame_index",
    "DETECT_CONTRACT",
]

DETECT_CONTRACT = "sightline.detect.fallback/1.0"


@runtime_checkable
class DetectorPlugin(Protocol):
    """What ``POST /detect`` calls. Returns detection dicts shaped like `schemas.Detection`."""

    name: str
    is_stub: bool

    def detect(self, image: bytes, meta: dict[str, Any]) -> list[dict[str, Any]]: ...


class StubDetector:
    """STUB — no model is loaded. Returns an empty box list and says so in every response.

    It exists so the route, the merge and the map can be exercised end to end offline; a real deployment
    passes the ML lane's detector into ``create_app(detector=...)``.
    """

    name = "stub"
    is_stub = True

    def detect(self, image: bytes, meta: dict[str, Any]) -> list[dict[str, Any]]:
        return []


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_by_frame_index(
    local: list[dict[str, Any]],
    cloud: list[dict[str, Any]],
    *,
    iou_thresh: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge one frame's cloud result into its local result. Pure, order-stable and **idempotent**.

    * a cloud box that overlaps a local box of the same class by more than ``iou_thresh`` updates that box
      only if it scores higher (``source`` becomes ``"merged"``);
    * anything else is appended with ``source = "cloud"``;
    * merging the same cloud list twice returns the same list, because the second pass matches the boxes it
      already inserted. Late cloud results therefore cannot double-count a detection.
    """
    out = [dict(d) for d in local]
    for d in out:
        d.setdefault("source", "local")
    for c in cloud:
        cb = tuple(c["bbox_px"])
        best_i, best_iou = -1, 0.0
        for i, o in enumerate(out):
            if o.get("cls", "human") != c.get("cls", "human"):
                continue
            v = iou_xyxy(tuple(o["bbox_px"]), cb)
            if v > best_iou:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= iou_thresh:
            if float(c.get("score", 0.0)) > float(out[best_i].get("score", 0.0)):
                keep_src = out[best_i].get("source", "local")
                out[best_i] = dict(c)
                out[best_i]["source"] = "cloud" if keep_src == "cloud" else "merged"
            elif out[best_i].get("source") == "local":
                out[best_i]["source"] = "merged"
        else:
            n = dict(c)
            n["source"] = "cloud"
            out.append(n)
    return out


class FrameMerger:
    """Bounded per-clip buffer of local results, so a late cloud reply can be merged by frame index.

    The edge writes its local result the moment it has one; the cloud reply for the same ``frame_idx`` may
    arrive several frames later, or never (link down). ``maxlen`` frames are kept; older ones are evicted,
    and a cloud reply for an evicted frame is reported as ``late``.
    """

    def __init__(self, maxlen: int = 300):
        self.maxlen = maxlen
        self._frames: OrderedDict[tuple[str, int], list[dict[str, Any]]] = OrderedDict()
        self.merged = 0
        self.late = 0

    def _key(self, clip_id: str, frame_idx: int) -> tuple[str, int]:
        return (clip_id, int(frame_idx))

    def add_local(self, clip_id: str, frame_idx: int, dets: list[dict[str, Any]]) -> None:
        k = self._key(clip_id, frame_idx)
        self._frames[k] = [dict(d, source=d.get("source", "local")) for d in dets]
        self._frames.move_to_end(k)
        while len(self._frames) > self.maxlen:
            self._frames.popitem(last=False)

    def merge_cloud(
        self, clip_id: str, frame_idx: int, dets: list[dict[str, Any]], *, iou_thresh: float = 0.5
    ) -> dict[str, Any]:
        k = self._key(clip_id, frame_idx)
        if k not in self._frames:
            self.late += 1
            self._frames[k] = [dict(d, source="cloud") for d in dets]
            self._frames.move_to_end(k)
            while len(self._frames) > self.maxlen:
                self._frames.popitem(last=False)
            return {"frame_idx": frame_idx, "clip_id": clip_id, "late": True,
                    "detections": list(self._frames[k])}
        before = len(self._frames[k])
        self._frames[k] = merge_by_frame_index(self._frames[k], dets, iou_thresh=iou_thresh)
        self.merged += 1
        return {
            "frame_idx": frame_idx,
            "clip_id": clip_id,
            "late": False,
            "added": len(self._frames[k]) - before,
            "detections": list(self._frames[k]),
        }

    def get(self, clip_id: str, frame_idx: int) -> list[dict[str, Any]]:
        return list(self._frames.get(self._key(clip_id, frame_idx), []))

    def stats(self) -> dict[str, Any]:
        return {"frames_buffered": len(self._frames), "merged": self.merged, "late": self.late,
                "maxlen": self.maxlen, "t_utc": time.time()}
