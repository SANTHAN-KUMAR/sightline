"""F8 tiling: batched native-resolution tiles, inverse mapping and the cross-tile merge (SOLUTION_DOC §5.5, §5.11).

**Native resolution is the point.** §2.5's recall floor is "≥ 20 px on target", and §5.5c makes flight geometry the
single largest lever. A 3840×2160 frame letterboxed to 1024 shrinks a 28 px survivor to 7 px; cutting the same frame
into 1024×1024 tiles and running the detector at ``imgsz=1024`` leaves **one tile pixel = one frame pixel**. That
identity is what this module guarantees, and it is why the training tiler (``dataset.py``) uses this same grid: the
model sees the deployment scale during training.

Grid maths. For an axis of length ``E`` and tile size ``T`` with requested overlap fraction ``ov``:

    stride  = round(T * (1 - ov))
    n       = 1 if E <= T else ceil((E - T) / stride) + 1
    starts  = round(i * (E - T) / (n - 1))  for i in 0..n-1

so the last tile is flush with the far edge and the *effective* overlap is always **at least** the requested one.
The useful invariant that falls out is :attr:`TileGrid.max_safe_target_px` = ``T - effective_stride``: any target
whose longest side is below that is **fully contained in at least one tile**, so it is never seen only as a
fragment. At the default 3840×2160 / 1024² / 0.2 that is 320 px, roughly ten times the size of a survivor at 45 m.

Merging is therefore two separate problems and this module treats them separately:

1. **Duplicates** — the same target seen whole in two overlapping tiles. Greedy, class-wise, score-ordered
   suppression by IoU, plus an intersection-over-smaller ("containment") rule that absorbs an edge *fragment* into
   the whole box that another tile saw. Containment is only applied across tiles: two boxes from the same single
   tile are the detector's own output and are left alone.
2. **Targets wider than the overlap band** — a fallen tree, a roof, a vehicle. No tile contains them, so every tile
   sees a clipped fragment. :func:`merge_tile_detections` first runs a **seam union** to a fixed point: two
   fragments from different tiles are unioned when each is clipped against the *interior* tile edge that faces the
   other and their extents along the perpendicular axis overlap. Fragments of a target crossing three tiles merge
   as a chain.

Everything here is stdlib + numpy, deterministic, and has no dependency on torch, ultralytics or a GPU: it is the
part of the detector that can be tested exactly (``tests/test_detect.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Sequence

import numpy as np

from sightline.schemas import Detection

__all__ = [
    "Tile",
    "TileGrid",
    "DEFAULT_TILE_PX",
    "DEFAULT_OVERLAP",
    "box_area",
    "intersection_area",
    "iou_xyxy",
    "containment_xyxy",
    "map_box_to_frame",
    "to_frame",
    "merge_tile_detections",
    "tile_batches",
]

#: Native tile size and overlap. 1024² matches the §5.5c training resolution exactly, so inference introduces no
#: rescale; 0.2 is the overlap §5.5 prescribes (SAHI's slicing-aware fine-tuning number).
DEFAULT_TILE_PX = (1024, 1024)
DEFAULT_OVERLAP = 0.2


# --- box maths (shared with fusion.py) ---------------------------------------------------------------------
def box_area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    h = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    return w * h if w > 0.0 and h > 0.0 else 0.0


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the *smaller* area (IoS). 1.0 means one box lies entirely inside the other.

    This is the measure that catches a tile-edge fragment against the whole box another tile saw: their IoU can be
    tiny (a 24 px sliver against a 300 px body) while the containment is exactly 1.
    """
    inter = intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0.0 else 0.0


# --- the grid ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Tile:
    """One native-resolution crop window. ``x1``/``y1`` are exclusive, so ``x1 - x0 == width_px``."""

    idx: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    frame_w: int
    frame_h: int

    @property
    def width_px(self) -> int:
        return self.x1 - self.x0

    @property
    def height_px(self) -> int:
        return self.y1 - self.y0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """View (not a copy) of the tile's pixels. Copy it before handing it to a library that may mutate."""
        return frame[self.y0:self.y1, self.x0:self.x1]

    def contains_box(self, bbox_px: Sequence[float]) -> bool:
        return (float(bbox_px[0]) >= self.x0 and float(bbox_px[1]) >= self.y0
                and float(bbox_px[2]) <= self.x1 and float(bbox_px[3]) <= self.y1)


@dataclass(frozen=True, slots=True)
class TileGrid:
    """A frame's tiling. Immutable and hashable, so it can be cached per (frame size, tile size, overlap)."""

    frame_w: int
    frame_h: int
    tile_w: int
    tile_h: int
    overlap: float
    n_cols: int
    n_rows: int
    tiles: tuple[Tile, ...]

    # -- construction ---------------------------------------------------------------------------------------
    @staticmethod
    def _starts(extent: int, tile: int, overlap: float) -> tuple[list[int], int]:
        """Evenly spaced starts, last tile flush with the far edge. Returns (starts, effective tile size)."""
        if extent <= 0 or tile <= 0:
            raise ValueError(f"extent and tile must be positive, got extent={extent}, tile={tile}")
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        tile = min(tile, extent)
        if extent == tile:
            return [0], tile
        stride = max(1, int(round(tile * (1.0 - overlap))))
        n = int(math.ceil((extent - tile) / stride)) + 1
        span = extent - tile
        return [int(round(i * span / (n - 1))) for i in range(n)], tile

    @classmethod
    def build(
        cls,
        frame_w: int,
        frame_h: int,
        tile_px: tuple[int, int] = DEFAULT_TILE_PX,
        overlap: float = DEFAULT_OVERLAP,
    ) -> "TileGrid":
        tw, th = int(tile_px[0]), int(tile_px[1])
        xs, tw = cls._starts(int(frame_w), tw, overlap)
        ys, th = cls._starts(int(frame_h), th, overlap)
        tiles: list[Tile] = []
        for r, y0 in enumerate(ys):
            for c, x0 in enumerate(xs):
                tiles.append(Tile(len(tiles), r, c, x0, y0, x0 + tw, y0 + th, int(frame_w), int(frame_h)))
        return cls(int(frame_w), int(frame_h), tw, th, float(overlap), len(xs), len(ys), tuple(tiles))

    @classmethod
    def for_frame(cls, frame: np.ndarray, tile_px: tuple[int, int] = DEFAULT_TILE_PX,
                  overlap: float = DEFAULT_OVERLAP) -> "TileGrid":
        h, w = frame.shape[:2]
        return cls.build(int(w), int(h), tile_px, overlap)

    # -- access ---------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tiles)

    def __iter__(self) -> Iterator[Tile]:
        return iter(self.tiles)

    def __getitem__(self, idx: int) -> Tile:
        return self.tiles[idx]

    def crops(self, frame: np.ndarray, indices: Sequence[int] | None = None) -> list[np.ndarray]:
        """The tile crops, in tile order. ``indices`` supports the §5.11 dynamic-tiling optimisation.

        Crops are contiguous copies: TensorRT/Ultralytics preprocessing wants contiguous memory, and a numpy view
        of a strided slice would be copied anyway.
        """
        h, w = frame.shape[:2]
        if (w, h) != (self.frame_w, self.frame_h):
            raise ValueError(f"frame is {w}x{h}, grid was built for {self.frame_w}x{self.frame_h}")
        chosen = self.tiles if indices is None else [self.tiles[i] for i in indices]
        return [np.ascontiguousarray(t.crop(frame)) for t in chosen]

    # -- properties the design depends on -------------------------------------------------------------------
    @property
    def stride_x(self) -> int:
        return self.tiles[1].x0 - self.tiles[0].x0 if self.n_cols > 1 else self.tile_w

    @property
    def stride_y(self) -> int:
        return self.tiles[self.n_cols].y0 - self.tiles[0].y0 if self.n_rows > 1 else self.tile_h

    @property
    def overlap_px(self) -> tuple[int, int]:
        return (self.tile_w - self.stride_x, self.tile_h - self.stride_y)

    @property
    def effective_overlap(self) -> tuple[float, float]:
        ox, oy = self.overlap_px
        return (ox / self.tile_w, oy / self.tile_h)

    @property
    def max_safe_target_px(self) -> int:
        """Longest side below which a target is guaranteed to be **whole** inside at least one tile.

        Above this the seam-union path in :func:`merge_tile_detections` is what reassembles it.
        """
        ox, oy = self.overlap_px
        return int(min(ox, oy))

    @property
    def megapixels(self) -> float:
        """Pixels the detector actually processes — the quantity §5.11's latency scales with, not the frame size."""
        return len(self.tiles) * self.tile_w * self.tile_h / 1e6

    def describe(self) -> str:
        ox, oy = self.overlap_px
        return (f"{self.n_cols}x{self.n_rows}={len(self.tiles)} tiles of {self.tile_w}x{self.tile_h} over "
                f"{self.frame_w}x{self.frame_h}; overlap {ox}x{oy} px "
                f"({self.effective_overlap[0]:.0%}x{self.effective_overlap[1]:.0%}); "
                f"whole targets up to {self.max_safe_target_px} px; {self.megapixels:.2f} MPix/frame")


def tile_batches(n_tiles: int, max_batch: int) -> list[list[int]]:
    """Split tile indices into engine-batch-sized groups (a TensorRT engine has a fixed batch dimension)."""
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    return [list(range(i, min(i + max_batch, n_tiles))) for i in range(0, n_tiles, max_batch)]


# --- inverse mapping ---------------------------------------------------------------------------------------
def map_box_to_frame(tile: Tile, bbox_px: Sequence[float]) -> tuple[float, float, float, float]:
    """Tile-local xyxy -> full-frame xyxy, clipped to the tile rect and then to the frame."""
    x1 = min(max(float(bbox_px[0]) + tile.x0, tile.x0), tile.x1)
    y1 = min(max(float(bbox_px[1]) + tile.y0, tile.y0), tile.y1)
    x2 = min(max(float(bbox_px[2]) + tile.x0, tile.x0), tile.x1)
    y2 = min(max(float(bbox_px[3]) + tile.y0, tile.y0), tile.y1)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def to_frame(
    per_tile: Sequence[Sequence[Detection]],
    grid: TileGrid,
    indices: Sequence[int] | None = None,
    frame_idx: int | None = None,
) -> list[Detection]:
    """Map per-tile detections (tile-local coordinates) into full-frame coordinates, stamping ``tile_idx``.

    ``per_tile[k]`` are the detections of ``grid[indices[k]]`` (or ``grid[k]`` when ``indices`` is None), so a
    dynamic-tiling run maps back exactly as a full run does.
    """
    idxs = list(range(len(per_tile))) if indices is None else list(indices)
    if len(idxs) != len(per_tile):
        raise ValueError(f"{len(per_tile)} tile results for {len(idxs)} tile indices")
    out: list[Detection] = []
    for k, tile_dets in enumerate(per_tile):
        tile = grid[idxs[k]]
        for det in tile_dets:
            mapped = replace(det, bbox_px=map_box_to_frame(tile, det.bbox_px), tile_idx=tile.idx)
            if frame_idx is not None:
                mapped.frame_idx = frame_idx
            if det.bbox_amodal_px is not None:
                mapped.bbox_amodal_px = map_box_to_frame(tile, det.bbox_amodal_px)
            out.append(mapped)
    return out


# --- the cross-tile merge ----------------------------------------------------------------------------------
@dataclass(slots=True)
class _Item:
    """A detection plus the set of tiles it was assembled from (grows as seam fragments are unioned)."""

    det: Detection
    tiles: set[int]


def _union_rect(grid: TileGrid, tiles: Iterable[int]) -> tuple[int, int, int, int]:
    ts = [grid[i] for i in tiles]
    return (min(t.x0 for t in ts), min(t.y0 for t in ts), max(t.x1 for t in ts), max(t.y1 for t in ts))


def _seam_union(grid: TileGrid, a: _Item, b: _Item, edge_tol_px: float, seam_overlap_frac: float) -> bool:
    """True when ``a`` and ``b`` are two fragments of one target split by the tile seam between them.

    Both must be clipped against the *interior* tile edge that faces the other fragment, and must overlap along
    the perpendicular axis. Requiring **both** sides to be clipped is what keeps this from merging two genuinely
    separate targets: a target smaller than the overlap band is always whole in one of the two tiles, so at most
    one of the pair is ever clipped (see :attr:`TileGrid.max_safe_target_px`).
    """
    for lo, hi in ((a, b), (b, a)):  # try both orderings; "lo" is the one on the low side of the seam
        lx0, ly0, lx1, ly1 = _union_rect(grid, lo.tiles)
        hx0, hy0, hx1, hy1 = _union_rect(grid, hi.tiles)
        lbox, hbox = lo.det.bbox_px, hi.det.bbox_px
        # horizontal seam: lo's right edge and hi's left edge are the same interior boundary region
        if lx1 < hx1 and hx0 > lx0 and lx1 < grid.frame_w and hx0 > 0:
            if abs(lbox[2] - lx1) <= edge_tol_px and abs(hbox[0] - hx0) <= edge_tol_px:
                inter = min(lbox[3], hbox[3]) - max(lbox[1], hbox[1])
                smaller = min(lbox[3] - lbox[1], hbox[3] - hbox[1])
                if smaller > 0 and inter / smaller >= seam_overlap_frac:
                    return True
        # vertical seam
        if ly1 < hy1 and hy0 > ly0 and ly1 < grid.frame_h and hy0 > 0:
            if abs(lbox[3] - ly1) <= edge_tol_px and abs(hbox[1] - hy0) <= edge_tol_px:
                inter = min(lbox[2], hbox[2]) - max(lbox[0], hbox[0])
                smaller = min(lbox[2] - lbox[0], hbox[2] - hbox[0])
                if smaller > 0 and inter / smaller >= seam_overlap_frac:
                    return True
    return False


def _merge_items(a: _Item, b: _Item) -> _Item:
    """Union of two fragments. The higher-scoring detection donates every non-geometric field."""
    keep, other = (a, b) if a.det.score >= b.det.score else (b, a)
    ka, kb = keep.det.bbox_px, other.det.bbox_px
    box = (min(ka[0], kb[0]), min(ka[1], kb[1]), max(ka[2], kb[2]), max(ka[3], kb[3]))
    det = replace(keep.det, bbox_px=box, score=max(a.det.score, b.det.score))
    return _Item(det, set(a.tiles) | set(b.tiles))


def merge_tile_detections(
    detections: Sequence[Detection],
    grid: TileGrid,
    *,
    iou_thr: float = 0.55,
    containment_thr: float = 0.8,
    edge_tol_px: float = 2.0,
    seam_overlap_frac: float = 0.5,
    max_union_passes: int = 8,
) -> list[Detection]:
    """Collapse per-tile detections (already in full-frame coordinates) into one list, score-ordered.

    Two mechanisms, in this order:

    1. **Seam union** to a fixed point — reassembles a target wider than the overlap band from its fragments.
    2. **Greedy class-wise suppression** — IoU for ordinary duplicates, plus intersection-over-smaller for a
       fragment sitting inside the whole box another tile saw. Containment is skipped for two boxes that came from
       exactly the same single tile, where nesting is the detector's own (deliberate) output.

    ``tile_idx`` on the survivor is the tile of the highest-scoring contributor, so provenance survives the merge.
    """
    # A detection with tile_idx = -1 (whole-frame pass, or an out-of-range index) carries no provenance: it takes
    # part in suppression but never in a seam union. Nothing is ever dropped for a bad index.
    items = [_Item(d, {d.tile_idx} if 0 <= d.tile_idx < len(grid) else set()) for d in detections]

    # 1. seam union to a fixed point
    for _ in range(max_union_passes):
        merged_any = False
        out: list[_Item] = []
        used = [False] * len(items)
        for i, a in enumerate(items):
            if used[i]:
                continue
            cur = a
            for j in range(i + 1, len(items)):
                b = items[j]
                if used[j] or b.det.cls != cur.det.cls or not cur.tiles or not b.tiles:
                    continue
                if cur.tiles & b.tiles:
                    continue  # same tile: not a seam pair
                if _seam_union(grid, cur, b, edge_tol_px, seam_overlap_frac):
                    cur = _merge_items(cur, b)
                    used[j] = True
                    merged_any = True
            used[i] = True
            out.append(cur)
        items = out
        if not merged_any:
            break

    # 2. greedy suppression, score-ordered
    order = sorted(range(len(items)), key=lambda i: items[i].det.score, reverse=True)
    kept: list[_Item] = []
    for i in order:
        cand = items[i]
        suppressed = False
        for k in kept:
            if k.det.cls != cand.det.cls:
                continue
            if iou_xyxy(k.det.bbox_px, cand.det.bbox_px) >= iou_thr:
                suppressed = True
                break
            same_single_tile = len(k.tiles) == 1 and k.tiles == cand.tiles
            if not same_single_tile and containment_xyxy(k.det.bbox_px, cand.det.bbox_px) >= containment_thr:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return [it.det for it in kept]
