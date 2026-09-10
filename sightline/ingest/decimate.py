"""Decimation and the full-rate frame index (SOLUTION_DOC §5.4 "Decimation").

    "Process every k-th frame (k = 3 -> 10 FPS, k = 6 -> 5 FPS); the tracker's buffers are set in
     processed-frame units (§5.6). **Every decoded frame is still available for the evidence thumbnail of the
     best observation.**"

That last sentence is the whole reason this module exists. Decimation must never *discard* frames, only skip
processing them, so `FrameIndex` records where every frame lives and `FrameIndex.load()` can fetch any of them
later — which is what `Track.best()` -> `Evidence.thumb_uri` needs (§5.6, §5.8).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, TypeVar

import numpy as np

__all__ = ["FrameRef", "FrameIndex", "decimate", "decimation_for_fps", "K_10FPS", "K_5FPS"]

#: The two decimation factors §5.4 names, assuming a 30 FPS source.
K_10FPS = 3
K_5FPS = 6

T = TypeVar("T")


def decimation_for_fps(source_fps: float, target_fps: float) -> int:
    """k = round(source / target), clamped to >= 1. 30 -> 10 gives 3; 30 -> 5 gives 6 (§5.4)."""
    if source_fps <= 0.0 or target_fps <= 0.0:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def decimate(items: Iterable[T], k: int, *, offset: int = 0) -> Iterator[T]:
    """Yield every k-th item, starting at `offset`. `k = 1` is the identity."""
    if k < 1:
        raise ValueError("k must be >= 1")
    for i, item in enumerate(items):
        if i >= offset and (i - offset) % k == 0:
            yield item


@dataclass(slots=True)
class FrameRef:
    """Where one DECODED frame lives. Cheap enough (a few dozen bytes) to keep one per frame of a whole clip."""

    frame_idx: int
    t_utc: float
    pts_s: float = 0.0
    path: str = ""            # frame-file captures: the image path
    video_path: str = ""      # container captures: the MP4
    video_index: int = -1     # index of this frame inside the container
    thermal_path: str = ""
    processed: bool = False   # True when the decimator passed it downstream for detection
    aux: dict[str, str] = field(default_factory=dict)   # seg/depth/annotation paths, carried untouched


class FrameIndex:
    """Every decoded frame of one clip, in decode order, with time-ordered lookup.

    Guardrail R10 note: this index only ever grows. Nothing here removes a frame, and `mark_processed` only
    annotates. Dropping evidence is exactly the kind of quiet data loss R10 exists to prevent.
    """

    def __init__(self, clip_id: str = "") -> None:
        self.clip_id = clip_id
        self._refs: list[FrameRef] = []
        self._by_idx: dict[int, FrameRef] = {}
        self._times: list[float] = []   # kept sorted for bisect; equals [r.t_utc] when times are monotonic

    # -- building ----------------------------------------------------------------------------------------
    def add(self, ref: FrameRef) -> FrameRef:
        self._refs.append(ref)
        self._by_idx[ref.frame_idx] = ref
        bisect.insort(self._times, ref.t_utc)
        return ref

    def mark_processed(self, frame_idx: int) -> None:
        ref = self._by_idx.get(frame_idx)
        if ref is not None:
            ref.processed = True

    # -- reading -----------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._refs)

    def __iter__(self) -> Iterator[FrameRef]:
        return iter(self._refs)

    def __getitem__(self, frame_idx: int) -> FrameRef:
        return self._by_idx[frame_idx]

    def get(self, frame_idx: int) -> FrameRef | None:
        return self._by_idx.get(frame_idx)

    @property
    def refs(self) -> list[FrameRef]:
        return list(self._refs)

    def processed_indices(self) -> list[int]:
        return [r.frame_idx for r in self._refs if r.processed]

    def times(self) -> np.ndarray:
        return np.asarray([r.t_utc for r in self._refs], dtype=np.float64)

    def nearest(self, t_utc: float) -> FrameRef | None:
        """The decoded frame closest in time to `t_utc` — how a track's best observation finds its frame."""
        if not self._refs:
            return None
        ts = self.times()
        return self._refs[int(np.argmin(np.abs(ts - float(t_utc))))]

    def span_s(self) -> float:
        return (max(self._times) - min(self._times)) if self._times else 0.0

    # -- fetching pixels ---------------------------------------------------------------------------------
    def load(self, frame_idx: int, *, video_reader: Any = None,
             image_loader: Callable[[str], np.ndarray] | None = None) -> np.ndarray | None:
        """Fetch the pixels of ANY decoded frame, processed or skipped (the evidence-thumbnail path).

        Frame-file captures are read straight off disk. Container captures need the clip's `VideoReader`
        (`Clip.video_reader`), which seeks to the nearest keyframe and decodes forward.
        """
        ref = self._by_idx.get(frame_idx)
        if ref is None:
            return None
        if ref.path:
            loader = image_loader
            if loader is None:
                from sightline.ingest.decode import read_image_bgr

                loader = read_image_bgr
            return loader(ref.path)
        if ref.video_index >= 0 and video_reader is not None:
            return video_reader.frame_at(ref.video_index)
        return None
