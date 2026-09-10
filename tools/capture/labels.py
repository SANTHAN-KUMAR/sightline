"""Turn a Cosys-AirSim instance-segmentation frame into labels (SOLUTION_DOC 5.1 step 6, 6.3).

The visible-extent box IS the training box (6.3). It is read exactly from the instance mask - no estimation,
no projection - which is the whole reason the simulator is worth using as the label source.

Naming, measured against Cosys-AirSim 3.4.1 on 2026-09-10: `simListInstanceSegmentationObjects()` does NOT
return the actor name. It returns, per rendered component:
    skeletal actor : `<ActorName>_<uid>`                     e.g. Human_007_71939
    static actor   : `<MeshName>_<n>_<ActorName>_<uid>`      e.g. SM_FloodValley_0_Ground_72567
so an exact match on "Human_007" finds nothing. Actors are recovered by regex on the id instead.

`simGetSegmentationColorMap()` returns an RGB palette indexed by the position of the object in
`simListInstanceSegmentationObjects()`. OpenCV reads PNGs as BGR, so the palette is reversed once on load
rather than per pixel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

HUMAN_RE = re.compile(r"^Human_(\d+)_\d+$")
ANIMAL_RE = re.compile(r"^Animal_(\d+)_\d+$")
#: static actors are `<Mesh>_<n>_<ActorName>_<uid>`; this pulls the actor name out
STATIC_RE = re.compile(r"^.*?_\d+_(?P<actor>[A-Za-z][A-Za-z0-9_]*?)_\d+$")


@dataclass(slots=True)
class MaskLabel:
    """One instance found in the mask. `bbox_px` is xyxy in full-frame pixels, inclusive of both edges."""

    actor_id: int
    name: str
    cls: str
    bbox_px: tuple[int, int, int, int]
    visible_px: int
    rgb: tuple[int, int, int]
    #: filled by `attach_truth` from data/scene/actors.json
    pose: str = "unknown"
    submersion: str = "unknown"
    occlusion: int | None = None
    zone: str = "unknown"
    aerially_detectable: bool = True
    group: str | None = None
    visible_fraction: float | None = None
    amodal_bbox_px: tuple[int, int, int, int] | None = None

    @property
    def width_px(self) -> int:
        return self.bbox_px[2] - self.bbox_px[0] + 1

    @property
    def height_px(self) -> int:
        return self.bbox_px[3] - self.bbox_px[1] + 1

    @property
    def size_px(self) -> int:
        """Longest side: the axis the 5.12 pixel-size slices bin on."""
        return max(self.width_px, self.height_px)


def actor_index(names: Iterable[str]) -> dict[int, tuple[int, str, str]]:
    """palette index -> (actor_id, canonical name, class) for every Human_/Animal_ instance."""
    out: dict[int, tuple[int, str, str]] = {}
    for i, n in enumerate(names):
        m = HUMAN_RE.match(n)
        if m:
            out[i] = (int(m.group(1)), f"Human_{int(m.group(1)):03d}", "human")
            continue
        m = ANIMAL_RE.match(n)
        if m:
            out[i] = (int(m.group(1)), f"Animal_{int(m.group(1)):03d}", "animal")
    return out


def palette_bgr(colour_map) -> np.ndarray:
    """Palette as BGR uint8 so it can be compared directly against an OpenCV-loaded PNG."""
    p = np.asarray(colour_map, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"unexpected colour map shape {p.shape}")
    return p[:, ::-1].copy()


def labels_from_mask(mask_bgr: np.ndarray, names: list[str], colour_map, *,
                     min_px: int = 4) -> list[MaskLabel]:
    """Exact visible-extent boxes for every survivor present in the mask.

    `min_px` drops instances of only a pixel or two: below that the box corners are noise, and 6.3 says an
    unresolvable target is not a usable training box. Anything dropped is still counted by the caller against
    the ground truth, so a missed survivor shows up as a miss rather than silently disappearing.
    """
    if mask_bgr.ndim != 3 or mask_bgr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 BGR mask, got {mask_bgr.shape}")
    pal = palette_bgr(colour_map)
    idx = actor_index(names)
    if not idx:
        raise ValueError("no Human_/Animal_ instances in the segmentation object list")

    # One pass over the frame: pack BGR into a single int32 key and bucket by colour.
    flat = mask_bgr.reshape(-1, 3).astype(np.int32)
    key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    present, first, counts = np.unique(key, return_index=True, return_counts=True)
    want = {}
    for i, (aid, name, cls) in idx.items():
        b, g, r = int(pal[i, 0]), int(pal[i, 1]), int(pal[i, 2])
        want[(b << 16) | (g << 8) | r] = (i, aid, name, cls)

    h, w = mask_bgr.shape[:2]
    out: list[MaskLabel] = []
    for k, n_px in zip(present, counts):
        hit = want.get(int(k))
        if hit is None or n_px < min_px:
            continue
        i, aid, name, cls = hit
        ys, xs = np.where(np.all(mask_bgr == pal[i], axis=2))
        out.append(MaskLabel(
            actor_id=aid, name=name, cls=cls,
            bbox_px=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            visible_px=int(n_px), rgb=(int(pal[i, 2]), int(pal[i, 1]), int(pal[i, 0])),
        ))
    out.sort(key=lambda m: m.actor_id)
    del h, w
    return out


def attach_truth(labels: list[MaskLabel], actors_json: dict) -> list[MaskLabel]:
    """Join the mask labels to the generator's ground truth (pose, submersion, occlusion, zone)."""
    by_id = {a["id"]: a for a in actors_json["actors"]}
    for m in labels:
        a = by_id.get(m.actor_id)
        if a is None:
            continue
        m.pose = a["pose"]
        m.submersion = a["submersion"]
        m.occlusion = a["occlusion"]
        m.zone = a["zone"]
        m.aerially_detectable = a["aerially_detectable"]
        m.group = a["group"]
    return labels


def attach_visible_fraction(labels: list[MaskLabel], amodal: list[MaskLabel]) -> list[MaskLabel]:
    """visible_fraction = visible px / unoccluded px, from a pixel-aligned reference pass (6.3).

    The reference pass is the same camera pose rendered with the occluder removed - in this scene that means
    the flood surface dropped, which is the dominant occluder. It is EXACT for water occlusion. Debris and
    roof-edge occlusion are not covered by it, so a label whose actor is missing from the reference keeps
    `visible_fraction = None` rather than a fabricated number.
    """
    ref = {m.actor_id: m for m in amodal}
    for m in labels:
        r = ref.get(m.actor_id)
        if r is None or r.visible_px <= 0:
            continue
        m.visible_fraction = round(min(1.0, m.visible_px / r.visible_px), 4)
        m.amodal_bbox_px = r.bbox_px
    return labels


def to_yolo(labels: list[MaskLabel], width: int, height: int, classes=("human", "animal")) -> list[str]:
    """YOLO lines from the VISIBLE box (6.3: the visible extent is the training box)."""
    cls_id = {c: i for i, c in enumerate(classes)}
    lines = []
    for m in labels:
        if m.cls not in cls_id:
            continue
        x1, y1, x2, y2 = m.bbox_px
        cx, cy = (x1 + x2 + 1) / 2.0 / width, (y1 + y2 + 1) / 2.0 / height
        bw, bh = m.width_px / width, m.height_px / height
        lines.append(f"{cls_id[m.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines
