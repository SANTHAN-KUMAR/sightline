"""Turn a Cosys-AirSim instance-segmentation frame into labels (SOLUTION_DOC 5.1 step 6, 6.3).

The visible-extent box IS the training box (6.3). It is read exactly from the instance mask - no estimation,
no projection - which is the whole reason the simulator is worth using as the label source.

Naming, measured against Cosys-AirSim 3.4.1 on 2026-09-10: `simListInstanceSegmentationObjects()` does NOT
return the actor name. It returns, per rendered component:
    skeletal actor : `<ActorName>_<uid>`                     e.g. Human_007_71939
    static actor   : `<MeshName>_<n>_<ActorName>_<uid>`      e.g. SM_FloodValley_0_Ground_72567
so an exact match on "Human_007" finds nothing. Actors are recovered by regex on the id instead.

Channel order, measured 2026-09-10 and the source of a bug that silently mislabelled everything:
`simGetImages(..., compress=False).image_data_uint8` is **RGB**, not BGR (verified two ways: the silt-brown
flood surface gives channel means 199/182/154, and reversing the segmentation palette attributes the centre of
a 45 m frame to a survivor 283 m away, while not reversing attributes it to a house 2 m away). The palette from
`simGetSegmentationColorMap()` is RGB too, indexed by position in `simListInstanceSegmentationObjects()`, so
raw buffer and palette are compared **directly, with no channel swap**.

The trap: OpenCV works in BGR. A raw buffer handed straight to `cv2.imwrite` is saved with red and blue
swapped, and a PNG loaded with `cv2.imread` is BGR and must be swapped before being matched against the
palette. `mask_to_rgb()` below is the single place that conversion is expressed.
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
    #: Section 6.3 `ignore`: no line of sight to the body, so neither a recall target nor a false positive.
    ignore: bool = False

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


def palette_rgb(colour_map) -> np.ndarray:
    """The instance palette as RGB uint8 — the same order as the raw AirSim image buffer, so no swap."""
    p = np.asarray(colour_map, dtype=np.uint8)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"unexpected colour map shape {p.shape}")
    return p


def mask_to_rgb(mask: np.ndarray, *, source: str) -> np.ndarray:
    """Normalise a segmentation frame to RGB.

    `source="airsim"` for a raw `image_data_uint8` buffer (already RGB); `source="cv2"` for an image loaded
    with `cv2.imread`, which is BGR and must be reversed. Getting this wrong does not raise - it silently
    attributes every pixel to the wrong instance - so the caller must say which it has.
    """
    if source == "airsim":
        return mask
    if source == "cv2":
        return mask[:, :, ::-1]
    raise ValueError(f"source must be 'airsim' or 'cv2', got {source!r}")


def labels_from_mask(mask_rgb: np.ndarray, names: list[str], colour_map, *,
                     min_px: int = 4) -> list[MaskLabel]:
    """Exact visible-extent boxes for every survivor present in the mask.

    `min_px` drops instances of only a pixel or two: below that the box corners are noise, and 6.3 says an
    unresolvable target is not a usable training box. Anything dropped is still counted by the caller against
    the ground truth, so a missed survivor shows up as a miss rather than silently disappearing.
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB mask, got {mask_rgb.shape}")
    pal = palette_rgb(colour_map)
    idx = actor_index(names)
    if not idx:
        raise ValueError("no Human_/Animal_ instances in the segmentation object list")

    # One pass over the frame: pack RGB into a single int32 key and bucket by colour.
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)
    key = (flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]
    present, first, counts = np.unique(key, return_index=True, return_counts=True)
    want = {}
    for i, (aid, name, cls) in idx.items():
        r, g, b = int(pal[i, 0]), int(pal[i, 1]), int(pal[i, 2])
        want[(r << 16) | (g << 8) | b] = (i, aid, name, cls)

    h, w = mask_rgb.shape[:2]
    out: list[MaskLabel] = []
    for k, n_px in zip(present, counts):
        hit = want.get(int(k))
        if hit is None or n_px < min_px:
            continue
        i, aid, name, cls = hit
        ys, xs = np.where(np.all(mask_rgb == pal[i], axis=2))
        out.append(MaskLabel(
            actor_id=aid, name=name, cls=cls,
            bbox_px=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            visible_px=int(n_px), rgb=(int(pal[i, 0]), int(pal[i, 1]), int(pal[i, 2])),
        ))
    out.sort(key=lambda m: m.actor_id)
    del h, w
    return out


#: Depth-gate slack in metres. ABOVE absorbs the camera's mounting offset below the telemetry reference and
#: any error in the body height; BELOW absorbs terrain slope under the body and the quantisation of
#: `base_asl_m`. Both are far smaller than the 3-16 m a crown or a slab stands above a survivor, which is the
#: separation this gate actually has to resolve.
DEPTH_SLACK_ABOVE_M = 1.0
DEPTH_SLACK_BELOW_M = 1.5

#: A box has to clear BOTH floors to be a training target. `MIN_VISIBLE_PX` alone lets specks seen through
#: gaps in a canopy hold a full-size box open; `MIN_VISIBLE_FRACTION` is what actually catches that case.
MIN_VISIBLE_PX = 12
MIN_VISIBLE_FRACTION = 0.04


def apply_depth_visibility(
    labels: list[MaskLabel], mask_rgb: np.ndarray, depth_planar: np.ndarray, *,
    cam_alt_asl_m: float, actors_json: dict, min_px: int = MIN_VISIBLE_PX,
    min_visible_fraction: float = MIN_VISIBLE_FRACTION,
    slack_above_m: float = DEPTH_SLACK_ABOVE_M, slack_below_m: float = DEPTH_SLACK_BELOW_M,
) -> tuple[list[MaskLabel], list[MaskLabel]]:
    """Cut each mask instance down to the pixels a camera can really see, using the depth buffer.

    Returns `(visible, fully_occluded)`.

    WHY THIS IS NEEDED. Cosys-AirSim renders its instance mask with two engine show flags switched off
    (`Source/Annotation/ObjectAnnotator.cpp:SetViewForAnnotationRender`)::

        show_flags.SetInstancedFoliage(false);
        show_flags.SetInstancedGrass(false);

    Every plant in this scene is an instance on a HierarchicalInstancedStaticMeshComponent, because the
    Windows commit limit killed the editor at ~5,500 individual actors. So the mask renders the terrain
    straight through the canopy: a survivor lying under a fern appears in it whole and unoccluded. Measured on
    seed23_alt35, 7 of 110 boxes sat on pure leaf texture with no subject visible in RGB at all, and every
    vegetation-occluded survivor carried a full-body box where guideline section 1 requires a visible-extent
    one.

    DepthPlanar is the ordinary scene depth buffer and both Nanite and instanced meshes write it, which
    `tools/capture/check_depth_sees_foliage.py` demonstrates on a frame where the mask shows flat terrain and
    depth shows every crown. Because it is *planar* depth - distance along the optical axis, not radial range
    - a nadir frame gives `depth = camera_altitude - surface_altitude` at every pixel, with no dependence on
    where the pixel sits in the image. So the test is simply whether the frontmost surface at that pixel is
    the survivor's own body or something above it::

        visible(px)  <=>  d_ground - body_height - slack_above  <=  depth(px)  <=  d_ground + slack_below

    THE DIVIDEND. Because foliage is missing from the mask, the mask silhouette is the **amodal** body - what
    the camera would see with the occluder removed. That makes `visible_px / amodal_px` a *measured*
    visibility budget rather than the reference-silhouette estimate `measure_occlusion.py` has to fall back
    on, and it is exact per observation. It is kept in `visible_fraction`, with the pre-gate box preserved in
    `amodal_bbox_px`.
    """
    if depth_planar.shape[:2] != mask_rgb.shape[:2]:
        raise ValueError(f"depth {depth_planar.shape[:2]} does not match mask {mask_rgb.shape[:2]}")
    by_id = {a["id"]: a for a in actors_json["actors"]}
    visible: list[MaskLabel] = []
    occluded: list[MaskLabel] = []

    for m in labels:
        a = by_id.get(m.actor_id)
        if a is None:
            raise KeyError(f"{m.name} is in the mask but not in actors.json - the palette and the ground "
                           f"truth disagree, which invalidates every box in this frame")
        x1, y1, x2, y2 = m.bbox_px
        sub_m = mask_rgb[y1:y2 + 1, x1:x2 + 1]
        sub_d = depth_planar[y1:y2 + 1, x1:x2 + 1]
        amodal = np.all(sub_m == np.array(m.rgb, dtype=mask_rgb.dtype), axis=2)

        d_ground = float(cam_alt_asl_m) - float(a["base_asl_m"])
        body_m = float(a["height_cm"]) / 100.0
        lo = d_ground - body_m - slack_above_m
        hi = d_ground + slack_below_m
        sel = amodal & np.isfinite(sub_d) & (sub_d >= lo) & (sub_d <= hi)

        n_amodal, n_vis = int(amodal.sum()), int(sel.sum())
        m.amodal_bbox_px = m.bbox_px
        m.visible_fraction = float(n_vis) / float(n_amodal) if n_amodal else 0.0
        # BOTH a pixel floor and a fraction floor. A pixel count alone is not enough: measured on
        # seed47_alt55, a survivor under a jacaranda leaked a handful of pixels through gaps between fronds
        # and kept a 22 px box over solid foliage with no subject visible in RGB at all. A body that is
        # 96 % hidden is not a training target whatever the absolute pixel count happens to be.
        if n_vis < min_px or m.visible_fraction < min_visible_fraction:
            # Nothing of this survivor reaches the camera. This is ground truth, not a labelling failure:
            # section 2.7 already has buried survivors an aerial search cannot find. Dropping the box is the
            # only honest option - writing one would train the detector to invent a person from foliage.
            m.visible_px = n_vis
            occluded.append(m)
            continue
        ys, xs = np.where(sel)
        m.bbox_px = (x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()), y1 + int(ys.max()))
        m.visible_px = n_vis
        # section 6.3's three-level scale, now measured per observation instead of inherited from the layout
        # generator, which never knew what the other five generators scattered on top.
        vf = m.visible_fraction
        m.occlusion = 0 if vf >= 0.99 else (1 if vf >= 0.5 else 2)
        visible.append(m)

    return visible, occluded


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
        if m.cls not in cls_id or m.ignore:
            continue
        x1, y1, x2, y2 = m.bbox_px
        cx, cy = (x1 + x2 + 1) / 2.0 / width, (y1 + y2 + 1) / 2.0 / height
        bw, bh = m.width_px / width, m.height_px / height
        lines.append(f"{cls_id[m.cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def truth_for_run(run_dir):
    """The ground truth THIS run was captured against, not whatever `data/scene/actors.json` holds now.

    `gen_actors.py --seed N` overwrites the global file, so the moment a second scenario is generated every
    tool that reads the global path starts checking one seed's labels against another seed's actor
    placements. Measured on 2026-09-11 that produced "50 labels disagree with actors.json" on a run whose
    labels were in fact correct. The capture copies its own `actors.json` beside the frames; prefer it, and
    say out loud which file was used so a mismatch can never be silent again.
    """
    import json as _json
    from pathlib import Path as _Path

    run = _Path(run_dir)
    own = run / "actors.json"
    if own.exists():
        return _json.loads(own.read_text(encoding="utf-8")), own
    glob = _Path(__file__).resolve().parents[2] / "data/scene/actors.json"
    return _json.loads(glob.read_text(encoding="utf-8")), glob
