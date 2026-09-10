"""F8b dataset conversion: a captured simulator run -> tiled Ultralytics YOLO data, and -> `EvalDataset`.

This is the module that turns `tools/capture/run.py`'s output into something a detector can train on and something
`sightline.eval` can score against. Three rules from the doc are implemented literally, because each of them is a
way the resulting number can be a lie:

1. **Tile at native resolution, on the SAME grid inference uses** (§5.5c step 1, `tiler.py`). A 3840x2160 frame
   letterboxed to 1024 shrinks a 28 px survivor to 7 px. Training crops are cut with :class:`TileGrid`, so a
   training pixel and a deployment pixel are the same pixel.
2. **Split by scenario seed, never by frame** (§5.5c step 3, §6.5). Consecutive frames of one flight are nearly
   identical; splitting them across train and val leaks the answer and inflates every number. :func:`split_by_seed`
   assigns whole *runs* (one seed each) to splits and :func:`assert_no_seed_leak` refuses to build a dataset whose
   seeds appear in two splits.
3. **A box the truth says is not aerially detectable is `uncertain`, not a positive and not a negative** (§2.7,
   §6.3). Actor 55 and 62 in `data/scene/actors.json` are buried; if one leaks into a mask (it has happened, see
   `docs/QUALITY_GATE.md`) it must not become a training target and must not be scored as a miss.

Coordinate convention, stated once because it is off-by-one bait. `tools/capture/labels.py` writes `bbox_px` as
**inclusive integer pixel indices** (`width = x2 - x1 + 1`), while `Detection`/`GtBox` and this module use
**continuous half-open** xyxy. :func:`capture_box_to_xyxy` is the single conversion `(x1, y1, x2 + 1, y2 + 1)`,
and `test_detect.py` checks it reproduces the run's own YOLO `.txt` line to 1e-6.

Everything here is stdlib + numpy + (optionally) cv2 for writing crops. No torch, no ultralytics, no GPU.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sightline.detect.tiler import DEFAULT_OVERLAP, DEFAULT_TILE_PX, Tile, TileGrid, intersection_area

__all__ = [
    "CLASSES",
    "MAX_TARGET_GROUND_M",
    "CaptureFrame",
    "CaptureLabel",
    "CaptureRun",
    "TiledBox",
    "assert_boxes_are_plausible",
    "assert_no_seed_leak",
    "build_yolo_dataset",
    "capture_box_to_xyxy",
    "clip_box_to_tile",
    "implausible_boxes",
    "load_run",
    "load_runs",
    "plausible_labels",
    "split_by_seed",
    "tile_boxes",
    "to_eval_dataset",
    "urgency_for",
    "yolo_line",
]

#: Class order is the training contract: index 0 = human, 1 = animal. It must match `tools/capture/labels.to_yolo`.
CLASSES: tuple[str, ...] = ("human", "animal")

#: Minimum fraction of a box's area that must survive the tile clip for the fragment to be a training target.
#: Below this the crop shows a sliver of a person with no identifiable part, which teaches the model that a
#: two-pixel smear is a human. SAHI's own slicing default is 0.2; 0.35 is stricter because our targets are tiny.
MIN_VISIBLE_FRAC = 0.35
#: §6.3: "minimum size >= 8 px on the longer side at the resolution the detector sees".
MIN_BOX_PX = 8.0
#: The largest ground extent a single labelled person may plausibly occupy, in metres. A prone adult is 1.8 m;
#: 3.0 m allows for off-nadir foreshortening, a long shadow included in the mask, and a sprawled pose. Anything
#: beyond this is not a person: it is a mask instance colour that has leaked onto the terrain or the water plane.
#: `tools/capture/validate.py` prints the box-size distribution but does not fail on it, so this lane fails on it
#: instead -- a training set of "people" 68 m across is worse than no training set.
MAX_TARGET_GROUND_M = 3.0


# --- 1. the capture run in memory ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CaptureLabel:
    """One labelled instance from `labels/<stem>.json`, converted to continuous xyxy."""

    bbox_px: tuple[float, float, float, float]
    cls: str = "human"
    actor_id: int = -1
    name: str = ""
    size_px: float = 0.0
    pose: str = "unknown"
    submersion: str = "unknown"
    occlusion: int | None = None
    zone: str = "unknown"
    group: str | None = None
    aerially_detectable: bool = True
    visible_px: int = 0

    @property
    def width_px(self) -> float:
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height_px(self) -> float:
        return self.bbox_px[3] - self.bbox_px[1]

    @property
    def longest_px(self) -> float:
        return max(self.width_px, self.height_px)


@dataclass(slots=True)
class CaptureFrame:
    """One captured frame: where the pixels are, what is in them, and the telemetry the slices need."""

    stem: str
    frame_idx: int
    image_path: Path
    mask_path: Path | None = None
    labels: list[CaptureLabel] = field(default_factory=list)
    width_px: int = 3840
    height_px: int = 2160
    t_utc: float = 0.0
    agl_m: float = 0.0
    alt_msl_m: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    gsd_cm_px: float = 0.0
    gimbal_pitch_deg: float = -90.0
    hfov_deg: float = 0.0
    mode: str = "AUTO"
    clip_id: str = ""
    time_of_day: str = "day"

    @property
    def scored_labels(self) -> list[CaptureLabel]:
        """Boxes that are recall targets: detectable actors only (§2.7)."""
        return [m for m in self.labels if m.aerially_detectable]


@dataclass(slots=True)
class CaptureRun:
    """One flight = one scenario seed = the atom the train/val/test split is made of (§5.5c step 3)."""

    root: Path
    clip_id: str
    scenario_seed: int
    frames: list[CaptureFrame] = field(default_factory=list)
    card: dict[str, Any] = field(default_factory=dict)

    @property
    def seed_group(self) -> str:
        return f"seed{self.scenario_seed}"

    @property
    def n_boxes(self) -> int:
        return sum(len(f.labels) for f in self.frames)

    @property
    def altitude_m(self) -> float:
        return float(self.card.get("altitude_m_agl", 0.0))

    @property
    def median_gsd_cm_px(self) -> float:
        """The run's own GSD, for frames whose telemetry row has not been written yet (a live capture).

        A survey flies one commanded altitude, so the median over the frames that DO have telemetry is a sound
        stand-in for the ones that do not, and it is only ever used to decide whether a box is person-sized.
        """
        vals = sorted(f.gsd_cm_px for f in self.frames if f.gsd_cm_px > 0)
        if vals:
            return vals[len(vals) // 2]
        return float(self.card.get("gsd_cm_px", 0.0) or 0.0)

    def describe(self) -> str:
        return (f"{self.clip_id}: {len(self.frames)} frames, {self.n_boxes} boxes, seed {self.scenario_seed}, "
                f"{self.altitude_m:g} m AGL, randomisation="
                f"{self.card.get('randomisation', 'unknown')}")


def capture_box_to_xyxy(bbox_px: Sequence[float]) -> tuple[float, float, float, float]:
    """Inclusive integer pixel indices -> continuous half-open xyxy. The ONE place the +1 lives."""
    return (float(bbox_px[0]), float(bbox_px[1]), float(bbox_px[2]) + 1.0, float(bbox_px[3]) + 1.0)


def _time_of_day(t_utc: float, card: dict[str, Any]) -> str:
    """The capture card is the authority when it states one; otherwise the run's own label, else 'day'."""
    for k in ("time_of_day", "tod"):
        if card.get(k):
            return str(card[k])
    return "day"


def load_run(path: str | Path) -> CaptureRun:
    """Read one `_artifacts/dataset/<run>` directory. Does not open a single image (this stays cheap)."""
    root = Path(path)
    card_p = root / "data_card.json"
    card: dict[str, Any] = json.loads(card_p.read_text(encoding="utf-8")) if card_p.exists() else {}
    clip_id = str(card.get("clip_id") or root.name)
    seed = int(card.get("scenario_seed", -1))

    tel: dict[int, dict[str, str]] = {}
    tel_p = root / "telemetry.csv"
    if tel_p.exists():
        with tel_p.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                tel[int(row["frame_idx"])] = row

    frames: list[CaptureFrame] = []
    for img in sorted((root / "images").glob("*.png")):
        stem = img.stem
        idx = int(stem.rsplit("_", 1)[-1]) if stem.rsplit("_", 1)[-1].isdigit() else len(frames)
        lj = root / "labels" / f"{stem}.json"
        raw = json.loads(lj.read_text(encoding="utf-8")) if lj.exists() else []
        labels = [
            CaptureLabel(
                bbox_px=capture_box_to_xyxy(m["bbox_px"]),
                cls=str(m.get("cls", "human")),
                actor_id=int(m.get("actor_id", -1)),
                name=str(m.get("name", "")),
                size_px=float(m.get("size_px", 0.0)),
                pose=str(m.get("pose", "unknown")),
                submersion=str(m.get("submersion", "unknown")),
                occlusion=(None if m.get("occlusion") is None else int(m["occlusion"])),
                zone=str(m.get("zone", "unknown")),
                group=m.get("group"),
                aerially_detectable=bool(m.get("aerially_detectable", True)),
                visible_px=int(m.get("visible_px", 0)),
            )
            for m in raw
        ]
        t = tel.get(idx, {})
        mask = root / "masks" / f"{stem}.png"
        frames.append(CaptureFrame(
            stem=stem, frame_idx=idx, image_path=img, mask_path=mask if mask.exists() else None, labels=labels,
            width_px=int(float(t.get("width_px", 3840) or 3840)),
            height_px=int(float(t.get("height_px", 2160) or 2160)),
            t_utc=float(t.get("t_utc", 0.0) or 0.0),
            agl_m=float(t.get("agl_m", card.get("altitude_m_agl", 0.0)) or 0.0),
            alt_msl_m=float(t.get("alt_msl_m", 0.0) or 0.0),
            lat=float(t.get("lat", 0.0) or 0.0), lon=float(t.get("lon", 0.0) or 0.0),
            gsd_cm_px=float(t.get("gsd_cm_px", card.get("gsd_cm_px", 0.0)) or 0.0),
            gimbal_pitch_deg=float(t.get("gimbal_pitch_deg", -90.0) or -90.0),
            hfov_deg=float(t.get("hfov_deg", 0.0) or 0.0),
            mode=str(t.get("mode", "AUTO") or "AUTO"),
            clip_id=clip_id,
            time_of_day=_time_of_day(float(t.get("t_utc", 0.0) or 0.0), card),
        ))
    return CaptureRun(root=root, clip_id=clip_id, scenario_seed=seed, frames=frames, card=card)


def load_runs(paths: Iterable[str | Path]) -> list[CaptureRun]:
    return [load_run(p) for p in paths]


# --- 2. the split (by seed, never by frame) -----------------------------------------------------------------
def assert_no_seed_leak(splits: dict[str, Sequence[CaptureRun]]) -> None:
    """§6.5 leakage guard. Raises rather than warns: a leaked seed makes every downstream number worthless."""
    where: dict[int, set[str]] = {}
    for name, runs in splits.items():
        for r in runs:
            where.setdefault(r.scenario_seed, set()).add(name)
    bad = {s: sorted(v) for s, v in where.items() if len(v) > 1}
    if bad:
        raise ValueError(
            f"scenario seed(s) appear in more than one split: {bad}. SOLUTION_DOC 5.5c step 3 forbids this; "
            "split by scenario seed, never by frame."
        )


def split_by_seed(
    runs: Sequence[CaptureRun],
    *,
    val_seeds: Sequence[int] | None = None,
    test_seeds: Sequence[int] | None = None,
    val_frac: float = 0.25,
    seed: int = 0,
) -> dict[str, list[CaptureRun]]:
    """Assign whole runs to `train` / `val` / `test`. Explicit seed lists win; otherwise a deterministic draw.

    The unit is the *run* because one run is one scenario seed. There is deliberately no frame-level path.
    """
    if not runs:
        raise ValueError("no runs to split")
    by_seed: dict[int, list[CaptureRun]] = {}
    for r in runs:
        by_seed.setdefault(r.scenario_seed, []).append(r)

    val_set = set(val_seeds or ())
    test_set = set(test_seeds or ())
    if val_set & test_set:
        raise ValueError(f"seeds {sorted(val_set & test_set)} listed as both val and test")

    if not val_set and not test_set:
        pool = sorted(by_seed)
        if len(pool) < 2:
            raise ValueError(
                f"only {len(pool)} scenario seed(s) captured ({pool}); a seed-held-out split needs at least two. "
                "Capture another run with a different scenario seed before training (SOLUTION_DOC 5.5c step 3)."
            )
        rng = random.Random(seed)
        shuffled = pool[:]
        rng.shuffle(shuffled)
        n_val = max(1, round(val_frac * len(pool)))
        val_set = set(shuffled[:n_val])

    out: dict[str, list[CaptureRun]] = {"train": [], "val": [], "test": []}
    for s in sorted(by_seed):
        name = "test" if s in test_set else "val" if s in val_set else "train"
        out[name].extend(by_seed[s])
    if not out["train"]:
        raise ValueError("the split left no training runs")
    assert_no_seed_leak(out)
    return out


def implausible_boxes(
    runs: Sequence[CaptureRun],
    *,
    max_ground_m: float = MAX_TARGET_GROUND_M,
) -> list[dict[str, Any]]:
    """Labels whose ground extent is impossible for one person, given the frame's own GSD.

    This is the check `tools/capture/validate.py` prints but does not enforce. It caught a real capture on
    2026-09-10 in which the segmentation mask collapsed to a single colour for whole stretches of the flight,
    so the terrain itself was labelled `Human_019` at 3840 x 2160 px = 68 x 38 m of ground.

    A frame with no GSD (telemetry not yet written) is skipped and reported in the returned dict's `basis`, so
    the absence of a check is visible rather than silently passing.
    """
    out: list[dict[str, Any]] = []
    for run in runs:
        fallback = run.median_gsd_cm_px
        for fr in run.frames:
            gsd = fr.gsd_cm_px if fr.gsd_cm_px > 0 else fallback
            if gsd <= 0:
                continue
            m_per_px = gsd / 100.0
            for lab in fr.labels:
                extent = lab.longest_px * m_per_px
                if extent > max_ground_m:
                    out.append({
                        "clip_id": run.clip_id, "frame": fr.stem, "actor_id": lab.actor_id, "name": lab.name,
                        "bbox_px": list(lab.bbox_px), "longest_px": lab.longest_px,
                        "ground_m": round(extent, 2), "gsd_cm_px": gsd, "agl_m": fr.agl_m,
                        "gsd_source": "frame telemetry" if fr.gsd_cm_px > 0 else "run median (no telemetry row)",
                        "basis": f"a single {lab.cls} may not span more than {max_ground_m} m of ground",
                    })
    return out


def plausible_labels(fr: CaptureFrame, *, gsd_cm_px: float = 0.0,
                     max_ground_m: float = MAX_TARGET_GROUND_M) -> list[CaptureLabel]:
    """The subset of a frame's labels that could be a person. Use this when a broken capture must still be
    looked at -- the mask-leak boxes are not ground truth and must not be drawn or scored as if they were."""
    g = fr.gsd_cm_px if fr.gsd_cm_px > 0 else gsd_cm_px
    if g <= 0:
        return list(fr.labels)
    return [m for m in fr.labels if m.longest_px * g / 100.0 <= max_ground_m]


def assert_boxes_are_plausible(runs: Sequence[CaptureRun], *, max_ground_m: float = MAX_TARGET_GROUND_M) -> None:
    """Raise, with the worst offenders named, if any label is not a person-sized thing. DO NOT TRAIN past this."""
    bad = implausible_boxes(runs, max_ground_m=max_ground_m)
    if not bad:
        return
    worst = sorted(bad, key=lambda b: -b["ground_m"])[:5]
    lines = "\n".join(f"  {b['frame']} {b['name']} {b['longest_px']:.0f} px = {b['ground_m']} m of ground"
                      for b in worst)
    raise ValueError(
        f"{len(bad)} labelled box(es) are larger than {max_ground_m} m on the ground and cannot be people. "
        f"The usual cause is a segmentation instance colour leaking onto the terrain or water plane. "
        f"DO NOT TRAIN ON THIS DATASET.\nWorst offenders:\n{lines}"
    )


# --- 3. tiling the labels -----------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TiledBox:
    """One training target inside one tile. `bbox_px` is TILE-LOCAL continuous xyxy."""

    cls_id: int
    bbox_px: tuple[float, float, float, float]
    visible_frac: float
    truncated: bool
    source: CaptureLabel

    @property
    def longest_px(self) -> float:
        return max(self.bbox_px[2] - self.bbox_px[0], self.bbox_px[3] - self.bbox_px[1])


def clip_box_to_tile(bbox_px: Sequence[float], tile: Tile) -> tuple[tuple[float, float, float, float], float]:
    """Clip a full-frame box to a tile and return (tile-local xyxy, fraction of the original area kept)."""
    x1 = max(float(bbox_px[0]), float(tile.x0))
    y1 = max(float(bbox_px[1]), float(tile.y0))
    x2 = min(float(bbox_px[2]), float(tile.x1))
    y2 = min(float(bbox_px[3]), float(tile.y1))
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0, 0.0), 0.0
    area = (float(bbox_px[2]) - float(bbox_px[0])) * (float(bbox_px[3]) - float(bbox_px[1]))
    frac = ((x2 - x1) * (y2 - y1) / area) if area > 0 else 0.0
    return (x1 - tile.x0, y1 - tile.y0, x2 - tile.x0, y2 - tile.y0), frac


def tile_boxes(
    labels: Sequence[CaptureLabel],
    tile: Tile,
    *,
    min_visible_frac: float = MIN_VISIBLE_FRAC,
    min_box_px: float = MIN_BOX_PX,
    classes: Sequence[str] = CLASSES,
) -> tuple[list[TiledBox], list[CaptureLabel]]:
    """Targets for one tile, plus the labels that were *dropped* as fragments.

    The second list is the honest half: a fragment that fails the visible-fraction test is not a background
    region either, so the caller may want to skip the tile rather than teach the model that a visible arm is
    background. :func:`build_yolo_dataset` does exactly that.
    """
    cls_id = {c: i for i, c in enumerate(classes)}
    keep: list[TiledBox] = []
    dropped: list[CaptureLabel] = []
    for m in labels:
        if m.cls not in cls_id:
            continue
        if intersection_area(m.bbox_px, tile.rect) <= 0.0:
            continue
        local, frac = clip_box_to_tile(m.bbox_px, tile)
        longest = max(local[2] - local[0], local[3] - local[1])
        if frac < min_visible_frac or longest < min_box_px:
            dropped.append(m)
            continue
        keep.append(TiledBox(cls_id[m.cls], local, frac, frac < 0.999, m))
    return keep, dropped


def yolo_line(box: TiledBox, tile_w: int, tile_h: int) -> str:
    """`cls cx cy w h`, normalised to the tile. Six decimals matches `tools/capture/labels.to_yolo`."""
    x1, y1, x2, y2 = box.bbox_px
    cx, cy = (x1 + x2) / 2.0 / tile_w, (y1 + y2) / 2.0 / tile_h
    bw, bh = (x2 - x1) / tile_w, (y2 - y1) / tile_h
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
        raise ValueError(f"box {box.bbox_px} is not inside a {tile_w}x{tile_h} tile")
    return f"{box.cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# --- 4. writing the Ultralytics dataset ---------------------------------------------------------------------
def build_yolo_dataset(
    splits: dict[str, Sequence[CaptureRun]],
    out_dir: str | Path,
    *,
    tile_px: tuple[int, int] = DEFAULT_TILE_PX,
    overlap: float = DEFAULT_OVERLAP,
    negative_frac: float = 0.05,
    min_visible_frac: float = MIN_VISIBLE_FRAC,
    min_box_px: float = MIN_BOX_PX,
    rng_seed: int = 0,
    write_images: bool = True,
    dry_run: bool = False,
    max_ground_m: float = MAX_TARGET_GROUND_M,
    allow_implausible_boxes: bool = False,
) -> dict[str, Any]:
    """Cut every frame into native-resolution tiles and write an Ultralytics dataset. Returns the manifest.

    Tile selection, in order of precedence:

    * a tile with at least one kept target is written;
    * a tile whose only overlap with a target is a rejected fragment is **skipped** (it is neither a positive nor
      an honest negative);
    * a genuinely empty tile is written with probability ``negative_frac`` -- background images are worth a few
      per cent of the set and are worthless above that.

    ``dry_run=True`` computes the whole manifest, including per-tile counts, without touching the disk: that is
    what `tests/test_detect.py` runs, and what a caller should run before committing to an hour of PNG writing.
    """
    assert_no_seed_leak(splits)
    all_runs = [r for runs in splits.values() for r in runs]
    if not allow_implausible_boxes:
        assert_boxes_are_plausible(all_runs, max_ground_m=max_ground_m)
    out = Path(out_dir)
    rng = random.Random(rng_seed)
    grid_cache: dict[tuple[int, int], TileGrid] = {}
    manifest: dict[str, Any] = {
        "product": "sightline.detect.dataset",
        "domain": "sim",
        "format": "ultralytics",
        "classes": list(CLASSES),
        "tile_px": list(tile_px),
        "overlap": overlap,
        "min_visible_frac": min_visible_frac,
        "min_box_px": min_box_px,
        "negative_frac": negative_frac,
        "rng_seed": rng_seed,
        "max_target_ground_m": max_ground_m,
        "implausible_boxes_allowed": allow_implausible_boxes,
        "split_rule": "by scenario seed (SOLUTION_DOC 5.5c step 3); never by frame",
        "splits": {},
        "root": str(out),
    }

    cv2 = None
    if write_images and not dry_run:
        import cv2 as _cv2

        cv2 = _cv2

    for split, runs in splits.items():
        if not runs:
            continue
        img_dir, lab_dir = out / "images" / split, out / "labels" / split
        if not dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)
        n_tiles = n_pos = n_neg = n_boxes = n_dropped = n_skipped = 0
        n_uncertain = 0
        seeds: set[int] = set()
        clips: set[str] = set()
        sizes: list[float] = []
        for run in runs:
            seeds.add(run.scenario_seed)
            clips.add(run.clip_id)
            for fr in run.frames:
                key = (fr.width_px, fr.height_px)
                if key not in grid_cache:
                    grid_cache[key] = TileGrid.build(fr.width_px, fr.height_px, tile_px, overlap)
                grid = grid_cache[key]
                scored = [m for m in fr.labels if m.aerially_detectable]
                n_uncertain += len(fr.labels) - len(scored)
                image = None
                for tile in grid:
                    keep, dropped = tile_boxes(scored, tile, min_visible_frac=min_visible_frac,
                                               min_box_px=min_box_px)
                    if not keep:
                        if dropped:
                            n_skipped += 1
                            n_dropped += len(dropped)
                            continue
                        if rng.random() >= negative_frac:
                            continue
                        n_neg += 1
                    else:
                        n_pos += 1
                        n_boxes += len(keep)
                        n_dropped += len(dropped)
                        sizes.extend(b.longest_px for b in keep)
                    n_tiles += 1
                    stem = f"{fr.stem}_t{tile.idx:02d}"
                    lines = [yolo_line(b, tile.width_px, tile.height_px) for b in keep]
                    if dry_run:
                        continue
                    (lab_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                         encoding="utf-8")
                    if write_images and cv2 is not None:
                        if image is None:
                            image = cv2.imread(str(fr.image_path))
                            if image is None:
                                raise FileNotFoundError(f"could not read {fr.image_path}")
                        cv2.imwrite(str(img_dir / f"{stem}.png"), tile.crop(image))
                image = None  # release the 4K frame before the next one
        manifest["splits"][split] = {
            "runs": len(runs), "seeds": sorted(seeds), "clips": sorted(clips),
            "frames": sum(len(r.frames) for r in runs),
            "tiles": n_tiles, "positive_tiles": n_pos, "background_tiles": n_neg,
            "skipped_fragment_tiles": n_skipped,
            "boxes": n_boxes, "dropped_fragments": n_dropped,
            "uncertain_boxes_excluded": n_uncertain,
            "median_box_px": (sorted(sizes)[len(sizes) // 2] if sizes else 0.0),
            "min_box_px_seen": (min(sizes) if sizes else 0.0),
            "max_box_px_seen": (max(sizes) if sizes else 0.0),
            "images": str(out / "images" / split),
            "labels": str(out / "labels" / split),
        }

    yaml_text = _data_yaml(out, manifest)
    manifest["data_yaml"] = str(out / "data.yaml")
    if not dry_run:
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["data_yaml_text"] = yaml_text
    return manifest


def _data_yaml(out: Path, manifest: dict[str, Any]) -> str:
    """Hand-written so no yaml dependency is needed to *produce* the file Ultralytics reads."""
    lines = [
        "# Written by sightline.detect.dataset (SOLUTION_DOC 5.5c). domain: sim, randomisation: off.",
        f"path: {out.as_posix()}",
    ]
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        if split in manifest["splits"]:
            lines.append(f"{key}: images/{split}")
    lines.append("names:")
    for i, c in enumerate(CLASSES):
        lines.append(f"  {i}: {c}")
    return "\n".join(lines) + "\n"


# --- 5. the same run, as evaluation ground truth --------------------------------------------------------------
def urgency_for(pose: str, submersion: str, cls: str = "human") -> str:
    """§5.8 urgency class from the simulator's own attributes. A derived mapping, not a measurement."""
    if cls == "animal":
        return "animal"
    if submersion in ("half", "head_only", "partial") or pose == "half_submerged":
        return "immersed"
    if pose == "trapped":
        return "trapped"
    return "stranded"


def to_eval_dataset(
    run: CaptureRun,
    *,
    split: str = "test",
    fps_processed: float = 5.0,
    actors_json: str | Path | None = None,
    scene_json: str | Path | None = None,
):
    """Build the eval lane's `EvalDataset` from a capture run. Ground truth in, `MetricRow`s out.

    `aerially_detectable = False` boxes become **`uncertain`** GtBoxes (§6.3): a detection on one is dropped
    rather than counted as a false positive, and missing one is not a miss -- which is the only treatment
    consistent with §2.7's "aerial search cannot find them" and with guardrail R10.
    """
    from sightline.eval.groundtruth import EvalDataset, GtBox, GtFrame, GtSurvivor

    frames: list[GtFrame] = []
    for fr in run.frames:
        boxes: list[GtBox] = []
        for m in fr.labels:
            x1, y1, x2, y2 = m.bbox_px
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append(GtBox(
                bbox_px=(x1, y1, x2, y2), gt_id=m.actor_id,
                cls=("animal" if m.cls == "animal" else "human"),
                frame_idx=fr.frame_idx, occlusion=m.occlusion,
                posture=m.pose if m.pose in _POSTURES else "unknown",
                submersion=m.submersion if m.submersion in _SUBMERSIONS else "unknown",
                truncated=(x1 <= 0.0 or y1 <= 0.0 or x2 >= fr.width_px or y2 >= fr.height_px),
                uncertain=not m.aerially_detectable,
                is_group=False, group_count=1,
                context=_context_for(m.zone, m.submersion),
            ))
        frames.append(GtFrame(
            frame_idx=fr.frame_idx, boxes=boxes, t_utc=fr.t_utc, clip_id=fr.clip_id,
            zone=_zone_of(fr), agl_m=fr.agl_m, time_of_day=fr.time_of_day, modality="rgb",
            gimbal_pitch_deg=fr.gimbal_pitch_deg, gsd_cm_px=fr.gsd_cm_px,
            image_path=str(fr.image_path), width_px=fr.width_px, height_px=fr.height_px,
        ))

    survivors: list[GtSurvivor] = []
    a_path = Path(actors_json) if actors_json else _repo() / "data" / "scene" / "actors.json"
    s_path = Path(scene_json) if scene_json else _repo() / "data" / "scene" / "flood_valley.json"
    if a_path.exists() and s_path.exists():
        survivors = _survivors_from_scene(a_path, s_path)

    return EvalDataset(
        domain="sim", frames=frames, survivors=survivors, fps_processed=fps_processed,
        clip_id=run.clip_id, split=split, seed_group=run.seed_group,  # type: ignore[arg-type]
        randomisation=bool(run.card.get("randomisation", "off") not in ("off", False, "false")),
        notes=f"capture run {run.root.name}; {run.describe()}",
    )


_POSTURES = ("standing", "sitting", "prone", "supine", "half_submerged", "trapped", "waving", "unknown")
_SUBMERSIONS = ("dry", "wet", "partial", "half", "head_only", "unknown")


def _repo() -> Path:
    return Path(__file__).resolve().parents[2]


def _zone_of(fr: CaptureFrame) -> str:
    """Frame zone = the zone of what is in it. With no boxes there is nothing to attribute it to."""
    zones = {m.zone for m in fr.labels if m.zone in ("fan", "settlement", "channel", "hillslope")}
    return zones.pop() if len(zones) == 1 else "unknown"


def _context_for(zone: str, submersion: str) -> str:
    if submersion in ("partial", "half", "head_only"):
        return "water"
    return {"settlement": "structure", "fan": "debris", "channel": "water", "hillslope": "vegetation"}.get(
        zone, "")


def _survivors_from_scene(actors_json: Path, scene_json: Path) -> list:
    from sightline.common.geodesy import offset_ne
    from sightline.eval.groundtruth import GtSurvivor

    actors = json.loads(actors_json.read_text(encoding="utf-8"))["actors"]
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    c = scene["map_centre_geopoint"]
    out = []
    for a in actors:
        lat, lon = offset_ne(float(c["lat"]), float(c["lon"]), float(a["north_m"]), float(a["east_m"]))
        out.append(GtSurvivor(
            gt_id=int(a["id"]), lat=lat, lon=lon, alt_msl_m=float(a.get("base_asl_m", 0.0)),
            cls=("animal" if a.get("cls") == "animal" else "human"),
            count=1,
            urgency_class=urgency_for(str(a.get("pose", "unknown")), str(a.get("submersion", "unknown")),
                                      str(a.get("cls", "human"))),  # type: ignore[arg-type]
            posture=(a.get("pose") if a.get("pose") in _POSTURES else "unknown"),  # type: ignore[arg-type]
            submersion=(a.get("submersion") if a.get("submersion") in _SUBMERSIONS
                        else "unknown"),  # type: ignore[arg-type]
            zone=a.get("zone", "unknown"),  # type: ignore[arg-type]
            buried=not bool(a.get("aerially_detectable", True)),
        ))
    return out


def summarise_boxes(runs: Sequence[CaptureRun]) -> dict[str, Any]:
    """Pixel-size and attribute census; the numbers a training decision should be made on, not on a feeling."""
    sizes: list[float] = []
    by_pose: dict[str, int] = {}
    by_sub: dict[str, int] = {}
    by_occ: dict[str, int] = {}
    for r in runs:
        for f in r.frames:
            for m in f.scored_labels:
                sizes.append(m.longest_px)
                by_pose[m.pose] = by_pose.get(m.pose, 0) + 1
                by_sub[m.submersion] = by_sub.get(m.submersion, 0) + 1
                by_occ[str(m.occlusion)] = by_occ.get(str(m.occlusion), 0) + 1
    sizes.sort()

    def q(p: float) -> float:
        return sizes[min(len(sizes) - 1, max(0, math.floor(p * (len(sizes) - 1))))] if sizes else 0.0

    return {
        "n_boxes": len(sizes), "min_px": q(0.0), "p10_px": q(0.10), "median_px": q(0.5),
        "p90_px": q(0.90), "max_px": q(1.0),
        "below_20_px": sum(1 for s in sizes if s < 20), "posture": by_pose, "submersion": by_sub,
        "occlusion": by_occ, "domain": "sim",
    }
