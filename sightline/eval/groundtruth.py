"""Ground-truth containers for the evaluation harness (§5.12 held-out clip protocol, §6.3 label vocabulary).

`schemas.py` is frozen and describes the *pipeline's output*; it has no ground-truth type. These are the
evaluation lane's own input types, and they are deliberately a mirror image of `Detection` / `Record` so a
simulator export (which knows every attribute exactly) or a COCO file written to `docs/annotation_guideline.md`
maps onto them one-for-one.

Vocabulary note: `posture`, `submersion` and `occlusion` use exactly the `POSTURES`, `SUBMERSIONS` and
`OCCLUSION_BINS` of `schemas.py`. `uncertain`, `ignore`, `is_group`/`group_count` and `truncated` come from
§6.3 and change how a box is *scored*, which `matching.py` implements.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from sightline.schemas import (
    OCCLUSION_BINS,
    POSTURES,
    SUBMERSIONS,
    ClassName,
    Modality,
    Posture,
    Submersion,
    UrgencyClass,
    Zone,
)

#: §5.8: the triage ordering the ranked list is scored against. Higher = more urgent.
URGENCY_ORDER: dict[str, int] = {"immersed": 4, "trapped": 3, "stranded": 2, "animal": 1, "unknown": 0}


@dataclass(slots=True)
class GtBox:
    """One annotated instance in one frame. `bbox_px` is the **visible-extent** box (§6.3), xyxy, full frame."""

    bbox_px: tuple[float, float, float, float]
    gt_id: int = -1  # stable survivor/animal identity across frames and passes; -1 = not tracked
    cls: ClassName = "human"
    frame_idx: int = -1
    occlusion: int | None = None  # OCCLUSION_BINS: 0 = <25 %, 1 = 25-75 %, 2 = >75 % hidden
    posture: Posture = "unknown"
    submersion: Submersion = "unknown"
    visible_fraction: float | None = None
    bbox_amodal_px: tuple[float, float, float, float] | None = None
    truncated: bool = False
    uncertain: bool = False  # §6.3: neither positive nor negative; overlapping detections are dropped
    ignore: bool = False  # §6.3 ignore region: matched by intersection-over-detection
    is_group: bool = False  # §6.3 human_group; scored like COCO iscrowd = 1
    group_count: int = 1
    context: str = ""  # open_ground | water | debris | vegetation | structure | vehicle
    modality: Modality = "rgb"
    thermal_only: bool = False

    def __post_init__(self) -> None:
        if self.posture not in POSTURES:
            raise ValueError(f"posture {self.posture!r} not in POSTURES {POSTURES}")
        if self.submersion not in SUBMERSIONS:
            raise ValueError(f"submersion {self.submersion!r} not in SUBMERSIONS {SUBMERSIONS}")
        if self.occlusion is not None and self.occlusion not in OCCLUSION_BINS:
            raise ValueError(f"occlusion {self.occlusion!r} not in OCCLUSION_BINS {OCCLUSION_BINS}")
        x1, y1, x2, y2 = self.bbox_px
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"bbox_px must be xyxy with positive extent, got {self.bbox_px}")

    @property
    def width_px(self) -> float:
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height_px(self) -> float:
        return self.bbox_px[3] - self.bbox_px[1]

    @property
    def size_px(self) -> float:
        """Longest side — the axis the §5.12 pixel-size slices bin on (same rule as `Detection.size_px`)."""
        return max(self.width_px, self.height_px)

    @property
    def scored(self) -> bool:
        """True when this box is a recall target. `uncertain` and `ignore` boxes are neither TP nor FN."""
        return not (self.uncertain or self.ignore)


@dataclass(slots=True)
class GtFrame:
    """One processed frame of ground truth plus the frame-level slice attributes (§6.2 writes them for free)."""

    frame_idx: int
    boxes: list[GtBox] = field(default_factory=list)
    t_utc: float = 0.0
    clip_id: str = ""
    zone: Zone = "unknown"
    agl_m: float = 60.0
    time_of_day: str = "day"
    modality: Modality = "rgb"
    weather: dict[str, float] = field(default_factory=dict)
    gimbal_pitch_deg: float = -90.0
    gsd_cm_px: float = 0.0
    # for the FiftyOne error browser: where the rendered frame lives and how big it is (4K default, §5.5c)
    image_path: str = ""
    width_px: int = 3840
    height_px: int = 2160

    def __post_init__(self) -> None:
        for b in self.boxes:
            if b.frame_idx < 0:
                b.frame_idx = self.frame_idx


@dataclass(slots=True)
class GtSurvivor:
    """One living being on the ground: the record-level and geolocation-level truth (§5.6, §5.7)."""

    gt_id: int
    lat: float
    lon: float
    alt_msl_m: float = 0.0
    cls: ClassName = "human"
    count: int = 1  # people at this position (a rooftop group is one survivor entry with count > 1)
    urgency_class: UrgencyClass = "stranded"
    posture: Posture = "unknown"
    submersion: Submersion = "unknown"
    zone: Zone = "unknown"
    buried: bool = False  # §6.2: present in truth as NOT VISIBLE. Excluded from recall, reported separately.

    @property
    def findable(self) -> bool:
        """A buried actor cannot be found from the air. Counting it as a miss would understate a real recall;
        hiding it would overstate what the search cleared (R10). It is excluded and reported on its own row."""
        return not self.buried

    @property
    def urgency_value(self) -> int:
        return URGENCY_ORDER.get(str(self.urgency_class), 0)


@dataclass(slots=True)
class EvalDataset:
    """A held-out clip with its domain. §6.5: splits are by flight or scenario seed, NEVER by frame."""

    domain: Literal["sim", "real"]
    frames: list[GtFrame] = field(default_factory=list)
    survivors: list[GtSurvivor] = field(default_factory=list)
    fps_processed: float = 5.0  # §5.11: processed FPS, not capture FPS. FP/min divides by this.
    clip_id: str = ""
    split: Literal["train", "val", "test"] = "test"
    seed_group: str = ""  # scenario seed / flight id: the unit the split was made on (§6.5)
    randomisation: bool = False  # §5.5c: domain randomisation off by default; never compare across the switch
    notes: str = ""

    def __post_init__(self) -> None:
        if self.domain not in ("sim", "real"):
            raise ValueError(f"domain must be 'sim' or 'real' (hard rule 5), got {self.domain!r}")

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def minutes_processed(self) -> float:
        """§5.12 FP/min denominator: `frames_processed / fps_processed / 60`."""
        if self.fps_processed <= 0:
            raise ValueError("fps_processed must be > 0 to report FP/min")
        return self.n_frames / self.fps_processed / 60.0

    def frame_index(self) -> dict[int, GtFrame]:
        return {f.frame_idx: f for f in self.frames}

    def scored_boxes(self, cls: str | None = "human") -> list[GtBox]:
        return [b for f in self.frames for b in f.boxes if b.scored and (cls is None or b.cls == cls)]

    def findable_survivors(self, cls: str | None = "human") -> list[GtSurvivor]:
        return [s for s in self.survivors if s.findable and (cls is None or s.cls == cls)]

    def buried_survivors(self) -> list[GtSurvivor]:
        return [s for s in self.survivors if s.buried]


# --- serialisation -----------------------------------------------------------------------------------------
def dataset_to_json(ds: EvalDataset) -> dict[str, Any]:
    return {
        "domain": ds.domain,
        "clip_id": ds.clip_id,
        "split": ds.split,
        "seed_group": ds.seed_group,
        "fps_processed": ds.fps_processed,
        "randomisation": ds.randomisation,
        "notes": ds.notes,
        "frames": [{**asdict(f), "boxes": [asdict(b) for b in f.boxes]} for f in ds.frames],
        "survivors": [asdict(s) for s in ds.survivors],
    }


def dataset_from_json(d: dict[str, Any]) -> EvalDataset:
    frames = []
    for fd in d.get("frames", []):
        boxes = [GtBox(**{**bd, "bbox_px": tuple(bd["bbox_px"])}) for bd in fd.get("boxes", [])]
        frames.append(GtFrame(**{**fd, "boxes": boxes}))
    survivors = [GtSurvivor(**sd) for sd in d.get("survivors", [])]
    return EvalDataset(
        domain=d["domain"],
        frames=frames,
        survivors=survivors,
        fps_processed=float(d.get("fps_processed", 5.0)),
        clip_id=d.get("clip_id", ""),
        split=d.get("split", "test"),
        seed_group=d.get("seed_group", ""),
        randomisation=bool(d.get("randomisation", False)),
        notes=d.get("notes", ""),
    )


def save_dataset(ds: EvalDataset, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dataset_to_json(ds), indent=1), encoding="utf-8")
    return p


def load_dataset(path: str | Path) -> EvalDataset:
    return dataset_from_json(json.loads(Path(path).read_text(encoding="utf-8")))


def check_split_disjoint(datasets: Iterable[EvalDataset]) -> dict[str, list[str]]:
    """§6.5 leakage guard: the same scenario seed / flight must not appear in two splits.

    Returns {seed_group: [splits]} for every group that appears in more than one split. An empty dict is the
    only acceptable result before a number is published.
    """
    seen: dict[str, set[str]] = {}
    for ds in datasets:
        if not ds.seed_group:
            continue
        seen.setdefault(ds.seed_group, set()).add(ds.split)
    return {g: sorted(s) for g, s in seen.items() if len(s) > 1}


__all__ = [
    "URGENCY_ORDER", "EvalDataset", "GtBox", "GtFrame", "GtSurvivor", "check_split_disjoint", "dataset_from_json",
    "dataset_to_json", "load_dataset", "save_dataset",
]
