"""F10: the crop verifier and the posture / submersion head (SOLUTION_DOC §5.5a).

One small model on candidate crops, four outputs: ``is_real``, ``posture``, ``submersion``, ``occlusion``. §5.5a
promotes it from stretch to MVP because it buys two things in one forward pass -- precision recovery at the low
operating threshold §5.5c step 4 asks for, and the attribute evidence the §5.8 triage ordering needs.

**The safety rule is implemented here, structurally, not left to the triage lane.** §5.5a:

    "Posture may raise urgency, never lower it."

    1. If ``posture = unknown`` or the head's confidence is below a threshold, the triage score uses the **base**
       weight for a stranded survivor. The unknown case is never the cheapest case.
    2. Posture can promote a record but can never push it below the base rank.
    3. The predicted class is displayed with its confidence, so a commander can overrule it.

:func:`apply_verifier_output` enforces 1 and 2 by construction: a prediction whose urgency rank is **below** the
base is discarded and the detection keeps ``posture = "unknown"``, with the rejected prediction preserved in the
returned :class:`VerifierDecision` so rule 3 (show it, with its confidence) is still possible. There is no code
path in this module through which a low-confidence or low-urgency posture can lower a detection's standing.

Suppression is likewise never silent: :func:`suppress_false_positives` **returns both lists**. Guardrail R10
applies to records, but the same discipline is applied to detections -- nothing this lane drops disappears
without the caller being handed it.

The feature extractor (DINOv2/DINOv3 ViT-S + four linear heads) is behind a lazy import in
:class:`CropVerifier`; everything above it -- cropping, the safety rule, calibration, the batching -- is numpy
and runs on CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from sightline.schemas import OCCLUSION_BINS, POSTURES, SUBMERSIONS, Detection

__all__ = [
    "BASE_URGENCY",
    "CROP_PAD_FRAC",
    "DEFAULT_MIN_CONF",
    "DEFAULT_MIN_IS_REAL",
    "POSTURE_URGENCY",
    "SUBMERSION_URGENCY",
    "URGENCY_RANK",
    "CropVerifier",
    "VerifierDecision",
    "VerifierOutput",
    "apply_batch",
    "apply_verifier_output",
    "crop_for",
    "crops_for",
    "suppress_false_positives",
    "urgency_of",
]

#: §5.8 urgency ordering, mirrored from `sightline.eval.groundtruth.URGENCY_ORDER` (which this lane may read but
#: may not import into a hot path, since eval is an offline module).
URGENCY_RANK: dict[str, int] = {"immersed": 4, "trapped": 3, "stranded": 2, "animal": 1, "unknown": 0}
#: §5.5a rule 1: the unknown case is never the cheapest case -- it sits at the *stranded* base, not below it.
BASE_URGENCY = "stranded"

POSTURE_URGENCY: dict[str, str] = {
    "half_submerged": "immersed",
    "trapped": "trapped",
    "standing": BASE_URGENCY, "sitting": BASE_URGENCY, "prone": BASE_URGENCY, "supine": BASE_URGENCY,
    "waving": BASE_URGENCY, "unknown": BASE_URGENCY,
}
SUBMERSION_URGENCY: dict[str, str] = {
    "head_only": "immersed", "half": "immersed", "partial": "immersed",
    "wet": BASE_URGENCY, "dry": BASE_URGENCY, "unknown": BASE_URGENCY,
}

#: Below this the head's answer is treated as "unknown" (§5.5a rule 1).
DEFAULT_MIN_CONF = 0.5
#: Below this `is_real` a candidate is a suppression *candidate*; the caller decides, and gets both lists.
DEFAULT_MIN_IS_REAL = 0.35
#: Crops are padded so the model sees context (a body against water vs against a roof). 40 % of the longest side.
CROP_PAD_FRAC = 0.4


@dataclass(frozen=True, slots=True)
class VerifierOutput:
    """One crop's four predictions. Confidences are per-head maxima, already softmaxed."""

    is_real: float
    posture: str = "unknown"
    posture_conf: float = 0.0
    submersion: str = "unknown"
    submersion_conf: float = 0.0
    occlusion: int | None = None
    occlusion_conf: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.is_real <= 1.0:
            raise ValueError(f"is_real must be a probability, got {self.is_real}")
        if self.posture not in POSTURES:
            raise ValueError(f"posture {self.posture!r} not in POSTURES")
        if self.submersion not in SUBMERSIONS:
            raise ValueError(f"submersion {self.submersion!r} not in SUBMERSIONS")
        if self.occlusion is not None and self.occlusion not in OCCLUSION_BINS:
            raise ValueError(f"occlusion {self.occlusion!r} not in OCCLUSION_BINS")


@dataclass(frozen=True, slots=True)
class VerifierDecision:
    """What the safety rule did, kept so the UI can show an overruled prediction (§5.5a rule 3)."""

    detection: Detection
    output: VerifierOutput
    applied_posture: str
    applied_submersion: str
    base_urgency: str
    final_urgency: str
    rejected_reason: str = ""

    @property
    def promoted(self) -> bool:
        return URGENCY_RANK[self.final_urgency] > URGENCY_RANK[self.base_urgency]

    @property
    def demoted(self) -> bool:
        """Must ALWAYS be False. `tests/test_detect.py` asserts it over the whole prediction cross-product."""
        return URGENCY_RANK[self.final_urgency] < URGENCY_RANK[self.base_urgency]


# --- cropping ------------------------------------------------------------------------------------------------
def crop_for(frame: np.ndarray, det: Detection, *, pad_frac: float = CROP_PAD_FRAC,
             min_px: int = 16) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Padded, clamped crop around a detection, plus the rect it came from (in full-frame pixels).

    Returns a **square-ish** window: a 12 x 30 px standing body squashed into a square input loses the aspect
    that separates standing from prone, so the pad is computed on the longest side and applied to both axes.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = det.bbox_px
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(max(x2 - x1, y2 - y1) * (0.5 + pad_frac), min_px / 2.0)
    rx1 = int(max(0, np.floor(cx - half)))
    ry1 = int(max(0, np.floor(cy - half)))
    rx2 = int(min(w, np.ceil(cx + half)))
    ry2 = int(min(h, np.ceil(cy + half)))
    if rx2 <= rx1 or ry2 <= ry1:
        raise ValueError(f"detection {det.bbox_px} produced an empty crop in a {w}x{h} frame")
    return frame[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)


def crops_for(frame: np.ndarray, dets: Sequence[Detection], **kw: Any) -> list[np.ndarray]:
    return [crop_for(frame, d, **kw)[0] for d in dets]


# --- the safety rule -----------------------------------------------------------------------------------------
def urgency_of(posture: str, submersion: str) -> str:
    """The more urgent of the two attribute readings. Never below :data:`BASE_URGENCY`."""
    a = POSTURE_URGENCY.get(posture, BASE_URGENCY)
    b = SUBMERSION_URGENCY.get(submersion, BASE_URGENCY)
    best = a if URGENCY_RANK[a] >= URGENCY_RANK[b] else b
    return best if URGENCY_RANK[best] >= URGENCY_RANK[BASE_URGENCY] else BASE_URGENCY


def apply_verifier_output(
    det: Detection,
    out: VerifierOutput,
    *,
    min_conf: float = DEFAULT_MIN_CONF,
    base_urgency: str = BASE_URGENCY,
) -> VerifierDecision:
    """Fold one crop result into its `Detection`, under §5.5a's promote-only rule.

    Applied when, and only when, the head is confident enough AND the result does not lower the record's
    standing. Everything else is preserved on the decision for display, never written onto the detection.
    """
    if base_urgency not in URGENCY_RANK:
        raise ValueError(f"unknown base urgency {base_urgency!r}")
    posture, submersion = "unknown", "unknown"
    reason = ""
    if out.posture_conf >= min_conf and out.posture in POSTURES:
        posture = out.posture
    elif out.posture != "unknown":
        reason = f"posture {out.posture!r} at {out.posture_conf:.2f} below min_conf {min_conf:.2f}"
    if out.submersion_conf >= min_conf and out.submersion in SUBMERSIONS:
        submersion = out.submersion
    elif out.submersion != "unknown" and not reason:
        reason = f"submersion {out.submersion!r} at {out.submersion_conf:.2f} below min_conf {min_conf:.2f}"

    urgency = urgency_of(posture, submersion)
    if URGENCY_RANK[urgency] < URGENCY_RANK[base_urgency]:
        # Unreachable through `urgency_of` today; kept as a hard stop so a future vocabulary change cannot
        # introduce a demotion silently. §5.5a rule 2.
        posture, submersion, urgency = "unknown", "unknown", base_urgency
        reason = "prediction would have demoted the record; discarded (SOLUTION_DOC 5.5a rule 2)"

    updated = replace(
        det,
        is_real=float(out.is_real),
        posture=posture,  # type: ignore[arg-type]
        posture_conf=float(out.posture_conf if posture != "unknown" else 0.0),
        submersion=submersion,  # type: ignore[arg-type]
        submersion_conf=float(out.submersion_conf if submersion != "unknown" else 0.0),
        occlusion=(out.occlusion if out.occlusion_conf >= min_conf else det.occlusion),
    )
    return VerifierDecision(updated, out, posture, submersion, base_urgency, urgency, reason)


def apply_batch(dets: Sequence[Detection], outs: Sequence[VerifierOutput], **kw: Any) -> list[VerifierDecision]:
    if len(dets) != len(outs):
        raise ValueError(f"{len(dets)} detections but {len(outs)} verifier outputs")
    return [apply_verifier_output(d, o, **kw) for d, o in zip(dets, outs)]


def suppress_false_positives(
    dets: Sequence[Detection],
    *,
    min_is_real: float = DEFAULT_MIN_IS_REAL,
) -> tuple[list[Detection], list[Detection]]:
    """Split into (kept, suppressed). **Both lists are returned**: nothing vanishes without the caller seeing it.

    A detection the verifier never saw (``is_real is None``) is always kept: absence of evidence is not evidence.
    """
    kept: list[Detection] = []
    dropped: list[Detection] = []
    for d in dets:
        (dropped if (d.is_real is not None and d.is_real < min_is_real) else kept).append(d)
    return kept, dropped


# --- the model -------------------------------------------------------------------------------------------------
class CropVerifier:
    """DINOv2/DINOv3 ViT-S features + four linear heads (§5.5a). **Imports torch; needs a free GPU.**

    STUB STATUS: the *inference* path is written and the safety rule around it is complete and tested, but no
    head weights exist yet -- they are trained from simulator crops once the F8b dataset is built, which is
    minutes of GPU time. Constructing this class without a `heads` checkpoint raises rather than returning
    uniform guesses, so a caller can never mistake an untrained head for a trained one.
    """

    def __init__(self, heads_path: str | Path, *, backbone: str = "dinov2_vits14", device: str = "cuda:0",
                 image_size: int = 98):
        p = Path(heads_path)
        if not p.exists():
            raise FileNotFoundError(
                f"verifier head weights not found: {p}. F10 is not trained yet; run the head training on "
                "simulator crops first. Do not substitute an untrained head (docs/QUALITY_GATE.md)."
            )
        import torch

        self.device = device
        self.image_size = image_size
        self.backbone_name = backbone
        state = torch.load(str(p), map_location="cpu")
        self.heads = state["heads"]
        self.classes: dict[str, list[str]] = state["classes"]
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)  # cached under TORCH_HOME (D:)
        self.backbone.eval().to(device)

    def __call__(self, crops: Sequence[np.ndarray]) -> list[VerifierOutput]:  # pragma: no cover - needs GPU
        import torch
        import torch.nn.functional as F

        if not crops:
            return []
        import cv2

        batch = np.stack([cv2.resize(c, (self.image_size, self.image_size)) for c in crops])
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0).to(self.device)
        with torch.inference_mode():
            f = self.backbone(x)
            probs = {k: F.softmax(f @ w.to(self.device), dim=-1).cpu().numpy() for k, w in self.heads.items()}
        out: list[VerifierOutput] = []
        for i in range(len(crops)):
            po = probs["posture"][i]
            su = probs["submersion"][i]
            oc = probs["occlusion"][i]
            out.append(VerifierOutput(
                is_real=float(probs["is_real"][i][1]),
                posture=self.classes["posture"][int(po.argmax())], posture_conf=float(po.max()),
                submersion=self.classes["submersion"][int(su.argmax())], submersion_conf=float(su.max()),
                occlusion=int(oc.argmax()), occlusion_conf=float(oc.max()),
            ))
        return out


#: A verifier is any callable ``list[np.ndarray] -> list[VerifierOutput]``; `CropVerifier` is one implementation
#: and a hand-written function is another (that is what `tests/test_detect.py` uses).
VerifierFn = Callable[[Sequence[np.ndarray]], list[VerifierOutput]]
