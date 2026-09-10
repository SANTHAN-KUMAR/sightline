"""Camera-motion compensation with an observable source (SOLUTION_DOC §5.6 rule 4).

    "Run `sparseOptFlow` camera-motion compensation on a 640-px-wide downscale; ORB/ECC on 4K is far too slow.
     Rippling water breaks optical-flow compensation, so fall back to a *telemetry-predicted* homography (the
     previous frame's boxes warped analytically from attitude and GPS change)."

Two things make this more than a config line.

**Detecting the failure.** Water does not starve optical flow of correspondences — it supplies plenty, all of
them moving with the ripples instead of with the ground. So an inlier count cannot see the failure and neither
can RANSAC: the wrong answer is self-consistent. What does see it is the second opinion the drone already has,
its own attitude and GPS. `SightlineCMC` computes both transforms every frame and compares where they send the
frame corners; past `cmc_max_disagreement_px` the telemetry answer wins. When telemetry is unavailable, only the
degenerate failures (no correspondences at all) are catchable and that is stated in the log, not glossed over.

**Making the switch observable.** Every frame appends a `CmcResult` saying which source was used and why, so the
evaluation lane can attribute an identity failure to the compensation rather than to the tracker. `source_counts`
is the one-line summary.

The class subclasses `trackers.utils.cmc.CMC` so it can be dropped into `BoTSORTTracker.cmc` and called at
exactly the point in the update where BoT-SORT warps its Kalman states — after `predict()`, before association.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from trackers.utils.cmc import CMC, CMCConfig

from sightline.schemas import Intrinsics, Telemetry
from sightline.track.config import TrackerConfig
from sightline.track.geometry import affine_disagreement_px, telemetry_affine

CmcSource = Literal["optical_flow", "telemetry_homography", "identity"]

_IDENTITY = np.eye(2, 3, dtype=np.float32)

#: Sentinel meaning "this frame has no image, use telemetry only".
#:
#: `BoTSORTTracker.update` skips compensation entirely when `frame is None`, which would silently disable CMC on
#: a metadata-only replay (§5.12's evaluation harness does exactly that) and on any run that chooses not to pay
#: for optical flow. The backend therefore hands this array through instead of `None`; `SightlineCMC.estimate`
#: recognises it **by identity** and treats it as no image at all. It is never read.
NO_FRAME: np.ndarray = np.zeros((2, 2, 3), dtype=np.uint8)
NO_FRAME.flags.writeable = False


@dataclass(slots=True)
class CmcResult:
    """One frame's compensation decision. `reason` is written to be read in a log, not parsed."""

    frame_idx: int
    source: CmcSource
    reason: str
    affine: np.ndarray = field(default_factory=lambda: _IDENTITY.copy())
    #: Corner disagreement between the optical-flow and telemetry estimates, px. NaN when only one was computed.
    disagreement_px: float = float("nan")
    #: Pixel motion the chosen transform implies at the frame corners (0 for identity). Useful as a sanity plot.
    motion_px: float = 0.0
    downscale: int = 1
    optflow_failed: bool = False
    telemetry_available: bool = False
    #: Worst-case error of reducing the telemetry homography to a 2x3 affine, px. NaN when no telemetry.
    telemetry_affine_residual_px: float = float("nan")
    elapsed_ms: float = 0.0

    def __str__(self) -> str:
        return (
            f"frame {self.frame_idx}: cmc={self.source} ({self.reason}) motion={self.motion_px:.1f}px "
            f"disagree={self.disagreement_px:.1f}px"
        )


class SightlineCMC(CMC):
    """`trackers` CMC with the §5.6 telemetry fallback and a per-frame decision log.

    Use it either as a plain estimator or by assignment into a `BoTSORTTracker`:

        cmc = SightlineCMC(cfg, intrinsics)
        tracker.cmc = cmc                       # tracker was built with enable_cmc=True
        cmc.set_context(frame_idx, prev_tel, cur_tel)
        tracker.update(detections, frame=bgr)   # calls cmc.estimate() at the right moment
    """

    def __init__(self, cfg: TrackerConfig, intrinsics: Intrinsics | None = None) -> None:
        # NOTE: `self.cfg` belongs to the base CMC (a CMCConfig); ours lives under a distinct name.
        self.track_cfg = cfg
        self.intrinsics = intrinsics
        width = intrinsics.width_px if intrinsics is not None else cfg.cmc_target_width_px
        downscale = cfg.cmc_downscale(width)
        super().__init__(CMCConfig(method=cfg.cmc_method, downscale=downscale))
        self.history: list[CmcResult] = []
        self.last: CmcResult | None = None
        self._frame_idx = -1
        self._prev_tel: Telemetry | None = None
        self._cur_tel: Telemetry | None = None

    # --- context ------------------------------------------------------------------------------------------
    def set_context(
        self,
        frame_idx: int,
        prev_tel: Telemetry | None,
        cur_tel: Telemetry | None,
        intrinsics: Intrinsics | None = None,
    ) -> None:
        """Tell the estimator which frame it is about to see. Call once per frame before `estimate`."""
        self._frame_idx = frame_idx
        self._prev_tel = prev_tel
        self._cur_tel = cur_tel
        if intrinsics is not None and intrinsics is not self.intrinsics:
            self.intrinsics = intrinsics
            self.downscale = max(1, self.track_cfg.cmc_downscale(intrinsics.width_px))

    # --- estimation ---------------------------------------------------------------------------------------
    def estimate(self, frame_bgr: np.ndarray | None, dets_xyxy: np.ndarray | None = None) -> np.ndarray:
        """Return the 2x3 affine mapping previous-frame pixels to current-frame pixels, and log the choice."""
        t0 = time.perf_counter()
        if not self.track_cfg.cmc_enabled:
            return self._record(CmcResult(self._frame_idx, "identity", "cmc disabled by config"), t0)

        tel_fit = None
        if self.track_cfg.cmc_telemetry_enabled and self._prev_tel is not None and self._cur_tel is not None:
            intr = self.intrinsics
            if intr is not None:
                tel_fit = telemetry_affine(self._prev_tel, self._cur_tel, intr)

        if frame_bgr is NO_FRAME:
            frame_bgr = None

        # Optical flow. The library returns identity both when it cannot estimate and when the camera did not
        # move, so `frames_failed` is read as the unambiguous failure signal.
        flow: np.ndarray | None = None
        optflow_failed = False
        bootstrapping = False
        if frame_bgr is not None:
            before = self.frames_failed
            bootstrapping = not self._initialized  # first frame after construction or reset: identity by design
            flow = np.asarray(super().estimate(frame_bgr, dets_xyxy), dtype=np.float32)
            optflow_failed = self.frames_failed > before

        width, height = self._frame_size(frame_bgr)
        base = CmcResult(
            frame_idx=self._frame_idx,
            source="identity",
            reason="",
            downscale=self.downscale,
            optflow_failed=optflow_failed,
            telemetry_available=tel_fit is not None,
            telemetry_affine_residual_px=(float("nan") if tel_fit is None else tel_fit.residual_px),
        )

        if flow is None:
            if tel_fit is not None:
                base.source, base.reason = "telemetry_homography", "no frame supplied; telemetry available"
                base.affine = tel_fit.affine
            else:
                base.reason = "no frame and no telemetry"
            return self._record(base, t0, width, height)

        if optflow_failed or bootstrapping:
            why = (
                "optical flow has no previous frame yet (bootstrap)"
                if bootstrapping
                else "optical flow found too few correspondences"
            )
            if tel_fit is not None:
                base.source = "telemetry_homography"
                base.reason = why
                base.affine = tel_fit.affine
            else:
                base.reason = f"{why}; no telemetry to fall back to"
            return self._record(base, t0, width, height)

        if tel_fit is None:
            base.source, base.reason, base.affine = "optical_flow", "no telemetry to cross-check against", flow
            return self._record(base, t0, width, height)

        disagreement = affine_disagreement_px(flow, tel_fit.affine, width, height)
        base.disagreement_px = disagreement
        if disagreement > self.track_cfg.cmc_max_disagreement_px:
            # The rippling-water case: plenty of correspondences, all of them wrong.
            base.source = "telemetry_homography"
            base.reason = (
                f"optical flow disagrees with telemetry by {disagreement:.1f} px "
                f"(> {self.track_cfg.cmc_max_disagreement_px:.1f})"
            )
            base.affine = tel_fit.affine
        else:
            base.source = "optical_flow"
            base.reason = f"agrees with telemetry to {disagreement:.1f} px"
            base.affine = flow
        return self._record(base, t0, width, height)

    # --- bookkeeping --------------------------------------------------------------------------------------
    def _frame_size(self, frame_bgr: np.ndarray | None) -> tuple[int, int]:
        if frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
            return int(w), int(h)
        if self.intrinsics is not None:
            return self.intrinsics.width_px, self.intrinsics.height_px
        return self.track_cfg.cmc_target_width_px, self.track_cfg.cmc_target_width_px

    def _record(self, res: CmcResult, t0: float, width: int | None = None, height: int | None = None) -> np.ndarray:
        if width is None or height is None:
            width, height = self._frame_size(None)
        res.motion_px = affine_disagreement_px(res.affine, _IDENTITY, width, height)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        res.downscale = self.downscale
        self.last = res
        if self.track_cfg.cmc_log and len(self.history) < self.track_cfg.cmc_log_max:
            self.history.append(res)
        return res.affine

    def source_counts(self) -> dict[str, int]:
        """How many frames each compensation source was used on — the §5.6 "make the switch observable" line."""
        counts: dict[str, int] = {"optical_flow": 0, "telemetry_homography": 0, "identity": 0}
        for r in self.history:
            counts[r.source] += 1
        return counts

    def reset(self) -> None:
        super().reset()
        self._prev_tel = None
        self._cur_tel = None
