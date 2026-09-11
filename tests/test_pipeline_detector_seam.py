"""The `--detector rgb` seam must actually be callable, and must apply the operating threshold.

WHY THIS FILE EXISTS. `pipeline.detections_from_model` called
``detect_frame(frame_png, weights=weights, conf=conf)`` while `detect/rgb.py` defined
``detect_frame(frame: np.ndarray, infer: TileInferencer, *, cfg, grid, frame_idx, tile_indices)``. Wrong first
argument, a missing required positional, and two keyword arguments that do not exist. Every call would have
raised `TypeError`.

It survived because **every caller, test and demo tool in the repo runs `--detector truth`**, so the one seam
between the finished detector and the finished pipeline was never executed. It would have failed on the first
frame of the first run with real weights - in the middle of the live demo.

These tests need no torch, no GPU and no weights: they substitute a fake detector at the cache seam, which is
the whole point - the contract can be checked without the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sightline import pipeline  # noqa: E402
from sightline.schemas import Detection  # noqa: E402


class FakeDetector:
    """Stands in for `RgbDetector`: same `.detect(frame)` surface, no torch."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[int, int]] = []

    def detect(self, frame: np.ndarray, **kw) -> list[Detection]:
        self.calls.append((frame.shape[1], frame.shape[0]))
        return [Detection(bbox_px=(10.0, 10.0, 30.0, 50.0), score=s, cls="human", modality="rgb")
                for s in self.scores]


@pytest.fixture
def fake(monkeypatch):
    det = FakeDetector([0.9, 0.4, 0.1])
    monkeypatch.setattr(pipeline, "_detector", lambda weights, device="cuda:0": det)
    return det


def test_the_array_path_is_callable_with_the_documented_arguments(fake) -> None:
    """The regression: this raised TypeError for the whole life of the file."""
    frame = np.zeros((2160, 3840, 3), np.uint8)
    out = pipeline.detections_from_model_array(frame, "best.pt", 0.0)
    assert len(out) == 3
    assert all(isinstance(d, Detection) for d in out)
    assert fake.calls == [(3840, 2160)], "the frame must reach the detector unmodified"


def test_the_operating_threshold_is_applied(fake) -> None:
    """SOLUTION_DOC 5.5c step 4: detect at a LOW threshold, then filter at the frozen operating point."""
    frame = np.zeros((64, 64, 3), np.uint8)
    assert len(pipeline.detections_from_model_array(frame, "best.pt", 0.0)) == 3
    assert len(pipeline.detections_from_model_array(frame, "best.pt", 0.5)) == 1
    assert len(pipeline.detections_from_model_array(frame, "best.pt", 0.95)) == 0


def test_the_threshold_is_inclusive_at_its_own_value(fake) -> None:
    """A frozen threshold of c must KEEP a detection scoring exactly c, or the reported recall is not the
    recall that was measured when the threshold was chosen."""
    frame = np.zeros((64, 64, 3), np.uint8)
    assert len(pipeline.detections_from_model_array(frame, "best.pt", 0.4)) == 2


def test_the_file_path_reads_the_frame_and_reuses_the_array_path(fake, tmp_path) -> None:
    import cv2

    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), np.full((120, 200, 3), 128, np.uint8))
    out = pipeline.detections_from_model(p, "best.pt", 0.5)
    assert len(out) == 1
    assert fake.calls == [(200, 120)]


def test_a_missing_frame_is_an_error_not_an_empty_result(fake, tmp_path) -> None:
    """Returning [] for an unreadable frame would read as 'no survivors here' - the worst possible failure."""
    with pytest.raises(FileNotFoundError):
        pipeline.detections_from_model(tmp_path / "nope.png", "best.pt", 0.5)


def test_the_detector_is_built_once_not_per_frame(monkeypatch) -> None:
    """Rebuilding per frame reloads the weights onto the GPU every ~1.2 s; the live loop would not survive it."""
    built: list[str] = []

    class Once:
        def detect(self, frame, **kw):
            return []

    def fake_build(cfg):
        built.append(cfg.weights)
        return Once()

    import sightline.detect.rgb as rgb

    monkeypatch.setattr(rgb, "RgbDetector", fake_build)
    pipeline._DETECTOR_CACHE.clear()
    frame = np.zeros((32, 32, 3), np.uint8)
    for _ in range(5):
        pipeline.detections_from_model_array(frame, "best.pt", 0.0)
    assert built == ["best.pt"], f"weights loaded {len(built)} times, expected once"
    pipeline._DETECTOR_CACHE.clear()
