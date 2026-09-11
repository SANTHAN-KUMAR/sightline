"""The confirmation gate must be reachable at the frame rate the loop actually runs at.

SOLUTION_DOC section 5.6 rule 3 is "3 hits within 2 seconds". That is written against the ~5 FPS loop
section 5.11 sizes. The live loop is latency-bound and was MEASURED on 2026-09-11 at a median 1.368 s per
frame - 0.73 FPS - because a 4K capture plus tiled inference on a 4060 sharing VRAM with the editor costs
about 1.3 s. At 0.73 FPS three hits span 2.74 s, so the gate can never close, and a pipeline that was
detecting and geolocating correctly produced **zero records** for that reason alone.

`TrackerConfig` already refuses an unreachable gate (`(min_hits - 1) / fps > min_hits_window_s` raises), so
the bug was not that nobody had thought about it - it was that the window was pinned at the doc's 2 s while
the frame rate was a fraction of the doc's assumption.

The fix keeps the rule's intent ("seen repeatedly in quick succession") and scales the window to the frame
rate, never below the doc's 2 s. These tests pin both halves: it must be reachable, and it must not quietly
become more permissive than the doc at the frame rate the doc assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sightline.pipeline import FramePipeline  # noqa: E402
from sightline.track.config import TrackerConfig  # noqa: E402


def window_for(fps: float) -> float:
    return FramePipeline(detector="truth", tracker_fps=fps).track_window_s


def test_the_doc_rate_keeps_the_doc_window() -> None:
    """At the 5 FPS section 5.11 assumes, the gate must stay exactly as section 5.6 specifies."""
    assert window_for(5.0) == pytest.approx(2.0)


def test_a_slow_loop_gets_a_reachable_window() -> None:
    """0.73 FPS was measured. Three hits need 2.74 s there, so a 2 s window is unsatisfiable."""
    w = window_for(0.73)
    assert w > 2.74, f"window {w:.2f}s still cannot hold 3 hits at 0.73 FPS"
    TrackerConfig(fps=0.73, min_hits_window_s=w)   # __post_init__ validates; must not raise


def test_the_window_never_shrinks_below_the_doc() -> None:
    for fps in (0.5, 0.73, 1.0, 2.0, 5.0, 30.0):
        assert window_for(fps) >= 2.0, f"{fps} FPS produced a window under the doc's 2 s"


def test_an_explicit_window_is_honoured() -> None:
    assert FramePipeline(detector="truth", tracker_fps=1.0, track_window_s=7.5).track_window_s == 7.5


def test_the_gate_the_pipeline_builds_is_always_reachable() -> None:
    """The real guard: whatever frame rate it is handed, TrackerConfig must accept the pairing."""
    for fps in (0.4, 0.73, 1.0, 3.0, 5.0):
        TrackerConfig(fps=fps, min_hits_window_s=window_for(fps))


def test_an_unreachable_pairing_is_still_rejected() -> None:
    """Guard against the fix turning the validator into a no-op."""
    with pytest.raises(ValueError):
        TrackerConfig(fps=0.5, min_hits_window_s=2.0)
