"""Why does the tracker return 0 tracks from 136 detections? Print its own bookkeeping.

    uv run python tools/train/_track_probe.py _artifacts/dataset/seed47_alt55

`TrackerStats` counts every stage of association separately - detections in, detections associated, tracks
started, tracks confirmed, tracks pruned unconfirmed - so the break is identifiable rather than guessed:

  * detections_in > 0 but detections_associated == 0   -> the backend is not linking anything at all;
  * tracks_started > 0 but tracks_confirmed == 0       -> association works, the GATE never closes (density
                                                          or window);
  * tracks_pruned_unconfirmed > 0                      -> tracklets are dying before they can confirm.

`cmc_sources` shows which camera-motion source was used per frame, which is the other half of the answer:
an "unavailable" majority means the 454 px inter-frame shift is never being compensated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sightline.pipeline import FramePipeline  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "_artifacts/dataset/seed47_alt55")
out = Path("_artifacts/_track_probe_out")

fp = FramePipeline(detector="truth", tracker_fps=0.73, clip_id="probe", domain="sim")
res = fp.run(run, out) if hasattr(fp, "run") else None
if res is None:
    from sightline.pipeline import run as offline_run

    offline_run(run, out, detector="truth", tracker_fps=0.73)

trk = getattr(fp, "_tracker", None)
if trk is None:
    print("pipeline did not expose its tracker; reading the manifest instead")
    import json

    m = json.loads((out / "manifest.json").read_text())
    print(json.dumps({k: v for k, v in m.items() if "track" in k.lower()}, indent=1)[:1200])
else:
    d = trk.describe()
    print("=== tracker ===")
    for k in ("backend_name", "fps", "min_hits", "min_hits_window_s", "cmc_enabled",
              "track_high_thresh", "track_low_thresh", "match_thresh", "backend_lost_buffer_frames"):
        if k in d:
            print(f"  {k}: {d[k]}")
    print("=== cmc sources (which motion model was used per frame) ===")
    print(f"  {d.get('cmc_sources')}")
    print("=== stats: where the chain breaks ===")
    for k, v in (d.get("stats") or {}).items():
        print(f"  {k}: {v}")
