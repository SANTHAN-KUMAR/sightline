"""The operator's camera view: the frame the drone just took, with what the model found drawn on it.

The map answers "where are the survivors". It does not answer the question an operator asks first, which is
"what is the drone looking at RIGHT NOW, and is the model seeing what I am seeing?" Without that, a
detection is a number in a list and there is no way to tell a real find from a false positive without
opening a record.

So this renders one JPEG per frame - the downscaled frame with every detection boxed, scored and labelled -
written atomically to `<out>/live_view.jpg` with `<out>/live_view.json` beside it. The API serves both and
the dashboard refreshes them, so the panel is always the latest frame rather than a stream that can fall
behind the aircraft.

It is written for a live loop, so it is deliberately cheap:

* the frame is downscaled ONCE to `WIDTH_PX` before anything is drawn - annotating a 4K frame and scaling
  afterwards costs several times more for a picture nobody views at 4K;
* it never raises into the flight. A drawing bug must not ground an aircraft, so everything is wrapped and
  a failure is reported once and then skipped;
* the write is atomic (temp file then replace), because the dashboard polls this path and a half-written
  JPEG renders as a broken image.

Box colour carries the score, using the same red-to-green ramp the record cards use, so a weak detection
looks weak at a glance.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

WIDTH_PX = 960          # what the panel displays; a 4K frame is pointless here
JPEG_Q = 82
_WARNED = False


def _colour(score: float) -> tuple[int, int, int]:
    """BGR, red (weak) -> amber -> green (strong). Matches the record card's score ramp."""
    s = max(0.0, min(1.0, float(score)))
    if s < 0.5:
        t = s / 0.5
        return (0, int(90 + 165 * t), 255)                  # red -> amber
    t = (s - 0.5) / 0.5
    return (0, 255, int(255 * (1.0 - t)))                   # amber -> green


def write_live_view(out_dir: Path, rgb, detections: Sequence[Any], *, frame_idx: int,
                    agl_m: float = 0.0, mode: str = "", n_records: int = 0,
                    detect_ms: float = 0.0, detector: str = "") -> bool:
    """Render one operator frame. Returns True if it was written. NEVER raises into the flight loop."""
    global _WARNED
    try:
        import cv2
        import numpy as np

        if rgb is None:
            return False
        h, w = rgb.shape[:2]
        scale = WIDTH_PX / float(w)
        im = cv2.resize(rgb[:, :, ::-1], (WIDTH_PX, max(1, int(round(h * scale)))))  # RGB->BGR, then shrink

        shown = []
        for d in detections or ():
            try:
                x1, y1, x2, y2 = (float(v) * scale for v in d.bbox_px)
                score = float(getattr(d, "score", 0.0))
            except Exception:                                # noqa: BLE001
                continue
            col = _colour(score)
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            label = f"{getattr(d, 'cls', 'human')} {score:.2f}"
            post = getattr(d, "posture", None)
            if post and post != "unknown":
                label += f" · {post}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty = max(int(y1) - 4, th + 4)
            cv2.rectangle(im, (int(x1), ty - th - 4), (int(x1) + tw + 6, ty + 2), col, -1)
            cv2.putText(im, label, (int(x1) + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1,
                        cv2.LINE_AA)
            shown.append({"bbox": [round(v, 1) for v in (x1, y1, x2, y2)], "score": round(score, 3),
                          "cls": getattr(d, "cls", "human"), "posture": post or "unknown"})

        # a status strip, so the picture is self-describing when someone screenshots it
        strip = f"frame {frame_idx}  |  {len(shown)} detection(s)  |  {agl_m:.0f} m AGL  |  {mode}"
        if detect_ms:
            strip += f"  |  {detect_ms:.0f} ms detect"
        if detector:
            strip += f"  |  {detector}"
        strip += f"  |  {n_records} record(s)"
        cv2.rectangle(im, (0, 0), (im.shape[1], 24), (24, 24, 28), -1)
        cv2.putText(im, strip, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (225, 225, 230), 1, cv2.LINE_AA)

        out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_jpeg(out_dir / "live_view.jpg", im, cv2)
        _atomic_write_text(out_dir / "live_view.json", json.dumps({
            "frame_idx": frame_idx, "detections": shown, "agl_m": round(agl_m, 1), "mode": mode,
            "n_records": n_records, "detect_ms": round(detect_ms, 1), "detector": detector,
            "width_px": im.shape[1], "height_px": im.shape[0],
        }, indent=1))
        return True
    except Exception as exc:                                 # noqa: BLE001
        # ONE bad frame must not cost the camera for the rest of the flight. The message is printed once
        # so the log does not fill with it, but every later frame is still attempted: the common causes
        # (a file lock, a momentarily odd frame) are transient, and a panel that goes dark and stays dark
        # for a whole sortie is worse than the error it is reporting.
        if not _WARNED:
            _WARNED = True
            print(f"  live view: dropped a frame ({type(exc).__name__}: {exc}). "
                  f"Later frames will still be attempted; the flight is unaffected.")
        return False


#: Windows holds a share lock while anything reads the file, and the dashboard polls this path once a
#: second - per open dashboard, per server. `os.replace` onto a file somebody is mid-read of fails with
#: PermissionError (WinError 5). It is transient by nature: the reader is done in microseconds. Retrying
#: over a quarter of a second turns "the camera died for the rest of the flight" into a dropped frame.
_REPLACE_TRIES = 6
_REPLACE_BACKOFF_S = 0.04


def _replace_with_retry(tmp: str, path: Path) -> None:
    """Retry the rename, then give up leaving the temp file in place to be reused.

    The temp path is FIXED per destination (`live_view.jpg.tmp`), not unique per frame, specifically so a
    failure needs no cleanup: the next frame writes over the same file. A unique name per attempt would
    mean deleting the leftover, and delete-shaped code in this repo has to earn an R10 allowance - which
    is a lot of justification for a problem that a stable filename simply does not have.
    """
    last: Exception | None = None
    for i in range(_REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:                   # a reader holds it; it will let go
            last = exc
            time.sleep(_REPLACE_BACKOFF_S * (i + 1))
    raise last if last is not None else OSError("replace failed")


def _atomic_write_jpeg(path: Path, im, cv2) -> None:
    """Temp file then replace: the dashboard polls this path and must never see a half-written JPEG."""
    tmp = str(path) + ".tmp"
    cv2.imwrite(tmp, im, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
    _replace_with_retry(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    _replace_with_retry(tmp, path)
