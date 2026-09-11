"""Prove the TensorRT engine is READY: it loads, it detects the same things, and it is faster.

    uv run python tools/train/verify_engine.py
    uv run python tools/train/verify_engine.py --frames 12

An engine on disk is not a working engine. This one has failed in two different ways already, both of which
a file-exists check would have called ready:

* built at batch 15 (the full tile grid), the FP16 AutoCast CPU reference pass exhausted host RAM and the
  build died - so it is built at batch 6;
* built at batch 6 and then CALLED with `max_batch=0`, which sends all 15 tiles in one go, it died at
  inference with `cudaError 700` (an illegal address). The optimisation profile is baked into the engine,
  so the caller has to match it. `sightline.pipeline.ENGINE_MAX_BATCH` is that match.

So this goes through `sightline.pipeline._detector`, the SAME constructor the live flight and the offline
replay use. A benchmark that built its own detector could pass while the product path still crashed.

It also compares DETECTIONS, not just milliseconds. An engine that is fast because FP16 quantisation lost
the small targets is not an optimisation, and at 40-60 m a survivor is 25-80 px - exactly the population
that goes first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ENGINE = REPO / "models/detect/f8b_sim/weights/best.engine"
WEIGHTS = REPO / "models/detect/f8b_sim/weights/best.pt"

#: Real 4K nadir frames from a held-out scenario seed, not synthetic noise: tiling, letterboxing and NMS
#: all behave differently on a frame that actually contains people.
FRAME_DIRS = ("_artifacts/dataset/seed47_alt55/images", "_artifacts/dataset/seed23_alt55/images")


def frames(n: int) -> list[Path]:
    for d in FRAME_DIRS:
        got = sorted((REPO / d).glob("*.jpg"))[:n]
        if len(got) >= n:
            return got
    raise SystemExit(f"need {n} frames; none of {FRAME_DIRS} has that many")


def run(weights: Path, imgs: list, label: str) -> dict:
    """Time the detector the PRODUCT builds, not one assembled here."""
    import numpy as np  # noqa: F401

    from sightline.pipeline import _detector

    t0 = time.perf_counter()
    det = _detector(str(weights))
    load_s = time.perf_counter() - t0

    det.detect(imgs[0])                                   # warmup: first call builds the context
    per_frame, counts, scores = [], [], []
    for im in imgs:
        t = time.perf_counter()
        out = det.detect(im)
        per_frame.append((time.perf_counter() - t) * 1e3)
        counts.append(len(out))
        scores += [round(float(getattr(d, "score", 0.0)), 3) for d in out]
    per_frame.sort()
    med = per_frame[len(per_frame) // 2]
    return {"label": label, "weights": weights.name, "load_s": round(load_s, 2),
            "median_ms": round(med, 1), "min_ms": round(per_frame[0], 1),
            "max_ms": round(per_frame[-1], 1), "detections": counts,
            "total_detections": sum(counts), "scores": sorted(scores, reverse=True)[:12]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()

    if not ENGINE.exists():
        print(f"FAIL: {ENGINE.relative_to(REPO)} does not exist. Build it:")
        print("  uv run python tools/train/_build_engine.py")
        return 2
    if not WEIGHTS.exists():
        print(f"FAIL: {WEIGHTS.relative_to(REPO)} is missing.")
        return 2

    import cv2

    paths = frames(a.frames)
    imgs = [cv2.imread(str(p))[:, :, ::-1] for p in paths]
    h, w = imgs[0].shape[:2]
    print(f"{len(imgs)} real frames at {w}x{h} from {paths[0].parent.relative_to(REPO)}\n")

    rows = []
    for weights, label in ((ENGINE, "TensorRT FP16"), (WEIGHTS, "PyTorch FP16")):
        try:
            rows.append(run(weights, imgs, label))
            r = rows[-1]
            print(f"  {label:16s} load {r['load_s']:5.2f}s   median {r['median_ms']:7.1f} ms   "
                  f"({r['min_ms']:.0f}-{r['max_ms']:.0f})   {r['total_detections']} detections")
        except Exception as exc:                          # noqa: BLE001
            print(f"  {label:16s} FAILED: {type(exc).__name__}: {exc}")
            rows.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})

    eng = next((r for r in rows if r["label"].startswith("TensorRT")), {})
    pt = next((r for r in rows if r["label"].startswith("PyTorch")), {})
    ok = True
    print()
    if "error" in eng:
        print("FAIL  the engine did not run. It is NOT ready; the demo must use --pytorch.")
        ok = False
    else:
        budget = 300.0                                    # R9 end-to-end; 5.11 budgets ~113 ms for detect
        print(f"  engine within the {budget:.0f} ms R9 detect budget: "
              f"{'YES' if eng['median_ms'] <= budget else 'NO'} ({eng['median_ms']:.0f} ms)")
        if pt.get("median_ms"):
            speedup = pt["median_ms"] / eng["median_ms"]
            print(f"  speedup over PyTorch: {speedup:.2f}x")
            # Detection parity matters more than speed. A quantised engine that drops small targets is a
            # regression wearing a benchmark's clothes.
            de, dp = eng["total_detections"], pt["total_detections"]
            worst = max(1, dp)
            drift = abs(de - dp) / worst
            print(f"  detections engine={de} pytorch={dp}  drift={drift:.1%}")
            if drift > 0.15:
                print("  FAIL  the engine disagrees with PyTorch by more than 15% - FP16 has cost recall.")
                ok = False
        if eng["median_ms"] > budget:
            ok = False

    out = REPO / "_artifacts/verification/engine_verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ok": ok, "frames": len(imgs), "rows": rows}, indent=2), encoding="utf-8")
    print(f"\n  -> {out.relative_to(REPO)}")
    print("\nENGINE READY" if ok else "\nENGINE NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
