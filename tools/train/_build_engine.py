"""Build the TensorRT engine for the demo detector and MEASURE what it bought. F8, SOLUTION_DOC 5.11.

    uv run python tools/train/_build_engine.py

Why: the live loop measured 1165-3205 ms per frame of PyTorch FP16 inference on an RTX 4060 that is sharing
its 8 GB with the Unreal editor, which puts the loop at 0.58 FPS. Section 5.11 budgets ~113 ms for detect and
5.6's 3-hits-in-2-seconds gate is written for the ~5 FPS that implies, so every timing constant downstream
inherits that assumption. TensorRT is the document's own answer to exactly this, and `detect/export.py` has
existed unused since it was written.

The engine is built at the batch the grid ACTUALLY uses. A 3840x2160 frame tiled at 1024 px with 0.2 overlap
is 5 x 3 = 15 tiles, and `DetectorConfig.max_batch = 0` sends the whole grid in one call. An engine built at
section 5.11's example batch of 6 would not take that call.

It then times both paths on the same real frames, because an engine that builds is not the same as an engine
that is faster.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sightline.detect.export import ExportConfig, export_engine  # noqa: E402
from sightline.detect.rgb import RAW_CONF, DetectorConfig, RgbDetector  # noqa: E402
from sightline.detect.tiler import DEFAULT_OVERLAP, DEFAULT_TILE_PX, TileGrid  # noqa: E402

WEIGHTS = REPO / "models/detect/f8b_sim/weights/best.pt"
FRAMES = sorted((REPO / "_artifacts/dataset/seed47_alt55/images").glob("*.jpg"))[:6]


def n_tiles(w: int = 3840, h: int = 2160) -> int:
    g = TileGrid.build(w, h, tile_px=DEFAULT_TILE_PX, overlap=DEFAULT_OVERLAP)
    return len(g.tiles)


def bench(weights: str, label: str, batch: int) -> float:
    import cv2

    det = RgbDetector(DetectorConfig(weights=weights, raw_conf=RAW_CONF, imgsz=1024,
                                     max_batch=batch, half=True, device="cuda:0"))
    frame = cv2.imread(str(FRAMES[0]))
    det.detect(frame)                                        # warm-up, never timed
    ts = []
    for f in FRAMES:
        im = cv2.imread(str(f))
        t0 = time.time()
        det.detect(im)
        ts.append((time.time() - t0) * 1000.0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"  {label:28s} median {med:8.1f} ms   min {ts[0]:8.1f}   max {ts[-1]:8.1f}")
    return med


if __name__ == "__main__":
    tiles = n_tiles()
    print(f"tile grid: {tiles} tiles of {DEFAULT_TILE_PX[0]} px at {DEFAULT_OVERLAP} overlap")
    print(f"frames for timing: {len(FRAMES)}")
    if not WEIGHTS.exists():
        sys.exit(f"weights missing: {WEIGHTS}")

    print("\n--- baseline: PyTorch FP16 ---")
    base = bench(str(WEIGHTS), "pytorch fp16", 0)

    print(f"\n--- building TensorRT engine (batch={tiles}) ---")
    t0 = time.time()
    man = export_engine(ExportConfig(weights=str(WEIGHTS), format="engine", imgsz=1024,
                                     half=True, batch=tiles, device=0))
    eng = man["exported"]
    print(f"  built in {time.time() - t0:.0f}s -> {eng}")

    print("\n--- tensorrt fp16 ---")
    trt = bench(eng, "tensorrt fp16", 0)

    speedup = base / trt if trt else 0.0
    fps = 1000.0 / trt if trt else 0.0
    print(f"\nspeedup {speedup:.2f}x   detect-only rate {fps:.2f} FPS")
    print(f"section 5.11 budget for detect is ~113 ms; this is {trt:.0f} ms")
    (REPO / "_artifacts/engine_bench.json").write_text(json.dumps({
        "tiles": tiles, "pytorch_fp16_ms": round(base, 1), "tensorrt_fp16_ms": round(trt, 1),
        "speedup": round(speedup, 2), "engine": eng, "manifest": man,
    }, indent=1, default=str), encoding="utf-8")
    print("wrote _artifacts/engine_bench.json")
