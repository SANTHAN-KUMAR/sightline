"""Day-1 test #8 (SOLUTION_DOC §5.11): measured per-frame detection latency on this machine for the three 4K
strategies, so the 300 ms budget (R9) rests on numbers instead of scaled estimates.

  C1  single 1920x1088 letterbox              - cheapest, marginal pixels-on-target above ~60 m
  C2  six native 1280x1088 tiles (2x3, ~10% overlap), batched in ONE engine call - the design's detection pass
  C3  C1 coarse pass + 4 ROI re-detects at 640 native

For each model (yolo26n/s/m) and strategy: TensorRT FP16 engine build time, warm inference latency (median/p90
over --iters), and end-to-end including tiling and NMS-free post-processing. Input is a real 4K simulator capture
when one exists under _artifacts/captures, else a synthetic frame.

Run (nothing else using the GPU):  uv run python -u tools/day1/latency_benchmark.py [--models n,s] [--iters 30]
Writes _artifacts/verification/latency_<ts>.json and prints the table for docs/verification/gpu_latency.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO / "models" / "latency"
TILE_W, TILE_H, TILES_X, TILES_Y = 1280, 1088, 3, 2


def source_frame() -> np.ndarray:
    """Newest 4K (or largest) simulator capture, else a synthetic 3840x2160 frame."""
    caps = sorted((REPO / "_artifacts" / "captures").glob("*/scene.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in caps:
        im = Image.open(p).convert("RGB")
        if im.width >= 1920:
            return np.array(im.resize((3840, 2160))) if im.width != 3840 else np.array(im)
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (2160, 3840, 3), dtype=np.uint8)


def tiles(frame: np.ndarray) -> list[np.ndarray]:
    h, w = frame.shape[:2]
    xs = np.linspace(0, w - TILE_W, TILES_X).astype(int)
    ys = np.linspace(0, h - TILE_H, TILES_Y).astype(int)
    return [frame[y:y + TILE_H, x:x + TILE_W] for y in ys for x in xs]


def timed(fn, iters: int, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ms), 1), "p90_ms": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 1),
            "min_ms": round(min(ms), 1)}


def build_engine(weights: str, imgsz, batch: int) -> tuple[YOLO, float, str]:
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(weights).stem}_{imgsz if isinstance(imgsz, int) else 'x'.join(map(str, imgsz))}_b{batch}"
    engine = ENGINE_DIR / f"{tag}.engine"
    build_s = 0.0
    if not engine.exists():
        t0 = time.time()
        src = YOLO(str(REPO / "models" / weights)) if (REPO / "models" / weights).exists() else YOLO(weights)
        out = src.export(format="engine", half=True, imgsz=imgsz, batch=batch, device=0, workspace=2)
        Path(out).replace(engine)
        build_s = round(time.time() - t0, 1)
    return YOLO(str(engine), task="detect"), build_s, str(engine)


def main(models: list[str], iters: int) -> None:
    frame = source_frame()
    tile_list = tiles(frame)
    results = []
    gpu = torch.cuda.get_device_name(0)
    for m in models:
        w = f"yolo26{m}.pt"
        # C1: one 1920x1088 letterboxed pass
        try:
            mdl, build, path = build_engine(w, [1088, 1920], 1)
            small = np.array(Image.fromarray(frame).resize((1920, 1088)))
            r = timed(lambda m=mdl, im=small: m.predict(im, imgsz=[1088, 1920], verbose=False, device=0), iters)
            results.append({"model": f"yolo26{m}", "strategy": "C1 1920x1088", "engine_build_s": build, **r, "engine": path})
        except Exception as e:  # noqa: BLE001 - a failed configuration is a result
            results.append({"model": f"yolo26{m}", "strategy": "C1 1920x1088", "error": f"{type(e).__name__}: {e}"[:300]})
        # C2: six native 1280x1088 tiles in one batched call
        try:
            mdl, build, path = build_engine(w, [TILE_H, TILE_W], TILES_X * TILES_Y)
            r = timed(lambda m=mdl: m.predict(tile_list, imgsz=[TILE_H, TILE_W], verbose=False, device=0), iters)
            results.append({"model": f"yolo26{m}", "strategy": f"C2 {TILES_X * TILES_Y}x{TILE_W}x{TILE_H} batched",
                            "engine_build_s": build, **r, "engine": path})
        except Exception as e:  # noqa: BLE001
            results.append({"model": f"yolo26{m}", "strategy": "C2 tiles", "error": f"{type(e).__name__}: {e}"[:300]})
        # C3: coarse C1 pass + 4 ROI re-detects at 640 native
        try:
            coarse, _, _ = build_engine(w, [1088, 1920], 1)
            roi_mdl, build, path = build_engine(w, 640, 4)
            rois = [frame[540:1180, 960:1600], frame[540:1180, 2240:2880],
                    frame[1080:1720, 960:1600], frame[1080:1720, 2240:2880]]
            small = np.array(Image.fromarray(frame).resize((1920, 1088)))

            def c3(c=coarse, im=small, rm=roi_mdl, rs=rois):
                c.predict(im, imgsz=[1088, 1920], verbose=False, device=0)
                rm.predict(rs, imgsz=640, verbose=False, device=0)
            r = timed(c3, iters)
            results.append({"model": f"yolo26{m}", "strategy": "C3 coarse + 4 ROI@640", "engine_build_s": build, **r})
        except Exception as e:  # noqa: BLE001
            results.append({"model": f"yolo26{m}", "strategy": "C3", "error": f"{type(e).__name__}: {e}"[:300]})

    out = REPO / "_artifacts" / "verification" / f"latency_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": gpu, "torch": torch.__version__, "iters": iters,
                               "frame": list(frame.shape), "results": results}, indent=2))
    print(f"\nGPU: {gpu}   torch {torch.__version__}   frame {frame.shape[1]}x{frame.shape[0]}   iters {iters}")
    print("\n| model | strategy | engine build s | median ms | p90 ms | < 300 ms? |\n|---|---|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['model']} | {r['strategy']} | - | ERROR {r['error'][:60]} | | |")
        else:
            print(f"| {r['model']} | {r['strategy']} | {r.get('engine_build_s', 0)} | {r['median_ms']} | {r['p90_ms']} "
                  f"| {'yes' if r['p90_ms'] < 300 else 'NO'} |")
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="n,s", help="comma list of yolo26 sizes, e.g. n,s,m")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    main([x.strip() for x in a.models.split(",") if x.strip()], a.iters)
