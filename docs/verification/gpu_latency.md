# V7 — GPU latency (day-1 test #8, SOLUTION_DOC §5.11)

Measured 2026-09-10 on the dev machine, nothing else using the GPU.
Script: `tools/day1/latency_benchmark.py` · raw: `_artifacts/verification/latency_20260910-202843.json`

**Setup**: RTX 4060 Laptop (8 GB), torch 2.14.0+cu130, TensorRT 11.3.0.99, **FP16** engines built by Ultralytics
8.4.146 from `yolo26n.pt` / `yolo26s.pt`. Input is a 3840×2160 frame (a real simulator capture when present).
30 timed iterations after 5 warm-up passes, `torch.cuda.synchronize()` around each. Latency is **detector only**
(tiling + inference + NMS-free post-processing); decode, tracking, geolocation and writing are extra (§5.11 budget).

| model | strategy | engine build (s) | median (ms) | p90 (ms) | < 300 ms |
|---|---|---|---|---|---|
| yolo26n | C1 single 1920×1088 letterbox | (cached) | 11.3 | 12.7 | yes |
| yolo26n | C2 6 × 1280×1088 native tiles, one batched call | 260.4 | 40.9 | 44.0 | yes |
| yolo26n | C3 C1 + 4 ROI re-detects @640 | 127.7 | 19.8 | 20.7 | yes |
| yolo26s | C1 single 1920×1088 letterbox | 197.3 | 16.7 | 17.6 | yes |
| **yolo26s** | **C2 6 × 1280×1088 native tiles** (the design's detection pass) | 276.6 | **65.0** | 68.3 | yes |
| yolo26s | C3 C1 + 4 ROI re-detects @640 | 146.5 | 29.2 | 30.2 | yes |

## What this settles
- **The doc's estimate was sound.** §5.11 predicted ≈60 ms for C2 with YOLO26s FP16 on a 4060 (T4 proxy);
  measured **65.0 ms**. The scaling method behind the Orin table can therefore be trusted to about ±10 %.
- **R9 (< 300 ms/frame) holds on this machine with a 4× margin** for the design configuration, so the pipeline can
  be developed at full tiling quality without latency compromises. The 4060 is an *upper bound* for Jetson Orin,
  not a proxy: the Orin numbers in §5.11 stay estimates until measured on the device (F20).
- **Tiling is the cost, not the model.** Going n → s costs +24 ms at C2; going C1 → C2 costs +48 ms (yolo26s).
  That is the price of native-resolution pixels on target, which §2.5 shows is what recall depends on.
- **C3 is the cheap middle**: 29 ms for yolo26s, half of C2, and the natural fallback if the Orin budget bites.

## Caveats
- FP16 only. INT8 (§5.11, needs a calibration set including tiny/occluded positives) is not measured yet.
- Engine builds are slow (2–5 min each) but cached in `models/latency/`; budget for a rebuild after any
  TensorRT/driver change.
- Measured on a synthetic/first-capture frame: detection *count* affects post-processing slightly, so re-measure on
  real flood-scene frames once F1 exists.
- **Never run this while the editor or PIE is open** — contention makes the numbers meaningless and triggers
  Windows memory-pressure warnings (16 GB / 8 GB VRAM).
