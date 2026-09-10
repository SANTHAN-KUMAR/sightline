"""Day-1 test #3 (SOLUTION_DOC §4 / §10): simGetImages throughput at 1080p and 4K, compressed PNG vs raw.

Needs a running sim with sim/settings/capture_4k.json (cameras "survey" = 4K, "survey_hd" = 1080p).
Measures wall-clock per request for: scene-only and scene+segmentation+infrared, each PNG-compressed and raw
uint8. Writes a JSON result to _artifacts/day1/capture_benchmark_<ts>.json and prints a table.
Run: uv run python tools/day1/capture_benchmark.py [--n 20] [--alt 40]
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cosysairsim as airsim

REPO = Path(__file__).resolve().parents[2]


def bench(c: airsim.MultirotorClient, camera: str, types: list[int], compress: bool, n: int) -> dict:
    reqs = [airsim.ImageRequest(camera, t, pixels_as_float=False, compress=compress) for t in types]
    times, sizes = [], []
    c.simGetImages(reqs, vehicle_name="Drone")  # warm-up (render targets allocate on first use)
    for _ in range(n):
        t0 = time.perf_counter()
        resp = c.simGetImages(reqs, vehicle_name="Drone")
        times.append(time.perf_counter() - t0)
        sizes.append(sum(len(r.image_data_uint8) for r in resp))
        if any(r.width == 0 for r in resp):
            return {"error": "empty image returned", "camera": camera, "types": types}
    ms = [t * 1000 for t in times]
    return {
        "camera": camera, "types": types, "compress": compress, "n": n,
        "w": resp[0].width, "h": resp[0].height,
        "ms_median": round(statistics.median(ms), 1), "ms_p90": round(sorted(ms)[int(0.9 * (n - 1))], 1),
        "fps": round(1000 / statistics.median(ms), 2), "mb_per_request": round(statistics.mean(sizes) / 2**20, 2),
    }


def main(n: int, alt: float) -> None:
    c = airsim.MultirotorClient()
    c.confirmConnection()
    c.enableApiControl(True, vehicle_name="Drone")
    c.armDisarm(True, vehicle_name="Drone")
    c.takeoffAsync(vehicle_name="Drone").join()
    c.moveToPositionAsync(0, 0, -alt, 5, vehicle_name="Drone").join()
    S, SEG, IR = airsim.ImageType.Scene, airsim.ImageType.Segmentation, airsim.ImageType.Infrared
    runs = []
    for cam in ("survey_hd", "survey"):
        for types in ([S], [S, SEG, IR]):
            for compress in (True, False):
                r = bench(c, cam, types, compress, n)
                runs.append(r)
                print(r)
    out = REPO / "_artifacts" / "day1" / f"capture_benchmark_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"alt_m": alt, "runs": runs}, indent=2))
    print("\n| camera | types | compress | res | median ms | p90 ms | FPS | MB/req |\n|---|---|---|---|---|---|---|---|")
    for r in runs:
        if "error" in r:
            print(f"| {r['camera']} | {r['types']} | - | - | ERROR {r['error']} | | | |")
            continue
        print(f"| {r['camera']} | {r['types']} | {r['compress']} | {r['w']}x{r['h']} | {r['ms_median']} | "
              f"{r['ms_p90']} | {r['fps']} | {r['mb_per_request']} |")
    print("saved", out)
    c.landAsync(vehicle_name="Drone").join()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--alt", type=float, default=40.0)
    a = ap.parse_args()
    main(a.n, a.alt)
