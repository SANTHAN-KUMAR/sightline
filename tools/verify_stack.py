"""Run the Python-stack verification (tests/test_stack.py) and write a JSON + Markdown summary.

    uv run python tools/verify_stack.py                 # everything (GPU, network, slow)
    uv run python tools/verify_stack.py -m "not slow"   # extra args go to pytest
    uv run python tools/verify_stack.py --smoke         # also re-run the sightline MCP smoke test

Outputs: docs/verification/stack_check.json and docs/verification/stack_check.md
(timings from _artifacts/stack/metrics.json, per-test outcomes from _artifacts/stack/junit.xml).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "_artifacts" / "stack"
OUT = REPO / "docs" / "verification"
PY = sys.executable
FO_PY = REPO / "envs" / "fiftyone" / ".venv" / "Scripts" / "python.exe"

# (tool, distribution in the main env or None, test name, metrics key)
TOOLS = [
    ("PyTorch (CUDA)", "torch", "test_torch_cuda_matmul", "torch"),
    ("torchvision", "torchvision", "test_torch_cuda_matmul", "torch"),
    ("Ultralytics YOLO26", "ultralytics", "test_yolo26n_predict", "ultralytics"),
    ("TensorRT (cu13)", "tensorrt-cu13-libs", "test_yolo26n_tensorrt_fp16_engine", "tensorrt"),
    ("ONNX", "onnx", "test_onnx_and_onnxruntime_gpu", "onnxruntime-gpu"),
    ("onnxslim", "onnxslim", "test_onnx_and_onnxruntime_gpu", "onnxruntime-gpu"),
    ("ONNX Runtime GPU", "onnxruntime-gpu", "test_onnx_and_onnxruntime_gpu", "onnxruntime-gpu"),
    ("Ultralytics BoT-SORT", "ultralytics", "test_ultralytics_botsort_track", "ultralytics-botsort"),
    ("RF-DETR", "rfdetr", "test_rfdetr_nano_predict", "rfdetr"),
    ("SAHI", "sahi", "test_sahi_sliced_prediction_4k", "sahi"),
    ("supervision (InferenceSlicer)", "supervision", "test_supervision_inference_slicer_4k", "supervision"),
    ("Weighted Boxes Fusion", "ensemble-boxes", "test_weighted_boxes_fusion", "ensemble-boxes"),
    ("ProbEn score rule (vendored)", None, "test_proben_score_rule", "proben"),
    ("DINOv2 ViT-S (timm)", "timm", "test_dinov2_vits_features", "timm-dinov2"),
    ("transformers (RF-DETR dep)", "transformers", "test_rfdetr_nano_predict", "rfdetr"),
    ("PyAV", "av", "test_pyav_decode_pts", "av"),
    ("PyNvVideoCodec", "pynvvideocodec", "test_pynvvideocodec_decode", "pynvvideocodec"),
    ("torchcodec", "torchcodec", "test_torchcodec_decode", "torchcodec"),
    ("pymavlink", "pymavlink", "test_pymavlink_tlog_roundtrip", "pymavlink"),
    ("pyulog", "pyulog", "test_pyulog_real_sample", "pyulog"),
    ("thermal_parser (DJI R-JPEG)", "thermal-parser", "test_thermal_parser_dji_rjpeg", "thermal-parser"),
    ("roboflow trackers", "trackers", "test_roboflow_trackers", "trackers"),
    ("BoxMOT (A/B only)", "boxmot", "test_boxmot_bytetrack_ab", "boxmot"),
    ("lap (assignment)", "lap", "test_ultralytics_botsort_track", "ultralytics-botsort"),
    ("scikit-learn DBSCAN", "scikit-learn", "test_dbscan_haversine_geo_dedup", "scikit-learn"),
    ("torchmetrics mAP", "torchmetrics", "test_torchmetrics_map_recall_at_05", "torchmetrics"),
    ("pycocotools", "pycocotools", "test_pycocotools_eval", "pycocotools"),
    ("py-motmetrics", "motmetrics", "test_motmetrics_mota_idf1", "motmetrics"),
    ("FiftyOne (separate env)", "@fiftyone", "test_fiftyone_separate_env", "fiftyone"),
    ("pyproj", "pyproj", "test_pyproj_geod_fwd_inv", "pyproj"),
    ("rasterio", "rasterio", "test_rasterio_geotiff_roundtrip", "rasterio"),
    ("dem-stitcher", "dem-stitcher", "test_dem_stitcher_glo30_wayanad", "dem-stitcher"),
    ("shapely", "shapely", "test_shapely_geopandas_pyogrio_gpkg", "geopandas"),
    ("geopandas", "geopandas", "test_shapely_geopandas_pyogrio_gpkg", "geopandas"),
    ("pyogrio", "pyogrio", "test_shapely_geopandas_pyogrio_gpkg", "geopandas"),
    ("DuckDB + spatial", "duckdb", "test_duckdb_spatial", "duckdb"),
    ("FastAPI", "fastapi", "test_fastapi_websocket_testclient", "fastapi"),
    ("httpx (TestClient)", "httpx", "test_fastapi_websocket_testclient", "fastapi"),
    ("uvicorn", "uvicorn", "test_uvicorn_websockets_real_socket", "uvicorn"),
    ("websockets", "websockets", "test_uvicorn_websockets_real_socket", "uvicorn"),
    ("persist-queue", "persist-queue", "test_persist_queue_sqliteackqueue_durability", "persist-queue"),
    ("simplekml", "simplekml", "test_simplekml_kmz_with_thumbnail", "simplekml"),
    ("geojson", "geojson", "test_geojson_feature", "geojson"),
    ("pytak (CoT)", "pytak", "test_pytak_cot_event", "pytak"),
    ("Fields2Cover", None, "test_fields2cover_windows_availability", "fields2cover"),
    ("pygame (joystick)", "pygame", "test_pygame_joystick_init_without_device", "pygame"),
    ("inputs", "inputs", "test_inputs_gamepad_enumeration", "inputs"),
    ("mavsdk", "mavsdk", "test_mavsdk_server_binary", "mavsdk"),
]


def licence_of(md) -> str:
    lic = md.get("License-Expression") or ""
    if not lic:
        raw = (md.get("License") or "").strip()
        lic = raw.splitlines()[0][:60] if raw and len(raw) < 200 else ""
    if not lic:
        cls = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
        lic = "; ".join(cls)
    return lic or "unknown"


def dist_info(name: str) -> tuple[str | None, str]:
    if name == "@fiftyone":
        if not FO_PY.exists():
            return None, "n/a"
        code = ("import importlib.metadata as m,json;d=m.metadata('fiftyone');"
                "print(json.dumps([m.version('fiftyone'),d.get('License-Expression') or d.get('License') or '']))")
        r = subprocess.run([str(FO_PY), "-c", code], capture_output=True, text=True)
        if r.returncode:
            return None, "n/a"
        v, lic = json.loads(r.stdout)
        return v, (lic.splitlines()[0][:60] if lic else "Apache-2.0")
    try:
        return metadata.version(name), licence_of(metadata.metadata(name))
    except metadata.PackageNotFoundError:
        return None, "not installed"


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def parse_junit(path: Path) -> dict:
    res = {}
    for tc in ET.parse(path).getroot().iter("testcase"):
        name = tc.get("name").split("[")[0]
        status, msg = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                status = {"failure": "FAILED", "error": "ERROR", "skipped": "skipped"}[tag]
                msg = (el.get("message") or "")[:300]
        res[name] = {"status": status, "time_s": round(float(tc.get("time", 0)), 2), "message": msg}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="also run tools/sightline_mcp/smoke_test.py")
    args, pytest_args = ap.parse_known_args()
    ART.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    junit = ART / "junit.xml"
    t0 = time.time()
    cmd = [PY, "-m", "pytest", str(REPO / "tests" / "test_stack.py"), "-v", "-rA", f"--junitxml={junit}",
           *pytest_args]
    proc = subprocess.run(cmd, cwd=REPO)
    tests = parse_junit(junit)
    metrics = json.loads((ART / "metrics.json").read_text()) if (ART / "metrics.json").exists() else {}

    smoke = None
    if args.smoke:
        r = subprocess.run([PY, str(REPO / "tools" / "sightline_mcp" / "smoke_test.py")], capture_output=True,
                           text=True, timeout=600)
        smoke = {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-4:]}

    rows = []
    for tool, dist, test, mkey in TOOLS:
        ver, lic = dist_info(dist) if dist else (None, "Apache-2.0 (vendor/proben)" if "ProbEn" in tool else "n/a")
        t = tests.get(test, {"status": "not run", "time_s": None, "message": ""})
        rows.append({"tool": tool, "distribution": dist, "version": ver, "licence": lic, "test": test,
                     "result": t["status"], "time_s": t["time_s"], "message": t["message"],
                     "metrics": metrics.get(mkey, {})})

    sizes = {k: round(dir_size(p) / 1e9, 2) for k, p in {
        ".venv (logical, hardlinked from uv cache)": REPO / ".venv",
        "envs/fiftyone/.venv": REPO / "envs" / "fiftyone" / ".venv",
        "models": REPO / "models", "data/dem": REPO / "data" / "dem", "data/samples": REPO / "data" / "samples",
        "data/fiftyone": REPO / "data" / "fiftyone", "D:/Tools/cache/uv": Path(r"D:\Tools\cache\uv"),
        "D:/Tools/cache/hf": Path(r"D:\Tools\cache\hf"),
        "D:/Tools/cache/duckdb": Path(r"D:\Tools\cache\duckdb")}.items() if p.exists()}

    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0],
               "pytest_returncode": proc.returncode, "duration_s": round(time.time() - t0, 1),
               "counts": {s: sum(1 for t in tests.values() if t["status"] == s)
                          for s in ("passed", "FAILED", "ERROR", "skipped")},
               "tools": rows, "tests": tests, "disk_gb": sizes, "mcp_smoke": smoke}
    (OUT / "stack_check.json").write_text(json.dumps(summary, indent=2, default=str))

    md = ["# Python stack check (generated by tools/verify_stack.py)", "",
          f"Generated {summary['generated']}, Python {summary['python']}, pytest rc={proc.returncode}, "
          f"{summary['duration_s']} s. Counts: {summary['counts']}.", "",
          "| Tool | Version | Licence | Test | Result | Time (s) | Key metrics |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        m = ", ".join(f"{k}={v}" for k, v in r["metrics"].items()
                      if not isinstance(v, (dict, list)) and k not in ("version", "weights", "file", "engine"))
        note = f" {r['message'][:120]}" if r["result"] != "passed" and r["message"] else ""
        md.append(f"| {r['tool']} | {r['version'] or '-'} | {r['licence']} | `{r['test']}` | {r['result']}{note} | "
                  f"{r['time_s'] if r['time_s'] is not None else '-'} | {m[:200]} |")
    md += ["", "## Disk (GB)", "", *[f"- {k}: {v}" for k, v in sizes.items()]]
    if smoke:
        md += ["", f"## MCP smoke test: rc={smoke['returncode']}", "", *[f"    {l}" for l in smoke["tail"]]]
    (OUT / "stack_check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'stack_check.json'} and {OUT / 'stack_check.md'}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
