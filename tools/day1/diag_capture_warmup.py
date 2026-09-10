"""Repro/regression for the cold-start capture defect (docs/verification/dev_workflow.md D1).

The FIRST `simGetImages` of a **fresh engine process** returns the post-processed image types before their
capture materials exist: float depth and depth_planar come back as an unconverted 0.0-0.61 buffer instead of
metres, and surface_normals is a ~1.7 MB (wrong) PNG. There is no error and width/height are correct.
`server.py::_capture_once` now discards one warm-up request per AirSim connection, so `sim_capture` never
returns such a frame.

Run it against a **cold** editor + fresh PIE (or a fresh -game / packaged run), with the drone on the ground:

    .venv\\Scripts\\python.exe -u tools\\day1\\diag_capture_warmup.py [--alt 20] [--batches 5]

--raw (default) talks to cosysairsim directly and SHOULD show the bad first batch on a cold process.
--through-mcp goes through the sightline MCP server's sim_capture and must NEVER show one.
Exit code 0 = as expected. Results: _artifacts/verification/capture_warmup_<mode>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cosysairsim as airsim
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TYPES = [("scene", airsim.ImageType.Scene, False),
         ("segmentation", airsim.ImageType.Segmentation, False),
         ("infrared", airsim.ImageType.Infrared, False),
         ("depth", airsim.ImageType.DepthPerspective, True),
         ("depth_planar", airsim.ImageType.DepthPlanar, True),
         ("surface_normals", airsim.ImageType.SurfaceNormals, False)]
NAMES = [n for n, _, _ in TYPES]


def climb(c, alt: float) -> float:
    c.enableApiControl(True, vehicle_name="Drone")
    c.armDisarm(True, vehicle_name="Drone")
    c.takeoffAsync(timeout_sec=40, vehicle_name="Drone").join()
    c.moveToPositionAsync(0, 0, -alt, 4, timeout_sec=60, vehicle_name="Drone").join()
    c.hoverAsync(vehicle_name="Drone").join()
    time.sleep(6)
    return -c.getMultirotorState(vehicle_name="Drone").kinematics_estimated.position.z_val


def raw(alt: float, batches: int) -> tuple[float, list[dict]]:
    c = airsim.MultirotorClient(ip="127.0.0.1", port=41451, timeout_value=90)
    c.confirmConnection()
    z = climb(c, alt)
    reqs = [airsim.ImageRequest("survey", t, is_f, not is_f) for _, t, is_f in TYPES]
    rows = []
    for b in range(batches):
        rs = c.simGetImages(reqs, vehicle_name="Drone")
        row = {"batch": b}
        for (name, _t, is_f), r in zip(TYPES, rs, strict=True):
            if is_f:
                a = np.array(r.image_data_float, dtype=np.float32)
                row[name] = [round(float(a.min()), 4), round(float(a.max()), 4)]
            else:
                row[name] = len(r.image_data_uint8)
        rows.append(row)
        print(json.dumps(row), flush=True)
        time.sleep(3)
    return z, rows


async def through_mcp(alt: float, batches: int) -> tuple[float, list[dict]]:
    import os

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cfg = json.loads((REPO / ".mcp.json").read_text())["mcpServers"]["sightline"]
    params = StdioServerParameters(command=cfg["command"], args=list(cfg["args"]),
                                   env={**os.environ, **cfg.get("env", {})}, cwd=str(REPO))
    rows: list[dict] = []
    z = 0.0
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        await s.call_tool("sim_fly", {"action": "takeoff", "vehicle": "Drone", "timeout_s": 40})
        await s.call_tool("sim_fly", {"action": "move_to", "x": 0.0, "y": 0.0, "z": -alt, "velocity": 4.0,
                                      "vehicle": "Drone", "timeout_s": 60})
        await asyncio.sleep(6)
        st = json.loads("\n".join(c.text for c in (await s.call_tool("sim_state", {"vehicle": "Drone"})).content
                                  if c.type == "text"))
        z = -st["position_ned"][2]
        for b in range(batches):
            res = await s.call_tool("sim_capture", {"camera": "survey", "vehicle": "Drone", "image_types": NAMES,
                                                    "preview": None, "save": False})
            meta = json.loads("\n".join(c.text for c in res.content if c.type == "text"))
            row = {"batch": b}
            for m in meta["images"]:
                row[m["type"]] = ([m.get("min"), m.get("max")] if "min" in m
                                  else f"{m['width']}x{m['height']}")
                if m.get("warning"):
                    row.setdefault("warnings", []).append(f"{m['type']}: {m['warning'][:60]}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            await asyncio.sleep(3)
    return z, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--through-mcp", action="store_true")
    a = ap.parse_args()
    mode = "mcp" if a.through_mcp else "raw"
    z, rows = asyncio.run(through_mcp(a.alt, a.batches)) if a.through_mcp else raw(a.alt, a.batches)
    bad = [r["batch"] for r in rows if isinstance(r["depth"], list) and (r["depth"][1] or 0) < z * 0.5]
    out = REPO / "_artifacts" / "verification" / f"capture_warmup_{mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": mode, "alt_m": z, "bad_batches": bad, "batches": rows}, indent=2))
    print(f"\nalt={z:.2f} m, unconverted-depth batches={bad or 'none'} -> {out}", flush=True)
    if a.through_mcp:
        return 1 if bad else 0                      # sim_capture must never expose one
    return 0 if (not bad or bad == [0]) else 1      # raw: only the very first batch may be bad


if __name__ == "__main__":
    sys.exit(main())
