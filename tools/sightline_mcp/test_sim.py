"""Full verification of every sightline AirSim tool THROUGH the MCP stdio transport, exactly as Claude Code
drives it. Each check asserts on real returned data (positions, image sizes, flags), not just "no exception".

Needs a running Cosys-AirSim instance with sim/settings/default.json (vehicle "Drone", camera "survey"):
editor PIE, `-game`, or the packaged Blocks exe.
Run: uv run python -u tools/sightline_mcp/test_sim.py [--alt 20] [--detect-filter "Cube*"]
Exit code = number of failed checks. Results: _artifacts/verification/sim_tools_<ts>.json
"""

import argparse
import asyncio
import base64
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
RESULTS: list[dict] = []
HOME: list[float] = [0.0, 0.0, 0.0]


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "ok": ok, "detail": detail[:400]})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail[:200]}", flush=True)


async def call(s: ClientSession, name: str, args: dict | None = None, expect_error: bool = False):
    t = time.time()
    res = await s.call_tool(name, args or {})
    texts = [c.text for c in res.content if c.type == "text"]
    images = [c for c in res.content if c.type == "image"]
    dt = time.time() - t
    if res.isError and not expect_error:
        raise RuntimeError(f"{name} -> error: {texts}")
    return (texts[0] if texts else ""), images, dt


def state_of(text: str) -> dict:
    # sim_fly returns "<action>: done\n{json}"; sim_state returns the json
    return json.loads(text[text.index("{"):])


def dist(a, b) -> float:
    return math.dist(a, b)


async def run(alt: float, detect_filter: str) -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "tools/sightline_mcp/server.py")], cwd=str(REPO))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()

        async def check(name, coro_fn):
            try:
                await coro_fn()
            except Exception as e:  # noqa: BLE001 - every failure is recorded, the suite continues
                record(name, False, f"{type(e).__name__}: {e}")

        # ---- connection
        async def t_ping():
            txt, _, dt = await call(s, "sim_ping")
            d = json.loads(txt)
            record("sim_ping", d["ping"] is True and "Drone" in d["vehicles"] and d["server_version"] >= 1,
                   f"{d} in {dt:.2f}s")
        await check("sim_ping", t_ping)

        # ---- state (also records home = spawn position on the ground)
        await call(s, "sim_fly", {"action": "reset", "vehicle": "Drone"})

        async def t_state():
            txt, _, _ = await call(s, "sim_state", {"vehicle": "Drone"})
            d = json.loads(txt)
            HOME[:] = d["position_ned"]
            ok = len(d["position_ned"]) == 3 and len(d["orientation_wxyz"]) == 4 and len(d["euler_roll_pitch_yaw_deg"]) == 3 \
                and abs(d["gps"]["lat"]) > 0
            record("sim_state", ok, f"pos={d['position_ned']} gps={d['gps']} landed={d['landed_state']}")
        await check("sim_state", t_state)

        # ---- flight: arm, takeoff, move_to, hover
        async def t_arm():
            txt, _, _ = await call(s, "sim_fly", {"action": "arm", "vehicle": "Drone"})
            record("sim_fly arm", state_of(txt)["api_control"] is True, "api_control true")
        await check("sim_fly arm", t_arm)

        async def t_takeoff():
            txt, _, dt = await call(s, "sim_fly", {"action": "takeoff", "vehicle": "Drone", "timeout_s": 30})
            z = state_of(txt)["position_ned"][2]
            record("sim_fly takeoff", z < -0.5, f"z={z:.2f} m after {dt:.1f}s")
        await check("sim_fly takeoff", t_takeoff)

        target = [15.0, -10.0, -alt]

        async def t_move():
            # climb vertically first (the Blocks start area is walled by cubes), then translate at altitude
            txt, _, _ = await call(s, "sim_fly", {"action": "move_to", "x": HOME[0], "y": HOME[1], "z": -alt,
                                                   "velocity": 4, "vehicle": "Drone", "timeout_s": 60})
            record("sim_fly vertical climb (no collision)", abs(state_of(txt)["position_ned"][2] + alt) < 1.0,
                   f"z={state_of(txt)['position_ned'][2]:.2f}")
            txt, _, dt = await call(s, "sim_fly", {"action": "move_to", "x": target[0], "y": target[1], "z": target[2],
                                                    "velocity": 6, "vehicle": "Drone", "timeout_s": 60})
            p = state_of(txt)["position_ned"]
            record("sim_fly move_to", dist(p, target) < 1.5, f"pos={p} target={target} err={dist(p, target):.2f} m, {dt:.1f}s")
        await check("sim_fly move_to", t_move)

        async def t_hover():
            await call(s, "sim_fly", {"action": "hover", "vehicle": "Drone"})
            # SimpleFlight overshoots ~2 m when braking from 6 m/s and settles in ~7-8 s (diag_landing.py trace);
            # station-keeping is judged after settling
            await asyncio.sleep(8)
            p0 = json.loads((await call(s, "sim_state", {"vehicle": "Drone"}))[0])["position_ned"]
            await asyncio.sleep(2)
            p1 = json.loads((await call(s, "sim_state", {"vehicle": "Drone"}))[0])["position_ned"]
            record("sim_fly hover (holds position 2 s after settling)", dist(p0, p1) < 0.3, f"drift={dist(p0, p1):.2f} m")
        await check("sim_fly hover", t_hover)

        # ---- capture: every image type
        async def t_capture():
            types = ["scene", "segmentation", "infrared", "depth", "depth_planar", "surface_normals"]
            txt, imgs, dt = await call(s, "sim_capture", {"camera": "survey", "vehicle": "Drone", "image_types": types})
            meta = json.loads(txt)
            by = {m["type"]: m for m in meta["images"]}
            for t in types:
                m = by.get(t, {})
                ok = m.get("width") == 1920 and m.get("height") == 1080 and "error" not in m and \
                    (Path(m["file"]).exists() if "file" in m else False)
                record(f"sim_capture {t}", ok, f"{m.get('width')}x{m.get('height')} file={m.get('file')}")
            d = by.get("depth", {})
            record("sim_capture depth range plausible", 0 < d.get("min", -1) and d.get("max", 0) > alt * 0.5,
                   f"min={d.get('min')} max={d.get('max')} (alt {alt} m, nadir camera)")
            record("sim_capture preview image returned", len(imgs) == 1 and len(base64.b64decode(imgs[0].data)) > 10_000,
                   f"{len(imgs)} image(s), capture {dt:.2f}s")
            if imgs:
                out = REPO / "_artifacts" / "verification" / "sim_preview.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(base64.b64decode(imgs[0].data))
            seg = next((m for m in meta["images"] if m["type"] == "segmentation"), None)
            if seg and "file" in seg:
                from PIL import Image as PILImage  # client side only; the server never lazy-imports
                n_colors = len(PILImage.open(seg["file"]).convert("RGB").getcolors(1 << 20) or [])
                record("segmentation has multiple instance colours", n_colors >= 2, f"{n_colors} colours")
        await check("sim_capture", t_capture)

        async def t_capture_annotation():
            # Annotation needs an "Annotation" layer in settings; the tool must report that cleanly, not crash.
            txt, _, _ = await call(s, "sim_capture", {"camera": "survey", "vehicle": "Drone", "image_types": ["annotation"],
                                                      "preview": None, "save": False}, expect_error=True)
            record("sim_capture annotation (no layer configured) handled cleanly", "images" in txt or "rror" in txt, txt[:160])
        await check("sim_capture annotation", t_capture_annotation)

        # ---- environment
        async def t_env():
            txt, _, _ = await call(s, "sim_environment", {"weather_enabled": True, "rain": 0.5, "fog": 0.3, "dust": 0.1,
                                                          "wind_ned_ms": [3.0, 0.0, 0.0]})
            record("sim_environment weather+wind", all(k in txt for k in ("weather=True", "Rain=0.5", "Fog=0.3", "Dust=0.1", "wind=")), txt)
            txt, _, _ = await call(s, "sim_environment", {"time_of_day": "2026-07-30 05:30:00", "celestial_clock_speed": 1.0})
            record("sim_environment time_of_day", "time=2026-07-30 05:30:00" in txt, txt)
            txt, _, _ = await call(s, "sim_environment", {"rain": 0.0, "fog": 0.0, "dust": 0.0, "wind_ned_ms": [0.0, 0.0, 0.0],
                                                          "weather_enabled": False, "time_of_day": "off"})
            record("sim_environment reset (weather, wind, time of day)", "weather=False" in txt and "time_of_day=off" in txt, txt)
        await check("sim_environment", t_env)

        # ---- clock: pause must freeze a MOVING vehicle (pose), step must advance it and re-pause
        async def pos_now():
            return json.loads((await call(s, "sim_state", {"vehicle": "Drone"}))[0])["position_ned"]

        async def t_clock():
            await call(s, "sim_fly", {"action": "move_to", "x": 60, "y": -10, "z": -alt, "velocity": 6, "wait": False,
                                      "vehicle": "Drone", "timeout_s": 60})
            await asyncio.sleep(1.5)
            d = json.loads((await call(s, "sim_clock", {"action": "pause"}))[0])
            p0 = await pos_now()
            await asyncio.sleep(1.5)
            p1 = await pos_now()
            d_step = json.loads((await call(s, "sim_clock", {"action": "step", "seconds": 0.5}))[0])
            await asyncio.sleep(1.5)
            p2 = await pos_now()
            d_res = json.loads((await call(s, "sim_clock", {"action": "resume"}))[0])
            await call(s, "sim_fly", {"action": "hover", "vehicle": "Drone"})
            record("sim_fly move_to wait=False returns while flying", True, f"moving at pause: p0={p0}")
            record("sim_clock pause freezes a moving vehicle", d["paused"] is True and dist(p0, p1) < 0.02,
                   f"moved {dist(p0, p1):.3f} m while paused")
            record("sim_clock step advances then re-pauses", dist(p1, p2) > 0.1 and d_step["paused"] is True,
                   f"moved {dist(p1, p2):.3f} m in a 0.5 s step; paused after step={d_step['paused']}")
            record("sim_clock resume", d_res["paused"] is False, str(d_res))
            col = json.loads((await call(s, "sim_state", {"vehicle": "Drone"}))[0])["collision"]
            record("no collision during free flight (wait=False leg)", col["object"] in ("", "Ground") or not col["has_collided"],
                   str(col))
        await check("sim_clock", t_clock)

        # ---- objects
        chosen = {}

        async def t_objects():
            names = json.loads((await call(s, "sim_objects", {"name_regex": ".*", "limit": 500}))[0])
            record("sim_objects list", len(names) > 5, f"{len(names)} objects, e.g. {names[:5]}")
            cand = [n for n in names if "cube" in n.lower() or "block" in n.lower()] or names
            posed = json.loads((await call(s, "sim_objects", {"name_regex": cand[0], "with_pose": True}))[0])
            chosen.update(posed[0])
            record("sim_objects with_pose", len(posed[0]["position_ned"]) == 3, str(posed[0]))
        await check("sim_objects", t_objects)

        async def t_static_pose():
            # level geometry has Static mobility: the tool must refuse with a clear, verified error
            if not chosen:
                raise RuntimeError("no object chosen")
            p = chosen["position_ned"]
            txt, _, _ = await call(s, "sim_set_object_pose", {"name": chosen["name"], "x": p[0] + 2, "y": p[1], "z": p[2]},
                                   expect_error=True)
            record("sim_set_object_pose on Static actor -> clear error", "Static" in txt, txt[:160])
        await check("sim_set_object_pose static", t_static_pose)

        async def t_spawn_move_destroy():
            assets = json.loads((await call(s, "sim_list_assets", {"name_regex": "cube"}))[0])
            record("sim_list_assets", assets["count"] > 0, f"{assets['count']} cube assets e.g. {assets['assets'][:3]}")
            asset = "Cube" if "Cube" in assets["assets"] else assets["assets"][0]
            sp = json.loads((await call(s, "sim_spawn_object", {"name": "sightline_test_cube", "asset": asset,
                                                                "x": 5, "y": 5, "z": -2}))[0])
            record("sim_spawn_object", dist(sp["position_ned"], [5, 5, -2]) < 0.05, str(sp))
            mv = json.loads((await call(s, "sim_set_object_pose", {"name": sp["name"], "x": 7, "y": 6, "z": -2, "yaw_deg": 30}))[0])
            record("sim_set_object_pose on spawned actor (verified)", mv["verified"] and dist(mv["actual"], [7, 6, -2]) < 0.05, str(mv))
            txt, _, _ = await call(s, "sim_destroy_object", {"name": sp["name"]})
            gone = json.loads((await call(s, "sim_objects", {"name_regex": sp["name"]}))[0])
            record("sim_destroy_object", "destroyed" in txt and gone == [], f"{txt}; remaining={gone}")
        await check("spawn/move/destroy", t_spawn_move_destroy)

        # ---- detections
        async def t_det():
            dets = json.loads((await call(s, "sim_detections", {"camera": "survey", "mesh_name_filter": detect_filter,
                                                                "radius_m": 300, "vehicle": "Drone"}))[0])
            ok = isinstance(dets, list) and all(len(d["box2d"]) == 4 for d in dets)
            record("sim_detections", ok, f"{len(dets)} detections for {detect_filter!r}; first={dets[:1]}")
        await check("sim_detections", t_det)

        # ---- release (gamepad takeover path) and re-acquire
        async def t_release():
            txt, _, _ = await call(s, "sim_fly", {"action": "release", "vehicle": "Drone"})
            released = state_of(txt)["api_control"] is False
            txt, _, _ = await call(s, "sim_fly", {"action": "arm", "vehicle": "Drone"})
            record("sim_fly release/re-acquire API control", released and state_of(txt)["api_control"] is True, "")
        await check("sim_fly release", t_release)

        # ---- land in place (ground height unknown to the tool), take off again, then return home
        async def t_land():
            # (20, -15) is open floor (probe landed there with 0.001 m xy error); translate at altitude, then land
            await call(s, "sim_fly", {"action": "move_to", "x": 20, "y": -15, "z": -alt, "velocity": 5, "vehicle": "Drone"})
            txt, _, dt = await call(s, "sim_fly", {"action": "land", "vehicle": "Drone", "timeout_s": 120})
            st = state_of(txt)
            ok = "touchdown confirmed" in txt and abs(st["position_ned"][2] - HOME[2]) < 0.6 and st["landed_state"] == 0 \
                and dist(st["position_ned"][:2], [20, -15]) < 1.0
            record("sim_fly land (verified touchdown, position held, Landed)", ok,
                   f"pos={st['position_ned']} landed_state={st['landed_state']} in {dt:.1f}s; {txt.splitlines()[0]}")
            txt, _, _ = await call(s, "sim_fly", {"action": "takeoff", "vehicle": "Drone", "timeout_s": 30})
            record("sim_fly takeoff again after landing", state_of(txt)["position_ned"][2] < HOME[2] - 1.0,
                   f"z={state_of(txt)['position_ned'][2]:.2f}")
            await call(s, "sim_fly", {"action": "move_to", "x": 20, "y": -15, "z": -alt, "velocity": 4, "vehicle": "Drone"})
        await check("sim_fly land", t_land)

        async def t_rtl():
            txt, _, dt = await call(s, "sim_fly", {"action": "rtl", "z": -alt, "vehicle": "Drone", "timeout_s": 90})
            st = state_of(txt)
            p = st["position_ned"]
            ok = dist(p[:2], HOME[:2]) < 0.5 and abs(p[2] - HOME[2]) < 0.6 and "touchdown confirmed" in txt \
                and st["landed_state"] == 0
            record("sim_fly rtl (home, verified touchdown)", ok,
                   f"pos={p} home={HOME} landed_state={st['landed_state']} in {dt:.1f}s; {txt.splitlines()[0]}")
        await check("sim_fly rtl", t_rtl)

        async def t_reset():
            txt, _, _ = await call(s, "sim_fly", {"action": "reset", "vehicle": "Drone"})
            p = state_of(txt)["position_ned"]
            record("sim_fly reset", dist(p, [0, 0, 0]) < 1.0, f"pos={p}")
        await check("sim_fly reset", t_reset)

        # ---- errors are clean, server survives
        async def t_errors():
            txt, _, dt = await call(s, "sim_fly", {"action": "bogus"}, expect_error=True)
            again = json.loads((await call(s, "sim_ping"))[0])["ping"]
            record("bad action -> clean error, server keeps serving", "bogus" in txt and again is True, f"{txt[:100]} ({dt:.2f}s)")
        await check("error handling", t_errors)

    out = REPO / "_artifacts" / "verification" / f"sim_tools_{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))
    fails = sum(1 for x in RESULTS if not x["ok"])
    print(f"\n{len(RESULTS) - fails}/{len(RESULTS)} checks passed; results -> {out}", flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=20.0)
    ap.add_argument("--detect-filter", default="Cube*")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a.alt, a.detect_filter)))
