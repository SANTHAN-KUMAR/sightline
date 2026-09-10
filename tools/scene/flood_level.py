"""Set or read the FloodValley flood level at runtime through the AirSim API (SOLUTION_DOC §5.1 step 3).

The water is one MOVABLE single-layer-water plane named "FloodWater" (tools/scene/build_flood_valley.py), so the
level is a simSetObjectPose away. Levels are given in metres above sea level, the unit the scenario, the
submersion attribute (pelvis height vs FloodLevel, §5.1 auto-labels) and the telemetry all use.

NED frame: AirSim's NED origin is the PlayerStart, z positive DOWN. With the PlayerStart at UE Z = ps_z cm and
UE Z = (asl - base_z) * 100 cm, the water plane's NED z is  ps_z/100 - (asl - base_z).

    uv run python tools/scene/flood_level.py            # print the current level
    uv run python tools/scene/flood_level.py 1062.5     # set it, verified by read-back
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cosysairsim as airsim

META = Path(__file__).resolve().parents[2] / "data" / "scene" / "flood_valley.json"
WATER = "FloodWater"


def _frame() -> tuple[float, list[float]]:
    meta = json.loads(META.read_text())
    return meta["base_z_m"], meta["ue_import"]["player_start_cm"]


def get_flood_level(client: airsim.MultirotorClient) -> float:
    base, ps = _frame()
    pose = client.simGetObjectPose(WATER)
    if pose.position.containsNan():
        raise RuntimeError(f"{WATER!r} not found in the scene (is FloodValley loaded?)")
    return base + ps[2] / 100.0 - pose.position.z_val


def set_flood_level(client: airsim.MultirotorClient, asl_m: float, tol_m: float = 0.01) -> float:
    base, ps = _frame()
    pose = airsim.Pose(airsim.Vector3r(-ps[0] / 100.0, -ps[1] / 100.0, ps[2] / 100.0 - (asl_m - base)),
                       airsim.Quaternionr(0.0, 0.0, 0.0, 1.0))
    if not client.simSetObjectPose(WATER, pose, teleport=True):
        raise RuntimeError(f"simSetObjectPose({WATER!r}) returned False: the actor must have Movable mobility")
    got = get_flood_level(client)
    if abs(got - asl_m) > tol_m:
        raise RuntimeError(f"flood level read back {got:.3f} m, requested {asl_m:.3f} m")
    return got


if __name__ == "__main__":
    c = airsim.MultirotorClient()
    c.confirmConnection()
    if len(sys.argv) > 1:
        print(f"flood level set to {set_flood_level(c, float(sys.argv[1])):.3f} m ASL")
    else:
        print(f"flood level {get_flood_level(c):.3f} m ASL")
