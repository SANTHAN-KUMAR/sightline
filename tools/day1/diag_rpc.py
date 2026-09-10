"""Direct (no MCP) Cosys-AirSim RPC diagnostic with per-step timing. Use it to tell whether a hang is in the sim or
in the MCP layer. Run: uv run python -u tools/day1/diag_rpc.py [--vehicle Drone] [--camera survey]"""

import argparse
import time

import cosysairsim as airsim


def step(name, fn):
    t = time.time()
    out = fn()
    shown = type(out).__name__ if hasattr(out, "image_data_uint8") or hasattr(out, "kinematics_estimated") else repr(out)
    print(f"{name}: {shown}  ({time.time() - t:.2f}s)", flush=True)
    return out


def main(vehicle: str, camera: str) -> None:
    c = step("client", lambda: airsim.MultirotorClient(timeout_value=15))
    step("ping", c.ping)
    step("server_version", c.getServerVersion)
    step("vehicles", c.listVehicles)
    s = step("state", lambda: c.getMultirotorState(vehicle_name=vehicle))
    print("  landed_state", s.landed_state, "pos", s.kinematics_estimated.position, flush=True)
    step("enableApiControl", lambda: c.enableApiControl(True, vehicle_name=vehicle))
    step("arm", lambda: c.armDisarm(True, vehicle_name=vehicle))
    step("takeoff", lambda: c.takeoffAsync(timeout_sec=20, vehicle_name=vehicle).join())
    s = c.getMultirotorState(vehicle_name=vehicle)
    print("  pos after takeoff", s.kinematics_estimated.position, flush=True)
    for t in ("Scene", "Segmentation", "Infrared", "DepthPerspective"):
        is_f = t.startswith("Depth")
        r = step(f"image {t}", lambda t=t, is_f=is_f: c.simGetImages(
            [airsim.ImageRequest(camera, getattr(airsim.ImageType, t), is_f, not is_f)], vehicle_name=vehicle)[0])
        print(f"  {t}: {r.width}x{r.height}", flush=True)
    step("land", lambda: c.landAsync(timeout_sec=30, vehicle_name=vehicle).join())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", default="Drone")
    ap.add_argument("--camera", default="survey")
    a = ap.parse_args()
    main(a.vehicle, a.camera)
