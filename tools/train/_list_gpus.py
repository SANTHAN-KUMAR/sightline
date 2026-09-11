"""Every GPU RunPod will actually rent us right now, biggest first. `plan` caps its table at 12 rows sorted
cheapest-first, which hides exactly the cards worth choosing when speed is the constraint."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runpod_train import gql  # noqa: E402

d = gql("query { gpuTypes { id displayName memoryInGb communityCloud secureCloud "
        "lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }")
rows = [g for g in d["gpuTypes"] if (g.get("lowestPrice") or {}).get("uninterruptablePrice")]
rows.sort(key=lambda g: (-int(g.get("memoryInGb") or 0), g["lowestPrice"]["uninterruptablePrice"]))

print(f"{'GPU':34s} {'VRAM':>5s} {'$/h':>7s}  cloud")
for g in rows:
    price = g["lowestPrice"]["uninterruptablePrice"]
    cloud = ("SECURE" if g.get("secureCloud") else "") + ("/COMMUNITY" if g.get("communityCloud") else "")
    print(f"{g['id'][:34]:34s} {int(g.get('memoryInGb') or 0):>5d} {price:>7.3f}  {cloud.strip('/')}")
print(f"\n{len(rows)} rentable GPU types")
