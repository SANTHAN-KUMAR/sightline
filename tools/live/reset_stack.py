"""ONE command to get back to exactly one clean dashboard.

    uv run python tools/live/reset_stack.py              # stop everything, start one empty C2 on :8781
    uv run python tools/live/reset_stack.py --seed       # ...with the SIM-00x demo fixtures loaded
    uv run python tools/live/reset_stack.py --port 8900  # somewhere else
    uv run python tools/live/reset_stack.py --stop-only  # stop everything, start nothing

Why this exists: several servers accumulate. Each demo session, each abandoned terminal and each restart
leaves one holding a port, and five of them on a 16 GB laptop is enough to make the machine crawl. Worse,
they are indistinguishable from the outside - a flight streams to ONE of them, and a dashboard showing any
of the others sits there looking like a dead pipeline.

So this stops every Sightline server and flight, then starts exactly one, and prints the one URL.

It starts the C2 **without fixtures by default**. `serve --demo` seeds SIM-001..SIM-008 so the map is not
blank, which is useful for looking at the interface and actively misleading during a live demo: eight canned
records sit in the list looking exactly like detections the aircraft just made. Empty, filling from the
flight, is the honest demo - and the one worth showing.

NOTHING HERE DELETES DATA. It terminates processes and opens a new database file. Existing record stores are
left on disk exactly as they are, which guardrail R10 requires.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

UV = r"D:\Tools\uv\uv.exe"


def health(port: int, timeout: float = 2.0) -> dict | None:
    try:
        import json

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:                                          # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--seed", action="store_true",
                    help="load the SIM-00x demo fixtures. Off by default: during a live demo they are "
                         "indistinguishable from real finds.")
    ap.add_argument("--stop-only", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print("SIGHTLINE - reset to one clean stack")
    print("=" * 74)

    # 1. stop everything ---------------------------------------------------------------------------
    from tools.live.stop_all import main as stop_main                        # noqa: PLC0415

    sys.argv = ["stop_all"]
    stop_main()

    if a.stop_only:
        print("\n--stop-only: nothing was started.")
        return 0

    # 2. wait for the port to actually free up ------------------------------------------------------
    for _ in range(20):
        if health(a.port, timeout=1.0) is None:
            break
        time.sleep(1.0)
    else:
        print(f"\nFAIL: something is still answering on :{a.port}. "
              f"Re-run, or pick another port with --port.")
        return 2

    # 3. start exactly one -------------------------------------------------------------------------
    cmd = [UV, "run", "python", "-m", "sightline.api.serve", "--port", str(a.port), "--fresh"]
    if a.seed:
        cmd.append("--demo")
    print(f"\nstarting: {' '.join(cmd[3:])}")
    subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):
        h = health(a.port)
        if h is not None:
            recs = h.get("store", {}).get("records", "?")
            print(f"\n  C2 up on :{a.port}   records: {recs}"
                  f"{'  (SEEDED FIXTURES)' if a.seed else '  (empty - it fills from the flight)'}")
            print(f"\n  OPEN THIS:  http://127.0.0.1:{a.port}/app/map/index.html")
            print("\n  Then: Run tab -> check the Connection panel -> press Start on a demo.")
            print("  The Connection panel names any link that is down and what to do about it.")
            return 0
        time.sleep(1.0)

    print(f"\nFAIL: the server did not come up on :{a.port} within 60 s.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
