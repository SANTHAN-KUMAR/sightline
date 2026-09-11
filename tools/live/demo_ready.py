"""Is the system actually ready to demonstrate? One command, one verdict, no optimism.

    uv run python tools/live/demo_ready.py
    uv run python tools/live/demo_ready.py --json _artifacts/demo_ready.json

Every check answers a question a demo can fail on, and each one either MEASURES something or reports that it
could not. A check that cannot run counts as a failure, never as a pass - this project has already shipped
one gate that printed "all checks passed" while silently skipping the check that mattered.

Nothing here starts a flight, opens the editor or spends money. It looks at what exists and what answers.
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WEIGHTS = REPO / "models/detect/f8b_sim/weights/best.pt"
ENGINE = REPO / "models/detect/f8b_sim/weights/best.engine"
RESULTS = REPO / "models/detect/f8b_sim/results.csv"
BENCH = REPO / "_artifacts/engine_bench.json"


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []   # (state, name, detail)

    def add(self, ok: bool | None, name: str, detail: str) -> None:
        self.rows.append(("PASS" if ok else ("WARN" if ok is None else "FAIL"), name, detail))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == "FAIL")


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect((host, port))
        return True
    except Exception:                                            # noqa: BLE001
        return False
    finally:
        s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--c2-port", type=int, default=8781)
    a = ap.parse_args()
    c = Check()

    # --- 1. the trained model ---------------------------------------------------------------------------
    if WEIGHTS.exists():
        mb = WEIGHTS.stat().st_size / 1048576
        c.add(True, "trained weights", f"{WEIGHTS.relative_to(REPO)} ({mb:.1f} MB)")
    else:
        c.add(False, "trained weights", f"missing: {WEIGHTS.relative_to(REPO)} - run the RunPod fine-tune")

    if RESULTS.exists():
        last = [ln for ln in RESULTS.read_text().splitlines() if ln.strip()][-1].split(",")
        try:
            c.add(True, "training metrics", f"final epoch mAP50 {float(last[7]):.3f}, "
                                            f"precision {float(last[5]):.3f}, recall {float(last[6]):.3f}")
        except (IndexError, ValueError):
            c.add(None, "training metrics", "results.csv present but the columns did not parse")
    else:
        c.add(False, "training metrics", "no results.csv - cannot show the run was healthy")

    # --- 2. the TensorRT engine -------------------------------------------------------------------------
    if ENGINE.exists():
        mb = ENGINE.stat().st_size / 1048576
        fresh = ENGINE.stat().st_mtime >= WEIGHTS.stat().st_mtime if WEIGHTS.exists() else True
        c.add(fresh, "tensorrt engine",
              f"{mb:.0f} MB" + ("" if fresh else " - OLDER than best.pt; rebuild or it is a stale model"))
    else:
        c.add(None, "tensorrt engine",
              "not built - the demo still runs on best.pt at ~167 ms/frame uncontended")

    if BENCH.exists():
        b = json.loads(BENCH.read_text())
        c.add(True, "engine benchmark",
              f"pytorch {b.get('pytorch_fp16_ms')} ms -> tensorrt {b.get('tensorrt_fp16_ms')} ms "
              f"({b.get('speedup')}x)")
    else:
        c.add(None, "engine benchmark", "not measured - do not quote a speedup without it")

    # --- 3. the command map -----------------------------------------------------------------------------
    if port_open(a.c2_port):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{a.c2_port}/health", timeout=5) as r:
                h = json.load(r)
            st = h.get("store", {})
            det = (h.get("detector") or {}).get("name", "?")
            c.add(True, "C2 server", f"port {a.c2_port}, {st.get('records', 0)} records, "
                                     f"schema {h.get('schema_version')}, detector={det}")
            # NOT a demo blocker, and asserting it was a mistake. This field describes the C2's OPTIONAL
            # /detect cloud-fallback endpoint (F18, "cloud detect stretch"), which exists so an edge device
            # that misses its deadline can ship a frame to the server. The demo's detection happens inside
            # sightline.mission.live, which loads the weights itself and POSTs records here - the C2 never
            # runs a detector on the demo path. Reported, not failed.
            c.add(None if det == "stub" else True, "C2 /detect fallback",
                  "a real detector is attached" if det != "stub"
                  else "stub (F18 stretch) - fine: the demo detects in live.py, not in the server")
        except Exception as exc:                                 # noqa: BLE001
            c.add(False, "C2 server", f"port open but /health failed: {type(exc).__name__}")
    else:
        c.add(False, "C2 server", f"nothing on 127.0.0.1:{a.c2_port} - "
                                  f"`uv run python -m sightline.api.serve --port {a.c2_port}`")

    basemap = REPO / "data/basemap/wayanad.pmtiles"
    c.add(basemap.exists(), "offline basemap",
          f"{basemap.stat().st_size/1048576:.1f} MB" if basemap.exists() else "missing wayanad.pmtiles")

    # --- 4. the simulator -------------------------------------------------------------------------------
    if port_open(41451):
        try:
            import cosysairsim as airsim                          # noqa: PLC0415

            cl = airsim.MultirotorClient()
            cl.confirmConnection()
            veh = list(cl.listVehicles())
            c.add(bool(veh), "simulator", f"AirSim up, vehicles={veh}" if veh else
                  "AirSim up but NO VEHICLE - PIE is in Simulate mode, not Play")
        except Exception as exc:                                 # noqa: BLE001
            c.add(False, "simulator", f"port 41451 open but RPC failed: {type(exc).__name__}")
    else:
        c.add(None, "simulator", "not running - needed for a LIVE flight, not for a replay demo")

    # --- 5. the data ------------------------------------------------------------------------------------
    runs = sorted((REPO / "_artifacts/dataset").glob("seed*")) if (REPO / "_artifacts/dataset").exists() else []
    seeds = set()
    boxes = 0
    for r in runs:
        card = r / "data_card.json"
        if card.exists():
            d = json.loads(card.read_text())
            seeds.add(d.get("scenario_seed"))
            boxes += int(d.get("total_boxes", 0))
    c.add(len(seeds) >= 2, "held-out split",
          f"{len(runs)} runs, seeds {sorted(x for x in seeds if x is not None)}, {boxes} boxes"
          + ("" if len(seeds) >= 2 else " - ONE seed cannot produce an honest val split"))

    # --- 6. the test suite ------------------------------------------------------------------------------
    try:
        # No --timeout flag: pytest-timeout is not a dependency here, and passing it made pytest exit 4
        # (usage error), which this gate then reported as a project failure. The subprocess timeout below
        # is the real guard. A gate that fails on its own invocation is worse than no gate.
        p = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                           cwd=str(REPO), capture_output=True, text=True, timeout=900)
        tail = [ln for ln in p.stdout.splitlines() if "passed" in ln or "failed" in ln]
        c.add(p.returncode == 0, "test suite", tail[-1].strip() if tail else f"exit {p.returncode}")
    except Exception as exc:                                     # noqa: BLE001
        c.add(False, "test suite", f"could not run: {type(exc).__name__}")

    # --- report -----------------------------------------------------------------------------------------
    print("=" * 78)
    print("SIGHTLINE — DEMO READINESS")
    print("=" * 78)
    for state, name, detail in c.rows:
        print(f"  {state:4s}  {name:22s}  {detail}")
    print("=" * 78)
    verdict = "READY" if c.failed == 0 else f"NOT READY — {c.failed} blocking failure(s)"
    print(verdict)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"verdict": verdict, "checks": [{"state": s, "name": n, "detail": d} for s, n, d in c.rows]},
            indent=1), encoding="utf-8")
        print(f"wrote {a.json}")
    return 1 if c.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
