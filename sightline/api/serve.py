"""Run the Sightline C2 server: map page + API + WebSocket + cloud sink, one process, loopback only.

    D:\\Tools\\uv\\uv.exe run python -m sightline.api.serve --port 8781 --demo
    then open  http://127.0.0.1:8781/app/map/index.html

Flags
-----
``--port``       TCP port on 127.0.0.1 (default 8781; **never** 8000 — that is the Unreal MCP server).
``--db``         record log path (default ``_artifacts/store/records.db``)
``--demo``       load the SIMULATED FloodValley scenario: synthetic records, plus a real plan-lane route,
                 its flown telemetry and a real `sightline.coverage` export written into ``--coverage``
``--fresh``      start the demo from an empty database file (a new file; nothing is ever deleted)
``--outbox``     also run the outbox + uploader against ``--cloud-url`` (default: this same server).
                 Point ``--cloud-url`` at a port with nothing on it to demonstrate the offline queue.
``--coverage``   directory holding the coverage lane's export (``coverage.json`` + per-layer PNG/GeoJSON;
                 default ``_artifacts/coverage/live``)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sightline.api.app import REPO, create_app
from sightline.api.mission_feed import MissionState
from sightline.store import Outbox, RecordStore, Uploader
from sightline.store.outbox import HttpTransport


def build(args: argparse.Namespace):
    db = Path(args.db)
    if args.fresh:
        db = db.with_name(f"{db.stem}_{int(time.time())}{db.suffix}")
    store = RecordStore(db)
    mission = MissionState()
    cov = Path(args.coverage)
    if args.demo:
        from sightline.api.demo_seed import seed_all

        seed_all(store, cov, mission)
    outbox = uploader = None
    if args.outbox:
        outbox = Outbox(Path(args.outbox_dir))
        url = args.cloud_url or f"http://127.0.0.1:{args.port}"
        uploader = Uploader(outbox, HttpTransport(url), base_backoff_s=0.5, max_backoff_s=30.0)
    app = create_app(store, outbox=outbox, uploader=uploader, mission=mission,
                     coverage_dir=cov, repo_root=REPO)
    #: --demo injects SIM-001..SIM-008 so the map is not blank. They are fixtures, and the dashboard
    #: labels them as such: nobody should ever mistake a seeded record for something the drone found.
    app.state.seeded = bool(args.demo)
    return app, store


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8781)
    p.add_argument("--db", default=str(REPO / "_artifacts" / "store" / "records.db"))
    p.add_argument("--coverage", default=str(REPO / "_artifacts" / "coverage" / "live"))
    p.add_argument("--outbox-dir", default=str(REPO / "_artifacts" / "store" / "outbox"))
    p.add_argument("--cloud-url", default="")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--outbox", action="store_true")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("refusing to bind a non-loopback address")

    import uvicorn

    app, store = build(args)
    # The demo buttons launch flights; those flights must stream back HERE, not to whatever port the
    # launcher's own default happened to be. See demo_control.SERVE_PORT.
    from sightline.api import demo_control

    demo_control.set_serve_port(args.port)
    print(f"Sightline C2  http://{args.host}:{args.port}/app/map/index.html   db={store.path}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, ws_ping_interval=20.0)


if __name__ == "__main__":
    main()
