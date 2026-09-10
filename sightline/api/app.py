"""F15 + F18 backend: the map's data source, the live WebSocket feed, and the cloud sink.

SOLUTION_DOC §5.9 ("a FastAPI backend pushing GeoJSON over a WebSocket at each record update") and §5.10 (the
outbox's idempotent upsert, and the ``POST /detect`` cloud-fallback route).

Routes
------
``GET  /``                                 -> redirect to the C2 map page
``GET  /health``                           liveness + store/outbox/live counters
``GET  /api/records.geojson``              RFC 7946 FeatureCollection, ranked (``?include_dismissed=0``)
``GET  /api/records/{id}``                 one Feature
``GET  /api/records/{id}/history``         every version of it (R10: history is never lost)
``GET  /api/records/{id}/audit``           who changed it, when, and why
``POST /api/records/{id}/dismiss``         ``{"reason": str, "by": str}`` — reason REQUIRED (R10)
``POST /api/records/{id}/status``          ``{"status": str, "actor": str, "reason": str}``
``POST /api/records/{id}/note``            free-text, e.g. "dispatched to Team Bravo" (§7 step 5)
``GET  /api/thumbs/{name}``                the evidence crop, as a file
``GET  /api/coverage/overlay.json``        the B6 search-quality overlay sidecar (see coverage_feed.py)
``GET  /api/coverage/pod.png``             the POD raster, north-up
``GET  /api/coverage/cannot_clear.geojson``the "aerial search cannot clear" polygons (hatched on the map)
``GET  /api/mission``                      drone / footprint / track / plan
``GET  /api/outbox``                       queue depth + link state for the status bar (§7 step 6)
``POST /api/upload``                       CLOUD SINK: idempotent upsert keyed by (kind, clip, record, ver)
``POST /detect``                           CLOUD FALLBACK: JPEG tile in, detections out (STUB detector)
``POST /api/detect/local``                 push the edge's own result for a frame, so /detect can merge it
``GET  /api/detect/frames/{clip}/{idx}``   the merged per-frame result
``WS   /ws``                               the live feed (see live.py for the message format)

Static: ``/app/...`` and ``/data/basemap/...`` are served from the repo with HTTP Range support (PMTiles reads
byte ranges), so one process serves the whole demo. ``app/map/serve.mjs`` remains the dependency-free
alternative and the two use identical URLs.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from sightline.api.coverage_feed import OVERLAY_CONTRACT, read_overlay
from sightline.api.detect_fallback import DETECT_CONTRACT, DetectorPlugin, FrameMerger, StubDetector
from sightline.api.live import LiveHub
from sightline.api.mission_feed import MissionState
from sightline.api.wire import feature_to_record
from sightline.schemas import SCHEMA_VERSION, Record, ce90_m
from sightline.store import GuardrailError, Outbox, RecordStore, StaleVersionError, Uploader

__all__ = ["create_app", "REPO"]

REPO = Path(__file__).resolve().parents[2]


def _feature(rec: Record) -> dict[str, Any]:
    f = rec.to_feature()
    # Derived conveniences the map needs but the schema does not store (never overriding a stored field).
    f["properties"]["ce90_m"] = round(ce90_m(rec.h_acc_m), 2)
    f["properties"]["score_total"] = round(rec.components.total(), 6)
    return f


def create_app(
    store: RecordStore,
    *,
    outbox: Outbox | None = None,
    uploader: Uploader | None = None,
    upload_store: RecordStore | None = None,
    mission: MissionState | None = None,
    coverage_dir: str | os.PathLike[str] | None = None,
    detector: DetectorPlugin | None = None,
    repo_root: str | os.PathLike[str] = REPO,
    serve_static: bool = True,
    title: str = "Sightline C2",
) -> FastAPI:
    """Build the app. ``upload_store`` is the CLOUD side of §5.10; leave it None to upsert into ``store``."""
    repo = Path(repo_root)
    hub = LiveHub()
    mission = mission or MissionState()
    detector = detector or StubDetector()
    merger = FrameMerger()
    cloud_store = upload_store or store
    cov_dir = Path(coverage_dir) if coverage_dir else repo / "_artifacts" / "coverage" / "live"
    seen_uploads: dict[str, float] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.bind_loop(asyncio.get_running_loop())
        unsub = store.subscribe(
            lambda rec, event: hub.publish_threadsafe("record", op="upsert", event=event, feature=_feature(rec))
        )
        if uploader is not None and not uploader.is_alive():
            uploader.start()
        try:
            yield
        finally:
            unsub()
            if uploader is not None:
                uploader.stop()

    app = FastAPI(title=title, version=SCHEMA_VERSION, lifespan=lifespan)
    app.state.store = store
    app.state.hub = hub
    app.state.mission = mission
    app.state.outbox = outbox
    app.state.uploader = uploader
    app.state.merger = merger
    app.state.detector = detector
    app.state.coverage_dir = cov_dir

    # ---- health & meta -------------------------------------------------------------------------------
    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "t_utc": time.time(),
            "schema_version": SCHEMA_VERSION,
            "store": store.stats(),
            "outbox": (uploader.stats() if uploader is not None else (outbox.stats() if outbox else None)),
            "live_clients": hub.clients,
            "live_seq": hub.seq,
            "detector": {"name": detector.name, "is_stub": getattr(detector, "is_stub", True)},
            "contracts": {"overlay": OVERLAY_CONTRACT, "detect": DETECT_CONTRACT},
        }

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/app/map/index.html")

    # ---- records -------------------------------------------------------------------------------------
    @app.get("/api/records.geojson")
    def records_geojson(
        include_dismissed: bool = Query(True),
        status: str | None = Query(None),
        limit: int | None = Query(None, ge=1, le=10000),
    ) -> dict[str, Any]:
        recs = store.records(include_dismissed=include_dismissed, status=status, limit=limit)
        return {
            "type": "FeatureCollection",
            "schema_version": SCHEMA_VERSION,
            "generated_utc": time.time(),
            "features": [_feature(r) for r in recs],
        }

    @app.get("/api/records/{record_id}")
    def one_record(record_id: str) -> dict[str, Any]:
        rec = store.get(record_id)
        if rec is None:
            raise HTTPException(404, f"no record {record_id}")
        return _feature(rec)

    @app.get("/api/records/{record_id}/history")
    def record_history(record_id: str) -> dict[str, Any]:
        hist = store.history(record_id)
        if not hist:
            raise HTTPException(404, f"no record {record_id}")
        return {"record_id": record_id, "versions": [_feature(r) for r in hist]}

    @app.get("/api/records/{record_id}/audit")
    def record_audit(record_id: str) -> dict[str, Any]:
        return {"record_id": record_id, "audit": store.audit_log(record_id)}

    @app.post("/api/records/{record_id}/dismiss")
    def dismiss(record_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """R10: dismissal keeps the row and demands a reason. There is no delete endpoint."""
        reason = str(body.get("reason", "") or "").strip()
        if not reason:
            raise HTTPException(422, "R10: dismissal requires a non-empty reason")
        try:
            rec = store.dismiss(record_id, reason=reason, by=str(body.get("by", "operator")))
        except KeyError:
            raise HTTPException(404, f"no record {record_id}") from None
        except GuardrailError as e:
            raise HTTPException(422, str(e)) from None
        _enqueue(rec)
        return {"ok": True, "feature": _feature(rec)}

    @app.post("/api/records/{record_id}/status")
    def set_status(record_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        st = str(body.get("status", ""))
        try:
            rec = store.set_status(
                record_id, st, actor=str(body.get("actor", "operator")), reason=str(body.get("reason", ""))
            )
        except KeyError:
            raise HTTPException(404, f"no record {record_id}") from None
        except (ValueError, GuardrailError) as e:
            raise HTTPException(422, str(e)) from None
        _enqueue(rec)
        return {"ok": True, "feature": _feature(rec)}

    @app.post("/api/records/{record_id}/note")
    def add_note(record_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        note = str(body.get("note", "") or "").strip()
        if not note:
            raise HTTPException(422, "note must not be empty")
        try:
            rec = store.add_note(record_id, note, actor=str(body.get("actor", "operator")))
        except KeyError:
            raise HTTPException(404, f"no record {record_id}") from None
        _enqueue(rec)
        return {"ok": True, "feature": _feature(rec)}

    def _enqueue(rec: Record) -> None:
        if outbox is not None:
            outbox.enqueue_record(rec)

    # ---- evidence ------------------------------------------------------------------------------------
    @app.get("/api/thumbs/{name}")
    def thumb(name: str) -> FileResponse:
        p = store.thumbnail_path(name)
        if p is None:
            raise HTTPException(404, f"no thumbnail {name}")
        return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})

    # ---- coverage (B6 contract) ----------------------------------------------------------------------
    @app.get("/api/coverage/overlay.json")
    def coverage_overlay() -> JSONResponse:
        try:
            side = read_overlay(cov_dir)
        except ValueError as e:
            raise HTTPException(500, str(e)) from None
        if side is None:
            return JSONResponse({"contract": OVERLAY_CONTRACT, "available": False, "dir": str(cov_dir)}, 404)
        side["available"] = True
        return JSONResponse(side)

    @app.get("/api/coverage/pod.png")
    def coverage_png() -> FileResponse:
        p = cov_dir / "pod.png"
        if not p.is_file():
            raise HTTPException(404, "no coverage raster yet")
        return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/api/coverage/cannot_clear.geojson")
    def coverage_cannot_clear() -> Response:
        p = cov_dir / "cannot_clear.geojson"
        if not p.is_file():
            return JSONResponse({"type": "FeatureCollection", "features": []})
        return Response(p.read_bytes(), media_type="application/geo+json")

    # ---- mission -------------------------------------------------------------------------------------
    @app.get("/api/mission")
    def mission_geojson() -> dict[str, Any]:
        return mission.as_geojson()

    # ---- outbox --------------------------------------------------------------------------------------
    @app.get("/api/outbox")
    def outbox_stats() -> dict[str, Any]:
        if uploader is not None:
            return uploader.stats()
        if outbox is not None:
            return outbox.stats()
        return {"depth": 0, "online": None, "note": "no outbox attached"}

    # ---- cloud sink: the idempotent upsert of §5.10 ---------------------------------------------------
    @app.post("/api/upload")
    def upload(job: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Idempotent by ``(kind, clip_id, record_id, version)``. A replayed job returns applied=false."""
        key = str(job.get("key") or "")
        kind = str(job.get("kind") or "")
        if not key or kind not in ("record", "thumb"):
            raise HTTPException(422, "job needs a key and kind in {record, thumb}")
        payload = job.get("payload") or {}
        if kind == "record":
            try:
                rec = feature_to_record(payload["feature"])
            except (KeyError, ValueError, TypeError) as e:
                raise HTTPException(422, f"bad record payload: {e}") from None
            existing = cloud_store.get(rec.record_id)
            if existing is not None and rec.version <= existing.version:
                same = cloud_store.get(rec.record_id, version=rec.version)
                if same is not None and same == rec:
                    seen_uploads[key] = time.time()
                    return {"ok": True, "applied": False, "reason": "duplicate", "key": key,
                            "version": rec.version}
                raise HTTPException(409, f"stale version {rec.version} for {rec.record_id}")
            try:
                cloud_store.put(rec, actor="uploader", reason="outbox upload")
            except StaleVersionError as e:
                raise HTTPException(409, str(e)) from None
            except GuardrailError as e:
                raise HTTPException(422, str(e)) from None
            seen_uploads[key] = time.time()
            return {"ok": True, "applied": True, "key": key, "record_id": rec.record_id,
                    "version": rec.version}

        data = base64.b64decode(payload.get("b64", ""))
        want = str(payload.get("sha256", ""))
        got = hashlib.sha256(data).hexdigest()
        if want and want != got:
            raise HTTPException(422, "thumbnail sha256 mismatch")
        name = Path(str(payload.get("name", ""))).name
        if not name:
            raise HTTPException(422, "thumbnail needs a name")
        dest = cloud_store.thumbs_dir / name
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == got:
            seen_uploads[key] = time.time()
            return {"ok": True, "applied": False, "reason": "duplicate", "key": key, "bytes": len(data)}
        dest.write_bytes(data)
        seen_uploads[key] = time.time()
        return {"ok": True, "applied": True, "key": key, "bytes": len(data)}

    @app.get("/api/upload/keys")
    def upload_keys() -> dict[str, Any]:
        return {"count": len(seen_uploads), "keys": sorted(seen_uploads)}

    # ---- cloud fallback detection (§5.10) — STUB detector ---------------------------------------------
    @app.post("/detect")
    async def detect(
        file: UploadFile = File(...),
        clip_id: str = Form(""),
        frame_idx: int = Form(-1),
        t_utc: float = Form(0.0),
        conf: float = Form(0.25),
        imgsz: int = Form(1920),
        merge: bool = Form(True),
        iou_thresh: float = Form(0.5),
    ) -> dict[str, Any]:
        """Ship a downscaled JPEG, get boxes back, merged into the frame's local result by ``frame_idx``.

        The detector is pluggable; the default is a labelled STUB that returns no boxes (the ML lane owns
        the real engine). ``latency_ms`` is measured so the edge can decide whether the round trip beat its
        per-frame deadline.
        """
        raw = await file.read()
        t0 = time.perf_counter()
        meta = {"clip_id": clip_id, "frame_idx": frame_idx, "t_utc": t_utc, "conf": conf, "imgsz": imgsz,
                "content_type": file.content_type, "bytes": len(raw)}
        try:
            dets = detector.detect(raw, meta)
        except Exception as e:  # a cloud failure must never take the edge down
            raise HTTPException(503, f"detector failed: {type(e).__name__}: {e}") from None
        latency_ms = (time.perf_counter() - t0) * 1e3
        merged = merger.merge_cloud(clip_id, frame_idx, dets, iou_thresh=iou_thresh) if merge else None
        return {
            "contract": DETECT_CONTRACT,
            "clip_id": clip_id,
            "frame_idx": frame_idx,
            "detector": detector.name,
            "is_stub": bool(getattr(detector, "is_stub", True)),
            "bytes_in": len(raw),
            "latency_ms": round(latency_ms, 3),
            "detections": dets,
            "merged": merged,
        }

    @app.post("/api/detect/local")
    def detect_local(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """The edge posts its own per-frame result so a late ``/detect`` reply has something to merge into."""
        merger.add_local(str(body.get("clip_id", "")), int(body.get("frame_idx", -1)),
                         list(body.get("detections") or []))
        return {"ok": True, "stats": merger.stats()}

    @app.get("/api/detect/frames/{clip_id}/{frame_idx}")
    def detect_frame(clip_id: str, frame_idx: int) -> dict[str, Any]:
        return {"clip_id": clip_id, "frame_idx": frame_idx,
                "detections": merger.get(clip_id, frame_idx), "stats": merger.stats()}

    # ---- live feed -----------------------------------------------------------------------------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        await hub.register(ws)
        try:
            await hub.send_to(ws, "hello", server="sightline-api", records=store.stats()["records"],
                              contracts={"overlay": OVERLAY_CONTRACT, "detect": DETECT_CONTRACT})
            await hub.send_to(
                ws, "snapshot",
                records={"type": "FeatureCollection", "schema_version": SCHEMA_VERSION,
                         "generated_utc": time.time(),
                         "features": [_feature(r) for r in store.records()]},
            )
            m = mission.as_geojson()
            if m["drone"] or m["plan"]:
                await hub.send_to(ws, "mission", **m)
            while True:
                text = await ws.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "ping":
                    await hub.send_to(ws, "pong", echo=msg.get("echo"))
                elif msg.get("type") == "resync":
                    await hub.send_to(
                        ws, "snapshot",
                        records={"type": "FeatureCollection", "schema_version": SCHEMA_VERSION,
                                 "generated_utc": time.time(),
                                 "features": [_feature(r) for r in store.records()]},
                    )
        except WebSocketDisconnect:
            pass
        finally:
            await hub.unregister(ws)

    # ---- broadcast helpers the sim/mission lane can call ----------------------------------------------
    @app.post("/api/mission/pose", include_in_schema=False)
    def push_pose(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Minimal ingress for the mission lane while it is out of process (fields mirror `Telemetry`)."""
        from sightline.schemas import Telemetry

        t = Telemetry(**{k: v for k, v in body.items() if k in Telemetry.__slots__})
        mission.update_pose(t, footprint=body.get("footprint"))
        hub.publish_threadsafe("mission", **mission.as_geojson())
        return {"ok": True}

    # ---- headless-check report sink (mirrors app/map/serve.mjs) ---------------------------------------
    @app.post("/__report", include_in_schema=False)
    async def report(request: Request) -> Response:
        body = (await request.body()).decode("utf-8", "replace")
        app.state.last_report = body
        print(f"REPORT {body}", flush=True)
        return Response(status_code=204)

    # ---- static: the map page, the vendored MapLibre stack, the PMTiles basemap -----------------------
    if serve_static:
        app_dir, basemap_dir = repo / "app", repo / "data" / "basemap"
        if app_dir.is_dir():
            app.mount("/app", StaticFiles(directory=str(app_dir)), name="app")
        if basemap_dir.is_dir():
            app.mount("/data/basemap", StaticFiles(directory=str(basemap_dir)), name="basemap")
    return app
