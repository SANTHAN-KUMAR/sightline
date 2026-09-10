"""Functional verification of every Python tool named in the solution document (docs/SOLUTION_DOC.md 3.4, 5.4-5.12,
Appendix A). One real functional test per tool, not just an import.

Markers: gpu (CUDA), network (downloads on first run; cached afterwards), slow (> ~30 s).
    uv run pytest tests/test_stack.py -m "not network"        # offline subset
    uv run pytest tests/test_stack.py -m "gpu"                # GPU subset
Results/timings land in _artifacts/stack/metrics.json; tools/verify_stack.py turns them into a summary.
"""
# NB: no `from __future__ import annotations` here: FastAPI resolves endpoint annotations at runtime, and a
# locally imported `WebSocket` would become an unresolvable string (the socket is then rejected with 403).
import json
import math
import os
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest
from conftest import DEM_DIR, MODELS, REPO, SAMPLES

WAYANAD = (11.4870, 76.1450)  # lat, lon (OriginGeopoint, docs/CONTEXT.md)


def _download(url: str, dst: Path, min_bytes: int = 1000) -> Path:
    if not dst.exists() or dst.stat().st_size < min_bytes:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(dst)
    return dst


# ----------------------------------------------------------------------------------------------------------------
# PyTorch / CUDA
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.gpu
def test_torch_cuda_matmul(cuda, record):
    import torch
    import torchvision

    a = torch.randn(2048, 2048, device=cuda, dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ a
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 10
    ref = (a.float().cpu() @ a.float().cpu())
    assert torch.allclose(c.float().cpu(), ref, rtol=2e-2, atol=2.0)
    # torchvision CUDA op
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float32, device=cuda)
    keep = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=cuda), 0.5)
    assert keep.tolist() == [0, 2]
    record("torch", version=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
           device=torch.cuda.get_device_name(0), fp16_matmul_2048_ms=round(dt * 1e3, 3),
           torchvision=torchvision.__version__)


# ----------------------------------------------------------------------------------------------------------------
# Ultralytics YOLO26 (+ TensorRT, ONNX Runtime, BoT-SORT)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.network
def test_yolo26n_predict(yolo26n_path, bus_image, record, ram_ok):
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ROOT

    model = YOLO(str(yolo26n_path))
    res = model.predict(bus_image, imgsz=640, conf=0.4, device=0, verbose=False)[0]
    names = [res.names[int(c)] for c in res.boxes.cls]
    assert names.count("person") >= 3, names
    assert "bus" in names
    p2 = sorted(p.name for p in (ROOT / "cfg" / "models").rglob("yolo26*p2*.yaml"))
    assert p2, "yolo26-p2.yaml (stride-4 head, doc 5.5) not found in the ultralytics package"
    record("ultralytics", version=ultralytics.__version__, weights=str(yolo26n_path),
           detections=len(names), p2_configs=p2, speed_ms=res.speed)


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
def test_yolo26n_tensorrt_fp16_engine(yolo26n_export_pt, bus_image, record, ram_ok):
    """TensorRT 11 is strongly typed: Ultralytics bakes FP16 into the ONNX with ModelOpt AutoCast, then builds."""
    import tensorrt as trt
    from ultralytics import YOLO

    engine = yolo26n_export_pt.with_suffix(".engine")
    t0 = time.perf_counter()
    if not engine.exists():
        out = YOLO(str(yolo26n_export_pt)).export(format="engine", imgsz=640, quantize=16, device=0, workspace=2,
                                                  verbose=False)
        assert Path(out) == engine
        build_s = time.perf_counter() - t0
    else:
        build_s = None
    m = YOLO(str(engine), task="detect")
    for _ in range(5):  # warm-up
        m.predict(bus_image, imgsz=640, device=0, verbose=False)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        res = m.predict(bus_image, imgsz=640, device=0, verbose=False)[0]
    e2e_ms = (time.perf_counter() - t0) / n * 1e3
    names = [res.names[int(c)] for c in res.boxes.cls]
    assert names.count("person") >= 3
    record("tensorrt", version=trt.__version__, engine=str(engine), engine_mb=round(engine.stat().st_size / 1e6, 1),
           build_s=None if build_s is None else round(build_s, 1), predict_e2e_ms=round(e2e_ms, 2),
           inference_ms=round(res.speed["inference"], 2))


@pytest.mark.gpu
@pytest.mark.network
def test_onnx_and_onnxruntime_gpu(yolo26n_export_pt, bus_image, record, ram_ok):
    import onnx
    import onnxruntime as ort
    import torch  # noqa: F401  (loads the CUDA 13 / cuDNN 9 DLLs that onnxruntime-gpu reuses)
    from ultralytics import YOLO

    onnx_path = yolo26n_export_pt.with_suffix(".onnx")
    if not onnx_path.exists():
        YOLO(str(yolo26n_export_pt)).export(format="onnx", imgsz=640, verbose=False)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    assert "CUDAExecutionProvider" in ort.get_available_providers()
    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert sess.get_providers()[0] == "CUDAExecutionProvider", sess.get_providers()
    import cv2

    x = cv2.resize(bus_image, (640, 640))[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    inp = sess.get_inputs()[0].name
    out = sess.run(None, {inp: x})[0]
    for _ in range(3):
        sess.run(None, {inp: x})
    t0 = time.perf_counter()
    for _ in range(20):
        sess.run(None, {inp: x})
    ms = (time.perf_counter() - t0) / 20 * 1e3
    if out.ndim == 3 and out.shape[-1] == 6:  # YOLO26 end-to-end (NMS-free) head: (1, 300, 6) x1 y1 x2 y2 conf cls
        n_person = int(((out[0, :, 4] > 0.4) & (out[0, :, 5] == 0)).sum())
    else:  # classic layout (1, 4 + 80, 8400): cx cy w h + class scores (overlapping anchors, before NMS)
        n_person = int((out[0, 4, :] > 0.4).sum())
    assert n_person >= 3, (out.shape, n_person)
    record("onnxruntime-gpu", version=ort.__version__, onnx=onnx.__version__, output_shape=list(out.shape),
           cuda_ep_ms=round(ms, 2))


@pytest.mark.gpu
@pytest.mark.network
def test_ultralytics_botsort_track(yolo26n_path, bus_image, record, ram_ok):
    from ultralytics import YOLO

    model = YOLO(str(yolo26n_path))
    ids_per_frame = []
    for i in range(8):  # simulated camera pan: shift the frame 6 px per step
        frame = np.roll(bus_image, shift=6 * i, axis=1)
        r = model.track(frame, persist=True, tracker="botsort.yaml", imgsz=640, conf=0.3, device=0,
                        verbose=False)[0]
        assert r.boxes.id is not None
        ids_per_frame.append(set(int(t) for t in r.boxes.id))
    common = set.intersection(*ids_per_frame[2:])
    assert len(common) >= 3, ids_per_frame  # the same people keep their IDs across the pan
    record("ultralytics-botsort", stable_ids=len(common), frames=len(ids_per_frame))


# ----------------------------------------------------------------------------------------------------------------
# RF-DETR, SAHI, supervision, WBF, ProbEn, DINOv2
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.slow
def test_rfdetr_nano_predict(bus_image, record, ram_ok):
    from importlib.metadata import version

    from PIL import Image
    from rfdetr import RFDETRNano
    from rfdetr.assets.model_weights import get_model_cache_dir

    cache = Path(get_model_cache_dir())
    assert cache == MODELS / "rfdetr", cache  # RF_HOME (conftest) keeps weights out of ~/.roboflow on C:
    model = RFDETRNano(device="cuda")  # published default checkpoint rf-detr-nano.pth, downloaded into RF_HOME
    img = Image.fromarray(bus_image[:, :, ::-1])
    det = model.predict(img, threshold=0.5)
    t0 = time.perf_counter()
    for _ in range(5):
        model.predict(img, threshold=0.5)
    ms = (time.perf_counter() - t0) / 5 * 1e3
    w = cache / "rf-detr-nano.pth"
    assert w.exists()
    assert len(det) >= 3, det
    record("rfdetr", version=version("rfdetr"), weights=str(w), weights_mb=round(w.stat().st_size / 1e6, 1),
           detections=len(det), predict_ms=round(ms, 1))


@pytest.mark.gpu
@pytest.mark.network
def test_sahi_sliced_prediction_4k(yolo26n_path, uhd_scene, record, ram_ok):
    import sahi
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    dm = AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(yolo26n_path),
                                            confidence_threshold=0.4, device="cuda:0")
    t0 = time.perf_counter()
    res = get_sliced_prediction(uhd_scene[:, :, ::-1].copy(), dm, slice_height=640, slice_width=640,
                                overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    dt = time.perf_counter() - t0
    persons = [p for p in res.object_prediction_list if p.category.name == "person"]
    xs = [p.bbox.minx for p in persons]
    assert len(persons) >= 6, len(persons)
    assert max(xs) > 3000  # boxes are mapped back to full-frame coordinates
    record("sahi", version=sahi.__version__, persons=len(persons), seconds_4k=round(dt, 2))


@pytest.mark.gpu
@pytest.mark.network
def test_supervision_inference_slicer_4k(yolo26n_path, uhd_scene, record, ram_ok):
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(str(yolo26n_path))

    def cb(tile: np.ndarray) -> sv.Detections:
        return sv.Detections.from_ultralytics(model(tile, imgsz=640, conf=0.4, device=0, verbose=False)[0])

    slicer = sv.InferenceSlicer(callback=cb, slice_wh=(640, 640), overlap_wh=(128, 128))
    t0 = time.perf_counter()
    det = slicer(uhd_scene)
    dt = time.perf_counter() - t0
    persons = det[det.class_id == 0]
    assert len(persons) >= 6
    assert persons.xyxy[:, 0].max() > 3000
    record("supervision", version=sv.__version__, persons=len(persons), seconds_4k=round(dt, 2))


def test_weighted_boxes_fusion(record):
    from ensemble_boxes import weighted_boxes_fusion

    rgb = [[0.10, 0.10, 0.20, 0.30], [0.60, 0.60, 0.70, 0.80]]
    thermal = [[0.11, 0.105, 0.21, 0.31], [0.40, 0.40, 0.45, 0.50]]
    boxes, scores, labels = weighted_boxes_fusion([rgb, thermal], [[0.9, 0.6], [0.7, 0.5]], [[0, 0], [0, 0]],
                                                  weights=[2, 1], iou_thr=0.5, skip_box_thr=0.05)
    assert len(boxes) == 3  # one fused pair + one RGB-only + one thermal-only box
    fused = boxes[np.argmax(scores)]
    assert 0.10 < fused[0] < 0.11 and 0.20 < fused[2] < 0.21
    record("ensemble-boxes", fused_boxes=len(boxes))


def test_proben_score_rule(record):
    from vendor.proben import bayesian_fusion, bayesian_fusion_multiclass, proben_single

    assert bayesian_fusion([0.7, 0.8]) == pytest.approx(0.56 / (0.56 + 0.06))  # agreement raises confidence
    assert bayesian_fusion([0.5, 0.5]) == pytest.approx(0.5)
    assert bayesian_fusion([0.9, 0.1]) == pytest.approx(0.5)  # disagreement cancels
    assert bayesian_fusion([1.0, 0.0]) == pytest.approx(0.5)  # clipped, no log(0)
    assert proben_single(0.42) == 0.42  # marginalisation: single-modality box keeps its posterior
    s, c = bayesian_fusion_multiclass([[0.7, 0.1], [0.6, 0.2]])
    assert c == 0 and s > 0.7
    record("proben", vendored="vendor/proben", fused_07_08=round(bayesian_fusion([0.7, 0.8]), 4))


@pytest.mark.gpu
@pytest.mark.network
def test_dinov2_vits_features(cuda, record, ram_ok):
    import timm
    import torch

    name = "vit_small_patch14_dinov2.lvd142m"
    model = timm.create_model(name, pretrained=True, num_classes=0, img_size=224,
                              cache_dir=str(MODELS / "hf")).eval().to(cuda).half()
    x = torch.randn(10, 3, 224, 224, device=cuda, dtype=torch.float16)  # ~10 crops per frame (doc 5.5)
    with torch.inference_mode():
        f = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f = model(x)
        torch.cuda.synchronize()
    assert tuple(f.shape) == (10, 384)
    assert torch.isfinite(f).all()
    record("timm-dinov2", version=timm.__version__, model=name, feat_dim=f.shape[1],
           ms_10_crops=round((time.perf_counter() - t0) * 1e3, 2))


# ----------------------------------------------------------------------------------------------------------------
# Ingest: PyAV, PyNvVideoCodec, torchcodec, pymavlink, pyulog, thermal_parser
# ----------------------------------------------------------------------------------------------------------------
N_FRAMES = 48


@pytest.fixture(scope="session")
def test_mp4(tmp_path_factory):
    """H.264 MP4 generated with PyAV (libx264 bundled in the PyAV wheel; NVENC / Media Foundation as fallbacks).
    Frame i has a white square at columns 20*i .. 20*i+100, rows 200..300, so decoders can be checked per frame."""
    import av

    path = tmp_path_factory.mktemp("video") / "synthetic.mp4"
    last_err = None
    for codec in ("libx264", "h264_nvenc", "h264_mf"):
        try:
            with av.open(str(path), "w") as out:
                st = out.add_stream(codec, rate=30)
                st.width, st.height, st.pix_fmt = 1280, 720, "yuv420p"
                for i in range(N_FRAMES):
                    img = np.zeros((720, 1280, 3), np.uint8)
                    img[:, :, 1] = 40
                    img[200:300, 20 * i:20 * i + 100] = 255  # moving white square
                    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                    frame.pts = i  # explicit PTS (stream time base 1/30)
                    for pkt in st.encode(frame):
                        out.mux(pkt)
                for pkt in st.encode():
                    out.mux(pkt)
            path.with_suffix(".codec").write_text(codec)
            return path
        except Exception as e:  # encoder not in this FFmpeg build
            last_err = e
    raise RuntimeError(f"no H.264 encoder available in PyAV: {last_err}")


def test_pyav_decode_pts(test_mp4, record):
    import av

    with av.open(str(test_mp4)) as c:
        frames = list(c.decode(video=0))
    pts = [f.pts for f in frames]
    assert len(frames) == N_FRAMES
    assert all(b > a for a, b in zip(pts, pts[1:]))
    record("av", version=av.__version__, encoder=test_mp4.with_suffix(".codec").read_text(), frames=len(frames))


@pytest.mark.gpu
def test_pynvvideocodec_decode(test_mp4, record):
    import PyNvVideoCodec as nvc
    import torch

    dec = nvc.SimpleDecoder(str(test_mp4), gpu_id=0, use_device_memory=True,
                            output_color_type=nvc.OutputColorType.RGB)
    n = len(dec)
    assert n == N_FRAMES
    t0 = time.perf_counter()
    cols = []
    for i in range(n):
        t = torch.from_dlpack(dec[i])  # zero-copy DLPack view of the CUDA surface
        assert t.is_cuda and tuple(t.shape) == (720, 1280, 3)
        white = torch.nonzero(t[250, :, 0] > 200)
        cols.append(int(white[0]) if len(white) else -1)
    torch.cuda.synchronize()
    fps = n / (time.perf_counter() - t0)
    expected = [20 * i for i in range(n)]
    assert sum(abs(c - e) <= 4 for c, e in zip(cols, expected)) == n, cols  # frame-accurate
    record("pynvvideocodec", version=getattr(nvc, "__version__", None), frames=n, frame_shape=[720, 1280, 3],
           decode_fps_720p_incl_dlpack=round(fps, 1))


def test_torchcodec_decode(test_mp4, record):
    """torchcodec's Windows wheel ships no FFmpeg. It decodes only with an external FFmpeg 4-8 *shared* build on
    PATH. PyAV's bundled FFmpeg must NOT be reused: it is MinGW-built, torchcodec's core is MSVC-built, and the
    mixed heaps crash the process (0xC0000374) - see tools/ffmpeg_shim.py."""
    from importlib.metadata import version

    from tools.ffmpeg_shim import ffmpeg_dll_dir_on_path

    ffmpeg_dir = ffmpeg_dll_dir_on_path()
    record("torchcodec", version=version("torchcodec"), ffmpeg_on_path=ffmpeg_dir,
           note="installs and imports; needs an external FFmpeg 4-8 shared build on PATH (none on this machine). "
                "Project decodes with PyAV + PyNvVideoCodec instead (doc 5.4).")
    if not ffmpeg_dir:
        pytest.skip("no FFmpeg shared build on PATH: install one on D: and add its bin\\ to PATH to enable "
                    "torchcodec")
    import torchcodec  # noqa: F401
    from torchcodec.decoders import VideoDecoder

    d = VideoDecoder(str(test_mp4))
    assert d.metadata.num_frames == N_FRAMES
    assert tuple(d[10].shape) == (3, 720, 1280)
    pts = [d.get_frame_at(i).pts_seconds for i in range(N_FRAMES)]
    assert all(b > a for a, b in zip(pts, pts[1:]))
    record("torchcodec", frames=d.metadata.num_frames)


def test_pymavlink_tlog_roundtrip(tmp_path, record):
    os.environ["MAVLINK20"] = "1"
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink2

    path = tmp_path / "flight.tlog"
    with open(path, "wb") as f:
        mav = mavlink2.MAVLink(f, srcSystem=1, srcComponent=1)
        t_us = 1_757_500_000_000_000
        for i in range(20):
            msgs = [
                mav.system_time_encode(t_us + i * 100_000, i * 100),
                mav.global_position_int_encode(i * 100, int(WAYANAD[0] * 1e7), int(WAYANAD[1] * 1e7) + i,
                                               960_000, 60_000, 0, 0, 0, 9000),
                mav.attitude_encode(i * 100, 0.01, -0.02, 1.57, 0, 0, 0),
                mav.gimbal_device_attitude_status_encode(0, 0, i * 100, 0, [1.0, 0.0, 0.0, 0.0],
                                                         0.0, 0.0, 0.0, 0),
            ]
            f.writelines((t_us + i * 100_000).to_bytes(8, "big") + m.pack(mav) for m in msgs)
    conn = mavutil.mavlink_connection(str(path))
    counts, last = {}, None
    while (m := conn.recv_match(blocking=False)) is not None:
        counts[m.get_type()] = counts.get(m.get_type(), 0) + 1
        if m.get_type() == "GLOBAL_POSITION_INT":
            last = m
    assert counts.get("GLOBAL_POSITION_INT") == 20 and counts.get("GIMBAL_DEVICE_ATTITUDE_STATUS") == 20
    assert last.lat / 1e7 == pytest.approx(WAYANAD[0]) and last.relative_alt == 60_000
    import pymavlink

    record("pymavlink", version=getattr(pymavlink, "__version__", None), messages=counts)


@pytest.mark.network
def test_pyulog_real_sample(record):
    import pyulog
    from pyulog import ULog

    p = _download("https://raw.githubusercontent.com/PX4/pyulog/main/test/sample_log_small.ulg",
                  SAMPLES / "px4_sample_log_small.ulg", 100_000)
    ulog = ULog(str(p))
    topics = sorted({d.name for d in ulog.data_list})
    assert len(topics) > 5
    d = ulog.data_list[0]
    ts = d.data["timestamp"]
    assert np.all(np.diff(ts.astype(np.int64)) >= 0)  # microsecond timestamps, monotonic
    record("pyulog", version=getattr(pyulog, "__version__", None), sample=str(p), topics=len(topics),
           has_attitude="vehicle_attitude" in topics, duration_s=round((ulog.last_timestamp - ulog.start_timestamp) / 1e6, 1))


@pytest.mark.network
def test_thermal_parser_dji_rjpeg(record):
    from tools.thermal import make_thermal

    p = _download("https://raw.githubusercontent.com/SanNianYiSi/thermal_parser/"
                  "b513647ef2318ba99dd2e2a543fa0fb071fd9579/images/DJI_H20T.jpg", SAMPLES / "DJI_H20T.jpg", 100_000)
    th = make_thermal()
    temp = th.parse(str(p))
    assert temp.ndim == 2 and temp.shape[0] >= 256
    assert -40 < float(np.nanmin(temp)) < float(np.nanmax(temp)) < 200
    record("thermal-parser", shape=list(temp.shape), t_min_c=round(float(temp.min()), 2),
           t_max_c=round(float(temp.max()), 2))


# ----------------------------------------------------------------------------------------------------------------
# Tracking / dedup / metrics
# ----------------------------------------------------------------------------------------------------------------
def _toy_detections(n_frames=10):
    """Three static targets seen by a slowly panning camera (xyxy, conf, cls)."""
    base = np.array([[100, 100, 130, 160], [300, 200, 330, 260], [500, 120, 530, 180]], dtype=np.float32)
    out = []
    for i in range(n_frames):
        b = base + np.array([3 * i, 1 * i, 3 * i, 1 * i], dtype=np.float32)
        out.append(b)
    return out


def test_roboflow_trackers(record):
    import supervision as sv
    import trackers

    tracker = trackers.ByteTrackTracker() if hasattr(trackers, "ByteTrackTracker") else trackers.SORTTracker()
    ids = []
    for b in _toy_detections():
        det = sv.Detections(xyxy=b, confidence=np.full(len(b), 0.9, np.float32), class_id=np.zeros(len(b), int))
        det = tracker.update(det)
        ids.append(tuple(int(t) for t in det.tracker_id if t >= 0))
    stable = [t for t in ids[3:] if len(t) == 3]
    assert stable and len(set(stable)) == 1, ids
    record("trackers", version=getattr(trackers, "__version__", None), tracker=type(tracker).__name__,
           ids=list(stable[0]))


def test_boxmot_bytetrack_ab(record):
    import boxmot
    from boxmot import ByteTrack

    tracker = ByteTrack()
    img = np.zeros((480, 640, 3), np.uint8)
    ids = []
    for b in _toy_detections():
        dets = np.hstack([b, np.full((3, 1), 0.9, np.float32), np.zeros((3, 1), np.float32)])
        out = tracker.update(dets, img)
        ids.append(tuple(sorted(int(r[4]) for r in out)))
    stable = [t for t in ids[3:] if len(t) == 3]
    assert stable and len(set(stable)) == 1, ids
    record("boxmot", version=boxmot.__version__, ids=list(stable[0]), lap_module="lap (lapx overridden)")


def test_dbscan_haversine_geo_dedup(record):
    import sklearn
    from pyproj import Geod
    from sklearn.cluster import DBSCAN

    g = Geod(ellps="WGS84")
    rng = np.random.default_rng(1)
    pts = []
    for az, dist in ((0, 0), (90, 40), (200, 120)):  # 3 survivors 40-120 m apart
        lon, lat, _ = g.fwd(WAYANAD[1], WAYANAD[0], az, dist)
        for _ in range(6):  # 6 observations each, ~2.5 m (1 sigma) geolocation noise
            lo, la, _ = g.fwd(lon, lat, rng.uniform(0, 360), abs(rng.normal(0, 2.5)))
            pts.append((la, lo))
    X = np.radians(np.array(pts))
    eps_m = 2 * 6.0  # 2 x CE90 (doc 5.6)
    labels = DBSCAN(eps=eps_m / 6_371_008.8, min_samples=1, metric="haversine").fit_predict(X)
    assert len(set(labels)) == 3
    record("scikit-learn", version=sklearn.__version__, clusters=len(set(labels)))


def test_torchmetrics_map_recall_at_05(record):
    import torch
    import torchmetrics
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True,
                                  extended_summary=True, backend="pycocotools")
    target = [dict(boxes=torch.tensor([[10, 10, 50, 90], [100, 100, 140, 180], [200, 50, 240, 130.]]),
                   labels=torch.tensor([0, 0, 0]))]
    preds = [dict(boxes=torch.tensor([[12, 11, 51, 88], [101, 99, 141, 178], [400, 400, 440, 480.]]),
                  scores=torch.tensor([0.9, 0.8, 0.7]), labels=torch.tensor([0, 0, 0]))]
    metric.update(preds, target)
    r = metric.compute()
    recall = r["recall"]  # (T, K, A, M): iou=0.5, class=human, area=all, maxDet=last
    assert recall.shape[0] == 1
    assert float(recall[0, 0, 0, -1]) == pytest.approx(2 / 3, abs=1e-3)
    assert 0.5 < float(r["map_50"]) < 0.8
    record("torchmetrics", version=torchmetrics.__version__, recall_iou05=round(float(recall[0, 0, 0, -1]), 4),
           map50=round(float(r["map_50"]), 4))


def test_pycocotools_eval(record):
    import pycocotools
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    m = np.zeros((50, 50), np.uint8, order="F")
    m[10:20, 10:30] = 1
    rle = mask_utils.encode(m)
    assert int(mask_utils.area(rle)) == 200
    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 100, "height": 100}], "categories": [{"id": 1, "name": "human"}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 40],
                                   "area": 800, "iscrowd": 0}]}
    gt.createIndex()
    dt = gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [11, 10, 20, 40], "score": 0.9}])
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate(), ev.accumulate()
    ev.summarize()
    assert ev.stats[1] == pytest.approx(1.0)  # AP@0.5
    record("pycocotools", version=getattr(pycocotools, "__version__", None), ap50=float(ev.stats[1]))


def test_motmetrics_mota_idf1(record):
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)
    gt = {1: (0.0, 0.0), 2: (10.0, 0.0)}
    for f in range(6):
        hyp_ids = [1, 2] if f < 3 else [1, 3]  # hypothesis 2 is renamed 3 halfway: one ID switch
        hyp_pos = [(0.1, 0.0), (10.1, 0.0)]
        d = mm.distances.norm2squared_matrix(np.array(list(gt.values())), np.array(hyp_pos), max_d2=1.0)
        acc.update(list(gt.keys()), hyp_ids, d)
    mh = mm.metrics.create()
    s = mh.compute(acc, metrics=["mota", "idf1", "num_switches"], name="toy")
    assert int(s["num_switches"].iloc[0]) == 1
    assert s["mota"].iloc[0] == pytest.approx(1 - 1 / 12)
    record("motmetrics", version=mm.__version__, mota=round(float(s["mota"].iloc[0]), 4),
           idf1=round(float(s["idf1"].iloc[0]), 4))


@pytest.mark.slow
def test_fiftyone_separate_env(record):
    py = REPO / "envs" / "fiftyone" / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        pytest.fail(f"FiftyOne env missing: run `uv sync` in {py.parents[2]}")
    r = subprocess.run([str(py), str(REPO / "envs" / "fiftyone" / "check_fiftyone.py")], capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    info = json.loads(r.stdout.strip().splitlines()[-1])
    assert info["samples"] == 3 and info["deleted"] and info["database_dir"].lower().startswith("d:")
    record("fiftyone", **info)


# ----------------------------------------------------------------------------------------------------------------
# Geo: pyproj, rasterio, dem-stitcher, shapely, geopandas/pyogrio, duckdb spatial
# ----------------------------------------------------------------------------------------------------------------
def test_pyproj_geod_fwd_inv(record):
    import pyproj
    from pyproj import Geod, Transformer

    g = Geod(ellps="WGS84")
    lon2, lat2, _ = g.fwd(WAYANAD[1], WAYANAD[0], 45.0, 100.0)
    _, _, d = g.inv(WAYANAD[1], WAYANAD[0], lon2, lat2)
    assert d == pytest.approx(100.0, abs=1e-6)
    x, y = Transformer.from_crs(4326, 32643, always_xy=True).transform(WAYANAD[1], WAYANAD[0])  # UTM 43N
    assert 600_000 < x < 700_000 and 1_200_000 < y < 1_300_000
    record("pyproj", version=pyproj.__version__, proj=pyproj.proj_version_str)


def test_rasterio_geotiff_roundtrip(tmp_path, record):
    import rasterio
    from rasterio.transform import from_origin

    arr = np.arange(100 * 120, dtype=np.float32).reshape(100, 120)
    tr = from_origin(76.10, 11.52, 1 / 3600, 1 / 3600)
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", width=120, height=100, count=1, dtype="float32", crs="EPSG:4326",
                       transform=tr, compress="deflate") as ds:
        ds.write(arr, 1)
    with rasterio.open(p) as ds:
        assert ds.crs.to_epsg() == 4326 and ds.transform == tr
        assert np.array_equal(ds.read(1), arr)
        v = next(ds.sample([(76.10 + 5.5 / 3600, 11.52 - 3.5 / 3600)]))[0]
    assert v == arr[3, 5]
    record("rasterio", version=rasterio.__version__, gdal=rasterio.__gdal_version__)


@pytest.mark.network
def test_dem_stitcher_glo30_wayanad(record):
    import dem_stitcher
    import rasterio
    from dem_stitcher import stitch_dem

    out = DEM_DIR / "wayanad_glo30_76.10_11.45_76.18_11.53.tif"
    if not out.exists():
        X, prof = stitch_dem([76.10, 11.45, 76.18, 11.53], dem_name="glo_30", dst_ellipsoidal_height=False,
                             dst_area_or_point="Point")
        with rasterio.open(out, "w", **prof) as ds:
            ds.write(X, 1)
    with rasterio.open(out) as ds:
        z = ds.read(1)
        origin_h = float(next(ds.sample([(WAYANAD[1], WAYANAD[0])]))[0])
        res_m = ds.res[0] * 111_320
    assert z.shape[0] > 200 and z.shape[1] > 200
    assert 300 < float(np.nanmedian(z)) < 2500
    record("dem-stitcher", version=getattr(dem_stitcher, "__version__", None), file=str(out), shape=list(z.shape),
           z_min=round(float(np.nanmin(z)), 1), z_max=round(float(np.nanmax(z)), 1), res_m=round(res_m, 1),
           origin_geopoint_height_m=round(origin_h, 1))


def test_shapely_geopandas_pyogrio_gpkg(tmp_path, record):
    import geopandas as gpd
    import pyogrio
    import shapely
    from shapely.geometry import Point, Polygon

    pts = gpd.GeoDataFrame({"record_id": [1, 2], "conf": [0.9, 0.6]},
                           geometry=[Point(76.1450, 11.4870), Point(76.1460, 11.4880)], crs="EPSG:4326")
    utm = pts.to_crs(32643)
    circles = utm.buffer(6.0).to_crs(4326)  # CE90 ring
    aoi = gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[Polygon([(76.14, 11.48), (76.15, 11.48), (76.15, 11.49),
                                                                 (76.14, 11.49)])], crs="EPSG:4326")
    p = tmp_path / "out.gpkg"
    pts.to_file(p, layer="records", engine="pyogrio")
    gpd.GeoDataFrame(geometry=circles, crs=4326).to_file(p, layer="rings", engine="pyogrio")
    aoi.to_file(p, layer="aoi", engine="pyogrio")
    layers = [l[0] for l in pyogrio.list_layers(p)]
    back = gpd.read_file(p, layer="records", engine="pyogrio")
    assert set(layers) == {"records", "rings", "aoi"}
    assert back.crs.to_epsg() == 4326 and back.geometry.equals(pts.geometry)
    assert shapely.contains(aoi.geometry[0], back.geometry[0])
    area = gpd.GeoSeries(circles).to_crs(32643).area.iloc[0]
    assert area == pytest.approx(math.pi * 36, rel=0.02)
    record("geopandas", version=gpd.__version__, pyogrio=pyogrio.__version__, shapely=shapely.__version__,
           gdal=pyogrio.__gdal_version_string__)
    (tmp_path / "keep").write_text(str(p))


@pytest.mark.network
def test_duckdb_spatial(tmp_path, record):
    import duckdb
    import geopandas as gpd
    from shapely.geometry import Point

    ext_dir = Path(r"D:\Tools\cache\duckdb\extensions")
    ext_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "pts.gpkg"
    gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(76.1450, 11.4870), Point(76.1460, 11.4870)],
                     crs=4326).to_file(p, engine="pyogrio")
    con = duckdb.connect()
    con.execute(f"SET extension_directory = '{ext_dir.as_posix()}'")
    con.execute("INSTALL spatial; LOAD spatial;")
    d = con.execute("SELECT ST_Distance_Sphere(ST_Point(11.4870, 76.1450), ST_Point(11.4870, 76.1460))").fetchone()[0]
    n = con.execute(f"SELECT count(*) FROM ST_Read('{p.as_posix()}')").fetchone()[0]
    con.close()
    assert d == pytest.approx(109.0, rel=0.02) and n == 2
    assert any(ext_dir.rglob("spatial.duckdb_extension")), "spatial extension not stored on D:"
    record("duckdb", version=duckdb.__version__, extension_dir=str(ext_dir), distance_m=round(d, 2))


# ----------------------------------------------------------------------------------------------------------------
# Output / C2 / offline
# ----------------------------------------------------------------------------------------------------------------
def _c2_app():
    from fastapi import FastAPI, WebSocket

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        msg = await sock.receive_json()
        await sock.send_json({"type": "FeatureCollection", "features": [msg], "ack": msg["properties"]["record_id"]})
        await sock.close()

    return app


FEATURE = {"type": "Feature", "geometry": {"type": "Point", "coordinates": [76.145, 11.487, 900.0]},
           "properties": {"record_id": "r-0001", "class": "human", "confidence": 0.91}}


def test_fastapi_websocket_testclient(record):
    import fastapi
    from fastapi.testclient import TestClient

    with TestClient(_c2_app()) as c:
        assert c.get("/health").json() == {"ok": True}
        with c.websocket_connect("/ws") as ws:
            ws.send_json(FEATURE)
            got = ws.receive_json()
    assert got["ack"] == "r-0001" and got["features"][0] == FEATURE
    record("fastapi", version=fastapi.__version__)


def test_uvicorn_websockets_real_socket(record):
    import socket
    import threading

    import uvicorn
    import websockets
    from websockets.sync.client import connect

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(_c2_app(), host="127.0.0.1", port=port, log_level="warning",
                                           ws="websockets"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        with connect(f"ws://127.0.0.1:{port}/ws") as ws:
            ws.send(json.dumps(FEATURE))
            got = json.loads(ws.recv(timeout=10))
        assert got["ack"] == "r-0001"
    finally:
        server.should_exit = True
        th.join(timeout=10)
    record("uvicorn", version=uvicorn.__version__, websockets=websockets.__version__)


def test_persist_queue_sqliteackqueue_durability(tmp_path, record):
    import persistqueue
    from persistqueue import SQLiteAckQueue

    path = str(tmp_path / "outbox")
    q = SQLiteAckQueue(path, multithreading=True, auto_commit=True)
    for i in range(3):
        q.put({"clip_id": "c1", "record_id": f"r-{i}", "version": 1})
    a = q.get()
    q.ack(a)  # delivered
    b = q.get()  # in flight when the process "dies" (never acked)
    del q
    q2 = SQLiteAckQueue(path, multithreading=True, auto_commit=True)  # restart: unacked work is resumed
    remaining = []
    while q2.size:
        item = q2.get(block=False)
        remaining.append(item["record_id"])
        q2.ack(item)
    assert sorted(remaining) == sorted(["r-1", "r-2"]) and b["record_id"] == "r-1"
    record("persist-queue", version=getattr(persistqueue, "__version__", None), resumed=remaining)


def test_simplekml_kmz_with_thumbnail(tmp_path, record):
    import simplekml
    from PIL import Image

    thumb = tmp_path / "r-0001.png"
    Image.new("RGB", (128, 128), (200, 30, 30)).save(thumb)
    kml = simplekml.Kml(name="Sightline records")
    href = kml.addfile(str(thumb))
    p = kml.newpoint(name="r-0001 human 0.91", coords=[(76.145, 11.487, 900)])
    p.description = f'<img src="{href}" width="128"/>'
    out = tmp_path / "records.kmz"
    kml.savekmz(str(out))
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        doc = z.read("doc.kml").decode()
    assert "doc.kml" in names and any(n.endswith("r-0001.png") for n in names)
    assert "76.145,11.487,900" in doc and href in doc
    record("simplekml", files=names)


def test_geojson_feature(record):
    import geojson

    f = geojson.Feature(geometry=geojson.Point((76.145123, 11.487456, 900.0)), properties=FEATURE["properties"])
    fc = geojson.FeatureCollection([f])
    assert fc.is_valid and f.geometry.is_valid
    back = geojson.loads(geojson.dumps(fc))
    assert back["features"][0]["geometry"]["coordinates"][:2] == [76.145123, 11.487456]
    record("geojson", version=geojson.__version__)


def test_pytak_cot_event(record):
    import xml.etree.ElementTree as ET

    import pytak

    el = pytak.gen_cot_xml(lat=WAYANAD[0], lon=WAYANAD[1], ce=6.0, hae=900.0, le=10.0, uid="sightline-r-0001",
                           stale=600, cot_type="a-u-G")
    xml = ET.tostring(el)
    root = ET.fromstring(xml)
    pt = root.find("point")
    assert root.tag == "event" and root.get("type") == "a-u-G" and root.get("uid") == "sightline-r-0001"
    assert float(pt.get("lat")) == pytest.approx(WAYANAD[0]) and float(pt.get("ce")) == 6.0
    record("pytak", version=getattr(pytak, "__version__", None), cot_bytes=len(xml))


def test_fields2cover_windows_availability(record):
    import importlib.util

    have = importlib.util.find_spec("fields2cover") is not None
    record("fields2cover", installed=have,
           note="PyPI 2.1.0 ships an sdist only (C++ build needing GDAL, OR-Tools, Eigen); no Windows wheel. "
                "Fallback: own boustrophedon generator (doc 5.3).")
    if not have:
        pytest.skip("Fields2Cover has no Windows wheel (sdist only); use the own boustrophedon fallback")


# ----------------------------------------------------------------------------------------------------------------
# Control
# ----------------------------------------------------------------------------------------------------------------
def test_pygame_joystick_init_without_device(record):
    import pygame

    pygame.init()
    pygame.joystick.init()
    try:
        assert pygame.joystick.get_init()
        n = pygame.joystick.get_count()
        assert n >= 0
    finally:
        pygame.quit()
    record("pygame", version=pygame.version.ver, sdl=".".join(map(str, pygame.get_sdl_version())), joysticks=n)


def test_inputs_gamepad_enumeration(record):
    import inputs

    pads = list(inputs.devices.gamepads)
    assert isinstance(pads, list)
    record("inputs", gamepads=len(pads))


def test_mavsdk_server_binary(record):
    import mavsdk

    exe = Path(mavsdk.__file__).parent / "bin" / "mavsdk_server.exe"
    assert exe.exists(), exe
    r = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=30)
    assert "Usage" in (r.stdout + r.stderr) or "usage" in (r.stdout + r.stderr)
    from mavsdk import System

    System()  # constructing the client does not start the server or open sockets
    record("mavsdk", version=getattr(mavsdk, "__version__", None), server=str(exe))
