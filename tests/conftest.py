"""Shared fixtures for the stack tests. Everything writes to D: (repo, models, D:\\Tools\\cache)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
SAMPLES = REPO / "data" / "samples"
DEM_DIR = REPO / "data" / "dem"
ART = REPO / "_artifacts" / "stack"

# Redirect every cache that would otherwise default to C:\Users\... (set before any library import).
os.environ.setdefault("MPLCONFIGDIR", r"D:\Tools\cache\matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", r"D:\Tools\cache\numba")
os.environ.setdefault("HF_HOME", r"D:\Tools\cache\hf")
os.environ.setdefault("TORCH_HOME", r"D:\Tools\cache\torch")
os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\Tools\cache\ultralytics")
os.environ["YOLO_AUTOINSTALL"] = "false"  # Ultralytics must never pip-install into the uv venv
os.environ.setdefault("RF_HOME", str(MODELS / "rfdetr"))  # rfdetr checkpoint cache (default ~/.roboflow/models)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MAVLINK20", "1")
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

for d in (MODELS, SAMPLES, DEM_DIR, ART):
    d.mkdir(parents=True, exist_ok=True)

import pytest

_METRICS: dict = {}


@pytest.fixture(scope="session")
def record():
    """record(tool, **values): collects timings/versions into _artifacts/stack/metrics.json."""

    def _rec(tool: str, **values):
        _METRICS.setdefault(tool, {}).update(values)

    yield _rec
    old = {}
    p = ART / "metrics.json"
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except Exception:
            old = {}
    old.update(_METRICS)
    p.write_text(json.dumps(old, indent=2, default=str))


def wait_for_ram(min_gb: float = 2.0, timeout_s: float = 600) -> float:
    """Back off while the machine is short on RAM (other agents run the Unreal editor/builds)."""
    import psutil

    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 1e9
        if free >= min_gb or time.time() - t0 > timeout_s:
            return free
        time.sleep(10)


@pytest.fixture
def ram_ok():
    free = wait_for_ram()
    if free < 1.0:
        pytest.skip(f"only {free:.2f} GB RAM free")
    return free


def free_vram_gb() -> float:
    import pynvml  # from nvidia-ml-py (ultralytics dependency)

    pynvml.nvmlInit()
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / 1e9
    finally:
        pynvml.nvmlShutdown()


def wait_for_vram(min_gb: float, timeout_s: float = 900) -> float:
    """The Unreal editor / sim shares the 8 GB GPU; wait for headroom instead of OOM-ing it."""
    t0 = time.time()
    while True:
        free = free_vram_gb()
        if free >= min_gb or time.time() - t0 > timeout_s:
            return free
        time.sleep(15)


@pytest.fixture
def vram_ok(request):
    need = 3.0 if request.node.get_closest_marker("slow") else 1.5
    free = wait_for_vram(need)
    if free < need:
        pytest.skip(f"GPU busy: only {free:.2f} GB VRAM free after waiting (need {need} GB)")
    return free


@pytest.fixture(scope="session")
def cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available to torch")
    return torch.device("cuda:0")


def pytest_collection_modifyitems(config, items):
    for item in items:  # every GPU test waits for VRAM headroom first
        if item.get_closest_marker("gpu") and "vram_ok" not in item.fixturenames:
            item.fixturenames.insert(0, "vram_ok")


@pytest.fixture(scope="session")
def bus_image():
    """Real photo shipped inside the ultralytics wheel (people + bus), 810x1080 BGR."""
    import cv2
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    assert img is not None
    return img


@pytest.fixture(scope="session")
def yolo26n_path():
    """yolo26n.pt downloaded once into D:\\Sightline\\models (explicit path, never the cwd)."""
    from ultralytics import YOLO

    p = MODELS / "yolo26n.pt"
    YOLO(str(p))  # attempt_download_asset() writes to this exact path if missing
    assert p.exists() and p.stat().st_size > 1e6
    return p


@pytest.fixture(scope="session")
def yolo26n_export_pt(yolo26n_path):
    """Hardlink of yolo26n.pt in models/stack_check so ONNX/TensorRT exports (written next to the .pt) never
    overwrite shared artifacts in models/ that other work uses."""
    d = MODELS / "stack_check"
    d.mkdir(exist_ok=True)
    p = d / "yolo26n.pt"
    if not p.exists():
        os.link(yolo26n_path, p)
    return p


@pytest.fixture(scope="session")
def uhd_scene(bus_image):
    """Synthetic 3840x2160 'aerial' frame: grey noise background with the bus photo pasted at 3 spots."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(0)
    canvas = rng.integers(90, 140, size=(2160, 3840, 3), dtype=np.uint8)
    small = cv2.resize(bus_image, (405, 540))
    for x, y in ((200, 300), (1700, 900), (3200, 1500)):
        canvas[y:y + 540, x:x + 405] = small
    return canvas
