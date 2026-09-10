# Python stack verification (2026-09-10)

Every Python tool named in `docs/SOLUTION_DOC.md` (§3.4, §5.4-§5.12, §6.4, Appendix A) was installed with uv and
checked with a **functional** test (`tests/test_stack.py`, one real test per tool), not an import.
Machine: Windows 11, RTX 4060 Laptop 8 GB (driver 592.82, CUDA 13.1), 16 GB RAM, Python 3.11.16.

Re-run: `uv run python tools/verify_stack.py --smoke` (writes `stack_check.json` / `stack_check.md` next to this
file). Subsets: `uv run pytest tests/test_stack.py -m "not network"`, `-m gpu`, `-m "not slow"`.

## 1. Environment layout

| Where | What |
|---|---|
| `D:\Sightline\.venv` (main, `pyproject.toml` + `uv.lock`) | MCP server deps (unchanged) + dependency groups `ml`, `ingest`, `track`, `eval`, `geo`, `c2`, `control`, `ab`, `dev`. `default-groups = "all"`, so a plain `uv sync` restores everything. |
| `D:\Sightline\envs\fiftyone` (own `pyproject.toml` + `uv.lock` + `.venv`) | FiftyOne 1.21.0 only (see §3, conflict C4). Run with `envs\fiftyone\.venv\Scripts\python.exe`. |
| `[tool.uv] environments` | Lock restricted to `win32/AMD64`: the project is Windows-only, and CUDA/TensorRT/NVDEC wheels do not exist for every platform. |
| Indexes | `pytorch-cu130` (`https://download.pytorch.org/whl/cu130`, explicit: torch, torchvision, torchcodec) and `nvidia` (`https://pypi.nvidia.com`, explicit: tensorrt-cu13, -libs, -bindings). |
| Git sources (pinned commits) | `thermal-parser` (SanNianYiSi/thermal_parser @ b513647, not on PyPI), `motmetrics` (cheind/py-motmetrics develop @ e5ae766, NumPy-2 fix). |
| Vendored | `vendor/proben/` (ProbEn score rule, Apache-2.0 header, tested in `test_proben_score_rule`). |
| Project shims | `tools/thermal.py` (thermal_parser plugin path, needed to reach its DJI SDK DLLs), `tools/ffmpeg_shim.py` (documents why torchcodec must **not** borrow PyAV's FFmpeg, and detects a real FFmpeg on PATH). |

**Why CUDA 13.0 wheels (cu130):** the driver reports CUDA 13.1, so cu130 runs natively. More importantly,
`onnxruntime-gpu` 1.29.0 on PyPI is built against CUDA 13 (its `cuda`/`cudnn` extras pull `nvidia-cuda-runtime~=13.0`
and `nvidia-cudnn-cu13`; the ORT install page still says 12.x, which is out of date). With torch cu130, ORT reuses
the `cublas64_13` / `cudart64_13` / `cudnn64_9` / `cufft64_12` DLLs that torch ships in `torch\lib` (import torch
first), so no second CUDA runtime is installed. TensorRT is the matching `cu13` 11.3 build. The cu132 index also has
torch 2.14 but needs a newer driver; the cu128 index stops at torch 2.11.

## 2. Results per tool

`uv run python tools/verify_stack.py --smoke` on 2026-09-10: **37 passed, 2 skipped, 0 failed** in 4:46
(machine-generated copy with every metric: `stack_check.md` / `stack_check.json`). Licences are read from the
installed package metadata. Timings are from a laptop RTX 4060 that was sharing the GPU with the Unreal editor,
so treat them as upper bounds, not benchmarks.

| Tool | Version | Licence | Functional test | Result | Measurements / notes |
|---|---|---|---|---|---|
| PyTorch (CUDA) | 2.14.0+cu130 | BSD-3 + Apache-2.0 (mixed) | FP16 2048² matmul on the GPU checked against a CPU reference | pass | CUDA 13.0, cuDNN 9.24, 19.4 ms per 2048² FP16 matmul |
| torchvision | 0.29.0+cu130 | BSD-3-Clause | `ops.nms` on CUDA tensors returns the expected boxes | pass | CUDA op path works |
| Ultralytics YOLO26 | 8.4.146 | AGPL-3.0 | YOLO26n detects 4 people + the bus in the packaged photo at imgsz 640 | pass | 5 detections; `yolo26-p2.yaml` (stride-4 head, doc §5.5) confirmed present |
| TensorRT (cu13) | 11.3.0.99 | Proprietary (NVIDIA) | export an FP16 engine (ModelOpt AutoCast + build), then infer through it | pass | **build 214 s**, engine 92.1 MB, **4.64 ms inference**, 12.7 ms end-to-end per frame |
| ONNX | 1.22.0 | Apache-2.0 | `onnx.checker` on the exported graph | pass | opset 18 |
| onnxslim | 0.1.96 | MIT | graph slimming during the Ultralytics ONNX export | pass | - |
| ONNX Runtime GPU | 1.29.0 | MIT | CUDA-EP session on the exported ONNX finds the people | pass | CUDA EP is first provider, 17.1 ms/frame; TensorRT EP also available; reuses torch's CUDA 13 DLLs |
| Ultralytics BoT-SORT | 8.4.146 | AGPL-3.0 | IDs stay stable over 8 panning frames | pass | 4 IDs kept across the pan |
| RF-DETR (Nano) | 1.10.1 | Apache-2.0 | detects the same people; weights land in `models/rfdetr` via `RF_HOME` | pass | 366 MB checkpoint, 60.9 ms/frame |
| SAHI | 0.12.6 | MIT | sliced inference over a synthetic 3840×2160 frame | pass | 12 persons, boxes remapped to full-frame coords, 1.01 s for the 4K frame (640 tiles, 0.2 overlap) |
| supervision (InferenceSlicer) | 0.30.2 | MIT | same 4K frame through `InferenceSlicer` with overlap NMS | pass | 16 persons, 0.88 s |
| Weighted Boxes Fusion | 1.0.9 | MIT | fuses an RGB and a thermal box set (weights 2:1) | pass | 1 fused + 2 single-modality boxes kept, fused box between the inputs |
| ProbEn score rule (vendored) | vendor/proben | Apache-2.0 | agreement raises, disagreement cancels, clipping, single-modality marginalisation | pass | fuse(0.7, 0.8) = 0.9032; fuse(0.9, 0.1) = 0.5; fuse(1.0, 0.0) = 0.5 (no log 0) |
| DINOv2 ViT-S (timm) | 1.0.29 | Apache-2.0 | frozen ViT-S/14 features for 10 crops | pass | (10, 384) finite, 66 ms per 10 crops FP16; weights in `models/hf` |
| transformers | 5.17.0 | Apache-2.0 | exercised as RF-DETR's backbone loader | pass | - |
| PyAV | 18.1.0 | BSD-3-Clause | encode 48 frames, decode them back, PTS strictly increasing | pass | libx264 in the wheel; h264_nvenc, hevc_nvenc, h264_mf and mpeg4 also available |
| PyNvVideoCodec | 2.2.2 | MIT | NVDEC decode to CUDA memory, DLPack into torch, frame-accurate | pass | 48/48 frames, square in the right place every frame, ~547 fps at 720p incl. DLPack |
| torchcodec | 0.16.0+cu130 | BSD-3-Clause | (not runnable here) | **skipped** | Wheel installs/imports but needs an external FFmpeg 4-8 *shared* build on PATH. **PyAV's bundled FFmpeg must not be reused: MinGW vs MSVC heap corruption, 0xC0000374, killed the test process.** Project uses PyAV + PyNvVideoCodec (doc §5.4) |
| pymavlink | 2.4.49 | LGPL-3.0 | write a `.tlog` (position/attitude/gimbal/system time) and parse it back | pass | 20 each of GLOBAL_POSITION_INT / ATTITUDE / GIMBAL_DEVICE_ATTITUDE_STATUS / SYSTEM_TIME; lat/alt round-trip exactly |
| pyulog | 1.2.4 | BSD-3-Clause | parse a **real** PX4 `.ulg` (pyulog test data, in `data/samples`) | pass | 55 topics incl. `vehicle_attitude`, 1174 s flight, monotonic µs timestamps |
| thermal_parser (DJI R-JPEG) | 20240826 (git b513647) | MIT (bundles DJI Thermal SDK v1.7 under the DJI EULA) | extract the temperature array from a **real** DJI H20T R-JPEG | pass | 512×640 float32, 13.1 – 28.1 °C; needs `tools/thermal.py` (plugin path) |
| roboflow trackers | 2.6.0 | Apache-2.0 | 3 static targets keep their IDs over 10 frames | pass | licence-clean alternative to the AGPL trackers |
| BoxMOT (A/B only) | 25.0.0 | AGPL-3.0 | same with BoxMOT ByteTrack | pass | proves `lap` works in place of the excluded `lapx` |
| lap | 0.5.13 | BSD-2-Clause | assignment solver behind both trackers | pass | - |
| scikit-learn (DBSCAN) | 1.9.0 | BSD-3-Clause | `DBSCAN(metric='haversine', eps=2×CE90)` on noisy lat/lon | pass | 18 noisy fixes (σ = 2.5 m) around 3 survivors → exactly 3 records |
| torchmetrics | 1.9.0 | Apache-2.0 | `MeanAveragePrecision(extended_summary=True)` recall at IoU 0.5 | pass | recall[iou=0.5] = 0.6667 (2 of 3), mAP50 = 0.663, pycocotools backend |
| pycocotools | 2.0.11 | BSD-2-Clause | RLE mask encode/area + `COCOeval` | pass | mask area exact, AP50 = 1.0 |
| py-motmetrics | 1.4.0 (git e5ae766) | MIT | MOTA / IDF1 / ID switches on toy tracks with one ID switch | pass | switches = 1, MOTA = 0.9167, IDF1 = 0.75 |
| FiftyOne | 1.21.0 (separate env) | Apache-2.0 | create dataset, `evaluate_detections`, delete; own MongoDB | pass | 3 samples, tp 3 / fp 3 / fn 0; DB in `data/fiftyone/db`; **no `~/.fiftyone` created**; mongod exits with the process |
| pyproj | 3.7.2 | MIT | `Geod.fwd` 100 m then `inv` round trip; WGS84 → UTM 43N | pass | round trip exact to 1e-6 m, PROJ 9.5.1 |
| rasterio | 1.4.4 | BSD-3-Clause | write + read a GeoTIFF (CRS, transform, values, sampling) | pass | GDAL 3.10.3 |
| dem-stitcher | 3.2.0 | Apache-2.0 | fetch Copernicus GLO-30 over the Wayanad valley | pass | tile saved in `data/dem`, 30.9 m/px, 294 – 2235 m. **Elevation at the OriginGeopoint (11.4870 N, 76.1450 E) = 1060 m**, vs 900 m in `DefaultEngine.ini` (see §7) |
| shapely | 2.1.2 | BSD-3-Clause | polygon/point predicates and buffering | pass | CE90 ring area matches πr² to 2 % |
| geopandas | 1.1.4 | BSD-3-Clause | 3-layer GeoPackage write + read | pass | records / rings / aoi layers, CRS preserved |
| pyogrio | 0.13.0 | MIT | GeoPackage I/O engine and layer listing | pass | GDAL 3.12.4 |
| DuckDB + spatial | 1.5.5 | MIT | `INSTALL`/`LOAD spatial`, `ST_Distance_Sphere`, `ST_Read` of the GeoPackage | pass | 108.97 m for 0.001° of longitude at 11.49 N; extension stored on D: |
| FastAPI | 0.141.1 | MIT | GeoJSON record over a WebSocket via `TestClient` | pass | round trip with ack |
| httpx | 0.28.1 | BSD-3-Clause | transport under `TestClient` | pass | - |
| uvicorn | 0.52.4 | BSD-3-Clause | same round trip over a **real** socket | pass | server started/stopped in-process |
| websockets | 17.1 | BSD-3-Clause | sync client for the real-socket round trip | pass | - |
| persist-queue | 1.1.0 | BSD-3-Clause | `SQLiteAckQueue`: ack one, leave one in flight, reopen | pass | unacked item is resumed after restart, acked one is gone (at-least-once, doc §5.10) |
| simplekml | 1.3.6 | LGPL-3.0+ | KMZ with an embedded thumbnail | pass | `doc.kml` + `files/r-0001.png` verified inside the zip |
| geojson | 3.3.0 | BSD-3-Clause | Feature with `[lon, lat, alt]` + properties | pass | valid, round trip, 6-decimal coords |
| pytak | 7.6.1 | Apache-2.0 | build a Cursor-on-Target XML event | pass | 338-byte event, uid/type/point/ce as expected. Warns that `aiohttp` is missing: only needed for TAK *transport*, not for CoT generation |
| Fields2Cover | - | BSD-3-Clause | (not installable) | **skipped** | no Windows wheel (§4) |
| pygame | 2.6.1 | LGPL-2.1 | joystick subsystem initialises | pass | SDL 2.28.4. **A gamepad is now present (1 detected)** - CONTEXT.md §2 says none was at setup; good news for F3 |
| inputs | 0.5 | BSD-3-Clause | enumerate XInput gamepads | pass | 1 gamepad |
| mavsdk | 3.17.2 | BSD-3-Clause | bundled `mavsdk_server.exe` runs; client constructs | pass | no sockets opened (avoids a Windows Firewall prompt); full check needs PX4 SITL |

## 3. Conflicts hit and how they were resolved

| # | Conflict | Resolution | Proof |
|---|---|---|---|
| C1 | **cv2 variants.** Main env needs `opencv-python` (ultralytics `!=4.13.0.90`, sahi `>=4.12.0.88`, trackers, boxmot `<5`); FiftyOne requires `opencv-python-headless`; both own `cv2`. | FiftyOne moved to its own env (C4). `override-dependencies = ["opencv-python-headless; sys_platform == 'never'"]` so no future dependency (e.g. albumentations via `rfdetr[augment]`) can pull it into the main env. | Only distribution `opencv-python 4.14.0.94` installed (unchanged); cv2 used by the YOLO, SAHI, supervision, BoT-SORT and PyAV tests. |
| C2 | **lap vs lapx.** boxmot 25 requires `lapx`; Ultralytics' tracker uses `lap` (and would pip-install it at runtime). Both install the `lap` module. | `lap>=0.5.12` declared in `track`; `lapx` excluded via override. | `test_boxmot_bytetrack_ab` and `test_ultralytics_botsort_track` both pass on `lap 0.5.13`. |
| C3 | **onnxruntime CPU vs GPU.** `ultralytics[export-base]` and `rfdetr[onnx]` ask for CPU `onnxruntime`; `onnxruntime-gpu` provides the same module. | Those extras are not installed; `onnxruntime` excluded via override. | Only `onnxruntime-gpu 1.29.0`; providers `TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider`. |
| C4 | **FiftyOne 1.21 pins**: `opencv-python-headless` (C1), `starlette<1.4` (main env is on 1.6.0, shared by `mcp`, `sse-starlette`, `fastapi`: sharing would downgrade the MCP server's HTTP stack), `pymongo~=4.9.2`, `motor~=3.6.0`, `mongoengine~=0.29.1`, `strawberry-graphql<0.317`, `hypercorn<0.19`. | Separate uv project `envs/fiftyone` (126 packages, 0.69 GB). The main test suite drives it through `envs/fiftyone/check_fiftyone.py`. | `test_fiftyone_separate_env`. |
| C5 | **motmetrics 1.4.0 (PyPI, 2022) calls `np.asfarray`**, removed in NumPy 2.0: `distances.norm2squared_matrix`/`iou_matrix` crash with our numpy 2.2.6 (numpy is pinned `<2.3` for the MCP env). | Git pin to the maintained develop branch, commit `e5ae766` (17 Jul 2026, HOTA + TrackEval parity; deps numpy/pandas/scipy only; drops xmltodict). Still versioned 1.4.0. | `test_motmetrics_mota_idf1`. |
| C6 | **pyproj 3.8.0 and rasterio 1.5.1** (Appendix A versions) require Python >= 3.12; the project stays on 3.11 (cosysairsim). | `pyproj>=3.7,<3.8` and `rasterio>=1.4.3,<1.5` resolve to 3.7.2 and 1.4.4, the last cp311 Windows wheels. | geo tests. |
| C7 | **TensorRT is sdist-only on PyPI** (`tensorrt-cu13` is an 18 KB metapackage; the libs/bindings wheels live on pypi.nvidia.com). | Explicit `nvidia` index for `tensorrt-cu13`, `-libs`, `-bindings`. The metapackage builds locally (plain setuptools) and provides the `tensorrt` shim over `tensorrt_bindings`; `tensorrt-cu13-libs` is a wheel-stub that fetches the real wheel during the first `uv sync` (network needed once, then cached). | `test_yolo26n_tensorrt_fp16_engine`. |
| C8 | **torchcodec ships no FFmpeg on Windows** (it looks for FFmpeg 4-9 shared DLLs and fails). | `tools/ffmpeg_shim.py` hardlinks PyAV 18.1's bundled FFmpeg 8 DLLs (hash-mangled names in `av.libs`) under canonical names into `D:\Tools\ffmpeg-pyav-shim\av-18.1.0` (hardlinks: no extra disk) and adds both folders with `os.add_dll_directory`. No third-party FFmpeg download. Alternative: an FFmpeg "shared" build on PATH. | `test_torchcodec_decode` (frames and PTS). |
| C9 | **thermal_parser packaging bug**: `setup.py` installs the DJI SDK DLLs and exiftool via `data_files` into `<venv>\plugins`, but the code looks in `site-packages\plugins`. | `tools/thermal.py` patches `get_default_filepaths()` to the real folder (no copying of the proprietary DLLs). | `test_thermal_parser_dji_rjpeg`. |
| C10 | **Ultralytics auto-installs missing packages** at runtime (lap, onnx, tensorrt, ...), which would bypass uv. | Every runtime dependency it asks for is declared in the groups; tests set `YOLO_AUTOINSTALL=false`. **Set it in any pipeline code too.** | No pip activity during the runs; lock diff clean. |
| C11 | FastAPI endpoint annotations under `from __future__ import annotations` with a locally imported `WebSocket` become unresolvable strings: the socket is rejected with HTTP 403. | Test code fixed (no future import). Note for future server code: import `WebSocket` at module level. | WebSocket tests. |
| C12 | **This Claude Code process did not inherit the user-level cache variables** (`YOLO_CONFIG_DIR`, `HF_HOME`, `TORCH_HOME`, `UV_CACHE_DIR` are empty in its shells; CONTEXT.md §3 predicts this for already-open apps). One Ultralytics `settings.json` landed in `C:\Users\kiran\AppData\Roaming\Ultralytics` during introspection. | File moved off C: (not deleted); `tests/conftest.py`, `tools/verify_stack.py`, `envs/fiftyone/check_fiftyone.py` and the uv commands set every variable explicitly. **User action:** restart Claude Code/terminals; consider adding user-level `MPLCONFIGDIR`, `RF_HOME`, `NUMBA_CACHE_DIR`, `FIFTYONE_DATABASE_DIR` (values in §5). | Post-run check of C: (§5). |

**MCP-protected packages: zero version changes.** mcp 1.30.0, cosysairsim 3.4.1, rpc-msgpack 0.6, numpy 2.2.6,
psutil 7.2.2, pillow 12.3.0, opencv-python 4.14.0.94, pydantic/pydantic-core, starlette 1.6.0, uvicorn 0.52.4,
httpx 0.28.1 are exactly as before (lock diff before each sync, §6).

## 4. Not installable / fallbacks

| Tool | Status | Fallback |
|---|---|---|
| **dji-log-parser** Python bindings | **Do not exist.** The project (MIT, v0.5.7, Apr 2025) is a Rust crate + `dji-log` CLI + npm binding; nothing on PyPI under any name. The doc's "Rust core with Python bindings" (§5.4) is wrong. | Call the `dji-log` CLI from its GitHub release (not downloaded here: executable download needs the user's OK) via `subprocess`, reading its JSON/CSV/GeoJSON output. Logs v13+ also need a DJI developer API key (doc §5.4). |
| **Fields2Cover** | **No Windows wheel**: PyPI 2.1.0 (3 Sep 2026) is sdist-only (C++ with GDAL, OR-Tools, Eigen, SWIG); no conda-forge package. Not installed. | Own ~60-line boustrophedon generator on shapely + pyproj (doc §5.3 allows it). Not written in this task. |
| DINOv3 ViT-S | Weights are gated on Hugging Face (licence acceptance). | DINOv2 ViT-S/14 (`timm vit_small_patch14_dinov2.lvd142m`, Apache-2.0), which the doc allows ("DINOv2 or DINOv3"). |
| mavsdk (stretch) | Installed (3.17.2). Only the bundled `mavsdk_server.exe --help` and client construction are verified: starting a server binds sockets (Windows Firewall prompt) and needs a SITL vehicle. | Verify against PX4 SITL when F3/PX4 work starts. |
| X-AnyLabeling | Out of scope (separate Windows exe, other agent). | - |

## 5. Everything stays on D:

| Item | Location | How |
|---|---|---|
| uv cache / Python | `D:\Tools\cache\uv`, `D:\Tools\uv-python` | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR` (set explicitly in every uv call). `.venv` files are hardlinks into the cache. |
| YOLO26 weights, ONNX, TensorRT engine | `D:\Sightline\models\yolo26n.{pt,onnx,engine}` | explicit path to `YOLO()` (`attempt_download_asset` writes to that path; exports land next to it) |
| RF-DETR checkpoint | `D:\Sightline\models\rfdetr\rf-detr-nano.pth` | `RF_HOME` (default is `~/.roboflow/models` on C:) |
| DINOv2 ViT-S | `D:\Sightline\models\hf\` | `timm.create_model(..., cache_dir=...)` |
| Ultralytics settings | `D:\Tools\cache\ultralytics` | `YOLO_CONFIG_DIR` |
| matplotlib font cache | `D:\Tools\cache\matplotlib` | `MPLCONFIGDIR` (default `~/.matplotlib`) |
| DuckDB extensions (spatial) | `D:\Tools\cache\duckdb\extensions` | `SET extension_directory = ...` per connection (default `~/.duckdb`) |
| FiftyOne MongoDB, datasets, zoo | `D:\Sightline\data\fiftyone\` | `FIFTYONE_DATABASE_DIR`, `FIFTYONE_DEFAULT_DATASET_DIR`, `FIFTYONE_*_ZOO_DIR`; `FIFTYONE_DO_NOT_TRACK=true`. Verified: no `~/.fiftyone` created. |
| DEM tile, samples | `D:\Sightline\data\dem`, `D:\Sightline\data\samples` | test code |
| FFmpeg for torchcodec | `D:\Tools\ffmpeg-pyav-shim` | `tools/ffmpeg_shim.py` (hardlinks) |

Transient files still go to `%TEMP%` on C: (pytest `tmp_path`, uv build dirs); they are small and short-lived.

## 6. Keeping the MCP server safe while changing the env

1. Backup of `pyproject.toml` and `uv.lock` before the first change.
2. `uv lock`, then a lock diff against the installed venv that flags any change to a protected package; sync only
   if the diff is clean.
3. `uv sync --inexact` (never removes a package while the server runs; Windows also locks loaded `.pyd` files).
4. RAM gate: syncs and heavy tests wait for >= 2 GB free RAM; GPU tests wait for 1.5 GB (3 GB for the TensorRT
   build) of free VRAM, because the editor/sim share the 8 GB GPU.
5. After every change: `tools/sightline_mcp/smoke_test.py` and `import cosysairsim, mcp`.

Smoke test after the last change: **pass** (`tools: 27; missing: none`, protocol 2025-11-25).

## 7. Findings and follow-ups for other agents

1. **DEM says 1060 m, the sim says 900 m.** Copernicus GLO-30 at the OriginGeopoint (11.4870 N, 76.1450 E) gives
   **1060.3 m**; `DefaultEngine.ini` uses 900 m (CONTEXT.md §5 explicitly asks for this to be verified against a
   DEM). The tile is in `data/dem/wayanad_glo30_76.10_11.45_76.18_11.53.tif` (294 - 2235 m over the AOI, 30.9 m/px).
   Owner of the sim settings should decide whether to move the origin height.
2. **A gamepad is now attached** (pygame and `inputs` both see 1), unlike CONTEXT.md §2. F3 takeover can be tested
   for real.
3. **Duplicate weights in `models/`.** `models/rf-detr-nano.pth` (366 MB, created 18:52 by another process)
   duplicates `models/rfdetr/rf-detr-nano.pth` (downloaded by the tests via `RF_HOME`). `models/yolo26n.onnx` at
   the root was re-exported by my first TensorRT attempt at 18:58, before I moved test exports into
   `models/stack_check/`. Nothing of mine writes to the root of `models/` any more; whoever owns those files
   should decide whether to delete the duplicate (I did not).
4. **`tests/_harness_server.py` was reformatted by my `ruff check --fix`** (safe fixes only: import order /
   unused import). It still compiles and is ruff-clean, but flagging it since it belongs to the MCP test work.
5. **`xmltodict` is installed but no longer in the lock** (the motmetrics develop branch dropped it). A plain
   `uv sync` (without `--inexact`) will remove it; nothing needs it.
6. **`YOLO_AUTOINSTALL=false` must be set by pipeline code too**, not just the tests, or Ultralytics will
   pip-install into the uv venv behind uv's back.
7. **PyNvVideoCodec frames are views of pooled decoder surfaces.** `DecodedFrame` objects are invalidated by
   later decodes: consume or `.clone()` a frame before decoding on. A first version of the test collected all
   frames into a list and read frame 10 afterwards - it saw the *last* frame's pixels.
8. **pytak transport needs `pytak[with-aiohttp]`** when the TAK client is actually wired up; CoT generation
   itself works without it.
9. **torchcodec** needs a user-installed FFmpeg shared build (§4, C8) if it is ever wanted; PyAV +
   PyNvVideoCodec cover the doc's decode requirements today.
10. **Empty folder left on C:** `C:\Users\kiran\AppData\Roaming\Ultralytics` (the stray `settings.json` was moved
    off C:, the now-empty directory was left in place).
