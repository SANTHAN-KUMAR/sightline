# External (non-Python) tools: install, staging and verification

Verified 2026-09-10 on the dev PC (Windows 11 26200, RTX 4060 8 GB, 16 GB RAM). Restore everything with
`powershell -ExecutionPolicy Bypass -File tools\setup\fetch_external.ps1` (idempotent, size + SHA-256 checked;
a re-run with everything present took 3 s). Nothing was installed on C:, no system setting was changed, and
no WSL distro was installed.

| # | Tool | Version | Status |
|---|---|---|---|
| 1 | Cesium for Unreal | v2.29.1 (UE 5.8 package) | **STAGED**: binaries match the engine; not yet in the project |
| 2 | X-AnyLabeling | v4.0.6 Windows CUDA 12 | **PASS**: launches, data kept on D:. GPU needs cuDNN 9 (open issue) |
| 3 | go-pmtiles CLI + Wayanad basemap | 1.31.2; Protomaps build 20260910 | **PASS** |
| 4 | MapLibre GL JS + pmtiles + Protomaps style/fonts/sprites | 6.9.0 / 4.5.0 / 5.7.2 | **PASS**: renders offline in headless Edge, 0 external requests |
| 5 | Fields2Cover | 2.1.0 | **REJECT on Windows**: use the in-house boustrophedon generator |
| 6 | PX4 SITL (WSL2) + QGroundControl | PX4 v1.18.0-beta2 / QGC v5.1.4 | **NOT INSTALLED**: stretch goal, steps documented |
| 7 | Theta-Limited DroneModels | commit edae834 (1 Sep 2026) | **PASS** (no M30T entry, see below) |
| 8 | TAK: WinTAK/ATAK, FreeTAKServer, pytak | n/a | **NOT INSTALLED**: documentation only |
| 9 | Gamepad | n/a | **none connected** |

---

## 1. Cesium for Unreal v2.29.1: STAGED

| Item | Value |
|---|---|
| Source | https://github.com/CesiumGS/cesium-unreal/releases/download/v2.29.1/CesiumForUnreal-58-v2.29.1.zip (release 2026-09-01; supports UE 5.6/5.7/5.8) |
| Download | `_downloads/CesiumForUnreal-58-v2.29.1.zip`, 1,226,845,642 B, sha256 `4c45422d…eae68b0b` (**matches the GitHub release digest**) |
| Staged at | `D:\Sightline\_staging\plugins\CesiumForUnreal` (5.45 GB extracted: Win64 dll+pdb 473 MB, Linux/Mac binaries, `Intermediate` 3.5 GB of per-platform link libs for packaging, `Source` 1.1 GB) |
| Licence | Apache-2.0 |

Note: the release-notes link text for UE 5.8 says "57", but the URL and the file both point to the 58 package.

**Verification (evidence):**
- `CesiumForUnreal.uplugin`: `"VersionName": "2.29.1"`, **`"EngineVersion": "5.8.0"`**, `"Installed": true`, modules
  `CesiumRuntime` (Runtime, PostConfigInit) and `CesiumEditor` (Editor, PostEngineInit), both allowed on Win64.
- `Binaries\Win64\`: `UnrealEditor-CesiumRuntime.dll` (36,955,648 B) and `UnrealEditor-CesiumEditor.dll`
  (18,153,472 B), each with its .pdb.
- `Binaries\Win64\UnrealEditor.modules` has **`"BuildId": "55116800"`**. `D:\UE_5.8\Engine\Build\Build.version`
  has `"Changelist": 56702186, "CompatibleChangelist": 55116800`, so the IDs **match** (as they do for Cosys-AirSim).
- The plugin depends on two engine plugins, `SunPosition` and `Water`. Both exist
  (`Engine\Plugins\Runtime\SunPosition`, `Engine\Plugins\Experimental\Water`), and `Water` is already enabled in
  `SightlineSim.uproject`.
- Not done here, because the UE build agent owns the project: copying the plugin into the project, adding it to
  the .uproject, and loading it in the editor.

**Integration steps (for the main agent, after the UE build agent is finished and the editor is closed):**
1. Add `sim/SightlineSim/Plugins/CesiumForUnreal/` to `.gitignore`, next to the AirSim line (it is 5.4 GB).
2. Move the staged plugin (same volume, instant):
   `Move-Item D:\Sightline\_staging\plugins\CesiumForUnreal D:\Sightline\sim\SightlineSim\Plugins\CesiumForUnreal`
   (optional, saves 0.4 GB: delete `Binaries\Linux` and `Binaries\Mac` in the moved copy).
3. Add `{ "Name": "CesiumForUnreal", "Enabled": true }` to the `Plugins` array of `SightlineSim.uproject`.
   SunPosition is pulled in automatically as a dependency.
4. Rebuild the editor target (`ue_build`). `"Installed": true` means UBT uses the precompiled DLLs. The
   BuildId match is what lets the editor load them without the "modules are missing or built with a different
   engine version" prompt.
5. Launch the editor. Expect `LogPluginManager: Mounting Project plugin CesiumForUnreal` and no `LogCesium`
   errors in the log. Then open `Window > Cesium`.
6. **Cesium ion token (user action):** "Connect to Cesium ion" opens a browser OAuth sign-in to a Cesium ion
   account. Claude must not create the account or enter credentials. After sign-in, pick or create the
   project default token from the Cesium panel. Cesium stores the project default token in
   `Config/DefaultGame.ini` (`[/Script/CesiumRuntime.CesiumRuntimeSettings]`; verify after the first sign-in).
   Use a token scoped to only the needed assets, and decide whether to commit it.
7. Scene: add `CesiumGeoreference` with origin at the Wayanad valley (lat 11.4870, lon 76.1450, height from DEM,
   matching `OriginGeopoint` in the AirSim settings), plus `Cesium World Terrain + Bing Aerial` from the Quick
   Add panel.

**Cesium ion Community (free) tier, read from cesium.com/platform/cesium-ion/pricing on 2026-09-10:**
personal and non-commercial use; individual account only; 10 GB storage; **15 GB/month streaming**; 1,000
Global Imagery (Bing/Google) sessions/month; 1,000 Google Photorealistic 3D Tiles root tiles/month; 50,000
geocodes/month; 5 h/month Reality Analysis.

**Open issues:**
- Cesium tiles stream from the internet, so an offline demo needs the tiles cached, or a static-mesh export
  of the valley.
- Watch RAM and VRAM: keep the tileset `Maximum Cached Bytes` low and the `Maximum Screen Space Error` high
  on this 16 GB machine.
- Hackathon use should fit "non-commercial" but confirm this.

## 2. X-AnyLabeling v4.0.6 (Windows CUDA 12): PASS (GPU inference needs cuDNN 9)

| Item | Value |
|---|---|
| Source | https://github.com/CVHub520/X-AnyLabeling/releases/download/v4.0.6/X-AnyLabeling-v4.0.6-Windows-CUDA12.exe (release 2026-09-05, newest) |
| File | `D:\Tools\X-AnyLabeling\X-AnyLabeling-v4.0.6-Windows-CUDA12.exe`, 583,328,533 B, sha256 `d2256b42…cd9b02` (**matches the release digest**) |
| Launcher | `D:\Tools\X-AnyLabeling\X-AnyLabeling.cmd` (use this, not the bare exe) |
| Licence | GPL-3.0 (an external desktop tool; not linked into Sightline) |

**Where it stores data**, read from the v4.0.6 source in the official wheel (`anylabeling/config.py`, `app.py`,
`services/auto_labeling/model.py`). Everything sits under a *work directory*, which defaults to `~`
(`C:\Users\<user>`):
- the config file `<work>\.xanylabelingrc`;
- auto-label model downloads in `<work>\xanylabeling_data\models\<model>\…`;
- the model list `<work>\xanylabeling_data\models.json`;
- trainer, chatbot, VQA and PaddleOCR data in `<work>\xanylabeling_data\…`.

**Redirect to D:** use the CLI flag **`--work-dir <dir>`**; there is no env var for this. The launcher passes
`--work-dir D:\Tools\X-AnyLabeling\work` and also sets:
- `TEMP`/`TMP` to `D:\Tools\X-AnyLabeling\tmp`. The exe is a one-file PyInstaller bundle that unpacks
  **~1.2 GB** into `%TEMP%\_MEIxxxx` on every start; without this it would land on C:.
- `HF_HOME` and `TORCH_HOME` to the project caches.
- `XANYLABELING_MODEL_HUB=github`, an env var the source reads (`github` or `modelscope`).
- `X_ANYLABELING_DEVICE`, which the source also honours, is not set.

**Verification (evidence):** launched via the launcher at 18:4x.
- t=5 s: bootloader process running.
- t=15, 20 and 25 s: child process `X-AnyLabeling-v4.0.6-Windows-CUDA12` with main window title
  **"X-AnyLabeling"**, working set ~246 MB.
- Closed with `CloseMainWindow()`: exited cleanly within 8 s and removed the `_MEI` dir (tmp back to 0 MB).
- It created `D:\Tools\X-AnyLabeling\work\.xanylabelingrc` (5,023 B; `language: en_US`, `model_hub: github`,
  `store_data: false`, `custom_models: []`).
- **Nothing was created in `C:\Users\<user>`**: `.xanylabelingrc` and `xanylabeling_data` are both absent there.

**Open issue: GPU.** The PyInstaller table of contents contains `onnxruntime_providers_cuda.dll` and
`onnxruntime_providers_tensorrt.dll`, but **no CUDA runtime, cuBLAS or cuDNN DLLs**; they come from PATH.
- On PATH: `cudart64_12.dll` and `cublas64_12.dll` from `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`.
- Not found: `cudnn64_9.dll`. onnxruntime's CUDA provider will fail to load and **fall back to CPU**.
- Fix, when auto-labelling starts: download the NVIDIA cuDNN 9.x for CUDA 12 Windows redist zip (official
  `developer.download.nvidia.com/compute/cudnn/redist/...`) and unpack it so that `D:\Tools\cudnn\bin\cudnn64_9.dll`
  exists. The launcher already prepends that folder to PATH when it is present.
- Confirm by running one model and checking the log for `CUDAExecutionProvider`.
- The model weights (Grounding DINO, SAM, and so on) download on first use into `work\xanylabeling_data\models`.

## 3. go-pmtiles CLI and Wayanad basemap: PASS

| Item | Value |
|---|---|
| Source | https://github.com/protomaps/go-pmtiles/releases/download/v1.31.2/go-pmtiles_1.31.2_Windows_x86_64.zip (17,842,310 B, sha256 `a658baa4…e785a1`, **matches**) |
| Installed | `D:\Tools\pmtiles\pmtiles.exe` (58,970,112 B), `pmtiles version` prints `pmtiles 1.31.2, commit a3e4951…, built 2026-07-22` |
| Basemap | `D:\Sightline\data\basemap\wayanad.pmtiles`, **1,770,913 B** (1.7 MB), sha256 `6C68F94E…D0FA21FE` |
| Source build | `https://build.protomaps.com/20260910.pmtiles` (137.9 GB planet, Last-Modified 2026-09-10 09:08 GMT; OSM replication time 2026-09-10T04:00Z) |
| Command | `pmtiles extract https://build.protomaps.com/20260910.pmtiles data\basemap\wayanad.pmtiles --bbox=76.05,11.42,76.25,11.58 --maxzoom=15` (14 s, 50 range requests, 1.8 MB transferred) |

`pmtiles show` output:
- spec v3, `tile type: mvt`, gzip;
- bounds (76.05, 11.42) to (76.25, 11.58); zoom 0 to 15;
- 439 addressed tiles, 435 entries, 429 contents; clustered;
- Protomaps Basemap **tile schema version 4.15.2** (planetiler 0.10.2), attribution © OpenStreetMap.

`pmtiles verify` completed with no errors.

**Notes:** daily builds are deleted after a while, so the exact build cannot be re-downloaded later. That is
why the 1.7 MB extract is not gitignored. `fetch_external.ps1 -RefreshBasemap` re-extracts from the newest
build. The India border caveat from doc §5.9 does not apply: the bbox contains no international border.

## 4. MapLibre offline map stack: PASS

| Package | Source | Size / integrity |
|---|---|---|
| maplibre-gl 6.9.0 (BSD-3) | npm `https://registry.npmjs.org/maplibre-gl/-/maplibre-gl-6.9.0.tgz` | 4,249,537 B, `sha512-vFMwMK0Z…pdJ9bA==` |
| pmtiles 4.5.0 (BSD-3) | npm `pmtiles-4.5.0.tgz` | 96,124 B, `sha512-CBeD4SoU…Wii5w==` |
| @protomaps/basemaps 5.7.2 (BSD-3) | npm `basemaps-5.7.2.tgz` | 59,103 B, `sha512-K1Yk6bWd…aaUQ==` |
| protomaps/basemaps-assets (fonts OFL, sprites) | `https://codeload.github.com/protomaps/basemaps-assets/zip/028c18f713baecad011301ff7a69acc39bcc2ae7` | 6,731,115 B, sha256 `e942a417…74b305` |

Tarballs are in `_downloads/npm`. npm was run with `--cache D:\Tools\cache\npm`, because this shell predated
the user-level `npm_config_cache` and `npm config get cache` still reported the C: default.

**Vendored into `D:\Sightline\app\map\vendor\`** (11.9 MB, 796 files):
- `maplibre-gl/`: `maplibre-gl.mjs`, `-shared.mjs`, `-worker.mjs`, `.css`, `LICENSE.txt`. v6 is ESM-only;
  the main module finds the worker and shared chunks through `import.meta.url`, so all three must sit side by side.
- `pmtiles/pmtiles.js`: IIFE, global `pmtiles`.
- `protomaps-basemaps/basemaps.js`: IIFE, global `basemaps`.
- `basemaps-assets/fonts/` (Noto Sans Regular, Medium and Italic; 256 glyph ranges each; `OFL.txt`) and
  `basemaps-assets/sprites/v4/` (light, dark, white, grayscale and black, at 1x and 2x).
- The `Noto Sans Devanagari Regular v1` font folder was skipped: its file paths are rejected by Windows
  (`tar: Can't create … Invalid argument`), and the English/Latin labels do not need it.

**Files created:**
- `app/map/serve.mjs`: dependency-free Node server, loopback only (127.0.0.1:8765). It serves `app/` and
  `data/basemap/` only, with **HTTP Range** support (206 and 416 responses), logs every request, and accepts
  `POST /__report`.
- `app/map/verify_offline.html`: MapLibre 6.9.0 with the `pmtiles://` protocol. It draws the Protomaps
  `light` flavour with `lang: en`, local glyphs and sprites, and a 3-point GeoJSON test layer (T1/T2/T3 rings,
  dots and labels). On `idle` it collects rendered-feature counts and every resource URL, sets
  `document.title` to MAP_OK or MAP_FAIL, and posts a report to the server.
- `app/map/headless_check.mjs`: launches headless Edge 152.0.4191.66 with
  `--host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE 127.0.0.1"`, so any non-loopback host fails to resolve.
  It drives Edge over the DevTools protocol, records every network request (page and auto-attached workers),
  waits for the page verdict, and writes the screenshot.
  Plain `msedge --headless --screenshot --virtual-time-budget` was tried first and captured before any tile
  had loaded (grey background only), which is why the CDP script exists.

**Evidence** (`node app/map/serve.mjs` then `node app/map/headless_check.mjs`, exit 0):
```
"title": "MAP_OK", "external": [], "request_count": 22, "request_hosts": ["http://127.0.0.1:8765"]
status: OFFLINE MAP OK  maplibre 6.9.0 / basemap features 90, markers 9, idle in 977 ms / external requests: 0, errors: 0
server REPORT {"ok":true,"basemap_features":90,"marker_features":9,
  "source_layers":["boundaries","buildings","earth","landuse","places","roads","water"],"external_requests":[],"errors":[]}
```
The server log shows the page, the vendor files, 206 range reads of `wayanad.pmtiles`, sprite and glyph
requests, and nothing else. (A 403 for `/favicon.ico` is expected: it is outside the allowed roots.)

Screenshot: `_artifacts/verification/map_offline.png`. Inspected: landuse, water bodies and rivers
(Punnappuzha, Muthappanpuzha), a road network, and place labels (Chooralmala, Mundakai, Attamala, Puthumala,
Kalladi, Punchiri Mattam Colony). The three test markers are drawn in red, orange and green with labels, with
a scale bar and OSM attribution.

**Integration:**
- The C2 map page should copy the `style` block from `verify_offline.html`. Build `glyphs` and `sprite` as
  absolute URLs from `location.origin`.
- FastAPI can serve the same two trees, as long as its static handler supports Range (Starlette's
  `FileResponse` does in recent versions; check) or keeps `serve.mjs`.
- Attribution "© OpenStreetMap, Protomaps" must stay visible.

## 5. Fields2Cover 2.1.0: REJECT on Windows

**Evidence:**
- **PyPI** (`pypi.org/pypi/fields2cover/json`) has only an sdist, `fields2cover-2.1.0.tar.gz`, and no wheels
  at all. Its `pyproject.toml` classifiers list only `Operating System :: POSIX :: Linux` and `MacOS`, and its
  README says "the package is built from source on your machine, so the system dependencies above must be
  installed".
- **conda-forge**: the package does not exist (`api.anaconda.org` returns "fields2cover could not be found").
- **vcpkg**: there is no `ports/fields2cover`.
- **Build attempt** (cheapest official route): `uv pip install fields2cover==2.1.0` in a throwaway Python 3.11
  venv under `_scratch/`. scikit-build-core 1.0.3 with CMake 4.4.3 picked VS 2026 BuildTools (MSVC 19.51).
  It **failed at configure** with `Could NOT find TinyXML2 (missing: TINYXML2_LIBRARY TINYXML2_INCLUDE_DIR)`
  at `CMakeLists.txt:49`. Full log: `_artifacts/verification/fields2cover_pip_install_windows.log`.
  TinyXML2 is only the first hard dependency. The same CMakeLists also requires GDAL ≥ 3.0, GEOS (falling
  back to the Linux-only `-lgeos_c`), Eigen3, OR-tools (C++), TBB, nlohmann_json and SWIG ≥ 4.1, and calls
  `find_library(m)`.
- **Upstream:** issue #25 "Document the installation process on Windows" has been open since 2022. In #181
  (Oct 2024) the maintainer wrote "Until now we only were able to compile it on Linux env".
- Getting it working would mean a vcpkg build of GDAL, OR-tools, GEOS and TBB (multiple hours, several GB)
  plus patching. That is well past the 30-minute budget.
- The throwaway venv was deleted.

**Decision:** use the fallback from doc §5.2 and §5.3a: an in-house boustrophedon/swath generator (~60 lines;
shapely is enough for polygon clipping). If Fields2Cover is ever wanted, the only realistic route is WSL2 with
`pip install fields2cover` on Ubuntu after `apt install libgdal-dev libgeos-dev libtinyxml2-dev …` plus
OR-tools, which ties into item 6.

**Side finding:** CMake found **Visual Studio 18 2026 BuildTools at `D:\VS\BuildTools2026` (MSVC 14.51)**,
alongside VS 2022 17.14. UE builds must keep using 14.44 (CONTEXT.md §3). If UBT ever picks the newer
toolchain, pin it with `CompilerVersion`/`-Compiler=VisualStudio2022` in BuildConfiguration.

## 6. PX4 SITL on WSL2 + QGroundControl: NOT INSTALLED (stretch goal)

**Current state:**
- WSL 2.3.26.0, kernel 5.15.167.4-1, WSLg 1.0.65, default version 2.
- `wsl -l -v` reports **no installed distributions** (`WSL_E_DEFAULT_DISTRO_NOT_FOUND`).

**Installing a distro onto D: (never `wsl --install -d`, which puts the VHDX under `%LOCALAPPDATA%` on C:):**
```powershell
# 1. Get an official Ubuntu 24.04 WSL root filesystem (Canonical: https://cloud-images.ubuntu.com/wsl/ ,
#    exact file name and size not checked today: pick the amd64 WSL rootfs and verify it against SHA256SUMS)
New-Item -ItemType Directory -Force D:\WSL\Ubuntu-24.04, D:\WSL\images
curl.exe -L -o D:\WSL\images\ubuntu-noble-wsl-amd64.rootfs.tar.gz <URL from cloud-images.ubuntu.com/wsl/noble/current/>
# 2. Import: the ext4.vhdx is created in the target folder on D:
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu-24.04 D:\WSL\images\ubuntu-noble-wsl-amd64.rootfs.tar.gz --version 2
# 3. Create a user (an imported distro starts as root):
wsl -d Ubuntu-24.04 -u root -- bash -c "useradd -m -G sudo -s /bin/bash px4 && passwd px4"
wsl -d Ubuntu-24.04 -u root -- bash -c "printf '[user]\ndefault=px4\n' >> /etc/wsl.conf"
wsl --terminate Ubuntu-24.04
```
- Alternative: `wsl --install -d Ubuntu-24.04`, then `wsl --manage Ubuntu-24.04 --move D:\WSL\Ubuntu-24.04`
  (supported since WSL 2.3.x). This briefly writes the image to C: first.
- Memory: cap the VM with `%UserProfile%\.wslconfig` → `[wsl2] memory=4GB swap=2GB swapFile=D:\\WSL\\swap.vhdx`.
  Without that, WSL can take up to 50 % of RAM and put its swap file on C:.

**PX4 v1.18 prerequisites:**
- Latest tag `v1.18.0-beta2` (2026-08-09). The stable release is still v1.17.0.
- Source: PX4 docs `dev_setup/dev_env_windows_wsl.md`: Ubuntu 22.04 or 24.04 in WSL2, then inside WSL:
```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive -b v1.18.0-beta2   # clone INSIDE the ext4 fs, not /mnt/d
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx     # "Ubuntu LTS (24.04, 22.04)"; --no-nuttx skips the ARM toolchain
cd PX4-Autopilot && make px4_sitl none_iris               # SITL without Gazebo, for an external simulator (AirSim)
```
- Budget ~5 to 8 GB inside the VHDX, on D:.

**Cosys-AirSim link requirements** (Cosys `docs/px4_sitl_wsl2.md`):
- PX4 ≥ v1.12 supports a remote simulator host: `export PX4_SIM_HOST_ADDR=<Windows vEthernet (WSL) IPv4>`.
- Open inbound **TCP 4560** (lockstep simulator link) and **UDP 14540** in Windows Firewall. This is a
  firewall change, so it is a user action.
- AirSim vehicle settings: `"VehicleType": "PX4Multirotor"`, `"UseTcp": true`, `"TcpPort": 4560`,
  `"LockStep": true`, `"ControlIp": "remote"`, `"ControlPortLocal": 14540`, `"ControlPortRemote": 14580`,
  `"LocalHostIp": "<vEthernet (WSL) IP>"`, `"ClockType": "SteppableClock"`, plus the barometer noise clamp
  (`PressureFactorSigma: 0.0001825`). Keep this as a separate `sim/settings/px4_wsl2.json` profile.
- PX4 must be restarted every time Unreal stops (doc §5.2).

**QGroundControl v5.1.4** (2026-08-30):
- Installer: `QGroundControl-installer-AMD64.exe`, **135,364,443 B**, sha256 `56f5f943…d49d20`.
- It is NSIS (`deploy/windows/nullsoft_installer.nsi`): `RequestExecutionLevel admin`, default
  `InstallDir "$PROGRAMFILES64\QGroundControl"` (C:).
- To keep it off C:, install silently to D: with `QGroundControl-installer-AMD64.exe /S /D=D:\Tools\QGroundControl`
  (`/D=` must be the last argument and unquoted). This needs a UAC prompt, which the user handles.
- QGC also writes settings and logs under `%LOCALAPPDATA%`/`Documents\QGroundControl` (small). The in-app
  "Application Load/Save Path" can point to D:.
- An alternative with no Windows install: run the QGC Linux AppImage inside WSLg.

Status: **NOT INSTALLED**. Time-box it to 2 h, per doc §5.2.

## 7. Theta-Limited DroneModels: PASS (no M30T entry)

| Item | Value |
|---|---|
| Source | `https://raw.githubusercontent.com/Theta-Limited/DroneModels/edae83415913edbf972cf25365c18e5f6a701dfb/droneModels.json` (main @ 2026-09-01; the file in the repo is `droneModels.json`, saved as `DroneModels.json`) |
| File | `D:\Sightline\data\reference\DroneModels.json`, 60,822 B, sha256 `0155D855…91499639`, plus `DroneModels.LICENSE` (Apache-2.0, 11,350 B) |

**Verification:** the file parses as JSON.
- Top-level keys: `lastUpdate` ("Tue Sep 1 07:17:06 EDT 2026") and `droneCCDParams`, an array of 122 entries.
- Keys used across the entries:
  - identity and camera: `makeModel`, `isThermal`, `comment`, `lensType` (`perspective` or fisheye);
  - intrinsics: `widthPixels`, `heightPixels`, `ccdWidthMMPerPixel` and `ccdHeightMMPerPixel` (strings such
    as `"6.4/4000.0"` that must be evaluated as a/b), `focalLength` (mm; present only for some entries);
  - perspective distortion: `radialR1..R3`, `tangentialT1..T2`;
  - fisheye distortion: `poly0..poly4`, `c`, `d`, `e`, `f`;
  - OpenAthena error model: `tle_model_y_intercept`, `tle_model_slant_range_coeff`.
- **DJI Mavic 3T**, `makeModel` **`djiM3T`**. There are three entries, told apart by `isThermal` and resolution:

| Camera | isThermal | Resolution | focalLength (mm) | Pixel pitch (mm) | Distortion |
|---|---|---|---|---|---|
| Thermal | true | 640×512 | 8.35 | 7.68/640 × 6.144/512 | radial/tangential all 0.0 |
| Wide colour | false | 8000×6000 | (none: take it from EXIF) | 6.430908/8000 × 4.838406/6000 | R1 0.189023, R2 −0.45647, R3 0.244984, T1 0.000203758, T2 −0.000163233 |
| Telephoto | false | 4000×3000 | 29.9 | 6.4/4000 × 4.8/3000 | all 0 |

  The same file also has `djiM3TA` (Mavic 3 Thermal Advanced: thermal 640×512, f 8.72 mm, pitch 5.12/640) and
  `djiM3E`.
- **DJI M30T: not present.** No `M30` string appears in any `makeModel` or `comment`. The nearest entries are the
  M300 payloads `djiZENMUSEH20T` / `djiZH20T` (colour 4056×3040 and thermal 640×512, no `focalLength`).
- Implication for §5.7: key lookups on (`makeModel`, `isThermal`, `widthPixels`). For M30T footage use DJI
  published intrinsics (thermal 640×512, 1280×1024 in super-resolution; wide 4000×3000), or calibrate. Record
  this as a gap. Check upstream for a newer commit later.

## 8. TAK clients and server: documentation only, NOT INSTALLED

| Component | Licence | Where | Needed for the demo? |
|---|---|---|---|
| **pytak** (snstac/pytak) | Apache-2.0; 254★; v7.6.1 (2026-08-23), active | PyPI `pytak` (Python; belongs to the Python-env agent) | Only for the CoT-output stretch feature (doc §5.9, F-stretch) |
| **ATAK-CIV** (deptofdefense/AndroidTacticalAssaultKit-CIV) | custom/"NOASSERTION" on GitHub; the **repo is archived** (last release 4.6.0.5, Oct 2024) | The current ATAK-CIV build is on Google Play ("ATAK-CIV") and at tak.gov | No. An Android phone could show CoT markers as a demo extra |
| **WinTAK-CIV** | Government-off-the-shelf, free, distribution-controlled | tak.gov after free account registration (not re-verified today; login-walled) | No. A Windows TAK client for a CoT demo; the account must be created by the user |
| **FreeTAKServer** (FreeTAKTeam/FreeTakServer) | **EPL-2.0**; 956★; latest release v2.2.1 (2024-05-10), last push 2024-10-29 (slow) | PyPI `FreeTAKServer`, or Docker | No. pytak can send CoT straight to a client over UDP multicast (239.2.3.1:6969) or TCP without a server |

**Recommendation:** skip TAK for the core demo. If the stretch is attempted, pytak → WinTAK over UDP multicast
on the same LAN is the lowest-effort path. It needs no server, no certificates and no install on this PC
beyond WinTAK.

## 9. Gamepad: none connected

- **XInput** (`xinput1_4.dll` `XInputGetState`, slots 0 to 3): every slot returns `1167 ERROR_DEVICE_NOT_CONNECTED`.
- **PnP:** no present device of class `XnaComposite`/`XboxComposite`, and no HID device with Generic Desktop
  usage 0x04 (joystick) or 0x05 (gamepad). The present HID devices are only the ELAN touchpad, the Intel HID
  event filter, consumer/system controls and a portable device control.
- **Registry history:** `HKCU\…\MediaProperties\PrivateProperties\Joystick\OEM\VID_045E&PID_028E` = "Controller
  (XBOX 360 For Windows)". An Xbox 360-compatible pad has been plugged in before, so the driver is known to Windows.
- **Plan for the F3 takeover tests:** connect an XInput pad (Xbox 360/One/Series, or an XInput-mode clone) and
  re-run the XInput probe. Cosys `"RC": {"RemoteControlID": 0}` should then map to slot 0. Until then, the
  mission loop's takeover logic can be unit-tested against a fake joystick source.

---

## Disk footprint (all on D:)

| Path | Size |
|---|---|
| `_downloads/CesiumForUnreal-58-v2.29.1.zip` | 1,226,845,642 B |
| `_staging/plugins/CesiumForUnreal` | 5.45 GB (moves into the project; see §1) |
| `D:\Tools\X-AnyLabeling` (exe + launcher + work) | 556 MB (+1.2 GB transient `tmp` while running) |
| `D:\Tools\pmtiles` | 56 MB |
| `_downloads/npm` + `basemaps-assets` zip/extract | ~41 MB |
| `app/map/vendor` | 11.9 MB |
| `data/basemap/wayanad.pmtiles` | 1.7 MB |
| `data/reference/DroneModels.json` | 60 KB |
| `_downloads/go-pmtiles…zip`, `x_anylabeling_cvhub-4.0.6-py3-none-any.whl` (inspection only, sha256 `fb1a37e4…30f8`, matches) | 17.8 MB, 3.4 MB |

C: free space swung 43.9 → 26.4 → 40.6 GB during this session. None of this work writes to C:, so the dip was
probably pagefile growth or the parallel UE build. Watch it.

`.gitignore` now also ignores `_staging/` and `_scratch/`.
