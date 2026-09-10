# Setup from scratch (reproducible)

Everything installs to **D:**. Commands are PowerShell. Verify at the end with `uv run python tools/doctor.py --live`.

## 1. Prerequisites already on the machine
- Unreal Engine **5.8.2** via the Epic Launcher at `D:\UE_5.8` (Launcher > Unreal Engine > Library > install location D:).
- Windows SDK 10.0.22621 (present under `C:\Program Files (x86)\Windows Kits\10`; it cannot be relocated).
- Git, NVIDIA driver (>= 592).

## 2. Visual Studio 2022 17.14 (C++ toolchain for UE 5.8 + Cosys-AirSim)
UE 5.8 accepts MSVC **14.44.35211+** (VS 2022 17.14) or **14.50.35723+** (VS 2026); 14.44.0-14.44.35210 is banned
(`Engine/Config/Windows/Windows_SDK.json`). Install with the bootstrapper directly. winget misreports a failed
install as "already installed".
```powershell
curl.exe -L -o D:\Sightline\_downloads\vs_community.exe https://aka.ms/vs/17/release/vs_community.exe
D:\Sightline\_downloads\vs_community.exe --installPath "D:\VS\2022\Community" --path cache="D:\VS\Cache" --path shared="D:\VS\Shared" `
  --add Microsoft.VisualStudio.Workload.NativeGame --add Microsoft.VisualStudio.Workload.NativeDesktop `
  --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64 `
  --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.Net.Component.4.6.2.TargetingPack `
  --add Component.Unreal.Ide --add Component.Unreal.Debugger --passive --norestart --wait
```
`--path cache/shared` only takes effect on the *first* VS install on a machine.

## 3. uv + caches on D:
```powershell
winget install --id astral-sh.uv --exact --location D:\Tools\uv
# user-level env vars (see docs/CONTEXT.md §3 for the full list)
[Environment]::SetEnvironmentVariable('UV_CACHE_DIR','D:\Tools\cache\uv','User')
[Environment]::SetEnvironmentVariable('UV_PYTHON_INSTALL_DIR','D:\Tools\uv-python','User')
[Environment]::SetEnvironmentVariable('UV_PYTHON_BIN_DIR','D:\Tools\uv-python\bin','User')
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath','D:\UE_Cache\DDC','User')
# ... PIP_CACHE_DIR, HF_HOME, TORCH_HOME, YOLO_CONFIG_DIR, npm_config_cache, UV_TOOL_DIR, UV_TOOL_BIN_DIR
```
Restart terminals and Claude Code afterwards.

## 4. Cosys-AirSim plugin + Python env
```powershell
cd D:\Sightline
powershell -ExecutionPolicy Bypass -File tools\setup\fetch_airsim.ps1   # plugin -> sim\SightlineSim\Plugins\AirSim
uv sync                                                                  # Python 3.11 + locked deps
uv run python tools\sightline_mcp\smoke_test.py                          # sightline MCP server OK?
```

## 5. Build and open the project
```powershell
D:\UE_5.8\Engine\Build\BatchFiles\Build.bat SightlineSimEditor Win64 Development -Project="D:\Sightline\sim\SightlineSim\SightlineSim.uproject" -WaitMutex
D:\UE_5.8\Engine\Binaries\Win64\UnrealEditor.exe D:\Sightline\sim\SightlineSim\SightlineSim.uproject -settings="D:\Sightline\sim\settings\default.json"
```
Or, from Claude Code: sightline MCP `ue_build` then `editor_launch`.

## 6. Claude Code MCP connection
`.mcp.json` at the repo root registers `unreal` (http://localhost:8000/mcp, live only while the editor runs) and
`sightline` (stdio). Open Claude Code with `D:\Sightline` as the working directory and approve both project MCP
servers when prompted. Verify with `uv run python tools\sightline_mcp\test_unreal_mcp.py` once the editor is up.
