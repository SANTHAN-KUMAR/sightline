"""Environment doctor: verifies every piece of the Sightline toolchain and says exactly what is missing.

Run at the start of every session:  uv run python tools/doctor.py   (add --live to also probe running services)
Exit code = number of FAIL checks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UE_ROOT = Path(os.environ.get("SIGHTLINE_UE_ROOT", r"D:\UE_5.8"))
PROJECT = REPO / "sim" / "SightlineSim" / "SightlineSim.uproject"
VS_ROOT = Path(r"D:\VS\2022\Community")
EXPECTED_ENV = {
    "UV_CACHE_DIR": r"D:\Tools\cache\uv", "UV_PYTHON_INSTALL_DIR": r"D:\Tools\uv-python",
    "PIP_CACHE_DIR": r"D:\Tools\cache\pip", "HF_HOME": r"D:\Tools\cache\hf", "TORCH_HOME": r"D:\Tools\cache\torch",
    "YOLO_CONFIG_DIR": r"D:\Tools\cache\ultralytics", "UE-LocalDataCachePath": r"D:\UE_Cache\DDC",
}

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn: bool = False) -> None:
    results.append(("PASS" if ok else ("WARN" if warn else "FAIL"), name, detail))


def user_env(name: str) -> str | None:
    """Read the persisted user-level value (the current process may predate the change)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, name)[0]
    except OSError:
        return None


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main(live: bool) -> int:
    # --- engine
    bv = UE_ROOT / "Engine" / "Build" / "Build.version"
    if bv.exists():
        v = json.loads(bv.read_text())
        check("Unreal Engine", (v["MajorVersion"], v["MinorVersion"]) == (5, 8),
              f"{v['MajorVersion']}.{v['MinorVersion']}.{v['PatchVersion']} CL{v['Changelist']} compat {v['CompatibleChangelist']}")
        compat = str(v["CompatibleChangelist"])
    else:
        check("Unreal Engine", False, f"missing {bv}")
        compat = ""
    mcp_plugin = UE_ROOT / "Engine/Plugins/Experimental/ModelContextProtocol/Binaries/Win64/UnrealEditor-ModelContextProtocol.dll"
    check("Epic MCP plugin binaries", mcp_plugin.exists(), str(mcp_plugin.parent))

    # --- MSVC toolchain acceptable to UE 5.8 (Windows_SDK.json)
    # UBT judges the toolset by cl.exe's file version (MicrosoftPlatformSDK.cs), not the folder name: VS 2022 17.14
    # keeps the folder at 14.44.35207 while servicing updates cl.exe.
    msvc_dir = VS_ROOT / "VC" / "Tools" / "MSVC"
    versions = []
    for d in sorted(msvc_dir.iterdir()) if msvc_dir.exists() else []:
        cl = d / "bin" / "Hostx64" / "x64" / "cl.exe"
        if cl.exists():
            r = subprocess.run(["powershell", "-NoProfile", "-Command", f"(Get-Item '{cl}').VersionInfo.ProductVersion"],
                               capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
            versions.append(r.stdout.strip())

    def ok_ver(v: str) -> bool:
        m = re.match(r"14\.(\d+)\.(\d+)", v)
        if not m:
            return False
        minor, build = int(m.group(1)), int(m.group(2))
        return (minor == 44 and build >= 35211) or (minor == 50 and build >= 35723)
    check("MSVC cl.exe (14.44.35211+ / 14.50.35723+)", any(ok_ver(v) for v in versions),
          f"cl.exe versions {versions or 'none'} under {msvc_dir}")
    sdk = Path(r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64\kernel32.lib")
    check("Windows SDK 10.0.22621 (libs)", sdk.exists(), str(sdk))

    # --- project + plugin
    check("SightlineSim.uproject", PROJECT.exists(), str(PROJECT))
    plugin_mod = PROJECT.parent / "Plugins/AirSim/Binaries/Win64/UnrealEditor.modules"
    if plugin_mod.exists():
        bid = json.loads(plugin_mod.read_text())["BuildId"]
        check("Cosys-AirSim plugin BuildId matches engine", bid == compat, f"plugin {bid} vs engine {compat}")
    else:
        check("Cosys-AirSim plugin", False, "run tools/setup/fetch_airsim.ps1")
    proj_bin = PROJECT.parent / "Binaries/Win64/UnrealEditor-SightlineSim.dll"
    check("SightlineSimEditor built", proj_bin.exists(), str(proj_bin), warn=True)
    check("AirSim settings profile", (REPO / "sim/settings/default.json").exists())

    # --- python env
    check("Python 3.11 venv", sys.version_info[:2] == (3, 11), sys.version.split()[0])
    for mod in ("cosysairsim", "mcp", "numpy", "cv2", "psutil", "PIL"):
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, str(e))
    check("uv.lock present", (REPO / "uv.lock").exists())

    # --- caches on D:
    for k, want in EXPECTED_ENV.items():
        got = user_env(k)
        check(f"env {k}", (got or "").lower() == want.lower(), f"{got!r}")
    check("uv on PATH", shutil.which("uv") is not None, shutil.which("uv") or "restart the terminal", warn=True)

    # --- disks / memory
    import psutil
    c_free = shutil.disk_usage("C:\\").free / 2**30
    d_free = shutil.disk_usage("D:\\").free / 2**30
    check("C: free > 25 GB", c_free > 25, f"{c_free:.1f} GB", warn=True)
    check("D: free > 60 GB", d_free > 60, f"{d_free:.1f} GB", warn=True)
    vm = psutil.virtual_memory()
    check("RAM available >= 9 GB for editor", vm.available / 2**30 >= 9, f"{vm.available / 2**30:.1f} GB available", warn=True)

    # --- MCP config
    mcp_json = REPO / ".mcp.json"
    try:
        cfg = json.loads(mcp_json.read_text())["mcpServers"]
        check(".mcp.json servers", {"unreal", "sightline"} <= set(cfg), ", ".join(cfg))
    except Exception as e:  # noqa: BLE001
        check(".mcp.json", False, str(e))

    if live:
        check("Epic MCP HTTP :8000 (editor running)", port_open(8000), warn=True)
        check("AirSim RPC :41451 (PIE/game running)", port_open(41451), warn=True)
        r = subprocess.run([sys.executable, str(REPO / "tools/sightline_mcp/smoke_test.py")], capture_output=True,
                           text=True, timeout=120, stdin=subprocess.DEVNULL)
        check("sightline MCP stdio smoke test", r.returncode == 0, (r.stdout.strip().splitlines() or [""])[-1])

    width = max(len(n) for _, n, _ in results)
    for status, name, detail in results:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    fails = sum(1 for s, _, _ in results if s == "FAIL")
    print(f"\n{fails} FAIL, {sum(1 for s, _, _ in results if s == 'WARN')} WARN")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also probe running editor/sim and the MCP server")
    sys.exit(main(ap.parse_args().live))
