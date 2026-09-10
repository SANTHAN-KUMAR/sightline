"""torchcodec on Windows needs an external FFmpeg. Do NOT reuse PyAV's bundled DLLs (verified crash).

torchcodec's Windows wheel ships no FFmpeg: it loads `avutil-<N>.dll` / `avcodec-<N>.dll` ... by canonical name
and raises "Could not load libtorchcodec" when they are absent.

**Tempting but broken shortcut.** PyAV's wheel bundles FFmpeg 8 under hash-mangled names in
`site-packages\\av.libs`. Exposing them under canonical names (hardlinks + `os.add_dll_directory`) *appears* to
work - a standalone script decoded frames and PTS correctly - but PyAV's FFmpeg is a MinGW build (it ships
libgcc/libstdc++/libwinpthread) while torchcodec's core is MSVC-built. They use different C runtime heaps, and
allocations crossing that boundary corrupt the heap: verified 2026-09-10, the pytest process died with
0xC0000374 (STATUS_HEAP_CORRUPTION) inside `create_decoder`, killing the whole test session. A crash like that
can also appear only under load, so a single successful run proves nothing. `enable()` therefore refuses unless
`allow_unsafe=True`, and nothing in the project calls it.

**Supported fix (user action, one-off).** Install an FFmpeg 4-8 *shared* build on D: (e.g. the LGPL "shared"
Windows build from ffmpeg.org / gyan.dev) and put its `bin` directory on PATH, or call
`os.add_dll_directory(r"D:\\Tools\\ffmpeg\\bin")` before importing torchcodec. Then drop the skip in
`tests/test_stack.py::test_torchcodec_decode` and re-run it.

Until then the project uses **PyAV** for PTS/metadata and **PyNvVideoCodec** for GPU decode (doc 5.4), which is
the primary path anyway; torchcodec is listed as ADAPT in Appendix A.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SHIM_ROOT = Path(r"D:\Tools\ffmpeg-pyav-shim")
_LIBS = ("avcodec", "avformat", "avutil", "swscale", "swresample", "avfilter", "avdevice")


def ffmpeg_dll_dir_on_path() -> str | None:
    """Return the first PATH entry that looks like a real FFmpeg shared build, if any."""
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        try:
            p = Path(entry)
            if p.is_dir() and any(re.match(r"^avutil-\d+\.dll$", f.name) for f in p.glob("avutil-*.dll")):
                return str(p)
        except OSError:
            continue
    return None


def enable(allow_unsafe: bool = False) -> Path:
    """Expose PyAV's bundled FFmpeg under canonical names. UNSAFE - see the module docstring."""
    if not allow_unsafe:
        raise RuntimeError(
            "Reusing PyAV's MinGW FFmpeg for torchcodec corrupts the heap (0xC0000374) because torchcodec is "
            "MSVC-built. Install an FFmpeg 4-8 shared build on D: and put its bin\\ on PATH instead."
        )
    import av

    src = Path(av.__file__).resolve().parent.parent / "av.libs"
    dst = SHIM_ROOT / f"av-{av.__version__}"
    dst.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^((?:{'|'.join(_LIBS)})-\d+)-[0-9a-f]{{32}}\.dll$")
    for f in src.glob("*.dll"):
        m = pat.match(f.name)
        if m and not (dst / f"{m.group(1)}.dll").exists():
            os.link(f, dst / f"{m.group(1)}.dll")
    if sys.platform == "win32":
        os.add_dll_directory(str(dst))
        os.add_dll_directory(str(src))
    return dst
