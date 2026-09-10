"""Reproduce (or rule out) duplicate execution of a timed-out ue_python command, through real stdio MCP sessions.

Scenario: the editor's game thread is busy (another client runs a 6 s script), a second client sends a small
command with timeout_s=2 that is queued behind it and times out. The pre-fix server then re-discovered the editor
and re-sent the same command; the fixed server reports the timeout and does not re-send.
Usage: python repro_timeout_reexec.py <server.py under test> [--label NAME]   (editor must be running)
Prints how many times the queued command actually executed in the editor.
"""

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
FIXED = str(REPO / "tools" / "sightline_mcp" / "server.py")


def params(server_py: str) -> StdioServerParameters:
    return StdioServerParameters(command=PY, args=[server_py], cwd=str(REPO),
                                 env={**os.environ, "SIGHTLINE_UPROJECT": str(REPO / "sim/SightlineSim/SightlineSim.uproject")})


async def main(server_under_test: str, label: str) -> int:
    errlog = open(REPO / "_artifacts" / "test_engine" / "repro_stderr.log", "a", encoding="utf-8")  # noqa: SIM115
    async with stdio_client(params(FIXED), errlog=errlog) as (r1, w1), ClientSession(r1, w1) as load, \
            stdio_client(params(server_under_test), errlog=errlog) as (r2, w2), ClientSession(r2, w2) as sut:
        await load.initialize()
        await sut.initialize()
        await sut.call_tool("ue_python", {"code": "import builtins; builtins._slt_q = 0", "mode": "file"})  # channel up
        marker = REPO / "_artifacts" / "test_engine" / "busy_marker.txt"
        marker.unlink(missing_ok=True)
        if HITCH:
            # Editor hitch on the SAME channel (like a map load / BP compile): a one-shot slate post-tick callback
            # stalls the game thread 6 s right after this call returns; the next command is queued behind it.
            stall = ("import unreal, time, pathlib\n_h = []\n"
                     "def _cb(dt):\n    unreal.unregister_slate_post_tick_callback(_h[0])\n"
                     f"    pathlib.Path(r'{marker}').write_text('busy')\n    time.sleep(6)\n"
                     "_h.append(unreal.register_slate_post_tick_callback(_cb))\nprint('stall armed')")
            await sut.call_tool("ue_python", {"code": stall, "mode": "file"})
            busy = asyncio.create_task(asyncio.sleep(0))
        else:
            busy = asyncio.create_task(load.call_tool("ue_python", {
                "code": f"import time, pathlib\npathlib.Path(r'{marker}').write_text('busy')\ntime.sleep(6)",
                "timeout_s": 60}))
        for _ in range(200):  # deterministic: send only once the game thread is inside the 6 s sleep
            if marker.exists():
                break
            await asyncio.sleep(0.05)
        print(f"{label}: editor busy={marker.exists()}")
        t0 = time.time()
        res = await sut.call_tool("ue_python", {"code": "import builtins; builtins._slt_q += 1; print('q ran')",
                                                "timeout_s": 2})
        took = time.time() - t0
        await busy
        await asyncio.sleep(8)  # let any orphaned/queued execution finish
        ev = await load.call_tool("ue_python", {"code": "__import__('builtins')._slt_q", "mode": "eval"})
        txt = "\n".join(c.text for c in ev.content if c.type == "text")
        m = re.search(r"^result: (\d+)", txt, re.M)
        runs = int(m.group(1)) if m else -1
        first = "\n".join(c.text for c in res.content if c.type == "text")[:160]
        print(f"{label}: queued command executed {runs}x; call isError={res.isError} took={took:.1f}s msg={first!r}")
        return runs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("server")
    ap.add_argument("--label", default="server")
    ap.add_argument("--hitch", action="store_true", help="stall the editor on the same channel (no second client)")
    a = ap.parse_args()
    HITCH = a.hitch
    sys.exit(0 if asyncio.run(main(a.server, a.label)) >= 0 else 1)
