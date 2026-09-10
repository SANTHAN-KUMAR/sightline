"""Runs the sightline MCP server over real stdio with faulthandler armed, calls one tool, and lets the server dump
every thread's stack to stderr if the call has not returned after N seconds. Pinpoints hangs inside the server.
Run: uv run python -u tools/day1/diag_mcp_hang.py [tool] [--dump-after 15]"""

import argparse
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]


async def main(tool: str, dump_after: int) -> None:
    boot = ("import faulthandler, sys, runpy; "
            f"faulthandler.dump_traceback_later({dump_after}, repeat=False, file=sys.stderr); "
            f"sys.argv=[r'{REPO / 'tools/sightline_mcp/server.py'}']; "
            f"runpy.run_path(r'{REPO / 'tools/sightline_mcp/server.py'}', run_name='__main__')")
    params = StdioServerParameters(command=sys.executable, args=["-u", "-c", boot], cwd=str(REPO))
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        print("initialized; calling", tool, flush=True)
        res = await asyncio.wait_for(s.call_tool(tool, {}), timeout=dump_after + 10)
        print("RESULT:", [getattr(c, "text", c)[:500] for c in res.content], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", nargs="?", default="sim_ping")
    ap.add_argument("--dump-after", type=int, default=15)
    a = ap.parse_args()
    try:
        asyncio.run(main(a.tool, a.dump_after))
    except Exception as e:  # noqa: BLE001
        print("FAILED:", type(e).__name__, e, flush=True)
        sys.exit(1)
