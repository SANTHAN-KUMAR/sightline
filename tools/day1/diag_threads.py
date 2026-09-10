"""Isolates why a cosysairsim call can hang inside the MCP server. Each scenario runs with a hard timeout.
A: call on a ThreadPoolExecutor worker thread
B: A + contextlib.redirect_stdout(sys.stderr)
C: inside a running asyncio loop, via anyio.to_thread -> executor (exactly the server's path)
Run: uv run python -u tools/day1/diag_threads.py <A|B|C>"""

import asyncio
import contextlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anyio
import cosysairsim as airsim

EX = ThreadPoolExecutor(max_workers=1)


def call():
    t = time.time()
    c = airsim.MultirotorClient(timeout_value=10)
    ping = c.ping()
    v = c.listVehicles()
    return f"ping={ping} vehicles={v} in {time.time() - t:.2f}s"


def call_redirected():
    with contextlib.redirect_stdout(sys.stderr):
        return call()


def main(mode: str) -> None:
    if mode == "A":
        print("A:", EX.submit(call).result(timeout=20), flush=True)
    elif mode == "B":
        print("B:", EX.submit(call_redirected).result(timeout=20), flush=True)
    elif mode == "C":
        async def run():
            return await anyio.to_thread.run_sync(lambda: EX.submit(call_redirected).result(timeout=20))
        print("C:", asyncio.run(run()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
