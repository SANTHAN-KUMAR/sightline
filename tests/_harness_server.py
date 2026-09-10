"""Test harness: runs the real tools/sightline_mcp/server.py (imported as a module, then server.main() over the real
stdio transport) with one fault injected. Used only by tests/test_mcp_protocol.py; never by Claude Code.

    python tests/_harness_server.py noise          # tool code writes to stdout (print, fd 1, Win32 STD_OUTPUT)
    python tests/_harness_server.py garbage        # negative control: non-JSON written to the pipe before main()
    python tests/_harness_server.py fake_editor    # a fake editor process + fake remote-execution channel

fake_editor interprets the ue_python `code` argument:
    "SLEEP:<s>"  run for <s> seconds, then succeed
    "TIMEOUT"    block for timeout_s, then raise TimeoutError (the command was delivered)
    "DROP_ONCE"  the first attempt loses the command channel (editor restarted), the retry succeeds
Every result/error carries the global attempt counter so the test can prove what was (not) re-sent.
"""

import ctypes
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools" / "sightline_mcp"))

mode = sys.argv[1] if len(sys.argv) > 1 else "none"

if mode == "garbage":
    # Deliberately bypasses the protection: proves the test's stray-output detector actually fires.
    os.write(1, b"THIS LINE IS NOT JSON-RPC\n")

import server
from ue_remote import CommandConnectionClosed

if mode == "noise":
    def _noisy_ue_processes():
        print("STRAY-PRINT-MARKER from a tool thread")
        os.write(1, b"STRAY-FD1-MARKER from native-style write\n")
        h = ctypes.windll.kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE, what C/C++ code writes to
        buf = b"STRAY-WIN32-MARKER via WriteFile(GetStdHandle)\n"
        ctypes.windll.kernel32.WriteFile(h, buf, len(buf), ctypes.byref(ctypes.c_ulong()), None)
        return []  # report no editor, so editor_wait_ready never talks to a real (other agent's) editor

    server._ue_processes = _noisy_ue_processes

elif mode == "fake_editor":
    _attempts = 0
    _count_lock = threading.Lock()
    _dropped = False

    def _fake_ue_processes():
        return [{"pid": 0, "name": "UnrealEditor.exe", "rss_gb": 0.0, "age_s": 1, "cmdline": "fake editor"}]

    class FakeRemote:
        def __init__(self, command_port: int = 0):
            self.command_port = command_port

        def nodes(self, wait_s: float = 2.5):
            return [{"node_id": "fake-node", "project_name": "Fake", "engine_version": "0", "machine": "test"}]

        def disconnect(self):
            pass

        def stop(self):
            pass

        def run(self, command, mode=None, unattended=True, timeout_s=300.0):
            global _attempts, _dropped
            with _count_lock:
                _attempts += 1
                n = _attempts
            if command.startswith("SLEEP:"):
                time.sleep(float(command.split(":", 1)[1]))
                return {"success": True, "result": f"slept attempt#{n}", "output": []}
            if command == "TIMEOUT":
                time.sleep(timeout_s)
                raise TimeoutError(f"fake socket timeout attempt#{n}")
            if command == "DROP_ONCE":
                if not _dropped:
                    _dropped = True
                    raise CommandConnectionClosed(f"fake channel closed attempt#{n}")
                return {"success": True, "result": f"reconnected attempt#{n}", "output": []}
            if command.startswith("import unreal"):  # editor_wait_ready's readiness probe: the editor never answers
                time.sleep(timeout_s)
                raise TimeoutError(f"fake probe timeout attempt#{n}")
            return {"success": True, "result": f"ok attempt#{n}", "output": []}

    server._ue_processes = _fake_ue_processes
    server.UnrealRemote = FakeRemote

server.main()
