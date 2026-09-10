"""Client for the Unreal Editor Python remote-execution protocol.

Implements the protocol defined by Epic in
Engine/Plugins/Experimental/PythonScriptPlugin/Content/Python/remote_execution.py
(UDP multicast discovery on 239.0.0.1:6766, TCP command channel we host on 127.0.0.1).

It is re-implemented rather than imported because the engine's
`_receive_message` stops reading at the first short TCP chunk, which truncates
large command results (e.g. actor dumps). Here we read until the JSON document
is complete.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass

PROTOCOL_VERSION = 1
PROTOCOL_MAGIC = "ue_py"

MULTICAST_GROUP = ("239.0.0.1", 6766)
MULTICAST_BIND = "127.0.0.1"
MULTICAST_TTL = 0
COMMAND_HOST = "127.0.0.1"

MODE_EXEC_FILE = "ExecuteFile"
MODE_EXEC_STATEMENT = "ExecuteStatement"
MODE_EVAL_STATEMENT = "EvaluateStatement"


@dataclass
class RemoteNode:
    node_id: str
    data: dict
    last_seen: float


def _message(type_: str, source: str, dest: str | None = None, data: dict | None = None) -> bytes:
    obj = {"version": PROTOCOL_VERSION, "magic": PROTOCOL_MAGIC, "type": type_, "source": source}
    if dest:
        obj["dest"] = dest
    if data:
        obj["data"] = data
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _parse(raw: bytes) -> dict | None:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if obj.get("version") != PROTOCOL_VERSION or obj.get("magic") != PROTOCOL_MAGIC:
        return None
    return obj


class CommandConnectionClosed(RuntimeError):
    """The editor closed the TCP command channel (it restarted, crashed or quit). Safe to reconnect and re-send."""


class UnrealRemote:
    """One discovery session plus at most one open command connection."""

    def __init__(self, command_port: int = 0):
        # command_port 0 = let the OS pick a free loopback port per connect. The editor connects to whatever
        # command_port the open_connection message names (PythonScriptRemoteExecution.cpp, FIPv4Endpoint), so an
        # ephemeral port is protocol-compatible. The old fixed 6776 + SO_REUSEADDR let two server instances (two
        # Claude sessions) listen on the same port at once on Windows, so an editor could connect to the wrong one.
        self.node_id = str(uuid.uuid4())
        self.command_port = command_port
        self._nodes: dict[str, RemoteNode] = {}
        self._lock = threading.RLock()
        self._running = False
        self._udp: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._cmd_sock: socket.socket | None = None
        self._connected_node: str | None = None

    # ---- discovery -------------------------------------------------------
    def start(self) -> None:
        with self._lock:  # status() and editor commands may call nodes() from different threads at once
            if self._running:
                return
            self._start_locked()

    def _start_locked(self) -> None:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind((MULTICAST_BIND, MULTICAST_GROUP[1]))
        udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
        udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)
        udp.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(MULTICAST_BIND))
        udp.setsockopt(
            socket.IPPROTO_IP,
            socket.IP_ADD_MEMBERSHIP,
            socket.inet_aton(MULTICAST_GROUP[0]) + socket.inet_aton(MULTICAST_BIND),
        )
        udp.settimeout(0.1)
        self._udp = udp
        self._running = True
        self._thread = threading.Thread(target=self._discovery_loop, daemon=True)
        self._thread.start()

    def _discovery_loop(self) -> None:
        last_ping = 0.0
        while self._running:
            try:
                raw, _ = self._udp.recvfrom(65535)
                msg = _parse(raw)
                if msg and msg["source"] != self.node_id and msg.get("dest") in (None, self.node_id):
                    if msg["type"] == "pong":
                        with self._lock:
                            self._nodes[msg["source"]] = RemoteNode(msg["source"], msg.get("data") or {}, time.time())
            except socket.timeout:
                pass
            except OSError:
                if not self._running:
                    return
            now = time.time()
            if now - last_ping >= 1.0:
                last_ping = now
                try:
                    self._udp.sendto(_message("ping", self.node_id), MULTICAST_GROUP)
                except OSError:
                    pass
            with self._lock:
                for nid in [n for n, node in self._nodes.items() if now - node.last_seen > 5.0]:
                    del self._nodes[nid]

    def nodes(self, wait_s: float = 2.5) -> list[dict]:
        """Return discovered editors, waiting up to wait_s for the first pong."""
        self.start()
        deadline = time.time() + wait_s
        while time.time() < deadline:
            with self._lock:
                if self._nodes:
                    break
            time.sleep(0.1)
        with self._lock:
            return [dict(n.data, node_id=n.node_id) for n in self._nodes.values()]

    # ---- command channel -------------------------------------------------
    def connect(self, node_id: str | None = None, wait_s: float = 5.0) -> str:
        found = self.nodes(wait_s)
        if not found:
            raise RuntimeError(
                "No Unreal Editor found via Python remote execution. Is the editor running with "
                "PythonScriptPlugin remote execution enabled (bRemoteExecution=True)?"
            )
        target = node_id or found[0]["node_id"]
        if self._cmd_sock and self._connected_node == target:
            return target
        self.disconnect()
        listener, port = self.open_listener()
        try:
            for _ in range(6):
                self._udp.sendto(
                    _message("open_connection", self.node_id, target,
                             {"command_ip": COMMAND_HOST, "command_port": port}),
                    MULTICAST_GROUP,
                )
                try:
                    sock, _ = listener.accept()
                    sock.setblocking(True)
                    self._cmd_sock = sock
                    self._connected_node = target
                    return target
                except socket.timeout:
                    continue
        finally:
            listener.close()
        raise RuntimeError("Editor was discovered but did not open the command connection.")

    def open_listener(self) -> tuple[socket.socket, int]:
        """Listening socket for the editor's command connection and the port to advertise in open_connection.
        Port 0 (default) binds an OS-assigned free port, so concurrent server instances never share a listener.
        A pinned port uses SO_EXCLUSIVEADDRUSE on Windows (never SO_REUSEADDR, which lets a second process bind the
        same listening port and steal the editor's connection)."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        if self.command_port and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        listener.bind((COMMAND_HOST, self.command_port))
        listener.listen(1)
        listener.settimeout(3)
        return listener, listener.getsockname()[1]

    def disconnect(self) -> None:
        if self._cmd_sock:
            try:
                self._udp.sendto(_message("close_connection", self.node_id, self._connected_node), MULTICAST_GROUP)
            except OSError:
                pass
            self._cmd_sock.close()
        self._cmd_sock = None
        self._connected_node = None

    def run(self, command: str, mode: str = MODE_EXEC_FILE, unattended: bool = True,
            timeout_s: float = 300.0) -> dict:
        """Run Python in the editor. Returns {'success', 'result', 'output': [{'type','output'}]}."""
        if not self._cmd_sock:
            self.connect()
        payload = _message("command", self.node_id, self._connected_node,
                           {"command": command, "unattended": unattended, "exec_mode": mode})
        try:
            self._cmd_sock.sendall(payload)
            return self._read_result(timeout_s)
        except (OSError, RuntimeError):
            # The editor may have restarted: drop the channel so the next call reconnects.
            self.disconnect()
            raise

    def _read_result(self, timeout_s: float) -> dict:
        self._cmd_sock.settimeout(timeout_s)
        buf = b""
        while True:
            part = self._cmd_sock.recv(1 << 20)
            if not part:
                raise CommandConnectionClosed("Editor closed the command connection.")
            buf += part
            msg = _parse(buf)
            if msg is None:
                continue  # incomplete JSON, keep reading
            if msg.get("type") == "command_result":
                return msg.get("data") or {}
            buf = b""

    def stop(self) -> None:
        self.disconnect()
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        if self._udp:
            self._udp.close()
        self._udp = None
