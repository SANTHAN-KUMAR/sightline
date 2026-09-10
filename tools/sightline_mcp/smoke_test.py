"""End-to-end check of the sightline MCP server over the real stdio MCP transport.

Spawns server.py exactly as Claude Code does (.mcp.json), performs the MCP handshake, lists tools and calls
`status`. Exit code 0 = server usable. Run: uv run python tools/sightline_mcp/smoke_test.py
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parents[2]
EXPECTED = {
    "status", "editor_launch", "editor_wait_ready", "editor_close", "sim_launch_game", "ue_build",
    "ue_generate_project_files", "ue_package", "job_status", "job_list", "job_kill", "ue_python",
    "ue_python_headless", "ue_console", "ue_log", "sim_ping", "sim_state", "sim_fly", "sim_capture",
    "sim_environment", "sim_clock", "sim_objects", "sim_set_object_pose", "sim_detections",
    "sim_list_assets", "sim_spawn_object", "sim_destroy_object",
}


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(REPO / "tools" / "sightline_mcp" / "server.py")],
                                   cwd=str(REPO))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        print("server:", init.serverInfo.name, init.serverInfo.version, "protocol", init.protocolVersion)
        tools = {t.name for t in (await session.list_tools()).tools}
        missing = EXPECTED - tools
        print(f"tools: {len(tools)}; missing: {sorted(missing) or 'none'}")
        res = await session.call_tool("status", {})
        status = json.loads(res.content[0].text)
        print("status.endpoints:", json.dumps(status["endpoints"]))
        print("status.ram_available_gb:", status["ram_available_gb"], "gpu:", status["gpu"])
        return 1 if missing or res.isError else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
