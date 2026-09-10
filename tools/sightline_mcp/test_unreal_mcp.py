"""Conformance check for Epic's in-editor MCP server (http://localhost:8000/mcp), using the same Streamable HTTP
client transport as Claude Code. Requires the editor to be running with the SightlineSim project.

Checks: initialize + protocol negotiation, tools/list, list_toolsets, describe_toolset for each toolset, and a
harmless real call through call_tool (IsPIERunning). Prints the toolset/tool inventory so it can be recorded in
docs/CONTEXT.md. Exit code 0 = usable.
Run: uv run python tools/sightline_mcp/test_unreal_mcp.py [--url http://localhost:8000/mcp]
"""

import argparse
import asyncio
import json
import re
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


def _text(result) -> str:
    return "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")


async def main(url: str, verbose: bool) -> int:
    async with streamablehttp_client(url) as (read, write, get_session_id), ClientSession(read, write) as session:
        init = await session.initialize()
        print(f"server: {init.serverInfo.name} {init.serverInfo.version}; protocol {init.protocolVersion}; "
              f"session {get_session_id()}")
        tools = (await session.list_tools()).tools
        names = [t.name for t in tools]
        print(f"native tools ({len(names)}): {names}")

        if "list_toolsets" not in names:
            print("tool search disabled: every toolset tool is native (bEnableToolSearch=False)")
            return 0

        lt = await session.call_tool("list_toolsets", {})
        toolsets_raw = _text(lt)
        print("list_toolsets ->", toolsets_raw[:3000] if verbose else toolsets_raw[:800])
        # The catalog is plain text, one "- <Name>: <description>" line per toolset (GetToolsetCatalogText), not JSON.
        # Descriptions contain their own "- " bullets; real toolset names are dotted "<Plugin>.<Toolset>".
        toolsets = [m.group(1) for m in re.finditer(r"^- ([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)(?::|$)", toolsets_raw, re.M)]
        if lt.isError or not toolsets:
            print("FAILED: list_toolsets returned no toolsets")
            return 1

        # describe each toolset (argument is `toolset_name`), so the schemas of the call_tool payloads are known
        failed = []
        for ts_name in toolsets:
            desc = await session.call_tool("describe_toolset", {"toolset_name": ts_name})
            text = _text(desc)
            if desc.isError:
                failed.append(f"{ts_name}: {text[:200]}")
                continue
            try:
                json.loads(text)
            except json.JSONDecodeError:
                failed.append(f"{ts_name}: schema is not JSON")
            if verbose:
                print(f"--- {ts_name}\n{text[:4000]}")
        print(f"described toolsets: {len(toolsets) - len(failed)}/{len(toolsets)}")

        schema = next(t for t in tools if t.name == "call_tool").inputSchema
        print("call_tool input schema:", json.dumps(schema))

        # one harmless real call through call_tool
        pie = await session.call_tool("call_tool", {"toolset_name": _find_toolset(toolsets_raw, "EditorApp", toolsets),
                                                    "tool_name": "IsPIERunning", "arguments": {}})
        print("call_tool IsPIERunning ->", pie.isError, _text(pie)[:200])
        if failed or pie.isError:
            print("FAILED:", failed or _text(pie)[:300])
            return 1
        return 0


def _find_toolset(catalog: str, hint: str, names: list[str]) -> str:
    return next((n for n in names if hint.lower() in n.lower()), names[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/mcp")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    try:
        sys.exit(asyncio.run(main(a.url, a.verbose)))
    except Exception as e:  # noqa: BLE001 - report connection failures plainly
        print(f"FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
