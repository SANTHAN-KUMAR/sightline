"""Real tool calls through Epic's in-editor MCP server (http://localhost:8000/mcp) via `call_tool`, using the same
Streamable HTTP client transport as Claude Code. Requires the editor running with the SightlineSim project.

Covers: initialize, tools/list, list_toolsets, describe_toolset (every toolset + an unknown name), and call_tool on
EditorToolset (IsPIERunning, GetSelectedActors, SearchCVars, CaptureViewport), the logs toolset (GetLogEntries,
GetLogCategories), ConfigSettings (ListContainers/ListCategories/ListSections/GetSectionPropertyValues, read-only,
cross-checked against Config/DefaultEngine.ini) and error paths. Tool/argument names are resolved from
describe_toolset output at run time, never hard-coded casing.

Writes the full toolset/tool inventory to _artifacts/unreal_mcp/inventory.json (+ raw describe_toolset schemas) and
the viewport capture to _artifacts/unreal_mcp/capture.png. Exit code = number of FAIL checks.
Run: .venv/Scripts/python.exe tools/sightline_mcp/test_unreal_calls.py [--url http://localhost:8000/mcp]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import re
import sys
from datetime import timedelta
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "_artifacts" / "unreal_mcp"
RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok, evidence: str) -> bool:
    RESULTS.append((name, bool(ok), evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)
    return bool(ok)


def _text(res) -> str:
    return "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _tool_specs(schema) -> list[dict]:
    """Find tool entries in a describe_toolset JSON document, whatever its exact nesting."""
    found: list[dict] = []

    def walk(o, key_name=None):
        if isinstance(o, dict):
            keys = set(o)
            if ("name" in keys or key_name) and keys & {"inputSchema", "input_schema", "parameters", "inputs"}:
                # describe_toolset reports qualified names ("EditorToolset.EditorAppToolset.IsPIERunning");
                # call_tool wants the bare tool name ("without toolset prefix").
                found.append({"name": str(o.get("name", key_name)).split(".")[-1],
                              "qualified": o.get("name", key_name), "description": o.get("description", ""),
                              "schema": o.get("inputSchema") or o.get("input_schema") or o.get("parameters")
                              or o.get("inputs")})
                return
            for k, v in o.items():
                walk(v, k if isinstance(v, dict) else None)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(schema)
    return found


def _props(spec: dict) -> dict:
    sch = spec.get("schema") or {}
    return sch.get("properties", {}) if isinstance(sch, dict) else {}


def map_args(spec: dict, logical: dict) -> dict:
    """Map logical C++ parameter names (ContainerName, bShowUI) onto the schema's property names."""
    props = list(_props(spec))
    out = {}
    for k, v in logical.items():
        cands = [p for p in props if norm(p) == norm(k)] or \
                [p for p in props if norm(p).lstrip("b") == norm(k).lstrip("b")]
        out[cands[0] if cands else k] = v
    return out


def _find_b64_image(o):
    if isinstance(o, dict):
        data = next((o[k] for k in o if k.lower() == "data" and isinstance(o[k], str) and len(o[k]) > 100), None)
        if data:
            mime = next((o[k] for k in o if k.lower() in ("mimetype", "mime_type")), "")
            return data, mime
        for v in o.values():
            r = _find_b64_image(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _find_b64_image(v)
            if r:
                return r
    return None


async def main(url: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "describe").mkdir(exist_ok=True)
    async with streamablehttp_client(url) as (read, write, get_sid), ClientSession(read, write) as s:
        init = await s.initialize()
        record("initialize", init.protocolVersion in ("2025-11-25", "2025-06-18", "2024-11-05") and get_sid(),
               f"server={init.serverInfo.name} {init.serverInfo.version} protocol={init.protocolVersion} "
               f"session={get_sid()}")
        tools = (await s.list_tools()).tools
        names = {t.name for t in tools}
        record("tools/list", {"list_toolsets", "describe_toolset", "call_tool"} <= names, f"native tools={sorted(names)}")

        lt = await s.call_tool("list_toolsets", {})
        catalog = _text(lt)
        # Descriptions contain their own "- " bullets; real toolset names are dotted ("EditorToolset.LogsToolset",
        # "editor_toolset.toolsets.actor.ActorTools").
        toolsets = [(m.group(1), (m.group(2) or "").strip())
                    for m in re.finditer(r"^- ([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)(?::\s*(.*))?$", catalog, re.M)]
        record("list_toolsets", not lt.isError and len(toolsets) >= 5, f"{len(toolsets)} toolsets: "
               f"{[n for n, _ in toolsets]}")

        inventory, where = [], {}
        bad = []
        for ts, ts_desc in toolsets:
            d = await s.call_tool("describe_toolset", {"toolset_name": ts})
            txt = _text(d)
            (OUT / "describe" / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', ts)}.json").write_text(txt, encoding="utf-8")
            try:
                specs = _tool_specs(json.loads(txt)) if not d.isError else []
            except json.JSONDecodeError:
                specs = []
            if d.isError or not specs:
                bad.append(ts)
            entry = {"toolset": ts, "description": ts_desc, "tools": []}
            for sp in specs:
                first = (sp.get("description") or "").strip().splitlines()
                entry["tools"].append({"name": sp["name"], "purpose": first[0] if first else "",
                                       "args": list(_props(sp))})
                where.setdefault(norm(sp["name"]), (ts, sp))
            inventory.append(entry)
        n_tools = sum(len(e["tools"]) for e in inventory)
        record("describe_toolset (all)", not bad, f"{len(toolsets) - len(bad)}/{len(toolsets)} toolsets parsed, "
               f"{n_tools} tools; unparsed={bad}")
        (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        d = await s.call_tool("describe_toolset", {"toolset_name": "NoSuchToolset"})
        record("describe_toolset (unknown -> error)", d.isError and "not found" in _text(d), _text(d)[:120])

        async def ct(tool: str, logical: dict | None = None, timeout: float = 120):
            if norm(tool) not in where:
                record(f"call_tool {tool}", False, "tool not found in any toolset")
                return None, "", []
            ts, sp = where[norm(tool)]
            args = map_args(sp, logical or {})
            res = await s.call_tool("call_tool", {"toolset_name": ts, "tool_name": sp["name"], "arguments": args},
                                    read_timeout_seconds=timedelta(seconds=timeout))
            raw = _text(res)
            imgs = [c for c in res.content if getattr(c, "type", "") == "image"]
            print(f"--> call_tool {ts}.{sp['name']}({json.dumps(args)[:300]}) isError={res.isError}\n{raw[:600]}\n",
                  flush=True)
            # toolset results are JSON objects {"returnValue": ...}; hand back the unwrapped value as JSON text
            txt = raw
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and "returnValue" in obj:
                    rv = obj["returnValue"]
                    txt = rv if isinstance(rv, str) else json.dumps(rv)
            except json.JSONDecodeError:
                pass
            return res, txt, imgs

        # --- EditorToolset
        res, txt, _ = await ct("IsPIERunning")
        if res:
            record("call_tool IsPIERunning", not res.isError and "false" in txt.lower(), f"-> {txt[:80]!r}")
        res, txt, _ = await ct("GetSelectedActors")
        if res:
            ok = not res.isError
            try:
                json.loads(txt)
            except json.JSONDecodeError:
                ok = False
            record("call_tool GetSelectedActors", ok, f"-> {txt[:120]!r}")
        res, txt, _ = await ct("SearchCVars", {"Name": "t.MaxFPS"})
        if res:
            record("call_tool SearchCVars", not res.isError and "t.MaxFPS" in txt, f"-> {txt[:160]!r}")
        # Epic quirk: CaptureViewport rejects {} ("input param captureTransform needs a default value") although the
        # C++ parameter is TOptional. Pass the current camera pose explicitly; add a neutral annotations config only
        # if that parameter is rejected the same way.
        _, cam, _ = await ct("GetCameraTransform")
        cap_args = {"captureTransform": json.loads(cam), "bShowUI": False}
        res, txt, imgs = await ct("CaptureViewport", cap_args, timeout=180)
        if res and res.isError and "annotations" in txt:
            cap_args["annotations"] = {"gridSpacing": 0, "gridExtent": 0, "gridHeight": 0, "maxLabelDistance": 0,
                                       "classFilter": {"refPath": "/Script/Engine.Actor"}, "maxLabels": 0}
            res, txt, imgs = await ct("CaptureViewport", cap_args, timeout=180)
        print("CaptureViewport args that worked:", sorted(cap_args), flush=True)
        if res:
            raw, mime, how = None, "", ""
            if imgs:
                raw, mime, how = base64.b64decode(imgs[0].data), imgs[0].mimeType, "MCP image content"
            else:
                try:
                    found = _find_b64_image(json.loads(txt))
                except json.JSONDecodeError:
                    found = None
                if found:
                    raw, mime, how = base64.b64decode(found[0]), found[1], "base64 in JSON text"
            info = "no image"
            ok = False
            if raw:
                im = Image.open(io.BytesIO(raw))
                im.load()
                ext = (im.format or "png").lower()
                (OUT / f"capture.{ext}").write_bytes(raw)
                ok = im.width >= 64 and im.height >= 64
                info = f"{how} {mime} {im.format} {im.width}x{im.height} {len(raw)} bytes -> {OUT / f'capture.{ext}'}"
            record("call_tool CaptureViewport", not res.isError and ok, info)

        # --- logs toolset
        res, txt, _ = await ct("GetLogEntries", {"Category": "", "Pattern": "", "MaxEntries": 5})
        if res:
            try:
                n = len(json.loads(txt))
            except (json.JSONDecodeError, TypeError):
                n = -1
            record("call_tool GetLogEntries", not res.isError and 0 < n <= 5, f"{n} entries, last={txt[-160:]!r}")
        # Epic quirk: the schema default for `category` is "LogsToolset" (not a log category), so omitting it fails
        # with "Log category 'LogsToolset' not found". Always pass category explicitly ("" = all categories).
        res, txt, _ = await ct("GetLogEntries", {"Category": "", "Pattern": "SLT_SPAWNED", "MaxEntries": 10})
        if res:
            record("call_tool GetLogEntries (sees sightline ue_python output)", not res.isError and "SLT_SPAWNED" in txt,
                   f"-> {txt[:200]!r}")
        res, txt, _ = await ct("GetLogCategories", {"Filter": "LogPython"})
        if res:
            record("call_tool GetLogCategories", not res.isError and "LogPython" in txt, f"-> {txt[:120]!r}")

        # --- ConfigSettings (read only)
        res, txt, _ = await ct("ListContainers")
        if res:
            record("call_tool ListContainers", not res.isError and "Project" in txt, f"-> {txt[:160]!r}")
        res, txt, _ = await ct("ListCategories", {"ContainerName": "Project"})
        cats = []
        if res:
            try:
                cats = json.loads(txt)
            except json.JSONDecodeError:
                pass
            record("call_tool ListCategories", not res.isError and "Project" in cats, f"{len(cats)} categories")
        res, txt, _ = await ct("ListSections", {"ContainerName": "Project", "CategoryName": "Project"})
        secs = []
        if res:
            try:
                secs = json.loads(txt)
            except json.JSONDecodeError:
                pass
            record("call_tool ListSections", not res.isError and bool(secs), f"-> {secs}")
        maps = next((x for x in secs if "map" in x.lower()), None)
        if maps:
            res, txt, _ = await ct("GetSectionPropertyValues", {"ContainerName": "Project", "CategoryName": "Project",
                                                               "SectionName": maps,
                                                               "PropertyNames": ["EditorStartupMap"]})
            ini = (REPO / "sim/SightlineSim/Config/DefaultEngine.ini").read_text()
            want = re.search(r"^EditorStartupMap=(\S+)", ini, re.M).group(1).split("/")[-1]
            record("call_tool GetSectionPropertyValues", res and not res.isError and want in txt,
                   f"section={maps!r} -> {txt[:200]!r} (DefaultEngine.ini EditorStartupMap ends with {want})")
        else:
            record("call_tool GetSectionPropertyValues", False, f"no maps section among {secs}")

        # --- error paths
        res = await s.call_tool("call_tool", {"toolset_name": toolsets[0][0], "tool_name": "NoSuchTool", "arguments": {}})
        record("call_tool unknown tool -> error", res.isError, _text(res)[:120])

    print("\n======== results")
    for n, ok, ev in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: {ev}")
    print("\n======== inventory")
    for e in inventory:
        print(f"## {e['toolset']} - {e['description']}")
        for t in e["tools"]:
            print(f"  - {t['name']}({', '.join(t['args'])}): {t['purpose']}")
    fails = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - fails} PASS, {fails} FAIL")
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/mcp")
    sys.exit(asyncio.run(main(ap.parse_args().url)))
