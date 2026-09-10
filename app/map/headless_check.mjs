// Headless offline-render check for verify_offline.html (Node >= 22, no dependencies).
// Launches Microsoft Edge (or Chrome) headless with every non-loopback hostname unresolvable, drives it over the
// DevTools protocol, records every network request (page + workers), waits for the page's own verdict
// (document.title MAP_OK / MAP_FAIL), and saves a screenshot.
//
//   node app/map/serve.mjs &            (serve on 127.0.0.1:8765 first)
//   node app/map/headless_check.mjs [--out _artifacts/verification/map_offline.png] [--url http://127.0.0.1:8765/app/map/verify_offline.html]
// Exit code 0 = rendered, basemap + markers present, zero external requests.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const URL_ = opt("--url", "http://127.0.0.1:8765/app/map/verify_offline.html");
const OUT = resolve(REPO, opt("--out", "_artifacts/verification/map_offline.png"));
const PORT = Number(opt("--cdp-port", "9333"));
const PROFILE = resolve(REPO, "_scratch", "edge-headless-cdp");
const BROWSERS = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];
const exe = opt("--browser", BROWSERS.find((p) => existsSync(p)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

mkdirSync(PROFILE, { recursive: true });
const browser = spawn(exe, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "--no-default-browser-check",
  "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync",
  "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1", "--window-size=1280,900", "about:blank",
], { stdio: "ignore" });

let ws;
const result = { browser: exe, url: URL_, requests: [], external: [], title: null, status: null, console: [] };
try {
  let target;
  for (let i = 0; i < 60 && !target; i++) {
    await sleep(500);
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      target = list.find((t) => t.type === "page");
    } catch { /* not up yet */ }
  }
  if (!target) throw new Error("browser DevTools endpoint did not come up");

  ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let seq = 0;
  const pending = new Map();
  const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
    const id = ++seq;
    pending.set(id, { res, rej });
    ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
  });
  const noteRequest = (url) => {
    result.requests.push(url);
    if (!/^(https?:\/\/127\.0\.0\.1[:/]|data:|blob:|about:)/.test(url)) result.external.push(url);
  };
  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) {
      const p = pending.get(m.id); pending.delete(m.id);
      return m.error ? p.rej(new Error(m.error.message)) : p.res(m.result);
    }
    if (m.method === "Network.requestWillBeSent") noteRequest(m.params.request.url);
    if (m.method === "Runtime.consoleAPICalled") result.console.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
    if (m.method === "Target.attachedToTarget") { // dedicated workers (MapLibre's tile worker)
      const sid = m.params.sessionId;
      await send("Network.enable", {}, sid).catch(() => {});
      await send("Runtime.runIfWaitingForDebugger", {}, sid).catch(() => {});
    }
  };
  await send("Network.enable");
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Target.setAutoAttach", { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
  await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: URL_ });

  const deadline = Date.now() + 90_000;
  while (Date.now() < deadline) {
    await sleep(500);
    const r = await send("Runtime.evaluate", { expression: "document.title", returnByValue: true });
    result.title = r.result.value;
    if (result.title === "MAP_OK" || result.title === "MAP_FAIL") break;
  }
  await sleep(1500); // let the last frame (labels) settle
  const st = await send("Runtime.evaluate", { expression: "document.getElementById('status')?.textContent", returnByValue: true });
  result.status = st.result.value;
  const shot = await send("Page.captureScreenshot", { format: "png" });
  mkdirSync(dirname(OUT), { recursive: true });
  writeFileSync(OUT, Buffer.from(shot.data, "base64"));
  result.screenshot = OUT;
} catch (e) {
  result.error = String(e);
} finally {
  try { ws?.close(); } catch {}
  browser.kill();
  await sleep(1500);
  try { rmSync(PROFILE, { recursive: true, force: true }); } catch {}
}
result.ok = result.title === "MAP_OK" && result.external.length === 0 && !result.error;
result.request_count = result.requests.length;
console.log(JSON.stringify({ ...result, requests: undefined, request_hosts: [...new Set(result.requests.map((u) => u.split("/").slice(0, 3).join("/")))] }, null, 2));
process.exit(result.ok ? 0 : 1);
