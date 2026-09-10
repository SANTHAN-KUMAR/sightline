// Headless offline-render check for the Sightline C2 map (Node >= 22, no dependencies).
//
// It launches Microsoft Edge (or Chrome) headless with EVERY non-loopback hostname made unresolvable, drives it
// over the DevTools protocol, records every network request made by the page AND its workers, waits for the
// page's own verdict (document.title -> MAP_OK / MAP_FAIL), pulls `window.__sightlineReport` out of the page,
// and saves screenshots. It exits non-zero when anything below is false, so it is a check that can FAIL:
//
//   * the page reached MAP_OK (basemap + markers + rings + track + plan + drone + footprint + hatched
//     cannot-clear polygons + coverage raster + score components + evidence thumbnails all present);
//   * ZERO external network requests (the whole point: this runs on a field laptop with no internet);
//   * no console errors, no MapLibre errors;
//   * with --interact: the evidence popup opens and contains the score components AND the thumbnail, and the
//     dismiss button REFUSES an empty reason (guardrail R10) before accepting one with a reason.
//
//   node app/map/serve.mjs &                                  # static-only, or:
//   D:\Tools\uv\uv.exe run python -m sightline.api.serve --port 8781 --demo --fresh
//   node app/map/headless_check.mjs --url http://127.0.0.1:8781/app/map/index.html?selfcheck=1 --interact
//
// Flags: --url, --out <png>, --shots <dir>, --cdp-port, --browser, --interact, --timeout-ms, --json <file>
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = process.argv.slice(2);
const opt = (n, d) => (args.includes(n) ? args[args.indexOf(n) + 1] : d);
const flag = (n) => args.includes(n);
const URL_ = opt("--url", "http://127.0.0.1:8765/app/map/verify_offline.html");
const OUT = resolve(REPO, opt("--out", "_artifacts/verification/map_offline.png"));
const SHOTS = resolve(REPO, opt("--shots", dirname(OUT)));
const PORT = Number(opt("--cdp-port", "9333"));
const TIMEOUT_MS = Number(opt("--timeout-ms", "90000"));
const JSON_OUT = opt("--json", "");
const INTERACT = flag("--interact");
const PROFILE = resolve(REPO, "_scratch", `edge-headless-cdp-${PORT}`);
const BROWSERS = [
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
];
const exe = opt("--browser", BROWSERS.find((p) => existsSync(p)));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

mkdirSync(PROFILE, { recursive: true });
mkdirSync(SHOTS, { recursive: true });
const browser = spawn(exe, [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${PROFILE}`,
  "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--no-first-run", "--no-default-browser-check",
  "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-sync",
  "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE 127.0.0.1", "--window-size=1600,1000", "about:blank",
], { stdio: "ignore" });

let ws;
const result = {
  browser: exe, url: URL_, requests: [], external: [], failed_requests: [], title: null, status: null,
  console: [], console_errors: [], screenshots: [], interactions: [], checks: {},
};
const fail = (name, detail) => { result.checks[name] = { ok: false, detail }; };
const pass = (name, detail) => { result.checks[name] = { ok: true, detail }; };

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
  const evaluate = async (expression) => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || "evaluate threw");
    return r.result.value;
  };
  const shot = async (name) => {
    const s = await send("Page.captureScreenshot", { format: "png" });
    const p = name === "main" ? OUT : join(SHOTS, `${name}.png`);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, Buffer.from(s.data, "base64"));
    result.screenshots.push(p);
    return p;
  };
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
    if (m.method === "Network.loadingFailed") result.failed_requests.push(m.params.errorText);
    if (m.method === "Runtime.consoleAPICalled") {
      const line = m.params.args.map((a) => a.value ?? a.description).join(" ");
      result.console.push(line);
      if (m.params.type === "error") result.console_errors.push(line);
    }
    if (m.method === "Runtime.exceptionThrown") {
      result.console_errors.push(m.params.exceptionDetails.exception?.description || "uncaught exception");
    }
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
  await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: URL_ });

  const deadline = Date.now() + TIMEOUT_MS;
  while (Date.now() < deadline) {
    await sleep(500);
    result.title = await evaluate("document.title");
    if (result.title === "MAP_OK" || result.title === "MAP_FAIL") break;
  }
  await sleep(1500); // let the last frame (labels) settle
  result.status = await evaluate("document.getElementById('status')?.textContent");
  result.report = await evaluate("window.__sightlineReport ?? null");
  await shot("main");

  if (INTERACT) {
    // 1. open the top-ranked record's evidence popup and prove what it contains.
    const opened = await evaluate(`(() => {
      const el = document.querySelector('#pane-recs .rec');
      if (!el) return { ok: false, why: 'no record rows in the side panel' };
      el.click();
      return { ok: true, id: el.dataset.id };
    })()`);
    result.interactions.push({ step: "click_top_record", ...opened });
    await sleep(1200);
    const popup = await evaluate(`(() => {
      const p = document.querySelector('.maplibregl-popup-content .pop');
      if (!p) return { ok: false, why: 'no evidence popup opened' };
      const img = p.querySelector('img');
      const rows = [...p.querySelectorAll('table.sc th')].map((th) => th.textContent.trim());
      const btn = p.querySelector('button[data-act="dismiss"]');
      const r = btn?.getBoundingClientRect();
      return { ok: true, has_thumbnail: !!img, thumb_src: img?.getAttribute('src') || null,
               thumb_complete: !!img && img.complete && img.naturalWidth > 0,
               thumb_px: img ? [img.naturalWidth, img.naturalHeight] : null,
               score_rows: rows, has_formula: /score =/.test(p.textContent),
               has_dismiss_input: !!p.querySelector('#dismiss-reason'),
               // an operator has to be able to SEE the dismiss control, not just have it in the DOM
               dismiss_rect: r ? [Math.round(r.left), Math.round(r.top), Math.round(r.right), Math.round(r.bottom)] : null,
               dismiss_on_screen: !!r && r.top >= 0 && r.bottom <= window.innerHeight &&
                                  r.left >= 0 && r.right <= window.innerWidth };
    })()`);
    result.interactions.push({ step: "read_popup", ...popup });
    await shot("evidence_popup");

    // 2. R10: the dismiss button must refuse an empty reason, then accept one WITH a reason and KEEP the record.
    const empty = await evaluate(`(async () => {
      const p = document.querySelector('.maplibregl-popup-content .pop');
      if (!p) return { ok: false, why: 'popup gone' };
      p.scrollTop = p.scrollHeight;   // the R10 controls live at the foot of the popup
      p.querySelector('#dismiss-reason').value = '   ';
      p.querySelector('button[data-act="dismiss"]').click();
      await new Promise((r) => setTimeout(r, 400));
      const el = document.querySelector('.maplibregl-popup-content .pop #act-err');
      return { ok: true, error_text: el ? el.textContent : null };
    })()`);
    result.interactions.push({ step: "dismiss_without_reason", ...empty });
    await shot("dismiss_refused");

    const withReason = await evaluate(`(async () => {
      const p = document.querySelector('.maplibregl-popup-content .pop');
      const id = document.querySelector('#pane-recs .rec.sel')?.dataset.id
              || document.querySelector('#pane-recs .rec')?.dataset.id;
      p.scrollTop = p.scrollHeight;
      p.querySelector('#dismiss-reason').value = 'headless check: verified duplicate of another record';
      p.querySelector('button[data-act="dismiss"]').click();
      await new Promise((r) => setTimeout(r, 1200));
      const r = await fetch('/api/records/' + encodeURIComponent(id));
      const f = await r.json();
      const h = await (await fetch('/api/records/' + encodeURIComponent(id) + '/history')).json();
      return { ok: r.ok, id, status: f.properties?.status, reason: f.properties?.dismissed_reason,
               versions: h.versions?.length ?? 0,
               still_listed: !!document.querySelector('#pane-recs .rec[data-id="' + id + '"]') };
    })()`);
    result.interactions.push({ step: "dismiss_with_reason", ...withReason });
    await sleep(900);
    await shot("after_dismiss");
    result.after_dismiss_report = await evaluate(`(async () => {
      const r = await fetch('/api/records.geojson');
      const fc = await r.json();
      return { total: fc.features.length,
               dismissed: fc.features.filter((f) => f.properties.status === 'dismissed').length };
    })()`);

    // 3. LIVE FEED: change a record from OUTSIDE the browser and require the page to move on its own.
    //    Node's fetch is not the page's fetch, so nothing the page does can explain the update.
    const before = await evaluate(`(() => {
      const st = window.__sightlineState;
      if (!st) return null;
      const f = [...st.records.values()].find((x) => x.properties.status !== 'dismissed');
      return f ? { id: f.id, version: f.properties.version, status: f.properties.status,
                   seq: st.lastSeq, ws: st.wsState,
                   fetches: performance.getEntriesByType('resource')
                              .filter((e) => e.name.includes('/api/records')).length } : null;
    })()`);
    if (before) {
      const base = new URL(URL_).origin;
      const res = await fetch(`${base}/api/records/${encodeURIComponent(before.id)}/status`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ status: "stale", actor: "headless_check", reason: "live-feed probe" }),
      });
      await sleep(1500);
      const after = await evaluate(`(() => {
        const st = window.__sightlineState;
        const f = st.records.get(${JSON.stringify(before.id)});
        return { version: f?.properties.version, status: f?.properties.status, seq: st.lastSeq,
                 fetches: performance.getEntriesByType('resource')
                            .filter((e) => e.name.includes('/api/records')).length,
                 in_list: !!document.querySelector('#pane-recs .rec[data-id="' + ${JSON.stringify(before.id)} + '"]') };
      })()`);
      result.interactions.push({ step: "live_push", http_status: res.status, before, after });
      await shot("live_update");
    }
  }
} catch (e) {
  result.error = String(e);
} finally {
  try { ws?.close(); } catch {}
  browser.kill();
  await sleep(1200);
  try { rmSync(PROFILE, { recursive: true, force: true }); } catch {}
}

// ---- the verdict: each of these can fail ------------------------------------------------------------------
const rep = result.report;
result.title === "MAP_OK" ? pass("page_verdict", result.title)
  : fail("page_verdict", `title=${result.title} report.ok=${rep?.ok} errors=${JSON.stringify(rep?.errors ?? [])}`);
result.external.length === 0 ? pass("no_external_requests", `${result.requests.length} requests, all loopback`)
  : fail("no_external_requests", result.external.slice(0, 8).join(", "));
result.console_errors.length === 0 ? pass("no_console_errors", "clean")
  : fail("no_console_errors", result.console_errors.slice(0, 5).join(" | "));
// The C2 page (index.html) must show every §5.9 layer; verify_offline.html is the basemap-only smoke page.
const C2 = /index\.html/.test(URL_);
if (rep) {
  const need = C2
    ? { basemap_features: 1, marker_features: 1, ring_features: 1, track_features: 1,
        plan_features: 1, drone_features: 1, footprint_features: 1, cannot_clear_features: 1,
        thumbnails_present: 1, cannot_clear_polygons: 1 }
    : { basemap_features: 1, marker_features: 1 };
  const short = Object.entries(need).filter(([k, v]) => !(Number(rep[k]) >= v));
  short.length === 0 ? pass("layers_rendered", Object.keys(need).map((k) => `${k}=${rep[k]}`).join(" "))
    : fail("layers_rendered", short.map(([k]) => `${k}=${rep[k]}`).join(" "));
  if (C2) {
    rep.coverage_raster_loaded ? pass("coverage_raster", `${rep.coverage_source} layer=${rep.coverage_layer} podmean=${rep.coverage_pod_mean}`)
      : fail("coverage_raster", "the POD raster never loaded");
    rep.score_components_present ? pass("score_components", "every record carries p_living")
      : fail("score_components", "a record reached the map with no score components");
  }
} else fail("page_report", "window.__sightlineReport was never set (did you pass ?selfcheck=1)");
if (INTERACT) {
  const pop = result.interactions.find((i) => i.step === "read_popup");
  (pop?.has_thumbnail && pop?.thumb_complete && pop?.score_rows?.length >= 4 && pop?.has_formula)
    ? pass("evidence_popup", `thumb ${pop.thumb_px} + ${pop.score_rows.length} score rows + the formula`)
    : fail("evidence_popup", JSON.stringify(pop));
  pop?.dismiss_on_screen ? pass("dismiss_control_visible", `rect ${pop.dismiss_rect}`)
    : fail("dismiss_control_visible", `the dismiss control is off-screen: rect ${pop?.dismiss_rect}`);
  const empty = result.interactions.find((i) => i.step === "dismiss_without_reason");
  /R10/.test(empty?.error_text || "") ? pass("r10_reason_required", empty.error_text)
    : fail("r10_reason_required", `no R10 error shown: ${JSON.stringify(empty)}`);
  const dis = result.interactions.find((i) => i.step === "dismiss_with_reason");
  (dis?.status === "dismissed" && dis?.reason && dis?.versions >= 2 && dis?.still_listed)
    ? pass("r10_record_kept", `v${dis.versions} kept, still listed, reason "${dis.reason}"`)
    : fail("r10_record_kept", JSON.stringify(dis));
  const live = result.interactions.find((i) => i.step === "live_push");
  (live && live.http_status === 200 && live.after.status === "stale" &&
   live.after.version === live.before.version + 1 && live.after.seq > live.before.seq &&
   live.after.fetches === live.before.fetches)
    ? pass("live_websocket_push", `v${live.before.version} -> v${live.after.version} in ${live.after.seq - live.before.seq} frame(s), no re-fetch`)
    : fail("live_websocket_push", JSON.stringify(live));
}
if (result.error) fail("driver", result.error);

result.ok = Object.values(result.checks).every((c) => c.ok);
result.request_count = result.requests.length;
result.request_hosts = [...new Set(result.requests.map((u) => u.split("/").slice(0, 3).join("/")))];
const printable = { ...result, requests: undefined, console: undefined };
console.log(JSON.stringify(printable, null, 2));
if (JSON_OUT) writeFileSync(resolve(REPO, JSON_OUT), JSON.stringify(result, null, 2));
console.log(`\n${result.ok ? "PASS" : "FAIL"}  ` +
  Object.entries(result.checks).map(([k, v]) => `${v.ok ? "ok" : "FAIL"}:${k}`).join("  "));
process.exit(result.ok ? 0 : 1);
