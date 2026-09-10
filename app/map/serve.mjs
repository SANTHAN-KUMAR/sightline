// Minimal offline static server for the Sightline map (no dependencies, Node >= 18).
// Serves the repo's app/ and data/basemap/ trees with HTTP Range support (PMTiles needs byte ranges;
// Python's http.server has none). Logs every request, so offline checks can assert "no external hosts".
//
//   node app/map/serve.mjs [--port 8765] [--log _logs/map_serve.log]
//   open http://127.0.0.1:8765/app/map/verify_offline.html
//
// The page may POST a JSON report to /__report; it is appended to the log (used by the headless check).
import { createServer } from "node:http";
import { createReadStream, promises as fs, appendFileSync, mkdirSync } from "node:fs";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const PORT = Number(opt("--port", "8765"));
const HOST = "127.0.0.1"; // loopback only
const LOG = opt("--log", "");
const ALLOWED = ["app", join("data", "basemap")].map((p) => join(REPO, p) + sep);

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".geojson": "application/geo+json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".pbf": "application/x-protobuf",
  ".pmtiles": "application/octet-stream",
  ".map": "application/json",
  ".txt": "text/plain; charset=utf-8",
};

if (LOG) mkdirSync(dirname(resolve(LOG)), { recursive: true });
function log(line) {
  const s = `${new Date().toISOString()} ${line}`;
  console.log(s);
  if (LOG) appendFileSync(LOG, s + "\n");
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}:${PORT}`);
  if (req.method === "POST" && url.pathname === "/__report") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      log(`REPORT ${body}`);
      res.writeHead(204).end();
    });
    return;
  }
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405).end();
    return log(`405 ${req.method} ${url.pathname}`);
  }
  const path = normalize(join(REPO, decodeURIComponent(url.pathname)));
  if (!ALLOWED.some((a) => path.startsWith(a))) {
    res.writeHead(403).end();
    return log(`403 ${req.method} ${url.pathname}`);
  }
  let st;
  try {
    st = await fs.stat(path);
    if (!st.isFile()) throw new Error("not a file");
  } catch {
    res.writeHead(404).end();
    return log(`404 ${req.method} ${url.pathname}`);
  }
  const headers = {
    "Content-Type": TYPES[extname(path).toLowerCase()] || "application/octet-stream",
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-cache",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Expose-Headers": "Content-Range, Content-Length, ETag",
    ETag: `"${st.size}-${Math.floor(st.mtimeMs)}"`,
  };
  const range = req.headers.range && /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
  if (range) {
    // "bytes=a-b", "bytes=a-" (to end) or "bytes=-n" (last n bytes)
    const start = range[1] === "" ? Math.max(0, st.size - Number(range[2])) : Number(range[1]);
    const end = Math.min(range[1] !== "" && range[2] !== "" ? Number(range[2]) : st.size - 1, st.size - 1);
    if (Number.isNaN(start) || start < 0 || start > end) {
      res.writeHead(416, { "Content-Range": `bytes */${st.size}` }).end();
      return log(`416 ${url.pathname} ${req.headers.range}`);
    }
    res.writeHead(206, { ...headers, "Content-Range": `bytes ${start}-${end}/${st.size}`, "Content-Length": end - start + 1 });
    log(`206 ${req.method} ${url.pathname} ${start}-${end}`);
    if (req.method === "HEAD") return res.end();
    return createReadStream(path, { start, end }).pipe(res);
  }
  res.writeHead(200, { ...headers, "Content-Length": st.size });
  log(`200 ${req.method} ${url.pathname}`);
  if (req.method === "HEAD") return res.end();
  createReadStream(path).pipe(res);
});

server.listen(PORT, HOST, () => log(`serving ${REPO} (app/, data/basemap/) at http://${HOST}:${PORT}/`));
