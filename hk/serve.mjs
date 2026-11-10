// Tiny static server so the plate page is same-origin (Google's tile endpoint
// only answers with CORS headers for a real origin, not file://).
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname } from "node:path";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".json": "application/json",
  ".css": "text/css",
};

export function serve(root, port) {
  const server = http.createServer(async (req, res) => {
    let path = decodeURIComponent(req.url.split("?")[0]);
    if (path === "/") path = "/hk/ortho.html";
    try {
      const body = await readFile(root + path);
      res.writeHead(200, {
        "Content-Type": TYPES[extname(path)] || "application/octet-stream",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
      });
      res.end(body);
    } catch {
      res.writeHead(404);
      res.end("not found");
    }
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}

// node hk/serve.mjs [port] — serve the repo so the pages can be opened by hand.
if (process.argv[1] && process.argv[1].endsWith("serve.mjs")) {
  const port = parseInt(process.argv[2] || "8130", 10);
  await serve(process.cwd(), port);
  console.log(`http://127.0.0.1:${port}/hk/map.html`);
}
