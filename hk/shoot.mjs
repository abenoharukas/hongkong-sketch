#!/usr/bin/env node
// Drive headless Chrome over CDP and pull one orthographic window at a time
// out of hk/ortho.html. Chrome writes true-size PNGs of the WebGL canvas.
//
//   node hk/shoot.mjs --out shots/plate --cols 6 --rows 3 --w 1024 --h 1024 \
//        --mpp 1.2 --lon 114.162 --lat 22.279 --az -15 --tilt 45 --overlap 32
//
// Every window is the same orthographic camera slid right/up by exactly one
// window, so the pieces butt together with no parallax to reconcile.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { serve } from "./serve.mjs";

const argv = process.argv.slice(2);
const arg = (name, dflt = null) => {
  const i = argv.indexOf("--" + name);
  return i >= 0 ? argv[i + 1] : dflt;
};

const OUT = arg("out", "shots/plate");
const COLS = parseInt(arg("cols", "6"), 10);
const ROWS = parseInt(arg("rows", "3"), 10);
const W = parseInt(arg("w", "1024"), 10);
const H = parseInt(arg("h", "1024"), 10);
const PORT = parseInt(arg("port", "9333"), 10);
const KEY = arg("key") || process.env.GOOGLE_MAPS_API_KEY;
if (!KEY) {
  console.error(
    "Missing Google Maps API key.\n" +
      "  export GOOGLE_MAPS_API_KEY=...   # or pass --key <KEY>\n" +
      "Never hardcode the key in source.",
  );
  process.exit(1);
}
const CHROME =
  arg("chrome") || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const params = new URLSearchParams({
  key: KEY,
  lon: arg("lon", "114.162"),
  lat: arg("lat", "22.279"),
  mpp: arg("mpp", "1.2"),
  az: arg("az", "-15"),
  tilt: arg("tilt", "45"),
  w: String(W),
  h: String(H),
  cols: String(COLS),
  rows: String(ROWS),
  ox: arg("ox", "0"),
  oy: arg("oy", "0"),
  overlap: arg("overlap", "0"),
  sse: arg("sse", "4"),
  dist: arg("dist", "20000"),
});

const HTTP = parseInt(arg("http", "8123"), 10);
const server = await serve(process.cwd(), HTTP);
const url = `http://127.0.0.1:${HTTP}/hk/ortho.html?` + params.toString();

mkdirSync(OUT, { recursive: true });

const profile = `/tmp/pixel-hk-chrome-${PORT}`;
const chrome = spawn(
  CHROME,
  [
    "--headless=new",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    `--window-size=${W},${H}`,
    "--hide-scrollbars",
    "--force-device-scale-factor=1",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--enable-unsafe-swiftshader",
    "--proxy-server=" + (process.env.HTTPS_PROXY || "http://127.0.0.1:7897"),
    "--proxy-bypass-list=<-loopback>;127.0.0.1;localhost",
    "--use-angle=metal",
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "pipe"] }
);
chrome.stderr.on("data", (d) => {
  const s = d.toString();
  if (/error|fail/i.test(s)) process.stderr.write("[chrome] " + s);
});

async function browserWs() {
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      const j = await r.json();
      if (j.webSocketDebuggerUrl) return j.webSocketDebuggerUrl;
    } catch {}
    await sleep(200);
  }
  throw new Error("chrome did not come up");
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.waiting = new Map();
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.waiting.has(msg.id)) {
        const { resolve, reject } = this.waiting.get(msg.id);
        this.waiting.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
  }
  send(method, params = {}, timeout = 0) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      if (timeout) {
        setTimeout(() => {
          if (this.waiting.delete(id)) reject(new Error(method + " timed out"));
        }, timeout);
      }
    });
  }
}

const open = (u) =>
  new Promise((resolve, reject) => {
    const ws = new WebSocket(u);
    ws.addEventListener("open", () => resolve(ws));
    ws.addEventListener("error", (e) => reject(new Error("ws error " + u)));
  });

const bws = await browserWs();
const browser = new CDP(await open(bws));
const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
const { sessionId } = await browser.send("Target.attachToTarget", {
  targetId,
  flatten: true,
});
const page = {
  send(method, params = {}, timeout = 0) {
    const id = ++browser.id;
    return new Promise((resolve, reject) => {
      browser.waiting.set(id, { resolve, reject });
      browser.ws.send(
        JSON.stringify({ id, method, params, sessionId })
      );
      if (timeout) {
        setTimeout(() => {
          if (browser.waiting.delete(id)) reject(new Error(method + " timed out"));
        }, timeout);
      }
    });
  },
};

await page.send("Page.enable");
await page.send("Runtime.enable");
await page.send("Emulation.setDeviceMetricsOverride", {
  width: W,
  height: H,
  deviceScaleFactor: 1,
  mobile: false,
});
await page.send("Page.navigate", { url });
await sleep(1500);

// Wait for the tileset to come up.
for (let i = 0; i < 600; i++) {
  const r = await page.send("Runtime.evaluate", {
    expression: "!!window.__ready",
    returnByValue: true,
  });
  if (r.result?.value) break;
  if (i === 5) {
    const err = await page.send("Runtime.evaluate", {
      expression: "document.body.innerHTML.slice(0,400)",
      returnByValue: true,
    });
    if (err.result?.value?.includes("Cesium")) {
      console.error("page:", err.result.value);
    }
  }
  await sleep(500);
}

const t0 = Date.now();
for (let j = 0; j < ROWS; j++) {
  for (let i = 0; i < COLS; i++) {
    const r = await page.send(
      "Runtime.evaluate",
      {
        expression: `window.__plate.shoot(${i},${j})`,
        awaitPromise: true,
        returnByValue: true,
      },
      900000
    );
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    const shot = await page.send("Page.captureScreenshot", {
      format: "png",
      fromSurface: true,
      captureBeyondViewport: false,
    });
    const file = `${OUT}/${i}_${j}.png`;
    writeFileSync(file, Buffer.from(shot.data, "base64"));
    const secs = ((Date.now() - t0) / 1000).toFixed(0);
    console.log(`${file}  window ${i},${j}  ${secs}s`);
  }
}

chrome.kill();
server.close();
process.exit(0);
