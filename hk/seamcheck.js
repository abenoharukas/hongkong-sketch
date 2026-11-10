// 瓦片衔接体检。打开 hk/map.html，把这段贴进控制台，它会逐个缩放比例去量
// 瓦片边界那一列/行的「列间差」，和左右 12 像素的邻域中位数比：
// 比值 ≈1 说明边界长得跟别处一样；比值几十上百说明那里有一条 1 像素的接缝线。
//
// 之前的做法（瓦片四周不留余量）在 0.3 / 0.7 / 1.0 这些比例上比值到过
// 5254 / 6377 / 8806 —— 就是一条孤立的线。给瓦片留 1 像素重叠之后 ≤1.21。
(async () => {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const dpr = window.devicePixelRatio || 1;
  const c = document.getElementById("map"), g = c.getContext("2d");
  const rows = [];
  for (const s of [0.3, 0.37, 0.5, 0.7, 0.9, 1.0, 1.7, 3.0]) {
    scale = s; camX = meta.width / 2; camY = meta.height / 2; apply();
    await wait(500); apply(); await wait(500); apply();
    const W = c.width, H = c.height, d = g.getImageData(0, 0, W, H).data;
    const col = new Float64Array(W), row = new Float64Array(H);
    for (let x = 0; x < W - 1; x++) {
      let a = 0;
      for (let y = 0; y < H; y += 4) { const i = (y * W + x) * 4; a += Math.abs(d[i] - d[i + 4]) + Math.abs(d[i + 1] - d[i + 5]) + Math.abs(d[i + 2] - d[i + 6]); }
      col[x] = a / (H / 4);
    }
    for (let y = 0; y < H - 1; y++) {
      let a = 0;
      for (let x = 0; x < W; x += 4) { const i = (y * W + x) * 4; a += Math.abs(d[i] - d[i + W * 4]) + Math.abs(d[i + 1] - d[i + W * 4 + 1]) + Math.abs(d[i + 2] - d[i + W * 4 + 2]); }
      row[y] = a / (W / 4);
    }
    const level = levelFor(s), span = Math.pow(2, MAX_LEVEL - level) * TILE, k = span * s * dpr;
    const left = (c.clientWidth / 2 - camX * s) * dpr, top = (c.clientHeight / 2 - camY * s) * dpr;
    const ratio = (arr, at) => {
      const nb = [];
      for (let o = -12; o <= 12; o++) { if (Math.abs(o) < 2) continue; const j = at + o; if (j > 1 && j < arr.length - 1) nb.push(arr[j]); }
      nb.sort((a, b) => a - b);
      return arr[at] / Math.max(nb[nb.length >> 1], 0.5);
    };
    let worst = 0;
    for (let n = Math.ceil(-left / k); n <= Math.floor((W - left) / k); n++) {
      const x = Math.round(left + n * k);
      if (x > 14 && x < W - 14) worst = Math.max(worst, ratio(col, x));
    }
    for (let n = Math.ceil(-top / k); n <= Math.floor((H - top) / k); n++) {
      const y = Math.round(top + n * k);
      if (y > 14 && y < H - 14) worst = Math.max(worst, ratio(row, y));
    }
    rows.push(`scale ${s}  level ${level}  worst boundary ratio ${worst.toFixed(2)}`);
  }
  console.log(rows.join("\n"));
  return rows;
})();
