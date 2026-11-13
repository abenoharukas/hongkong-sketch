"""Remove water the painter invented on land.

The pixel LoRA was captioned with turquoise water and obliged on hillsides: the
southern band of this plate came back with four reservoirs the source never had,
cut off by straight lines along the sampling windows. Captions are fixed for the
next run; this repairs the map already painted.

Each invented lake is refilled from a shifted copy of the painted surroundings,
so the forest keeps its own texture instead of a smeared nearest-neighbour fill.

    python hk/landfix.py hk/art.png --source /tmp/hk05/plate.png --out hk/art-fixed.png
"""

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def water_of(pixels, source, tol=60, min_area=8000):
    """Painted water where the source is dry vegetation - the model's favourite lie."""
    src = source.astype(np.int16)
    art = pixels.astype(np.int16)
    sr, sg, sb = src[..., 0], src[..., 1], src[..., 2]
    ar, ag, ab = art[..., 0], art[..., 1], art[..., 2]
    # blue-or-teal on the map, green forest or lawn below it
    blueish = ((ab - ar) > 40) & ((ag - ar) > 20)
    green = (sg > sr + 6) & (sg >= sb - 2)
    # never touch the harbour: it is flat water in the source
    f = source.astype(np.float32)
    flat = np.abs(f - f.mean((0, 1))).mean(-1)
    idx = np.argsort(flat.reshape(-1))[: max(1000, flat.size // 20)]
    harbour = np.median(f.reshape(-1, 3)[idx], 0)
    land = green & ~(np.linalg.norm(f - harbour, axis=-1) < 42)
    lake = blueish & land
    # the painted lake has gradients, piers and boats breaking it into specks:
    # close it hard, fill the holes, then keep only blobs worth repairing
    lake = ndi.binary_closing(lake, np.ones((17, 17), bool))
    lake = ndi.binary_fill_holes(lake)
    lab, n = ndi.label(lake)
    keep = np.zeros_like(lake)
    if n:
        sizes = ndi.sum(lake, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_area])
    return ndi.binary_dilation(keep, np.ones((9, 9), bool)) & (blueish | green) & land, None


def donor_patch(pixels, mask, cy, cx, size=256):
    """A clean painted square near the blob to borrow canopy texture from."""
    h, w = mask.shape
    for dist in (0, 160, 260, 380, 520, 700, 900):
        for dy, dx in ((0, -dist), (0, dist), (-dist, 0), (dist, 0)):
            y0, x0 = cy + dy - size // 2, cx + dx - size // 2
            if y0 < 0 or x0 < 0 or y0 + size > h or x0 + size > w:
                continue
            if mask[y0:y0 + size, x0:x0 + size].mean() < 0.01:
                return pixels[y0:y0 + size, x0:x0 + size]
    return None


def fill_blob(pixels, blob, mask, src, size=256):
    """Tile a clean canopy patch over the blob and follow the source's own shading."""
    ys, xs = np.nonzero(blob)
    cy, cx = int(ys.mean()), int(xs.mean())
    patch = donor_patch(pixels, mask, cy, cx, size)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if patch is None:
        return None
    tiles = []
    rng = np.random.default_rng(len(ys))
    for ty in range(y0, y1, size):
        for tx in range(x0, x1, size):
            p = patch
            if rng.random() < 0.5:
                p = p[:, ::-1]
            if rng.random() < 0.5:
                p = p[::-1]
            tiles.append((ty, tx, p))
    canvas = pixels.copy()
    for ty, tx, p in tiles:
        ey, ex = min(ty + size, canvas.shape[0]), min(tx + size, canvas.shape[1])
        canvas[ty:ey, tx:ex] = p[: ey - ty, : ex - tx]
    # modulate with the source's low-frequency shading so hills stay hills
    lum = src[y0:y1, x0:x1].mean(-1)
    lo, hi = np.percentile(lum, 5), np.percentile(lum, 95)
    t = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)[..., None]
    tile_part = canvas[y0:y1, x0:x1].astype(np.float32)
    tone = 0.72 + 0.56 * t
    return y0, y1, x0, x1, tile_part * tone


def refill(pixels, mask, src=None):
    """Copy painted surroundings over each blob, choosing a shift that stays off the blob."""
    out = pixels.copy()
    lab, n = ndi.label(mask)
    h, w = mask.shape[:2]
    for i in range(1, n + 1):
        blob = lab == i
        ys, xs = np.nonzero(blob)
        if not len(ys):
            continue
        cy, cx = int(np.median(ys)), int(np.median(xs))
        if (blob.sum() > 20000) and src is not None:
            got = fill_blob(pixels, blob, mask, src)
            if got is not None:
                y0, y1, x0, x1, block = got
                sub = out[y0:y1, x0:x1]
                m = blob[y0:y1, x0:x1][..., None]
                out[y0:y1, x0:x1] = np.where(m, np.clip(block, 0, 255), sub)
                continue
        best = None
        for dist in (96, 160, 240, 340, 460, 600, 760):
            for dy, dx in ((0, -dist), (0, dist), (-dist, 0), (dist, 0),
                           (-dist, -dist), (-dist, dist), (dist, -dist), (dist, dist)):
                ny, nx = ys + dy, xs + dx
                if ny.min() < 0 or nx.min() < 0 or ny.max() >= h or nx.max() >= w:
                    continue
                shifted = blob.copy()
                src_ys, src_xs = ys + dy, xs + dx
                if mask[src_ys, src_xs].any():
                    continue
                best = (dy, dx)
                break
            if best:
                break
        if not best:
            continue
        dy, dx = best
        out[ys, xs] = pixels[ys + dy, xs + dx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=60)
    ap.add_argument("--min-area", type=int, default=8000)
    a = ap.parse_args()
    with Image.open(a.image) as im:
        art = np.asarray(im.convert("RGB"))
    with Image.open(a.source) as im:
        src = np.asarray(im.convert("RGB"))
    if art.shape != src.shape:
        raise SystemExit("art and source must be the same size")
    mask, colour = water_of(art, src, a.tol, a.min_area)
    if not mask.any():
        raise SystemExit("no invented water found")
    out = refill(art, mask, src)
    dst = a.out or a.image
    Image.fromarray(out).save(dst)
    print(f"{dst}: invented water {mask.mean() * 100:.2f}% of frame refilled from nearby paint")


if __name__ == "__main__":
    main()
