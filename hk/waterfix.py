"""Keep the painter off open water.

The pixel LoRA invents land where the source is empty sea: give it a stretch of
harbour and it returns a housing estate with lawns. Both reference pipelines hit
this and both fix it in code, not in the model.

The rule here: where the *source* is water and the painter drew something that is
not water, replace it with the painter's own water colour. Boats, piers and
shorelines survive because the source is not flat water there.

    python hk/waterfix.py hk/art.png --source /tmp/hk05/plate.png --out hk/art-fixed.png
"""

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def flat_water(pixels, tol=42, min_area=150000, max_std=9.0):
    """The harbour colour and a mask of the flat water it covers.

    Blue glass facades are the same hue as harbour water, so colour alone paints
    over towers. Water is also *smooth*: keep only large components whose local
    standard deviation is tiny, which no facade or roof survives."""
    f = pixels.astype(np.float32)
    h, w = f.shape[:2]
    # the harbour is the largest low-variance patch in the bottom-left band of
    # the colour cube; take its median instead of hard-coding a value per plate
    coarse = f[::8, ::8].reshape(-1, 3)
    sat = (coarse.max(1) - coarse.min(1)) / np.maximum(coarse.max(1), 1)
    blue = coarse[:, 2] > coarse[:, 0] + 10
    cand = coarse[blue & (sat > 0.2) & (sat < 0.85)]
    if len(cand) < 64:
        return None, np.zeros((h, w), bool), None
    colour = np.median(cand, 0)
    for _ in range(6):  # refine around the mode, boats and shallows fall out
        d = np.linalg.norm(f - colour, axis=-1)
        keep = d < tol
        if keep.sum() < 1000:
            break
        colour = np.median(f[keep], 0)
    mask = np.linalg.norm(f - colour, axis=-1) < tol
    grey = f.mean(-1)
    mean = ndi.uniform_filter(grey, 7)
    var = np.maximum(ndi.uniform_filter(grey * grey, 7) - mean * mean, 0)
    mask &= np.sqrt(var) < max_std
    mask = ndi.binary_opening(mask, np.ones((5, 5), bool))
    mask = ndi.binary_closing(mask, np.ones((5, 5), bool))
    lab, n = ndi.label(mask)
    if n:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_area])
        mask = keep
    return colour, mask, tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tol", type=float, default=42)
    ap.add_argument("--paint-tol", type=float, default=50)
    ap.add_argument("--min-area", type=int, default=150000)
    ap.add_argument("--max-std", type=float, default=9.0)
    a = ap.parse_args()
    with Image.open(a.image) as im:
        art = np.asarray(im.convert("RGB"))
    with Image.open(a.source) as im:
        src = np.asarray(im.convert("RGB"))
    if art.shape != src.shape:
        raise SystemExit("art and source must be the same size")

    _, water, _ = flat_water(src, a.tol, a.min_area, a.max_std)
    if not water.any():
        raise SystemExit("no flat water found in the source")
    painted_water = art[water].astype(np.float32)
    colour = np.median(painted_water, 0)
    is_water_art = np.linalg.norm(art.astype(np.float32) - colour, axis=-1) < a.paint_tol
    bad = water & ~is_water_art
    out = art.copy()
    out[bad] = colour.astype(np.uint8)
    dst = a.out or a.image
    Image.fromarray(out).save(dst)
    print(f"{dst}: water {water.mean() * 100:.1f}% of frame, replaced {bad.sum()} px ({bad.mean() * 100:.2f}%), colour {colour.round().astype(int)}")


if __name__ == "__main__":
    main()
