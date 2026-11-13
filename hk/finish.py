"""Code finishing on the painted raster, after the model, before the tiles.

The model's ink is black and too thick at this map's density: every tree is a
dark blob and every wall a black grid, which reads as dirt. Real pixel-art maps
draw edges as a *deeper shade of the local colour*, not as black.

`local_ink` is ported from Statecraft-Labs/isometric-cities (MIT),
`source/material_finish.py:local_ink()` - kept byte-for-byte in behaviour so its
measured parameters apply. It only rewrites *thin* dark strokes (distance
transform <= 1.45 px, luma < luminance_max) toward the surrounding material's
dark tone, so genuine shadow masses are left alone.

    python hk/finish.py hk/art.png                 # in place, unless --out
    python hk/finish.py hk/art.png --out hk/art-fin.png --strength 0.65
"""

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def local_ink(pixels, luminance_max=70, strength=0.65):
    """Lighten only thin dark strokes toward the neighbouring material's dark tone."""
    f = pixels.astype(np.float32)
    lum = f @ LUMA
    bright = lum > 82
    weight = ndi.gaussian_filter(bright.astype(np.float32), 2.0)
    local = np.stack(
        [ndi.gaussian_filter(f[:, :, c] * bright, 2.0) / np.maximum(weight, 0.001) for c in range(3)],
        -1,
    )
    thin = ndi.distance_transform_edt(lum < luminance_max) <= 1.45
    mask = (lum < luminance_max) & (weight > 0.13) & thin & ((local @ LUMA) > lum * 1.7)
    target = np.maximum(f, local * 0.48)
    out = f.copy()
    out[mask] = (1 - strength) * f[mask] + strength * target[mask]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default=None)
    ap.add_argument("--luminance-max", type=float, default=70)
    ap.add_argument("--strength", type=float, default=0.65)
    a = ap.parse_args()
    with Image.open(a.image) as im:
        px = np.asarray(im.convert("RGB"))
    out, mask = local_ink(px, a.luminance_max, a.strength)
    dst = a.out or a.image
    Image.fromarray(out).save(dst)
    print(f"{dst}: ink {mask.mean() * 100:.2f}% of pixels, strength {a.strength}")


if __name__ == "__main__":
    main()
