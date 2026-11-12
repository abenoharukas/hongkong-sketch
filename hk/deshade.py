#!/usr/bin/env python3
"""Flatten the sun's cast shadows out of the plate before the repaint.

Google's photogrammetry carries the sunlight of the day it was flown: every
tower has a long, near-black shadow lying on the ground and on its neighbours'
roofs. The image model copies that faithfully (the finished map lines up with
the plate to within a pixel), and a cast shadow that dark is a photograph's
way of saying "sun", not a model's — which is why the redraw still reads as a
photo next to the reference maps, whose source mosaics were flown near noon.

A cast shadow is a *large, flat, dark blob*. A window row, an ink edge, a kerb
is small and thin; a sunlit roof is flat but not dark. So: find dark regions,
keep only the ones that survive an opening (thin structure disappears), and
lift those pixels until they sit at `--floor` of the local sunlit envelope.
Walls keep their own relative darkness where they are textured; the near-black
shadow lying on roads, plazas and roofs goes away.

    python hk/deshade.py --plate hk/plate.png --out hk/plate-flat.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import water_fill  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def deshade(image, floor=0.70, radius=26, blob=7, feather=3, water=None,
            max_sat=0.45, skip_veg=False):
    a = np.asarray(image.convert("RGB"), np.float32) / 255.0
    y = a @ LUMA

    # Local sunlit level: the brightest thing within ~60 m, smoothed.
    env = ndimage.gaussian_filter(ndimage.maximum_filter(y, 2 * radius + 1), radius / 2)
    ratio = y / np.maximum(env, 1e-3)

    # A cast shadow on a road, a plaza or a roof is a *grey* dark shape: the
    # surface it falls on has little saturation (asphalt, concrete, tile). The
    # dark side of a glass tower, a forest slope, a red brick wall are saturated
    # and belong to the building, not to the sun, so they are left alone.
    sat = (a.max(2) - a.min(2)) / np.maximum(a.max(2), 1e-3)
    # Vegetation is dark and matte too, but its darkness is canopy, not sun.
    # Off by default: on this plate the shadowed streets are greenish enough
    # that the guard swallows them (`--skip-veg` to turn it back on).
    veg = ((a[:, :, 1] > a[:, :, 0] * 1.15) & (a[:, :, 1] > a[:, :, 2] * 1.12)
           if skip_veg else np.zeros(y.shape, bool))

    dark = (ratio < floor) & (sat < max_sat) & ~veg
    mask = ndimage.binary_opening(dark, disk(blob))
    mask = ndimage.binary_closing(mask, disk(blob))
    if water is not None:
        mask &= ~water
    weight = ndimage.gaussian_filter(mask.astype(np.float32), feather)

    # Bring those pixels up to `floor` of the envelope, hue untouched.
    target = np.clip(ratio / floor, 0.0, 1.0)
    gain = 1.0 + (target / np.maximum(ratio, 1e-3) - 1.0) * weight
    out = np.clip(a * gain[..., None], 0.0, 1.0)
    return Image.fromarray((out * 255).round().astype(np.uint8)), mask.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default="hk/plate.png")
    ap.add_argument("--out", default="hk/plate-flat.png")
    ap.add_argument("--floor", type=float, default=None)
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--blob", type=int, default=None)
    ap.add_argument("--max-sat", type=float, default=0.45,
                    help="skip dark shapes this saturated (walls, forest, brick)")
    ap.add_argument("--feather", type=float, default=3.0)
    ap.add_argument("--passes", default=None,
                    help="comma-separated floor:radius:blob, applied in order"
                         " (default 0.70:26:7 then 0.60:20:5)")
    ap.add_argument("--skip-veg", action="store_true",
                    help="leave vegetation alone too (this plate then loses too much)")
    ap.add_argument("--no-water", action="store_true", help="do not mask out the harbour")
    ap.add_argument("--repaint-water", dest="repaint_water", action="store_true", default=True,
                    help="run water_fill again on the result (default)")
    ap.add_argument("--keep-water", dest="repaint_water", action="store_false")
    args = ap.parse_args()

    plate = Image.open(args.plate).convert("RGB")
    water = None
    if not args.no_water:
        _, _, water = water_fill(plate)
    if args.passes:
        passes = [tuple(float(v) for v in p.split(":")) for p in args.passes.split(",")]
    elif args.floor or args.radius or args.blob:
        passes = [(args.floor or 0.70, args.radius or 26, args.blob or 7)]
    else:
        passes = [(0.70, 26, 7), (0.60, 20, 5)]

    out = plate
    for floor, radius, blob in passes:
        out, frac = deshade(out, floor, int(radius), int(blob), args.feather, water,
                            args.max_sat, args.skip_veg)
        print(f"  floor {floor:.2f} radius {int(radius):d} blob {int(blob):d}: "
              f"lifted {frac * 100:.1f}%", flush=True)
    if args.repaint_water:
        # the lift brightens the shallows and the mesh edges near the shore
        out, colour, _ = water_fill(out)
    out.save(args.out)
    print(f"{args.plate} -> {args.out}")


if __name__ == "__main__":
    main()
