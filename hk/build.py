#!/usr/bin/env python3
"""Assemble the orthographic plate into one seamless pixel map and cut tiles.

hk/shoot.mjs renders the city through a single orthographic camera, one
1024-pixel window at a time, sliding the camera exactly one window along the
picture's right and up axes. Parallel rays mean the windows butt together
without any parallax, so the plate is one picture -- not sixteen that have to
be talked into agreeing.

Everything after that happens once, on the whole plate:

    flatten -> grade -> drop to the art grid -> quantise -> ink -> tiles

If a seam is ever visible, the bug is upstream: either the camera was not
orthographic, or the windows were not shifted by exactly one window, or the
image model was run per tile instead of over the whole canvas.

    .venv/bin/python hk/build.py --plate shots/plate
    .venv/bin/python hk/build.py --check          # seam report only
"""

import json
import os
import re
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

WIN = 1024            # window size shot by hk/ortho.html
TILE = 512            # viewer tile
OVERLAP = 0           # ortho windows need none
PX = 2                # plate pixels per art pixel
COLOURS = 48
INK = 28
VERSION = str(int(time.time()))


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def windows(dirname):
    """Every window PNG, ordered the way the plate grid is laid out."""
    found = {}
    for name in os.listdir(dirname):
        if not name.endswith(".png"):
            continue
        stem = name[:-4]
        if "_" not in stem:
            continue
        i, j = stem.split("_")
        found[(int(i), int(j))] = os.path.join(dirname, name)
    return found


def assemble(dirname):
    files = windows(dirname)
    cols = max(i for i, _ in files) + 1
    rows = max(j for _, j in files) + 1
    missing = [(i, j) for j in range(rows) for i in range(cols) if (i, j) not in files]
    if missing:
        raise SystemExit(f"missing windows: {missing}")

    step = WIN - OVERLAP
    width = cols * step + OVERLAP
    height = rows * step + OVERLAP
    plate = Image.new("RGB", (width, height))
    for (i, j), path in sorted(files.items()):
        image = Image.open(path).convert("RGB")
        if image.size != (WIN, WIN):
            image = image.resize((WIN, WIN), Image.LANCZOS)
        plate.paste(image, (i * step, j * step))
    return plate, (cols, rows)


def seam_report(plate, cols, rows):
    """A seam that is much sharper than ordinary texture means the windows
    disagree; if the numbers sit near the picture's own contrast, they agree."""
    px = plate.load()
    width, height = plate.size
    step = WIN - OVERLAP
    out = []
    for i in range(1, cols):
        x = i * step
        diffs = [abs(px[x, y][0] - px[x - 1, y][0]) for y in range(0, height, 3)]
        out.append(("v", x, sum(diffs) / len(diffs)))
    for j in range(1, rows):
        y = j * step
        diffs = [abs(px[x, y][0] - px[x, y - 1][0]) for x in range(0, width, 3)]
        out.append(("h", y, sum(diffs) / len(diffs)))
    # Interior reference: mean neighbour difference well away from any seam.
    sample = [abs(px[x, y][0] - px[x + 1, y][0])
              for y in range(20, height - 20, 37)
              for x in range(20, width - 40, 41)]
    base = sum(sample) / len(sample)
    return base, out


def water_fill(plate, radius=24, flat=10.0):
    """Water carries no photogrammetry, so Google hands back whatever tone its
    tile happened to use: bright cyan in the shallows, near-black offshore, a
    third teal in between, with hard edges where they meet. Those edges are not
    a window seam, but they look exactly like one, so paint every flat surface
    one harbour colour and keep the boats.

    Flatness is the whole trick: over a fifty-pixel window nothing in an aerial
    photograph is featureless except the places where there is no geometry."""
    rgb = plate.convert("RGB")
    a = np.asarray(rgb).astype(np.int16)
    r, b = a[:, :, 0], a[:, :, 2]
    grey = np.asarray(rgb.convert("L")).astype(np.float32)
    mean = ndimage.uniform_filter(grey, radius * 2 + 1)
    sq = ndimage.uniform_filter(grey * grey, radius * 2 + 1)
    sd = np.sqrt(np.maximum(0.0, sq - mean * mean))

    water = (sd < flat) & (b >= r - 6)
    water = ndimage.binary_closing(water, np.ones((9, 9)))
    water = ndimage.binary_dilation(water, np.ones((7, 7)))

    # The sea is the body of water that reaches the edge of the picture. (Look
    # a band inside the frame: the closing above eats a few rows at the border.)
    labels, count = ndimage.label(water)
    sizes = ndimage.sum(water, labels, range(1, count + 1))
    band = 16
    edge = set(np.unique(np.concatenate([
        labels[:band].ravel(), labels[-band:].ravel(),
        labels[:, :band].ravel(), labels[:, -band:].ravel()])).tolist())
    edge.discard(0)
    keep = np.zeros(count + 1, bool)
    for i in range(1, count + 1):
        keep[i] = sizes[i - 1] >= 50000 and i in edge
    water = keep[labels]

    # Anything smaller than a boat is noise on the flat surface; real objects
    # (junks, ferries, buoys) stay photographic.
    dry_labels, dry_count = ndimage.label(~water)
    dry_sizes = ndimage.sum(~water, dry_labels, range(1, dry_count + 1))
    big = np.zeros(dry_count + 1, bool)
    big[1:] = dry_sizes >= 450
    water |= ~big[dry_labels] & ~water

    # Harbour colour: the middle of the tones Google handed back, so it is
    # neither the near-black offshore nor the bright inshore shallows.
    px = np.asarray(rgb)[water]
    if len(px):
        colour = tuple(int(v) for v in np.median(px, axis=0))
    else:
        colour = (27, 107, 116)

    out = np.asarray(rgb).copy()
    out[water] = colour
    return Image.fromarray(out), colour


def stylise(plate, px, colours, ink_threshold):
    """One pass over the whole plate. Nothing here looks at tile edges, so
    nothing here can create one."""
    image = plate.filter(ImageFilter.MedianFilter(5)).filter(ImageFilter.GaussianBlur(1.1))
    r, g, b = image.split()
    r = r.point(lambda v: min(255, int(v * 1.05 + 6)))
    g = g.point(lambda v: min(255, int(v * 1.00 + 2)))
    b = b.point(lambda v: min(255, int(v * 0.96)))
    image = Image.merge("RGB", (r, g, b))
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Contrast(image).enhance(1.05)

    w, h = image.size
    image = image.resize((max(1, w // px), max(1, h // px)), Image.BOX)
    image = image.quantize(colors=colours, method=Image.MEDIANCUT,
                           dither=Image.Dither.NONE).convert("RGB")
    if ink_threshold > 0:
        edges = (image.convert("L")
                 .filter(ImageFilter.GaussianBlur(0.7))
                 .filter(ImageFilter.FIND_EDGES)
                 .point(lambda v: 255 if v > ink_threshold else 0))
        image = Image.composite(Image.new("RGB", image.size, (46, 40, 34)), image, edges)
    return image


def patch_region(image, patch, box, feather=44):
    """The image model stamps its own logo into the bottom-right corner, over
    whatever the map happens to be there. Rather than fill it (there is nothing
    to fill it with) or mirror a band (which shows), redraw that one corner from
    the plate at a higher zoom and paste the result back. It is the same model,
    the same prompt and the same ground, so it reads as the same picture."""
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = box
    b = np.asarray(patch.convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS),
                   dtype=np.float32)
    alpha = np.ones((y1 - y0, x1 - x0), np.float32)
    ramp_x = np.clip(np.arange(x1 - x0, dtype=np.float32) / feather, 0, 1)
    ramp_y = np.clip(np.arange(y1 - y0, dtype=np.float32) / feather, 0, 1)
    alpha = alpha * ramp_x[None, :] * ramp_y[:, None]
    alpha = alpha[:, :, None]
    a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - alpha) + b * alpha
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def copy_patch(image, box, dx, dy, feather=18):
    """Last-resort fill for the corner the model stamped its logo on: copy a
    block from the same picture, offset by (dx, dy), and feather the seam."""
    a = np.asarray(image.convert("RGB")).astype(np.float32)
    x0, y0, x1, y1 = box
    src = a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    alpha = np.ones((y1 - y0, x1 - x0), np.float32)
    fx = np.clip(np.arange(x1 - x0, dtype=np.float32) / feather, 0, 1)
    fy = np.clip(np.arange(y1 - y0, dtype=np.float32) / feather, 0, 1)
    alpha *= fx[None, :] ** 0.5 * fy[:, None] ** 0.5
    alpha = alpha[:, :, None]
    a[y0:y1, x0:x1] = a[y0:y1, x0:x1] * (1 - alpha) + src * alpha
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def from_redraw(path, size, colours):
    """The image model's pass over the whole plate. It keeps the layout and
    gives one consistent style, because it saw one picture rather than one
    tile at a time. Trim the edge the model likes to darken, then sit it on the
    plate's aspect ratio."""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    edge = max(2, int(min(w, h) * 0.006))
    image = image.crop((edge, edge, w - edge, h - edge))
    if size and image.size != size:
        image = image.resize(size, Image.LANCZOS)
    if colours:
        image = image.quantize(colors=colours, method=Image.MEDIANCUT,
                               dither=Image.Dither.NONE).convert("RGB")
    return image


def pyramid(image, out_dir, tile=TILE, overlap=1):
    """Deep-zoom levels, parent pixel = mean of its four children.

    Every tile is saved `overlap` pixels larger on each side, carrying a sliver
    of its neighbour. The viewer draws that sliver into the seam, so when a tile
    is resampled to a fractional scale its bilinear filter has real pixels to
    sample instead of clamping at its own edge - which is what put a one-pixel
    line on every tile boundary. Same trick as Deep Zoom."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    width, height = image.size
    levels = 0
    while (width >> levels) > tile or (height >> levels) > tile:
        levels += 1
    for level in range(levels + 1):
        factor = 1 << (levels - level)
        sw = max(1, -(-width // factor))
        sh = max(1, -(-height // factor))
        scaled = image if factor == 1 else image.resize((sw, sh), Image.LANCZOS)
        # pad by edge replication so the outermost tiles get a sliver too
        padded = Image.fromarray(np.pad(np.asarray(scaled), ((overlap, overlap), (overlap, overlap), (0, 0)), mode="edge"))
        cols = (sw + tile - 1) // tile
        rows = (sh + tile - 1) // tile
        folder = os.path.join(out_dir, "z%d" % level)
        os.makedirs(folder, exist_ok=True)
        for ty in range(rows):
            for tx in range(cols):
                box = (tx * tile, ty * tile, tx * tile + tile + 2 * overlap,
                       ty * tile + tile + 2 * overlap)
                padded.crop(box).save(os.path.join(folder, "%d_%d.png" % (tx, ty)))
    return levels, width, height


def write_page(levels, width, height, bg):
    path = os.path.join(HERE, "map.html")
    src = open(path).read()
    meta = (
        "const meta = {\n"
        '  dir: "tiles",\n'
        f"  tileSize: {TILE},\n"
        "  tileOverlap: 1,\n"
        f"  maxLevel: {levels},\n"
        f"  width: {width},\n"
        f"  height: {height},\n"
        f"  focusX: {width // 2},\n"
        f"  focusY: {height // 2},\n"
        "  maxScale: 8,\n"
        f'  bg: "rgb({bg[0]},{bg[1]},{bg[2]})",\n'
        f'  version: "{VERSION}"\n'
        "};"
    )
    src = re.sub(r"const meta = \{.*?\n\};", meta, src, count=1, flags=re.S)
    open(path, "w").write(src)


def main():
    plate_dir = _arg("--plate", os.path.join(HERE, "source", "plate"))
    check_only = "--check" in sys.argv
    px = int(_arg("--px", PX))
    colours = int(_arg("--colours", COLOURS))
    ink = int(_arg("--ink", INK))

    plate, (cols, rows) = assemble(plate_dir)
    base, seams = seam_report(plate, cols, rows)
    print(f"plate {plate.size[0]}x{plate.size[1]}  {cols}x{rows} windows")
    print(f"  interior neighbour difference {base:.2f}")
    for axis, at, value in seams:
        flag = "  <-- CHECK" if value > base * 3 + 6 else ""
        print(f"  seam {axis} at {at:5d}: {value:.2f}{flag}")
    if check_only:
        return

    plate, water = water_fill(plate)
    print(f"  water fill {water}")
    plate.save(os.path.join(HERE, "plate.png"))
    redraw = _arg("--redraw")
    if redraw:
        box = tuple(int(v) for v in _arg("--logo-box", "0,0,0,0").split(","))
        corner = _arg("--corner-redraw")
        if box[2] > 0 and corner:
            clean = os.path.join(HERE, "redraw-clean.png")
            im = patch_region(Image.open(redraw), Image.open(corner), box)
            dab = _arg("--corner-dab")
            if dab:
                b = tuple(int(v) for v in dab.split(","))
                im = copy_patch(im, b[:4], b[4], b[5])
            im.save(clean)
            redraw = clean
        art = from_redraw(redraw, None, int(_arg("--redraw-colours", "0")))
        print(f"  redraw {redraw}")
    elif "--raw" in sys.argv:
        art = plate                      # 原图直出，不做风格化，用来看瓦片衔接
    else:
        art = stylise(plate, px, colours, ink)
    art.save(os.path.join(HERE, "art.png"))
    levels, width, height = pyramid(art, os.path.join(HERE, "tiles"))
    write_page(levels, width, height, water)
    print(f"art {width}x{height}  levels 0..{levels}  version {VERSION}")


if __name__ == "__main__":
    main()
