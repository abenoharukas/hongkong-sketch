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
    return Image.fromarray(out), colour, water


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


WATER_RAMP = ((0x1A, 0x5E, 0x63), (0x24, 0x74, 0x76), (0x2F, 0x8A, 0x88),
              (0x3E, 0x9C, 0x97), (0x4F, 0xB0, 0xA8))
VEG_RAMP = ((0x2C, 0x4A, 0x1E), (0x3A, 0x5E, 0x27), (0x4E, 0x7A, 0x34),
            (0x66, 0x93, 0x3F), (0x86, 0xB1, 0x55))
BUILT_RAMP = ((0x6F, 0x61, 0x52), (0x8A, 0x7A, 0x65), (0xA7, 0x93, 0x79),
              (0xC0, 0xAC, 0x8E), (0xD8, 0xC9, 0xAE))
ROAD_RAMP = ((0x33, 0x32, 0x2F), (0x45, 0x43, 0x40))
# Hill sides are neither buildings nor lawns: smooth ground that the city's
# texture test leaves alone. Without a family of its own it takes the built-up
# beige and a mountain range reads as more city.
TERRAIN_RAMP = ((0x3B, 0x45, 0x28), (0x49, 0x55, 0x30), (0x59, 0x66, 0x38),
                (0x6D, 0x79, 0x46))


def _ramps():
    return (np.asarray(WATER_RAMP, np.float32),
            np.asarray(VEG_RAMP, np.float32),
            np.asarray(BUILT_RAMP, np.float32),
            np.asarray(ROAD_RAMP, np.float32),
            np.asarray(TERRAIN_RAMP, np.float32))


def family_shape(rgb):
    return rgb.shape[:2]


def luma_of(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def pixel_art(plate, px, colours, ink, water_mask, ink_min=26.0, tilt=0.0,
              sharpen=1.25, flat=5):
    """The reference look: flat colour families and a one-pixel ink line.

    Two things separate this from a hand-drawn (or model-drawn) pass:

    * **One cell, one colour, from its family.** Each art cell is classified as
      water, vegetation, built or road from its hue and then takes a shade from
      that family's ramp by its own brightness. A roof cannot pick a random
      teal and a lawn cannot pick a beige, so the picture reads as a map of flat
      surfaces instead of a photograph in 40 colours. `colours` limits how many
      ramp steps are actually used.
    * **The ink is the cell boundary.** A cell is inked when the cell to its
      right or below it belongs to a different family or shade, so every line is
      exactly one cell wide -- the same width on a tower, a coastline and a
      coastline corner. Nothing here draws a stroke, so no stroke can vary.

    """
    w, h = plate.size
    art = plate.filter(ImageFilter.MedianFilter(flat)).filter(
        ImageFilter.GaussianBlur(0.6))
    art = art.resize((max(1, w // px), max(1, h // px)), Image.BOX)
    art = art.filter(ImageFilter.MedianFilter(3))
    if sharpen > 1.0:
        art = art.filter(ImageFilter.UnsharpMask(
            radius=1.0, percent=int((sharpen - 1.0) * 100), threshold=2))

    rgb = np.asarray(art, np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    delta = mx - mn
    sat = np.where(mx > 0.0, delta / np.maximum(mx, 1e-6), 0.0)
    hue = np.zeros_like(mx)
    safe = np.maximum(delta, 1e-6)
    r_dom = (mx == r)
    g_dom = (mx == g) & ~r_dom
    b_dom = ~r_dom & ~g_dom
    hue[r_dom] = (((g - b)[r_dom] / safe[r_dom]) % 6.0)
    hue[g_dom] = ((b - r)[g_dom] / safe[g_dom]) + 2.0
    hue[b_dom] = ((r - g)[b_dom] / safe[b_dom]) + 4.0
    hue *= 60.0

    # Water is the mask water_fill() already found by flatness, not a hue band:
    # in this render the shaded glass, the blue-grey roofs and the hazy sides of
    # the hills all sit in the water band, and a hue test paints teal specks
    # over the whole city. Take the real sea, then classify the land.
    water = water_mask
    if water.shape != rgb.shape[:2]:
        water = np.asarray(
            Image.fromarray((water.astype(np.uint8) * 255))
            .resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127

    # Vegetation: green has to *dominate* red and blue, which survives haze and
    # shadow far better than a hue threshold does.
    green_excess = g - np.maximum(r, b)
    veg = (green_excess > 0.008) & (sat > 0.08) & ~water
    veg = ndimage.binary_opening(veg, np.ones((2, 2), bool))
    labels, count = ndimage.label(veg)
    if count:
        sizes = ndimage.sum(veg, labels, range(1, count + 1))
        small = np.concatenate(([0], np.nonzero(sizes < 40)[0] + 1))
        veg = veg & ~np.isin(labels, small)

    # A city is texture; a hillside is not. Measure how much the picture moves
    # in a small window: towers and roofs are busy, a slope under haze is even.
    grey = luma_of(rgb)
    mean = ndimage.uniform_filter(grey, 5)
    structure = np.sqrt(np.maximum(
        0.0, ndimage.uniform_filter(grey * grey, 5) - mean * mean))
    terrain = (structure < 0.022) & ~water & ~veg

    road = (sat <= 0.12) & (mx < 0.26) & ~water & ~veg & ~terrain
    family = np.where(water, 0,
             np.where(veg, 1,
             np.where(terrain, 4,
             np.where(road, 3, 2))))

    # Majority vote in a 5 x 5 cell window. Single cells of the wrong family
    # are photographic noise, not places: they put one teal or one beige speck
    # in the middle of a lawn and the ink then draws a box around each speck.
    best = np.zeros(family.shape, np.float32)
    voted = np.zeros(family.shape, np.int8)
    for index in range(5):
        score = ndimage.uniform_filter((family == index).astype(np.float32), 5)
        better = score > best
        best[better] = score[better]
        voted[better] = index
    family = voted
    del best

    ramps = _ramps()
    steps = max(2, min(5, int(colours) // 8))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    del mx, mn, delta, sat, hue, safe, green_excess, structure, terrain
    out = np.zeros(rgb.shape, np.float32)
    shade = np.zeros(family.shape, np.int32)
    for index, ramp in enumerate(ramps):
        mask = family == index
        if not mask.any():
            continue
        if index == 0:
            # one harbour colour: a brightness ramp here only re-bands the sea
            out[mask] = ramp[len(ramp) // 2]
            continue
        count = min(steps, len(ramp))
        ramp = ramp[:count] if count < len(ramp) else ramp
        values = luma[mask]
        ranks = np.argsort(np.argsort(values)).astype(np.float32) / max(
            1, values.size - 1)
        pick = np.clip((ranks * count).astype(np.int32), 0, count - 1)
        local = np.zeros(values.shape, np.int32)
        local[:] = pick
        shade[mask] = local
        out[mask] = ramp[local]

    # Ink: a cell is inked where its right or lower neighbour is a different
    # (family, shade) cell. Exactly one cell wide, by construction.
    right = np.zeros(family.shape, bool)
    lower = np.zeros(family.shape, bool)
    shade_jump = 3
    right[:, :-1] = (family[:, :-1] != family[:, 1:]) | (
        np.abs(shade[:, :-1] - shade[:, 1:]) >= shade_jump)
    lower[:-1, :] = (family[:-1, :] != family[1:, :]) | (
        np.abs(shade[:-1, :] - shade[1:, :]) >= shade_jump)
    contrast = np.zeros(family.shape, np.float32)
    contrast[:, :-1] = np.max(np.abs(out[:, :-1] - out[:, 1:]), axis=2)
    contrast[:-1, :] = np.maximum(
        contrast[:-1, :], np.max(np.abs(out[:-1, :] - out[1:, :]), axis=2))
    ink_mask = (right | lower) & (contrast >= ink_min)

    out[ink_mask] = np.asarray(ink, np.float32)
    image = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")
    if tilt > 0.0:
        image = tilt_shift(image, tilt)
    return image


def tilt_shift(image, amount=1.0):
    """Blur the top and bottom of the frame, keep a sharp band across the
    middle: the miniature effect the reference page does with two backdrop
    filters."""
    w, h = image.size
    blurred = image.filter(ImageFilter.GaussianBlur(max(1.0, h / 220.0 * amount)))
    # weight 1 in the middle third, falling to 0 at the edges
    rows = np.arange(h, dtype=np.float32) / max(1, h - 1)
    centre = 0.5
    weight = np.clip((np.abs(rows - centre) - 0.16) / 0.30, 0, 1) ** 1.3
    mask = np.repeat(weight[:, None], w, axis=1)
    mask = Image.fromarray((mask * 255).astype(np.uint8), "L")
    return Image.composite(blurred, image, mask)


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

    plate, water, water_mask = water_fill(plate)
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
        art = pixel_art(
            plate, px, colours,
            tuple(int(v) for v in _arg("--ink-rgb", "34,30,27").split(",")),
            water_mask,
            ink_min=float(_arg("--ink-min", "26")),
            tilt=float(_arg("--tilt", "0")))
        print(f"  pixel art: {px} plate px per cell, {colours} colours, 1 px ink")
    art.save(os.path.join(HERE, "art.png"))
    levels, width, height = pyramid(art, os.path.join(HERE, "tiles"))
    write_page(levels, width, height, water)
    print(f"art {width}x{height}  levels 0..{levels}  version {VERSION}")

if __name__ == "__main__":
    main()
