#!/usr/bin/env python3
"""Turn parallel shooter blocks into one plate window set.

A big plate is shot by several browsers at once, each covering a rectangle of
the window grid. `ortho.html` takes `ox`/`oy`, the block's offset in metres
along the picture's right/up axes, so the rectangles butt together exactly; all
this script does is move each block's local `i_j.png` onto its global index.

    .venv/bin/python hk/merge_blocks.py --blocks hk/source/blocks \
        --out hk/source/plate05 --cols 14 --rows 9 --block-cols 5 --block-rows 3
"""

import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--blocks", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--cols", type=int, required=True)
parser.add_argument("--rows", type=int, required=True)
parser.add_argument("--block-cols", type=int, default=5)
parser.add_argument("--block-rows", type=int, default=3)
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
missing = []
written = 0
for name in sorted(os.listdir(args.blocks)):
    folder = os.path.join(args.blocks, name)
    if not os.path.isdir(folder):
        continue
    bcol, brow = int(name[1]), int(name[2])
    for file in os.listdir(folder):
        if not file.endswith(".png"):
            continue
        i, j = (int(v) for v in file[:-4].split("_"))
        gi = bcol * args.block_cols + i
        gj = brow * args.block_rows + j
        if gi >= args.cols or gj >= args.rows:
            continue                      # overlap column/row, already covered
        shutil.copyfile(os.path.join(folder, file),
                        os.path.join(args.out, f"{gi}_{gj}.png"))
        written += 1

for j in range(args.rows):
    for i in range(args.cols):
        if not os.path.exists(os.path.join(args.out, f"{i}_{j}.png")):
            missing.append((i, j))
print(f"plate windows: {written} written, {len(missing)} missing")
if missing:
    print("missing (first 12):", missing[:12])
    raise SystemExit(1)
