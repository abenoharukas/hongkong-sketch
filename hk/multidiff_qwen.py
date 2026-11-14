#!/usr/bin/env python3
"""Redraw the plate with Qwen-Image-2.1-Turbo, the way the reference page does it.

https://www.yadongxie.com/blog/how-pixel-singapore-was-made

One latent covers the whole canvas, one cell per 16x16 pixels. Every denoising
step the transformer runs on overlapping 1024-pixel windows spaced 768 apart,
each conditioned on its own crop of the plate; the predicted velocities are
averaged with a weight that is flat across the middle 512 pixels and falls to
zero at the edges; every other step the whole grid is shifted by 384 pixels, so
the places where windows meet on one step land in the flat middle on the next.
Then one Euler step with the averaged velocity.

The model is a flow-matching transformer that conditions on the crop two ways:
its Qwen3-VL text encoder reads the crop as vision context, and the VAE encodes
it into latent tokens that are prepended to the target tokens. Both are needed,
so a window is not "a prompt" - it is a (vision, crop-latents) pair, and it has
to be computed for every window position of both grids. It is also the expensive
part, so it is cached to disk and the two stages run as separate processes: the
text encoder is 17.5 GB and the transformer is 14.2 GB, and on this machine they
cannot be in memory at the same time.

    # 1. conditioning (text encoder + VAE), cached per window
    python hk/multidiff_qwen.py --stage cond --region 3072,1024,2048,1024 \
        --work /tmp/hkverify
    # 2. denoise + decode
    python hk/multidiff_qwen.py --stage sample --region 3072,1024,2048,1024 \
        --work /tmp/hkverify --out /tmp/hkverify/multi.png --control

`--control` runs the same grid the old broken way - every window its own
trajectory, no shared latent - and writes `*_control.png` next to the output, so
the two can be measured against each other instead of argued about.
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import Qwen3VLProcessor

from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

WINDOW = 1024               # sampling window, pixels
STRIDE = 768                # 256 px of overlap with the next window
SHIFT = 384                 # the grid is shifted by half a stride on odd steps
CELL = 16                   # pixels covered by one latent cell
FLAT = 512                  # weight is flat across the middle 512 px
MODEL = os.path.expanduser("~/models/qwen-image-2.1-turbo")
PROMPT = "Redraw this aerial view as detailed isometric pixel art."


# ---------------------------------------------------------------- geometry

def parse_size(text):
    w, _, h = text.partition("x")
    return int(w), int(h)


def parse_region(text):
    if not text:
        return None
    x, y, w, h = (int(v) for v in text.split(","))
    return x, y, w, h


def window_grid(width, height, offset):
    """Top-left corners of every window. The last one is pulled back so it ends
    on the edge; a negative start is snapped onto the edge and the weight map
    deals with the extra overlap."""
    xs = list(range(-offset, max(0, width - WINDOW) + 1, STRIDE)) or [0]
    if xs[-1] + WINDOW < width:
        xs.append(width - WINDOW)
    ys = list(range(-offset, max(0, height - WINDOW) + 1, STRIDE)) or [0]
    if ys[-1] + WINDOW < height:
        ys.append(height - WINDOW)
    xs = sorted({max(0, min(x, width - WINDOW)) for x in xs})
    ys = sorted({max(0, min(y, height - WINDOW)) for y in ys})
    return [(x, y) for y in ys for x in xs]


def window_weight(size=WINDOW, flat=FLAT):
    """Flat across the middle `flat` pixels, cosine falloff to zero at the edge."""
    ramp = (size - flat) // 2
    w = np.ones(size, np.float32)
    t = np.linspace(0, math.pi / 2, ramp, dtype=np.float32)
    w[:ramp] = np.sin(t) ** 2
    w[-ramp:] = np.sin(t[::-1]) ** 2
    return torch.from_numpy(np.outer(w, w))


def cells(span):
    return span // CELL


def window_tokens(latents, x, y):
    """The (b, 4096, 64) view of a 1024 px window into the whole-canvas latent,
    which is packed row-major: (cells_y, cells_x, channels)."""
    n = cells(WINDOW)
    h = cells(latents.shape[-2])
    return latents[..., (y // CELL):(y // CELL) + n, (x // CELL):(x // CELL) + n, :].reshape(1, n * n, -1)


# ---------------------------------------------------------------- loading

def pick_dtype(name, device):
    if name == "auto":
        return torch.float16 if device == "mps" else torch.bfloat16
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def pipeline(device, dtype, want, lora=None, lora_scale=1.0):
    """A pipeline holding only the components `want` names, so the 17.5 GB text
    encoder and the 14.2 GB transformer never have to be resident together."""
    path = MODEL
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(path, "processor"))
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(path, "scheduler"))
    sigmas = json.load(open(os.path.join(path, "model_index.json")))["sample_sigmas"]
    text_encoder = transformer = vae = None
    if want == "cond":
        from transformers import Qwen3VLForConditionalGeneration
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            os.path.join(path, "text_encoder"), torch_dtype=dtype, low_cpu_mem_usage=True)
    elif want == "sample":
        transformer = QwenImage21Transformer2DModel.from_pretrained(
            os.path.join(path, "transformer"), torch_dtype=dtype, low_cpu_mem_usage=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        os.path.join(path, "vae"), torch_dtype=torch.float32, low_cpu_mem_usage=True)
    pipe = QwenImage21Pipeline(scheduler=scheduler, vae=vae, text_encoder=text_encoder,
                               processor=processor, transformer=transformer,
                               sample_sigmas=sigmas)
    pipe.to(device)
    if want == "sample" and lora:
        pipe.load_lora_weights(lora)
        pipe.fuse_lora(lora_scale=lora_scale)
    return pipe


def plate_crop(plate, region, x, y):
    """The 1024 px crop of the plate that conditions the window at (x, y)."""
    rx, ry = region[0], region[1]
    return plate.crop((rx + x, ry + y, rx + x + WINDOW, ry + y + WINDOW))


# --------------------------------------------------------------- stage cond

@torch.inference_mode()
def stage_cond(args, region):
    x0, y0, width, height = region
    device, dtype = args.device, pick_dtype(args.dtype, args.device)
    plate = Image.open(args.plate).convert("RGB")
    os.makedirs(args.work, exist_ok=True)

    offsets = sorted({0, SHIFT})
    todo = sorted({w for off in offsets for w in window_grid(width, height, off)})
    print(f"region {width}x{height} at {x0},{y0}: {len(todo)} distinct windows", flush=True)

    pipe = pipeline(device, dtype, "cond")
    print(f"text encoder on {device} {dtype} in {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
          if device == "mps" else "loaded", flush=True)

    t0 = time.time()
    for i, (x, y) in enumerate(todo):
        cache = os.path.join(args.work, f"cond_{x}_{y}.pt")
        if os.path.exists(cache) and not args.force:
            continue                      # already done - a stopped run resumes here
        crop = plate_crop(plate, region, x, y)
        # Exactly what QwenImage21Pipeline.__call__ does with a condition image:
        # one resize feeds the vision encoder and the VAE.
        rgba = crop.convert("RGBA")
        resized = pipe.image_processor.resize(rgba, width=WINDOW, height=WINDOW)
        embeds, mask, img_mask = pipe.encode_prompt(
            prompt=args.prompt, image=[resized], device=torch.device(device),
            num_images_per_prompt=1)
        vae_image = pipe.image_processor.preprocess(
            rgba, width=WINDOW, height=WINDOW).unsqueeze(2).to(device)
        tokens = pipe._encode_vae_image(vae_image, None)
        cond = pipe._pack_latents(tokens, 1, tokens.shape[1],
                                  tokens.shape[3], tokens.shape[4])
        torch.save({"embeds": embeds.half().cpu(),
                    "mask": None if mask is None else mask.cpu(),
                    "img_mask": img_mask.cpu(), "cond": cond.half().cpu()},
                   os.path.join(args.work, f"cond_{x}_{y}.pt"))
        per = (time.time() - t0) / (i + 1)
        print(f"  window {i + 1}/{len(todo)} at {x},{y}  {per:.1f}s  "
              f"eta {per * (len(todo) - i - 1) / 60:.1f} min", flush=True)


# ------------------------------------------------------------- stage sample

def load_cond(work, x, y, device):
    blob = torch.load(os.path.join(work, f"cond_{x}_{y}.pt"), weights_only=False)
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in blob.items()}


def sigmas_for(pipe, tokens):
    """The Turbo checkpoint ships its own 8-step sigmas; the terminal sigma is
    appended by the scheduler."""
    pipe.scheduler.set_timesteps(sigmas=pipe.config.sample_sigmas, mu=None)
    return pipe.scheduler.timesteps, pipe.scheduler.sigmas


def run_transformer(pipe, hidden, t, blob, n_tokens):
    """One window's velocity prediction. The condition tokens are already in
    `hidden`, so the transformer only reconstructs the layout."""
    n = cells(WINDOW)
    out = pipe.transformer(
        hidden_states=hidden,
        encoder_hidden_states=blob["embeds"].to(hidden.device, hidden.dtype),
        timestep=(t / 1000.0).expand(1).to(device=hidden.device, dtype=hidden.dtype),
        img_shapes=[[(1, n, n), (1, n, n)]],
        img_mask=blob["img_mask"].to(hidden.device),
        encoder_hidden_states_mask=(blob["mask"].to(hidden.device)
                                    if blob["mask"] is not None else None),
        return_dict=False,
    )[0]
    return out[:, -n_tokens:]


def audit_report(audit, grid):
    """A: how much neighbouring windows disagree inside the 256 px band they
    share, on the last denoising step. The window edge weight is zero there, so
    a disagreement that is large here is what would show up as a seam."""
    if not audit:
        return
    overlap = (WINDOW - STRIDE) // CELL          # 16 cells
    rows = []
    for (x, y) in grid:
        a = audit.get((x, y))
        if a is None:
            continue
        for (xx, yy) in ((x + STRIDE, y), (x, y + STRIDE)):
            b = audit.get((xx, yy))
            if b is None:
                continue
            if xx != x:                                    # vertical neighbour
                va, vb = a[:, -overlap:], b[:, :overlap]
            else:                                          # horizontal neighbour
                va, vb = a[-overlap:, :], b[:overlap, :]
            d = (va - vb).abs().mean()
            scale = 0.5 * (va.abs().mean() + vb.abs().mean())
            rows.append((float(d / scale), x, y, xx, yy))
    if not rows:
        return
    rows.sort(reverse=True)
    vals = [r[0] for r in rows]
    print(f"\n  audit: {len(rows)} neighbour pairs on the last step\n"
          f"    |dv|/|v| in the shared band: median {np.median(vals):.3f}  "
          f"worst {vals[0]:.3f}  mean {np.mean(vals):.3f}", flush=True)
    for r, x, y, xx, yy in rows[:5]:
        print(f"    worst {r:.3f} between ({x},{y}) and ({xx},{yy})", flush=True)


def target_mask(blob):
    """`img_mask` covers the vision slots; the transformer wants one more slot
    per 2x2 group of target cells appended."""
    return torch.cat([blob["img_mask"], blob["img_mask"].new_ones(1, cells(WINDOW * WINDOW // CELL // 4))], dim=1)


@torch.inference_mode()
def stage_sample(args, region):
    x0, y0, width, height = region
    device, dtype = args.device, pick_dtype(args.dtype, args.device)
    pipe = pipeline(device, dtype, "sample", args.lora, args.lora_scale)
    torch.manual_seed(args.seed)

    n = cells(WINDOW)
    # one weight per latent cell: the map is smooth (16 cells of ramp), so
    # sampling its centres is the same as averaging it over the cell
    weight = (window_weight()[CELL // 2::CELL, CELL // 2::CELL]
              .reshape(1, 1, n, n, 1).to(device))
    timesteps, sigmas = sigmas_for(pipe, cells(width) * cells(height))
    print(f"{len(timesteps)} steps, sigmas {[round(float(s), 4) for s in sigmas]}", flush=True)

    first = load_cond(args.work, *window_grid(width, height, 0)[0], device)
    img_mask = target_mask(first)

    # one cell per 16x16 px, packed row-major so a window is a contiguous slice
    lat = torch.randn((1, 1, cells(height), cells(width), 64), device=device, dtype=dtype)
    state = args.out + ".state.pt"
    start = 0
    if args.resume and os.path.exists(state):
        blob = torch.load(state, weights_only=False)
        lat, start = blob["lat"].to(device, dtype), blob["step"]
        print(f"resuming at step {start + 1}", flush=True)
    t0 = time.time()
    audit = {}
    for i, t in enumerate(timesteps):
        if i < start:
            continue
        offset = SHIFT if i % 2 else 0
        last = i == len(timesteps) - 1
        grid = window_grid(width, height, offset)
        acc = torch.zeros_like(lat)
        wsum = torch.zeros((1, 1, cells(height), cells(width), 1), device=device, dtype=dtype)
        for (x, y) in grid:
            blob = load_cond(args.work, x, y, device)
            blob["img_mask"] = img_mask
            hidden = torch.cat([blob["cond"].to(dtype), window_tokens(lat, x, y)], dim=1)
            vel = run_transformer(pipe, hidden, t, blob, n * n)
            if last and args.audit:
                audit[(x, y)] = vel.reshape(n, n, -1).float().cpu()
            sl = (slice(None), slice(None), slice(y // CELL, y // CELL + n), slice(x // CELL, x // CELL + n))
            acc[sl] += vel.reshape(1, 1, n, n, -1) * weight
            wsum[sl] += weight
        step = float(sigmas[i + 1] - sigmas[i]) * (acc / wsum.clamp(min=1e-6))
        lat = lat + step
        # every step is saved, so an interrupted run resumes instead of restarting
        torch.save({"lat": lat.cpu(), "step": i + 1}, state)
        per = (time.time() - t0) / (i + 1)
        print(f"  step {i + 1}/{len(timesteps)}  {len(grid)} windows  {per:.1f}s/step  "
              f"eta {per * (len(timesteps) - i - 1) / 60:.1f} min", flush=True)

    if args.audit:
        audit_report(audit, window_grid(width, height, SHIFT if (len(timesteps) - 1) % 2 else 0))
    save(pipe, lat, width, height, args.out, args.tile, args.overlap)
    if args.control:
        control(pipe, args, region, timesteps, sigmas)


@torch.inference_mode()
def control(pipe, args, region, timesteps, sigmas):
    """The old way: every window its own trajectory from its own noise, no
    shared latent, pasted together at the end. This is the baseline the numbers
    have to beat."""
    x0, y0, width, height = region
    device, dtype = args.device, pick_dtype(args.dtype, args.device)
    n = cells(WINDOW)
    canvas = torch.zeros((1, 1, cells(height), cells(width), 64), device=device, dtype=dtype)
    starts = window_grid(width, height, SHIFT)      # offset grid so every seam is interior
    t0 = time.time()
    for k, (x, y) in enumerate(starts):
        blob = load_cond(args.work, x, y, device)
        blob["img_mask"] = target_mask(blob)
        torch.manual_seed(args.seed + k)
        win = torch.randn((1, 1, n, n, 64), device=device, dtype=dtype)
        for i, t in enumerate(timesteps):
            hidden = torch.cat([blob["cond"].to(dtype), win.reshape(1, n * n, -1)], dim=1)
            vel = run_transformer(pipe, hidden, t, blob, n * n)
            win = win + float(sigmas[i + 1] - sigmas[i]) * vel.reshape(1, 1, n, n, -1)
        canvas[:, :, y // CELL:y // CELL + n, x // CELL:x // CELL + n] = win
        print(f"  control window {k + 1}/{len(starts)}  {(time.time() - t0) / (k + 1):.1f}s", flush=True)
    save(pipe, canvas, width, height, args.out.replace(".png", "_control.png"), args.tile, args.overlap)


def cell_weight(n, ramp):
    """Per-cell blend weight: 1 in the middle, cosine down to 0 over `ramp`
    cells at each edge. Upsampled to pixels with a plain repeat."""
    w = torch.ones(n, dtype=torch.float32)
    r = min(ramp, n // 2)
    if r > 0:
        t = torch.linspace(0, math.pi / 2, r)
        w[:r] = torch.sin(t) ** 2
        w[-r:] = torch.flip(torch.sin(t) ** 2, dims=[0])
    return w


@torch.inference_mode()
def decode_tiled(vae, lat, tile=32, overlap=8):
    """Decode a (1, 64, 1, h, w) latent in overlapping cell tiles and blend the
    results in pixel space. The VAE's upsampling multiplies the intermediate
    tensor, and a whole frame does not fit next to the transformer; nor would a
    whole map, so this is the only decode path."""
    _, _, _, hc, wc = lat.shape
    step = max(1, tile - overlap)
    out = wsum = None
    done = 0
    total = ((hc - overlap - 1) // step + 1) * ((wc - overlap - 1) // step + 1)
    for y in range(0, hc, step):
        for x in range(0, wc, step):
            y2, x2 = min(y + tile, hc), min(x + tile, wc)
            y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
            px = vae.decode(lat[:, :, :, y1:y2, x1:x2], return_dict=False)[0][:, :, 0][:, :3]
            if out is None:
                out = torch.zeros((1, 3, hc * CELL, wc * CELL), device=px.device, dtype=px.dtype)
                wsum = torch.zeros((1, 1, hc * CELL, wc * CELL), device=px.device, dtype=px.dtype)
            wy = cell_weight(y2 - y1, overlap // 2).repeat_interleave(CELL).to(px.device, px.dtype)
            wx = cell_weight(x2 - x1, overlap // 2).repeat_interleave(CELL).to(px.device, px.dtype)
            w = (wy[:, None] * wx[None, :])[None, None]
            w = w[..., :px.shape[2], :px.shape[3]]
            out[:, :, y1 * CELL:y1 * CELL + px.shape[2], x1 * CELL:x1 * CELL + px.shape[3]] += px * w
            wsum[:, :, y1 * CELL:y1 * CELL + px.shape[2], x1 * CELL:x1 * CELL + px.shape[3]] += w
            done += 1
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"  decode tile {done}/{total} at cell {x1},{y1}  "
                  f"mps {torch.mps.current_allocated_memory() / 2**30:.1f} GB", flush=True)
    return out / wsum.clamp(min=1e-6)


@torch.inference_mode()
def save(pipe, lat, width, height, path, tile=32, overlap=8):
    lat = pipe._unpack_latents(lat.reshape(1, -1, 64), height, width, pipe.vae_scale_factor)
    lat = lat.to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(lat.device, lat.dtype)
    image = decode_tiled(pipe.vae, lat * std + mean, tile, overlap)
    image = ((image.float() + 1) / 2).clamp(0, 1)
    out = (image[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(out).save(path)
    print(f"wrote {path} {out.shape[1]}x{out.shape[0]}", flush=True)


@torch.inference_mode()
def stage_decode(args, region):
    """Nothing but the VAE: finish an already-sampled canvas from its state
    file. The transformer is 13 GB and the decode does not need it."""
    x0, y0, width, height = region
    pipe = pipeline(args.device, pick_dtype(args.dtype, args.device), "decode")
    if args.decode_dtype == "fp16":
        pipe.vae.to(torch.float16)
    blob = torch.load(args.out + ".state.pt", weights_only=False)
    lat = blob["lat"].to(args.device)
    print(f"decoding step {blob['step']} latents {tuple(lat.shape)}", flush=True)
    save(pipe, lat, width, height, args.out, args.tile, args.overlap)


# ------------------------------------------------------------------- report

def report(path):
    """A seam is a line that is sharper than the picture's own texture. Compare
    the difference across every 512-px tile boundary (and every 256-px offset
    where a window can end) against the median difference nearby."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    h, w, _ = a.shape
    col = np.abs(np.diff(a, axis=1)).sum(axis=(0, 2)) / h
    row = np.abs(np.diff(a, axis=0)).sum(axis=(1, 2)) / w
    print(f"\n{os.path.basename(path)}  {w}x{h}  neighbour difference: "
          f"col median {np.median(col):.1f}  row median {np.median(row):.1f}")
    for name, series in (("x", col), ("y", row)):
      for pitch in (STRIDE, SHIFT, 256, 192):     # both window grids + decode joints
        at = range(pitch, len(series), pitch)
        worst = []
        for p in at:
            j = min(p - 1, len(series) - 1)
            near = np.r_[series[max(0, j - 40):j - 1], series[j + 2:j + 41]]
            ratio = series[j] / max(np.median(near), 0.5)
            worst.append((ratio, p))
        if worst:
            r, p = max(worst)
            verdict = "  OK" if r <= 1.5 else "  <-- FAIL"
            print(f"  {name} every {pitch:4d} px: worst {r:.2f} at {p}{verdict}")


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cond", "sample", "decode", "report"], required=True)
    ap.add_argument("--plate", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plate.png"))
    ap.add_argument("--region", default=None, help="x,y,w,h of the plate to redraw")
    ap.add_argument("--size", default=None, help="whole plate as WxH when --region is absent")
    ap.add_argument("--work", default="/tmp/hk-multidiff", help="conditioning cache")
    ap.add_argument("--out", default="/tmp/hk-multidiff/art.png")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--control", action="store_true", help="also run the per-window baseline")
    ap.add_argument("--audit", action="store_true",
                    help="measure neighbour disagreement on the last step (A)")
    ap.add_argument("--lora", default=None, help="style LoRA: local path or hub repo")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>.state.pt")
    ap.add_argument("--force", action="store_true", help="recompute cached conditioning")
    ap.add_argument("--tile", type=int, default=16, help="decode tile, in latent cells")
    ap.add_argument("--overlap", type=int, default=4, help="decode tile overlap, in cells")
    ap.add_argument("--decode-dtype", default="fp16", choices=["fp16", "fp32"],
                    help="VAE precision for the decode; fp32 needs about 3x the memory")
    args = ap.parse_args()

    if args.stage == "report":
        report(args.out)
        return
    region = parse_region(args.region)
    if region is None:
        w, h = parse_size(args.size or "6144x4096")
        region = (0, 0, w, h)
    if args.stage == "cond":
        stage_cond(args, region)
    elif args.stage == "decode":
        stage_decode(args, region)
    else:
        stage_sample(args, region)


if __name__ == "__main__":
    main()
