"""MultiDiffusion redraw of the whole plate, following
https://www.yadongxie.com/blog/how-pixel-singapore-was-made

One latent covers the whole canvas. At every denoising step the UNet runs on
overlapping 1024-pixel windows, each conditioned on the same text prompt, and
the predicted noise is averaged with a weight that is flat across the middle of
a window and falls to zero at its edge. Because the window grid is shifted by
half a stride on every other step, the places where windows meet on one step
land in the flat middle of a window on the next - which is what stops tiles from
making their own decisions at their borders.

The window contents come from one latent canvas, not from separate per-tile
latents, so a road cannot change colour at a tile boundary: there are no tiles.

    python hk/diffuse.py --plate hk/plate.png --out hk/art.png --size 6144x4096
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(os.path.expanduser("~"), "models")

WINDOW = 1024          # pixels
STRIDE = 768           # 256 px of overlap between neighbours
SHIFT = 384            # the grid is shifted by half a stride on odd steps


def parse_size(text):
    w, _, h = text.partition("x")
    return int(w), int(h)


def window_grid(width, height, offset):
    """Top-left corners of every window, spaced STRIDE apart, with a grid offset
    of `offset` pixels. The last window is pulled back so it ends on the edge."""
    xs, ys = [], []
    for x in range(-offset, width - WINDOW + 1, STRIDE):
        xs.append(x)
    if not xs or xs[-1] + WINDOW < width:
        xs.append(width - WINDOW)
    for y in range(-offset, height - WINDOW + 1, STRIDE):
        ys.append(y)
    if not ys or ys[-1] + WINDOW < height:
        ys.append(height - WINDOW)
    # snap negative starts onto the edge and let the weight handle the overlap
    xs = sorted({max(0, min(x, width - WINDOW)) for x in xs})
    ys = sorted({max(0, min(y, height - WINDOW)) for y in ys})
    return [(x, y) for y in ys for x in xs]


def window_weight(size=WINDOW, flat=512):
    """Flat in the middle `flat` pixels, cosine falloff to zero at the border."""
    ramp = (size - flat) // 2
    w = np.ones(size, np.float32)
    t = np.linspace(0, math.pi / 2, ramp, dtype=np.float32)
    w[:ramp] = np.sin(t) ** 2
    w[-ramp:] = np.sin(t[::-1]) ** 2
    return np.outer(w, w).astype(np.float32)


def encode(vae, image, device, dtype=torch.float32, tile=1024):
    """VAE-encode a big image in tiles so it fits in memory."""
    w, h = image.size
    lat = torch.zeros((1, 4, h // 8, w // 8), device=device, dtype=dtype)
    acc = torch.zeros((1, 1, h // 8, w // 8), device=device, dtype=dtype)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            crop = image.crop((x, y, min(x + tile, w), min(y + tile, h)))
            t = torch.from_numpy(np.asarray(crop)).permute(2, 0, 1)[None]
            t = (t.to(device, torch.float32) / 127.5 - 1.0)
            z = vae.encode(t).latent_dist.sample()
            z = z * vae.config.scaling_factor
            lw, lh = z.shape[3], z.shape[2]
            lat[:, :, y // 8:y // 8 + lh, x // 8:x // 8 + lw] = z
            acc[:, :, y // 8:y // 8 + lh, x // 8:x // 8 + lw] = 1
    del acc
    return lat


def decode(vae, lat, device, dtype=torch.float32, tile=128):
    """Windowed VAE decode with a flat-middle / faded-edge blend, same idea as
    the sampling loop."""
    _, _, lh, lw = lat.shape
    img = torch.zeros((1, 3, lh * 8, lw * 8), device=device, dtype=dtype)
    wsum = torch.zeros((1, 1, lh * 8, lw * 8), device=device, dtype=dtype)
    wt = torch.from_numpy(window_weight(tile * 8, tile * 8 // 2))[None, None].to(device, dtype)
    stride = tile * 3 // 4
    for y in range(0, lh, stride):
        for x in range(0, lw, stride):
            y = min(y, max(0, lh - tile))
            x = min(x, max(0, lw - tile))
            z = lat[:, :, y:y + tile, x:x + tile]
            if z.shape[2] < tile or z.shape[3] < tile:
                continue
            px = vae.decode((z.float() / vae.config.scaling_factor)).sample
            w = wt[..., :px.shape[2], :px.shape[3]]
            img[:, :, y * 8:y * 8 + px.shape[2], x * 8:x * 8 + px.shape[3]] += px * w
            wsum[:, :, y * 8:y * 8 + px.shape[2], x * 8:x * 8 + px.shape[3]] += w
    img = img / wsum.clamp(min=1e-6)
    return img.clamp(-1, 1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=os.path.join(HERE, "plate.png"))
    ap.add_argument("--out", default=os.path.join(HERE, "art.png"))
    ap.add_argument("--size", default="3072x2048", help="output pixels, WxH")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--strength", type=float, default=0.55,
                    help="how much of the render to keep (lower = closer to the plate)")
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--lora", default=os.path.join(MODELS, "lora", "pixel-art-xl.safetensors"))
    ap.add_argument("--lora-scale", type=float, default=0.9)
    ap.add_argument("--prompt", default="Redraw this aerial view as detailed isometric "
                                       "pixel art, pixel art")
    ap.add_argument("--negative", default="blurry, photo, text, watermark, gradient")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device, dtype = "mps", torch.float16
    W, H = parse_size(args.size)
    W, H = W // 8 * 8, H // 8 * 8

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        os.path.join(MODELS, "sdxl-base"), torch_dtype=dtype, variant="fp16",
        use_safetensors=True, add_watermarker=False)
    pipe.load_lora_weights(args.lora, adapter_name="pixel")
    pipe.fuse_lora(adapter_names=["pixel"], lora_scale=args.lora_scale)
    pipe.to(device)
    unet, vae = pipe.unet, pipe.vae
    vae.to(torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing")
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)

    plate = Image.open(args.plate).convert("RGB").resize((W, H), Image.LANCZOS)
    with torch.inference_mode():
        lat0 = encode(vae, plate, device, tile=512).to(dtype)
    print(f"encoded plate {lat0.shape}", flush=True)

    (pos, neg, pooled, neg_pooled) = pipe.encode_prompt(
        args.prompt, device, 1, True, args.negative,
        pooled_prompt_embeds=None, negative_pooled_prompt_embeds=None)
    add_time_ids = torch.tensor([[H, W, 0, 0, H, W]], device=device, dtype=dtype)

    sched = pipe.scheduler
    sched.set_timesteps(args.steps)
    timesteps = sched.timesteps
    k0 = max(0, int(round(args.steps * (1.0 - args.strength))))
    timesteps = timesteps[k0:]

    noise = torch.randn_like(lat0)
    latents = sched.add_noise(lat0, noise, timesteps[:1].repeat(lat0.shape[0]))
    wmap = torch.from_numpy(window_weight())[None, None].to(device, dtype)
    embeds = torch.cat([neg, pos])
    pooled_cat = torch.cat([neg_pooled, pooled])
    time_ids = add_time_ids.repeat(2, 1)

    t_loop = time.time()
    done = 0
    for i, t in enumerate(timesteps):
        offset = SHIFT if i % 2 else 0
        grid = window_grid(W, H, offset)
        acc = torch.zeros_like(latents)
        wsum = torch.zeros_like(latents[:, :1])
        for (x, y) in grid:
            win = latents[:, :, y // 8:y // 8 + WINDOW // 8, x // 8:x // 8 + WINDOW // 8]
            pred = unet(torch.cat([win, win]), t,
                        encoder_hidden_states=embeds,
                        added_cond_kwargs={"text_embeds": pooled_cat, "time_ids": time_ids},
                        return_dict=False)[0]
            pred = pred[:1] + args.cfg * (pred[1:] - pred[:1])
            w = wmap[..., :pred.shape[2], :pred.shape[3]]
            acc[:, :, y // 8:y // 8 + pred.shape[2], x // 8:x // 8 + pred.shape[3]] += pred * w
            wsum[:, :, y // 8:y // 8 + pred.shape[2], x // 8:x // 8 + pred.shape[3]] += w
        pred = acc / wsum.clamp(min=1e-6)
        latents = sched.step(pred, t, latents, return_dict=False)[0]
        done += 1
        if done % 2 == 0 or done == len(timesteps):
            per = (time.time() - t_loop) / done
            print(f"  step {done}/{len(timesteps)}  {len(grid)} windows  "
                  f"{per:.1f}s/step  eta {(len(timesteps) - done) * per / 60:.1f} min", flush=True)

    with torch.inference_mode():
        img = decode(vae, latents, device, torch.float32)
    img = ((img[0].permute(1, 2, 0).float().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"wrote {args.out} {img.shape[1]}x{img.shape[0]} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
