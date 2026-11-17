"""Render the README GIFs and filmstrip figures from trained models.

    python scripts/make_gifs.py            # writes assets/hero.gif, assets/textures.gif, assets/filmstrip.png ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nca.export import load_model
from nca.model import make_seed, to_rgb
from nca.targets import load_emoji, pad_target
from nca.train_texture import noise_state, texture_rgb

ROOT = Path(__file__).resolve().parents[1]
SURFACE = (252, 252, 251)
INK = (11, 11, 11)
MUTED = (82, 81, 78)


def save_gif(frames: list[np.ndarray], path: Path, fps: int, colors: int = 128) -> None:
    """Looping GIF with one shared palette (smaller than per-frame palettes)."""
    imgs = [Image.fromarray(f) for f in frames]
    sample = Image.fromarray(np.concatenate([np.asarray(imgs[i]) for i in np.linspace(0, len(imgs) - 1, 6).astype(int)], 0))
    pal = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)")


def upscale(rgb: torch.Tensor, scale: int) -> np.ndarray:
    """``(3, H, W)`` float in [0, 1] -> uint8 ``(H*scale, W*scale, 3)`` with nearest-neighbour upscaling."""
    a = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return a.repeat(scale, 0).repeat(scale, 1)


def caption_bar(width: int, text: str, height: int = 28) -> np.ndarray:
    img = Image.new("RGB", (width, height), SURFACE)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=15)
    box = d.textbbox((0, 0), text, font=font)
    d.text(((width - (box[2] - box[0])) // 2, 5), text, fill=INK, font=font)
    return np.asarray(img)


def label_bar(width: int, labels: list[str], height: int = 22) -> np.ndarray:
    img = Image.new("RGB", (width, height), SURFACE)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    cell = width // len(labels)
    for i, t in enumerate(labels):
        box = d.textbbox((0, 0), t, font=font)
        d.text((i * cell + (cell - (box[2] - box[0])) // 2, 3), t, fill=MUTED, font=font)
    return np.asarray(img)


def centroid(x: torch.Tensor) -> tuple[float, float]:
    a = x[0, 3].clamp(min=0)
    ys, xs = torch.meshgrid(torch.arange(a.shape[0]), torch.arange(a.shape[1]), indexing="ij")
    tot = a.sum().clamp(min=1e-6)
    return float((a.cpu() * ys).sum() / tot), float((a.cpu() * xs).sum() / tot)


def disc(h: int, w: int, cy: float, cx: float, r: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).float()


@torch.no_grad()
def growth_and_regeneration(name: str, device: torch.device, scale: int, stride: int, grow: int = 130,
                            regen: int = 140, seed: int = 0) -> tuple[list[np.ndarray], list[str], dict[str, torch.Tensor]]:
    """Frames of: grow from the seed, hold, get damaged, regenerate, hold."""
    torch.manual_seed(seed)
    model, _ = load_model(ROOT / "runs" / f"{name}_regenerating" / "model.pt")
    model.to(device)
    h = w = 72
    x = make_seed(1, model.channels, h, w, device)
    frames, stages, snaps = [], [], {}

    def emit(stage: str) -> None:
        frames.append(upscale(to_rgb(x)[0], scale))
        stages.append(stage)

    emit("grow")
    for t in range(1, grow + 1):
        x = model(x)
        if t % stride == 0:
            emit("grow")
        if t in (10, 30, 60, 130):
            snaps[f"grow{t}"] = to_rgb(x)[0].cpu()
    for _ in range(8):
        emit("hold")
    cy, cx = centroid(x)
    x = x * (1.0 - disc(h, w, cy, cx, 11.0).to(device))[None, None]
    snaps["damaged"] = to_rgb(x)[0].cpu()
    for _ in range(10):
        emit("damage")
    for t in range(1, regen + 1):
        x = model(x)
        if t % stride == 0:
            emit("regenerate")
        if t in (10, 40, 140):
            snaps[f"regen{t}"] = to_rgb(x)[0].cpu()
    for _ in range(14):
        emit("hold")
    return frames, stages, snaps


def hero(device: torch.device, names: list[str], scale: int = 3, stride: int = 2, fps: int = 20) -> dict:
    runs = [growth_and_regeneration(n, device, scale, stride) for n in names]
    n_frames = min(len(r[0]) for r in runs)
    width = sum(r[0][0].shape[1] for r in runs)
    out = []
    for i in range(n_frames):
        stage = runs[0][1][i]
        text = {"grow": "growing from a single cell", "hold": "", "damage": "damaged: a disc of cells is erased",
                "regenerate": "regenerating"}[stage]
        if stage == "hold":
            text = "grown pattern" if i < n_frames // 2 else "regenerated pattern"
        out.append(np.concatenate([caption_bar(width, text),
                                   np.concatenate([r[0][i] for r in runs], axis=1)], axis=0))
    save_gif(out, ROOT / "assets" / "hero.gif", fps)
    return {n: r[2] for n, r in zip(names, runs)}


def filmstrip(snaps: dict[str, dict[str, torch.Tensor]], names: list[str]) -> None:
    keys = ["grow10", "grow30", "grow60", "grow130", "damaged", "regen10", "regen40", "regen140"]
    labels = ["step 10", "step 30", "step 60", "step 130", "damaged", "+10 steps", "+40 steps", "+140 steps"]
    scale = 3
    rows = []
    for n in names:
        rows.append(np.concatenate([upscale(snaps[n][k], scale) for k in keys], axis=1))
    width = rows[0].shape[1]
    sheet = np.concatenate([label_bar(width, labels)] + rows, axis=0)
    Image.fromarray(sheet).save(ROOT / "assets" / "filmstrip.png", optimize=True)
    print("wrote filmstrip.png", sheet.shape)


@torch.no_grad()
def targets_sheet(names: list[str], device: torch.device) -> None:
    """Row 1: targets. Row 2: pattern grown for 200 steps by the regenerating model."""
    top, bottom = [], []
    for n in names:
        model, targs = load_model(ROOT / "runs" / f"{n}_regenerating" / "model.pt")
        model.to(device)
        tgt = pad_target(load_emoji(targs["target"], targs["size"]), targs["padding"])
        torch.manual_seed(0)
        x = make_seed(1, model.channels, 72, 72, device)
        x = model.rollout(x, 200)
        top.append(upscale(to_rgb(tgt[None])[0], 3))
        bottom.append(upscale(to_rgb(x)[0], 3))
    width = sum(t.shape[1] for t in top)
    sheet = np.concatenate([label_bar(width, names), np.concatenate(top, 1), np.concatenate(bottom, 1)], 0)
    Image.fromarray(sheet).save(ROOT / "assets" / "targets_vs_grown.png", optimize=True)
    print("wrote targets_vs_grown.png", sheet.shape)


@torch.no_grad()
def textures_gif(device: torch.device, size: int = 96, scale: int = 2, fps: int = 20) -> None:
    names = [n for n in ("coral", "spots", "cells") if (ROOT / "runs" / f"texture_{n}" / "model.pt").exists()]
    if not names:
        return
    torch.manual_seed(1)
    models = [load_model(ROOT / "runs" / f"texture_{n}" / "model.pt")[0].to(device) for n in names]
    xs = [noise_state(1, m.channels, size, device) for m in models]
    frames = []
    schedule = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    steps_done = 0
    caption = "noise to texture"
    for target_step in schedule + list(range(160, 481, 32)):
        while steps_done < target_step:
            xs = [m(x) for m, x in zip(models, xs)]
            steps_done += 1
        frames.append(np.concatenate([upscale(texture_rgb(x)[0], scale) for x in xs], 1))
    # Damage: replace a disc with fresh noise and let the texture heal.
    cy = cx = size // 2
    d = disc(size, size, cy, cx, size * 0.2).to(device)
    xs = [x * (1 - d) + noise_state(1, x.shape[1], size, device) * d for x in xs]
    n_before = len(frames)
    frames.append(np.concatenate([upscale(texture_rgb(x)[0], scale) for x in xs], 1))
    for t in range(1, 97):
        xs = [m(x) for m, x in zip(models, xs)]
        if t in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96):
            frames.append(np.concatenate([upscale(texture_rgb(x)[0], scale) for x in xs], 1))
    width = frames[0].shape[1]
    out = []
    for i, f in enumerate(frames):
        text = "noise grows into the texture" if i < n_before else "centre replaced by noise, then self-repair"
        out.append(np.concatenate([caption_bar(width, text), f], 0))
    out = out[:1] * 6 + out + out[-1:] * 14
    save_gif(out, ROOT / "assets" / "textures.gif", fps=fps // 2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    (ROOT / "assets").mkdir(exist_ok=True)
    names = ["lizard", "butterfly", "octopus", "ladybug"]
    snaps = hero(device, names)
    filmstrip(snaps, ["lizard", "butterfly", "octopus", "ladybug"])
    targets_sheet(["lizard", "butterfly", "mushroom", "octopus", "ladybug"], device)
    textures_gif(device)


if __name__ == "__main__":
    main()
