"""Target images: Noto Emoji PNGs (Apache-2.0) and procedurally generated textures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: Emoji used as growth targets, keyed by a short name -> Unicode code point.
EMOJI: dict[str, str] = {
    "lizard": "1f98e",
    "butterfly": "1f98b",
    "mushroom": "1f344",
    "octopus": "1f419",
    "ladybug": "1f41e",
    "cactus": "1f335",
    "fish": "1f420",
}


def load_emoji(name: str, size: int = 40) -> Tensor:
    """Load an emoji as a premultiplied RGBA tensor ``(4, size, size)`` in [0, 1].

    The 128px Noto PNG is downsampled with a Lanczos filter, like the 40px targets
    of the original article. RGB is premultiplied by alpha so that transparent
    pixels are exactly zero in every channel.
    """
    path = DATA_DIR / "emoji" / f"emoji_u{EMOJI[name]}.png"
    img = Image.open(path).convert("RGBA")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    h, w = arr.shape[:2]
    oy, ox = (size - h) // 2, (size - w) // 2
    canvas[oy:oy + h, ox:ox + w] = arr
    canvas[..., :3] *= canvas[..., 3:4]
    return torch.from_numpy(canvas).permute(2, 0, 1).contiguous()


def pad_target(target: Tensor, padding: int) -> Tensor:
    return torch.nn.functional.pad(target, (padding,) * 4)


# --------------------------------------------------------------------------- textures
def _hsv_palette(t: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Piecewise-linear colour map from scalar ``t`` in [0, 1] to RGB in [0, 1]."""
    xs = np.array([s for s, _ in stops])
    cols = np.array([c for _, c in stops], dtype=np.float32) / 255.0
    return np.stack([np.interp(t, xs, cols[:, k]) for k in range(3)], axis=-1)


def gray_scott(size: int = 128, steps: int = 12000, feed: float = 0.0545, kill: float = 0.062,
               seed: int = 0) -> np.ndarray:
    """Gray-Scott reaction-diffusion on a torus; returns the V concentration in [0, 1].

    Default (F, k) sits in the "coral / labyrinth" regime. Integrated with forward
    Euler on a 5-point Laplacian with ``D_u = 0.16, D_v = 0.08``.
    """
    rng = np.random.default_rng(seed)
    u = np.ones((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    for _ in range(20):
        cy, cx = rng.integers(0, size, 2)
        r = rng.integers(3, 7)
        yy, xx = np.ogrid[:size, :size]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
        u[m], v[m] = 0.5, 0.25
    lap = lambda z: np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1) - 4 * z
    for _ in range(steps):
        uvv = u * v * v
        u += 0.16 * lap(u) - uvv + feed * (1 - u)
        v += 0.08 * lap(v) + uvv - (feed + kill) * v
    return (v - v.min()) / (v.max() - v.min() + 1e-8)


def voronoi_cells(size: int = 128, n_points: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Periodic Voronoi diagram: per-pixel (distance to border, cell id)."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n_points, 2)) * size
    yy, xx = np.mgrid[:size, :size].astype(np.float64)
    d = np.full((n_points, size, size), np.inf)
    for i, (py, px) in enumerate(pts):
        dy = np.abs(yy - py)
        dx = np.abs(xx - px)
        dy = np.minimum(dy, size - dy)
        dx = np.minimum(dx, size - dx)
        d[i] = np.sqrt(dy**2 + dx**2)
    order = np.sort(d, axis=0)
    return order[1] - order[0], np.argmin(d, axis=0)


def make_texture(name: str, size: int = 128, seed: int | None = None) -> Tensor:
    """Procedural texture ``(3, size, size)`` in [0, 1] (tileable, generated on the fly).

    ``seed=None`` gives the canonical training target; other seeds give new samples of the same
    texture family (used to measure natural sample-to-sample variability).
    """
    if name == "coral":
        v = gray_scott(size, feed=0.0545, kill=0.062, seed=0 if seed is None else seed)
        rgb = _hsv_palette(v, [(0.0, (12, 30, 60)), (0.45, (20, 110, 140)),
                               (0.7, (250, 140, 90)), (1.0, (255, 235, 200))])
    elif name == "spots":
        v = gray_scott(size, feed=0.035, kill=0.065, seed=1 if seed is None else seed)
        rgb = _hsv_palette(v, [(0.0, (245, 222, 170)), (0.35, (215, 160, 80)),
                               (0.7, (90, 50, 20)), (1.0, (30, 15, 5))])
    elif name == "cells":
        edge, ids = voronoi_cells(size, seed=0 if seed is None else seed)
        rng = np.random.default_rng(3 if seed is None else seed + 3)
        hues = rng.random(ids.max() + 1)
        base = _hsv_palette(hues[ids], [(0.0, (90, 170, 80)), (0.5, (150, 200, 70)), (1.0, (60, 130, 90))])
        shade = np.clip(edge / 6.0, 0, 1)[..., None]
        rgb = base * (0.25 + 0.75 * shade)
    else:
        raise KeyError(name)
    return torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).contiguous()


TEXTURES = ("coral", "spots", "cells")
