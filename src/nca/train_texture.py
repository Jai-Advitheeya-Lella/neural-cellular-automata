"""Train a texture-synthesis NCA with a VGG Gram-matrix style loss (Niklasson et al., Distill 2021).

The cell grid is a torus (circular padding), perception adds a Laplacian to identity + Sobel, there is
no alive masking, and training starts from uniform noise. Because the rule is local and the world is
periodic, the trained texture can be synthesised on a grid larger than the training grid.

    python -m nca.train_texture --texture coral
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nca.model import NCA
from nca.style import StyleLoss
from nca.targets import make_texture
from nca.train import normalize_grads


def noise_state(n: int, channels: int, size: int, device: torch.device | str,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform noise in [-0.5, 0.5] in every channel: the initial state of the texture NCA."""
    return torch.rand(n, channels, size, size, device=device, generator=generator) - 0.5


def texture_rgb(x: torch.Tensor) -> torch.Tensor:
    """RGB view of a texture NCA state: channels 0-2 shifted to [0, 1] and clamped."""
    return (x[:, :3] + 0.5).clamp(0.0, 1.0)


def overflow_loss(x: torch.Tensor) -> torch.Tensor:
    """Mean absolute amount by which any channel leaves [-1, 1] (keeps the state bounded)."""
    return (x - x.clamp(-1.0, 1.0)).abs().mean()


def load_target(name: str, size: int) -> torch.Tensor:
    """Procedural texture resized to ``size`` x ``size`` with area filtering."""
    t = make_texture(name, 128)
    return F.interpolate(t[None], size=(size, size), mode="area")[0]


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    target = load_target(args.texture, args.size).to(device)
    style = StyleLoss(target).to(device)
    model = NCA(channels=args.channels, hidden=args.hidden, fire_rate=args.fire_rate, laplacian=True,
                alive_masking=False, padding="circular").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.iters * 0.6)], gamma=0.2)
    pool = noise_state(args.pool_size, args.channels, args.size, device)

    out_dir = Path(args.out_dir) / f"texture_{args.texture}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[int, float, float, int]] = []
    t0 = time.time()
    for it in range(args.iters + 1):
        idx = torch.randperm(args.pool_size, device=device)[: args.batch_size]
        x = pool[idx].clone()
        x[:1] = noise_state(1, args.channels, args.size, device)  # always train on one fresh start
        n_steps = int(np.random.randint(args.min_steps, args.max_steps + 1))
        for _ in range(n_steps):
            fire = torch.rand_like(x[:, :1]) <= args.fire_rate
            x = checkpoint(model, x, None, 0.0, fire, use_reentrant=False)
        s_loss = style(texture_rgb(x)).mean()
        o_loss = overflow_loss(x)
        loss = s_loss + args.overflow_weight * o_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        sched.step()
        pool[idx] = x.detach()
        rows.append((it, float(s_loss.detach()), float(o_loss.detach()), n_steps))
        if it % 250 == 0:
            print(f"[{args.texture}] it={it:5d} style={rows[-1][1]:.4f} overflow={rows[-1][2]:.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)

    train_time = time.time() - t0
    torch.save({"model": model.state_dict(), "config": model.config(), "args": vars(args)}, out_dir / "model.pt")
    with open(out_dir / "loss.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "style_loss", "overflow_loss", "steps"])
        w.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "texture": args.texture, "iters": args.iters, "train_time_s": train_time,
        "final_style_loss_avg_last100": float(np.mean([r[1] for r in rows[-100:]])),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }, indent=2))
    print(f"done in {train_time:.0f}s -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--texture", default="coral", choices=["coral", "spots", "cells"])
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--channels", type=int, default=12)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--fire-rate", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=256)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--min-steps", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--overflow-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="runs")
    return p


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
