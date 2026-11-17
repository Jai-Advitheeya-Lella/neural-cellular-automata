"""Train a growing NCA on an emoji target.

Three regimes from the Distill article:

* ``growing``      every batch starts from the single-cell seed (no pool);
* ``persistent``   sample-pool training, so the rule also learns to *maintain* the pattern;
* ``regenerating`` sample-pool training where the best samples are additionally
  damaged with random circular cut-outs, so the rule learns to repair.

    python -m nca.train --target lizard --mode regenerating
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from nca.model import NCA, make_seed, to_rgba
from nca.pool import SamplePool, circle_masks
from nca.targets import load_emoji, pad_target


def normalize_grads(model: torch.nn.Module) -> None:
    """Per-parameter gradient normalisation g <- g / (||g|| + 1e-8), as in the article."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(p.grad.norm() + 1e-8)


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    target = pad_target(load_emoji(args.target, args.size), args.padding).to(device)
    _, h, w = target.shape
    model = NCA(channels=args.channels, hidden=args.hidden, fire_rate=args.fire_rate).to(device)
    step_fn = torch.compile(model) if args.compile and device.type == "cuda" else model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[args.lr_drop], gamma=0.1)

    seed = make_seed(1, args.channels, h, w, device)
    use_pool = args.mode in ("persistent", "regenerating")
    pool = SamplePool(seed.expand(args.pool_size, -1, -1, -1).clone()) if use_pool else None

    def per_sample_loss(x: torch.Tensor) -> torch.Tensor:
        return (to_rgba(x) - target).pow(2).mean(dim=(1, 2, 3))

    out_dir = Path(args.out_dir) / f"{args.target}_{args.mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[tuple[int, float, float]] = []
    t0 = time.time()
    for it in range(args.iters + 1):
        if pool is not None:
            idx, x0 = pool.sample(args.batch_size)
            with torch.no_grad():
                order = per_sample_loss(x0).argsort(descending=True)
            idx, x0 = idx[order], x0[order]
            x0[:1] = seed  # replace the worst sample with a fresh seed
            if args.mode == "regenerating" and args.damage_n > 0:
                masks = circle_masks(args.damage_n, h, w, device=device)
                x0[-args.damage_n:] *= (1.0 - masks)[:, None]
        else:
            idx, x0 = None, seed.expand(args.batch_size, -1, -1, -1).clone()

        n_steps = int(np.random.randint(args.min_steps, args.max_steps + 1))
        x = x0
        for _ in range(n_steps):
            fire = torch.rand_like(x[:, :1]) <= args.fire_rate
            # Recompute each step's activations in the backward pass (O(steps) memory in the state only).
            x = checkpoint(step_fn, x, None, 0.0, fire, use_reentrant=False)
        losses = per_sample_loss(x)
        loss = losses.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        normalize_grads(model)
        opt.step()
        sched.step()

        if pool is not None:
            pool.commit(idx, x.detach())
        log_rows.append((it, float(loss.detach()), n_steps))
        if it % 500 == 0:
            print(f"[{args.target}/{args.mode}] it={it:5d} loss={log_rows[-1][1]:.5f} "
                  f"log10={np.log10(log_rows[-1][1]):.3f} {time.time() - t0:.0f}s", flush=True)

    train_time = time.time() - t0
    torch.save({"model": model.state_dict(), "config": model.config(), "args": vars(args)}, out_dir / "model.pt")
    with open(out_dir / "loss.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["iter", "loss", "steps"])
        wr.writerows(log_rows)
    (out_dir / "summary.json").write_text(json.dumps({
        "target": args.target, "mode": args.mode, "iters": args.iters, "train_time_s": train_time,
        "final_loss_avg_last100": float(np.mean([r[1] for r in log_rows[-100:]])),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }, indent=2))
    print(f"done in {train_time:.0f}s -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="lizard")
    p.add_argument("--mode", choices=["growing", "persistent", "regenerating"], default="regenerating")
    p.add_argument("--size", type=int, default=40, help="target size in cells")
    p.add_argument("--padding", type=int, default=16)
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--fire-rate", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--pool-size", type=int, default=1024)
    p.add_argument("--damage-n", type=int, default=3)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--min-steps", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-drop", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--out-dir", default="runs")
    return p


def main(argv: list[str] | None = None) -> None:
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
