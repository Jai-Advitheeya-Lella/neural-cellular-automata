"""Quantitative evaluation of the trained growing NCAs (long-horizon stability and regeneration).

For every ``runs/<target>_<mode>`` model this grows the pattern from the single-cell seed with
independent stochastic updates (``--trials`` parallel trials), records the RGBA mean-squared error
against the target at several horizons (training used 64-96 steps), then damages the grown pattern
and measures the error again after letting it regenerate.

    python scripts/evaluate.py            # writes assets/metrics.json, assets/metrics.md, assets/stability.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from nca.export import load_model  # noqa: E402
from nca.model import make_seed, to_rgba  # noqa: E402
from nca.pool import circle_masks  # noqa: E402
from nca.targets import load_emoji, pad_target  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = (96, 200, 500, 1000, 2000)
GROW_STEPS = 200      # pattern is grown for this long before damage
REGEN_STEPS = 150     # recovery time after damage
BLUE, ORANGE, GREEN, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def mse(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (to_rgba(x) - target).pow(2).mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(run: Path, device: torch.device, trials: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model, targs = load_model(run / "model.pt")
    model.to(device)
    target = pad_target(load_emoji(targs["target"], targs["size"]), targs["padding"]).to(device)
    _, h, w = target.shape
    x = make_seed(trials, model.channels, h, w, device)

    curve: list[float] = []
    horizon_mse: dict[int, float] = {}
    grown = None
    for t in range(1, max(HORIZONS) + 1):
        x = model(x)
        m = mse(x, target)
        curve.append(float(m.mean()))
        if t in HORIZONS:
            horizon_mse[t] = float(m.mean())
        if t == GROW_STEPS:
            grown = x.clone()
    assert grown is not None

    results = {}
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    kinds = {
        "half": lambda n: torch.cat([torch.zeros(n, h, w // 2), torch.ones(n, h, w - w // 2)], 2),  # right half destroyed
        "disc": lambda n: circle_masks(n, h, w, generator=g),
    }
    for kind, make_mask in kinds.items():
        mask = make_mask(trials).to(device)
        y = grown * (1.0 - mask)[:, None]
        after_damage = float(mse(y, target).mean())
        for _ in range(REGEN_STEPS):
            y = model(y)
        results[kind] = {"mse_after_damage": after_damage, "mse_after_regeneration": float(mse(y, target).mean()),
                         "alive_cells_after": float((y[:, 3] > 0.1).float().sum((1, 2)).mean())}
    return {"target": targs["target"], "mode": targs["mode"], "mse_at_step": horizon_mse,
            "mse_curve": curve, "mse_grown_at_200": float(mse(grown, target).mean()),
            "damage": results, "trials": trials,
            "baseline_mse_empty_canvas": float(target.pow(2).mean())}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=16)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    runs = sorted(r for r in (ROOT / "runs").glob("*_*") if (r / "model.pt").exists() and not r.name.startswith("texture_"))
    out = {}
    for run in runs:
        try:
            out[run.name] = evaluate(run, device, args.trials)
        except torch.cuda.OutOfMemoryError:
            out[run.name] = evaluate(run, torch.device("cpu"), args.trials)
        r = out[run.name]
        print(run.name, {k: f"{v:.5f}" for k, v in r["mse_at_step"].items()}, flush=True)
    (ROOT / "assets").mkdir(exist_ok=True)
    (ROOT / "assets" / "metrics.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "mse_curve"} for k, v in out.items()}, indent=1))

    lines = ["| Model (target, training regime) | " + " | ".join(f"MSE @{t}" for t in HORIZONS) +
             " | MSE after half cut + 150 steps | MSE after random disc cut + 150 steps | empty-canvas MSE |",
             "|---|" + "---|" * (len(HORIZONS) + 3)]
    for name, r in out.items():
        lines.append(f"| {r['target']}, {r['mode']} | " + " | ".join(f"{r['mse_at_step'][t]:.5f}" for t in HORIZONS)
                     + f" | {r['damage']['half']['mse_after_regeneration']:.5f}"
                     f" | {r['damage']['disc']['mse_after_regeneration']:.5f} | {r['baseline_mse_empty_canvas']:.4f} |")
    (ROOT / "assets" / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Stability of the three training regimes on the lizard.
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    for mode, color, label in (("growing", ORANGE, "growing (no pool)"), ("persistent", BLUE, "persistent (pool)"),
                               ("regenerating", GREEN, "regenerating (pool + damage)")):
        key = f"lizard_{mode}"
        if key in out:
            ax.plot(np.arange(1, len(out[key]["mse_curve"]) + 1), out[key]["mse_curve"], color=color, lw=2, label=label)
    ax.axvspan(64, 96, color=GRID, alpha=0.8, lw=0)
    ax.text(80, 0.03, "training\nhorizon", ha="center", va="bottom", fontsize=8, color=MUTED,
            transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("CA steps from the seed cell")
    ax.set_ylabel("RGBA MSE vs target (mean of 16 trials)")
    ax.set_title("Lizard: error vs rollout length for the three training regimes", fontsize=11, loc="left")
    ax.legend(loc="center right")
    fig.savefig(ROOT / "assets" / "stability.png", dpi=150)


if __name__ == "__main__":
    main()
