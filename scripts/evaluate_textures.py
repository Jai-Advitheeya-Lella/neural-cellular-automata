"""Evaluate the texture-synthesis NCAs: Gram-matrix style distance to the target and a comparison figure.

For each texture the NCA is run for 200 steps from uniform noise on a 64x64 torus (the training size)
and on a 128x128 torus (larger than anything seen in training). Style distance is the same relative Gram loss
used for training, measured on 16 independent runs, against two references: the noise initial state (upper
reference) and a random 64x64 crop of the 128x128 target itself (a lower reference: how far two views of the
*real* texture are from the 64x64 target statistics).

    python scripts/evaluate_textures.py        # writes assets/textures.png, assets/texture_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from nca.export import load_model  # noqa: E402
from nca.style import StyleLoss  # noqa: E402
from nca.targets import make_texture  # noqa: E402
from nca.train_texture import load_target, noise_state, texture_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEPS = 200


def load_target_seed(name: str, size: int, seed: int) -> torch.Tensor:
    """A different sample of the same procedural texture, area-downsampled like the training target."""
    return torch.nn.functional.interpolate(make_texture(name, 128, seed=seed)[None], size=(size, size), mode="area")[0]


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "text.color": "#0b0b0b"})
    names = ["coral", "spots", "cells"]
    fig, axes = plt.subplots(len(names), 4, figsize=(9, 2.3 * len(names)), constrained_layout=True)
    metrics = {}
    for r, name in enumerate(names):
        model = load_model(ROOT / "runs" / f"texture_{name}" / "model.pt")[0].to(dev)
        target = load_target(name, 64).to(dev)
        style = StyleLoss(target).to(dev)
        full = make_texture(name, 128).to(dev)
        x64 = noise_state(16, model.channels, 64, dev)
        noise_loss = float(style(texture_rgb(x64)).mean())
        out64 = model.rollout(x64, STEPS)
        loss64 = float(style(texture_rgb(out64)).mean())
        x128 = noise_state(1, model.channels, 128, dev)
        out128 = model.rollout(x128, STEPS)
        others = torch.stack([load_target_seed(name, 64, s).to(dev) for s in range(100, 104)])
        other_loss = float(style(others).mean())
        # 128x128 output evaluated on 64x64 tiles against the 64x64 target statistics.
        tiles = torch.stack([texture_rgb(out128)[0][:, i:i + 64, j:j + 64] for i in (0, 64) for j in (0, 64)])
        loss128 = float(style(tiles).mean())
        metrics[name] = {"noise_init": noise_loss, "nca_64x64": loss64, "nca_128x128_tiles": loss128,
                         "other_samples_of_same_texture": other_loss}
        print(name, {k: round(v, 4) for k, v in metrics[name].items()})
        panels = [("target (128px)", full), ("NCA, 64x64 torus", texture_rgb(out64[:1])[0]),
                  ("NCA, 128x128 torus", texture_rgb(out128)[0]), ("noise init", texture_rgb(x64[:1])[0])]
        for ax, (title, img) in zip(axes[r], panels):
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy(), interpolation="nearest")
            ax.set_title(title if r == 0 else "", fontsize=9)
            ax.axis("off")
        axes[r][0].text(-0.05, 0.5, name, transform=axes[r][0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.savefig(ROOT / "assets" / "textures.png", dpi=130)
    (ROOT / "assets" / "texture_metrics.json").write_text(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
