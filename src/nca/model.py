"""Neural Cellular Automata (Mordvintsev et al., Distill 2020).

A cell grid ``x`` of shape ``(B, C, H, W)`` is updated by a rule shared by every
cell. Each step:

1. *Perception*: every channel is convolved with fixed 3x3 filters (identity and
   Sobel gradients, plus a Laplacian for the texture model). This is the only
   way information moves between cells.
2. *Update*: a per-cell MLP (two 1x1 convolutions) maps the perception vector to
   a residual ``dx``. The last layer is zero-initialised so training starts from
   the "do nothing" rule.
3. *Stochastic update*: each cell applies ``dx`` independently with probability
   ``fire_rate``, which removes the need for a global clock.
4. *Alive masking* (growing model only): a cell is alive if any cell in its 3x3
   neighbourhood has alpha > 0.1, both before and after the update; dead cells
   are zeroed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

IDENTITY = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
SOBEL_Y = SOBEL_X.T.contiguous()
LAPLACIAN = torch.tensor([[1.0, 2.0, 1.0], [2.0, -12.0, 2.0], [1.0, 2.0, 1.0]]) / 16.0


def perception_filters(angle: float = 0.0, laplacian: bool = False) -> Tensor:
    """Stack of 3x3 filters ``(F, 3, 3)``: identity, rotated Sobel-x, rotated Sobel-y[, Laplacian].

    Rotating the gradient filters by ``angle`` rotates the frame in which cells
    perceive their neighbourhood, which rotates the grown pattern (Distill 2020, Fig. "rotation").
    """
    c, s = math.cos(angle), math.sin(angle)
    dx = c * SOBEL_X - s * SOBEL_Y
    dy = s * SOBEL_X + c * SOBEL_Y
    filters = [IDENTITY, dx, dy] + ([LAPLACIAN] if laplacian else [])
    return torch.stack(filters)


def depthwise_kernel(filters: Tensor, channels: int) -> Tensor:
    """Grouped-conv weight ``(C*F, 1, 3, 3)`` with outputs interleaved as [c0f0, c0f1, ..., c1f0, ...]."""
    return filters.repeat(channels, 1, 1).unsqueeze(1)


def alive_mask(x: Tensor, threshold: float = 0.1) -> Tensor:
    """``(B, 1, H, W)`` boolean: some cell in the 3x3 neighbourhood has alpha (channel 3) > threshold."""
    return F.max_pool2d(x[:, 3:4], kernel_size=3, stride=1, padding=1) > threshold


class NCA(nn.Module):
    """Growing / texture Neural Cellular Automaton.

    Args:
        channels: State channels per cell. Channels 0-3 are RGBA for the growing
            model (RGB for the texture model), the rest are hidden.
        hidden: Width of the per-cell update MLP.
        fire_rate: Probability that a cell applies its update in a step.
        laplacian: Add a Laplacian filter to the perception (texture model).
        alive_masking: Apply the alpha-based alive mask (growing model).
        padding: ``"zeros"`` (growing; the world has edges) or ``"circular"``
            (texture; a torus, so textures tile seamlessly).
        update_bias: Whether the final 1x1 conv has a bias.
    """

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 128,
        fire_rate: float = 0.5,
        laplacian: bool = False,
        alive_masking: bool = True,
        padding: str = "zeros",
        update_bias: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.laplacian = laplacian
        self.alive_masking = alive_masking
        self.padding = padding
        n_filters = 4 if laplacian else 3
        self.register_buffer("kernel", depthwise_kernel(perception_filters(0.0, laplacian), channels))
        self.w1 = nn.Conv2d(channels * n_filters, hidden, kernel_size=1)
        self.w2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=update_bias)
        nn.init.zeros_(self.w2.weight)
        if self.w2.bias is not None:
            nn.init.zeros_(self.w2.bias)

    def config(self) -> dict:
        return {"channels": self.channels, "hidden": self.w1.out_channels, "fire_rate": self.fire_rate,
                "laplacian": self.laplacian, "alive_masking": self.alive_masking,
                "padding": self.padding, "update_bias": self.w2.bias is not None}

    def perceive(self, x: Tensor, angle: float = 0.0) -> Tensor:
        kernel = self.kernel
        if angle != 0.0:
            kernel = depthwise_kernel(perception_filters(angle, self.laplacian), self.channels).to(x)
        if self.padding == "circular":
            x = F.pad(x, (1, 1, 1, 1), mode="circular")
            return F.conv2d(x, kernel, groups=self.channels)
        return F.conv2d(x, kernel, padding=1, groups=self.channels)

    def forward(
        self,
        x: Tensor,
        fire_rate: float | None = None,
        angle: float = 0.0,
        update_mask: Tensor | None = None,
    ) -> Tensor:
        """One CA step.

        Args:
            x: State ``(B, C, H, W)``.
            fire_rate: Override the stochastic update probability (1.0 = synchronous).
            angle: Perception rotation in radians.
            update_mask: Explicit ``(B, 1, H, W)`` mask of cells that fire; used to
                compare against the JavaScript runner with identical randomness.
        """
        pre_alive = alive_mask(x) if self.alive_masking else None
        dx = self.w2(F.relu(self.w1(self.perceive(x, angle))))
        if update_mask is None:
            rate = self.fire_rate if fire_rate is None else fire_rate
            update_mask = torch.rand_like(x[:, :1]) <= rate
        x = x + dx * update_mask.to(x.dtype)
        if self.alive_masking:
            life = pre_alive & alive_mask(x)
            x = x * life.to(x.dtype)
        return x

    def rollout(self, x: Tensor, steps: int, **kwargs) -> Tensor:
        for _ in range(steps):
            x = self(x, **kwargs)
        return x


def make_seed(batch: int, channels: int, height: int, width: int, device: torch.device | str = "cpu") -> Tensor:
    """All-zero grid with a single live cell in the centre (channels 3: set to 1)."""
    x = torch.zeros(batch, channels, height, width, device=device)
    x[:, 3:, height // 2, width // 2] = 1.0
    return x


def to_rgba(x: Tensor) -> Tensor:
    return x[:, :4]


def to_rgb(x: Tensor) -> Tensor:
    """Composite the (premultiplied) RGBA state over a white background, clipped to [0, 1]."""
    rgb, a = x[:, :3], x[:, 3:4].clamp(0.0, 1.0)
    return (1.0 - a + rgb).clamp(0.0, 1.0)
