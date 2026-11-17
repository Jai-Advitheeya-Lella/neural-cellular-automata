"""Sample pool and damage masks for persistent / regenerating training."""

from __future__ import annotations

import torch
from torch import Tensor


class SamplePool:
    """Fixed-size pool of CA states living on the training device.

    Training samples a batch, runs the CA, and writes the results back, so the
    pool gradually fills with states from long rollouts. This teaches the rule
    to keep the pattern stable well past the training horizon.
    """

    def __init__(self, states: Tensor) -> None:
        self.states = states

    def __len__(self) -> int:
        return self.states.shape[0]

    def sample(self, n: int) -> tuple[Tensor, Tensor]:
        idx = torch.randperm(len(self), device=self.states.device)[:n]
        return idx, self.states[idx].clone()

    def commit(self, idx: Tensor, states: Tensor) -> None:
        self.states[idx] = states


def circle_masks(n: int, h: int, w: int, device: torch.device | str = "cpu",
                 generator: torch.Generator | None = None) -> Tensor:
    """``n`` binary masks ``(n, h, w)`` with a random disc set to 1 (the region to destroy).

    Centres are uniform in the central half of the grid and radii uniform in
    [0.1, 0.4] of the half-width, in normalised [-1, 1] coordinates (as in the article).
    """
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    center = torch.rand(2, n, 1, 1, device=device, generator=generator) - 0.5
    r = 0.3 * torch.rand(n, 1, 1, device=device, generator=generator) + 0.1
    return (((x - center[0]) ** 2 + (y - center[1]) ** 2) < r**2).float()
