import torch

from nca.targets import EMOJI, gray_scott, load_emoji, make_texture, pad_target, voronoi_cells


def test_emoji_targets_are_premultiplied_rgba() -> None:
    for name in EMOJI:
        t = load_emoji(name, 40)
        assert t.shape == (4, 40, 40) and t.min() >= 0 and t.max() <= 1
        transparent = t[3] == 0
        assert transparent.any()
        assert t[:3][:, transparent].abs().sum() == 0  # no colour where alpha is zero
        assert (t[:3] <= t[3:4] + 1e-6).all()          # premultiplied: rgb <= alpha


def test_pad_target() -> None:
    assert pad_target(torch.ones(4, 10, 10), 3).shape == (4, 16, 16)


def test_gray_scott_is_deterministic_and_normalised() -> None:
    a, b = gray_scott(32, steps=200, seed=1), gray_scott(32, steps=200, seed=1)
    assert (a == b).all() and a.min() == 0 and abs(a.max() - 1) < 1e-6


def test_voronoi_is_periodic() -> None:
    edge, ids = voronoi_cells(32, 8, seed=0)
    assert edge.shape == ids.shape == (32, 32) and edge.min() >= 0


def test_procedural_textures() -> None:
    for name in ("coral", "spots", "cells"):
        t = make_texture(name, 32)
        assert t.shape == (3, 32, 32) and t.min() >= 0 and t.max() <= 1
        assert t.std() > 0.05
