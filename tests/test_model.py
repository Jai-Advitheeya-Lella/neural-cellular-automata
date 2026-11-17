import math

import pytest
import torch
import torch.nn.functional as F

from nca.model import (IDENTITY, NCA, SOBEL_X, SOBEL_Y, alive_mask, depthwise_kernel, make_seed,
                       perception_filters, to_rgb)
from nca.pool import SamplePool, circle_masks


def test_identity_and_sobel_filters() -> None:
    f = perception_filters()
    assert f.shape == (3, 3, 3)
    torch.testing.assert_close(f[0], IDENTITY)
    # Gradient filters annihilate constants and have unit response to a unit ramp.
    assert f[1].sum().abs() < 1e-7 and f[2].sum().abs() < 1e-7
    ramp_x = torch.arange(8.0).expand(8, 8)[None, None]
    out = F.conv2d(ramp_x, f[1][None, None])
    torch.testing.assert_close(out, torch.ones_like(out))
    out_y = F.conv2d(ramp_x.transpose(2, 3), f[2][None, None])
    torch.testing.assert_close(out_y, torch.ones_like(out_y))


def test_laplacian_filter_annihilates_constants() -> None:
    f = perception_filters(laplacian=True)
    assert f.shape == (4, 3, 3)
    assert f[3].sum().abs() < 1e-7


def test_rotated_sobel_is_a_rotation_of_the_gradient() -> None:
    f0, f90, f180 = (perception_filters(a) for a in (0.0, math.pi / 2, math.pi))
    torch.testing.assert_close(f90[1], -SOBEL_Y, atol=1e-6, rtol=0)
    torch.testing.assert_close(f90[2], SOBEL_X, atol=1e-6, rtol=0)
    torch.testing.assert_close(f180[1], -f0[1], atol=1e-6, rtol=0)
    # The rotation is orthogonal: gradient magnitude is unchanged for any angle.
    img = torch.randn(1, 1, 9, 9)
    for angle in (0.3, 1.1, 2.5):
        f = perception_filters(angle)
        g0 = torch.stack([F.conv2d(img, perception_filters()[k][None, None]) for k in (1, 2)])
        g = torch.stack([F.conv2d(img, f[k][None, None]) for k in (1, 2)])
        torch.testing.assert_close(g0.pow(2).sum(0), g.pow(2).sum(0), atol=1e-5, rtol=1e-5)


def test_depthwise_perception_matches_per_channel_convolution() -> None:
    torch.manual_seed(0)
    model = NCA(channels=5, hidden=8)
    x = torch.randn(2, 5, 10, 10)
    p = model.perceive(x)
    assert p.shape == (2, 15, 10, 10)
    filters = perception_filters()
    for c in range(5):
        for k in range(3):
            ref = F.conv2d(x[:, c:c + 1], filters[k][None, None], padding=1)
            torch.testing.assert_close(p[:, 3 * c + k:3 * c + k + 1], ref, atol=1e-6, rtol=0)
    assert depthwise_kernel(filters, 5).shape == (15, 1, 3, 3)


def test_last_layer_is_zero_initialised_so_rule_starts_as_identity() -> None:
    model = NCA(channels=16, hidden=32, alive_masking=False)
    assert torch.all(model.w2.weight == 0) and torch.all(model.w2.bias == 0)
    x = torch.randn(2, 16, 12, 12)
    torch.testing.assert_close(model(x), x)


def test_alive_mask_is_3x3_max_pool_of_alpha() -> None:
    x = torch.zeros(1, 4, 7, 7)
    x[0, 3, 3, 3] = 0.5
    x[0, 3, 0, 0] = 0.05  # below threshold
    m = alive_mask(x)[0, 0]
    assert m.sum() == 9 and m[2:5, 2:5].all()
    assert not m[0, 0]


def test_alive_masking_kills_cells_without_a_live_neighbour() -> None:
    model = NCA(channels=8, hidden=8)
    x = torch.zeros(1, 8, 9, 9)
    x[0, :, 4, 4] = 1.0   # live seed
    x[0, :3, 0, 0] = 1.0  # isolated colour with alpha 0 -> not alive
    out = model(x, fire_rate=1.0)
    assert out[0, :, 0, 0].abs().sum() == 0
    assert out[0, 3, 4, 4] == 1.0


def test_fire_rate_controls_fraction_of_updated_cells() -> None:
    torch.manual_seed(0)
    model = NCA(channels=4, hidden=8, alive_masking=False, update_bias=True)
    with torch.no_grad():
        model.w2.bias.fill_(1.0)  # every firing cell moves by exactly +1 in each channel
    x = torch.zeros(1, 4, 64, 64)
    for rate, expected in ((0.0, 0.0), (1.0, 1.0), (0.5, 0.5)):
        y = model(x, fire_rate=rate)
        frac = (y[0, 0] != 0).float().mean().item()
        assert frac == pytest.approx(expected, abs=0.03)


def test_explicit_update_mask_is_honoured() -> None:
    model = NCA(channels=4, hidden=8, alive_masking=False)
    with torch.no_grad():
        model.w2.bias.fill_(1.0)
    x = torch.zeros(1, 4, 6, 6)
    mask = torch.zeros(1, 1, 6, 6, dtype=torch.bool)
    mask[0, 0, 2, 3] = True
    y = model(x, update_mask=mask)
    assert y.sum() == 4.0 and y[0, :, 2, 3].eq(1.0).all()


def test_circular_padding_is_translation_equivariant() -> None:
    torch.manual_seed(0)
    model = NCA(channels=6, hidden=16, laplacian=True, alive_masking=False, padding="circular")
    with torch.no_grad():
        model.w2.weight.normal_(0, 0.1)
    x = torch.randn(1, 6, 16, 16)
    shifted = torch.roll(x, shifts=(3, -5), dims=(2, 3))
    mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    torch.testing.assert_close(torch.roll(model(x, update_mask=mask), (3, -5), (2, 3)),
                               model(shifted, update_mask=mask), atol=1e-5, rtol=1e-5)


def test_perception_angle_rotates_the_update_input() -> None:
    model = NCA(channels=3, hidden=4)
    x = torch.randn(1, 3, 8, 8)
    p0, p90 = model.perceive(x, 0.0), model.perceive(x, math.pi / 2)
    # identity channel unchanged; (dx, dy) -> (-dy, dx)
    torch.testing.assert_close(p90[:, 0::3], p0[:, 0::3])
    torch.testing.assert_close(p90[:, 1::3], -p0[:, 2::3], atol=1e-6, rtol=0)
    torch.testing.assert_close(p90[:, 2::3], p0[:, 1::3], atol=1e-6, rtol=0)


def test_make_seed_and_rendering() -> None:
    s = make_seed(2, 16, 20, 20)
    assert s[0, 3:, 10, 10].eq(1).all() and s[0, :3].sum() == 0 and s.sum() == 2 * 13
    rgb = to_rgb(s)
    assert rgb.shape == (2, 3, 20, 20) and rgb.min() >= 0 and rgb.max() <= 1
    assert rgb[0, :, 0, 0].eq(1).all()  # transparent -> white background


def test_gradients_flow_through_a_rollout() -> None:
    torch.manual_seed(0)
    model = NCA(channels=8, hidden=8)
    x = make_seed(1, 8, 12, 12)
    # With a zero last layer the first gradient is non-zero only for the output layer.
    loss = model.rollout(x, 6)[:, :4].pow(2).sum()
    loss.backward()
    assert model.w2.weight.grad.abs().sum() > 0


def test_circle_masks_and_pool() -> None:
    m = circle_masks(5, 40, 40)
    assert m.shape == (5, 40, 40) and set(m.unique().tolist()) <= {0.0, 1.0}
    assert (m.sum((1, 2)) > 0).all()
    pool = SamplePool(torch.zeros(10, 4, 3, 3))
    idx, batch = pool.sample(4)
    assert len(set(idx.tolist())) == 4 and batch.shape == (4, 4, 3, 3)
    pool.commit(idx, batch + 1)
    assert pool.states.sum() == 4 * 4 * 9
