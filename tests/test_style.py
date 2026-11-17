import torch

from nca.style import StyleLoss, gram_matrix


def test_gram_matrix_is_symmetric_psd_and_normalised() -> None:
    f = torch.randn(2, 5, 6, 7)
    g = gram_matrix(f)
    assert g.shape == (2, 5, 5)
    torch.testing.assert_close(g, g.transpose(1, 2))
    assert torch.linalg.eigvalsh(g).min() > -1e-4
    ref = f[0].reshape(5, -1) @ f[0].reshape(5, -1).T / 42
    torch.testing.assert_close(g[0], ref)


def test_style_loss_zero_on_target_and_positive_elsewhere() -> None:
    torch.manual_seed(0)
    target = torch.rand(3, 32, 32)
    loss = StyleLoss(target, pretrained=False)  # random VGG weights keep the test offline
    same = loss(target[None])
    other = loss(torch.rand(1, 3, 32, 32).pow(3))
    assert same.item() < 1e-8
    assert other.item() > 1e-3
    loss(target[None].requires_grad_()).sum().backward()  # differentiable
