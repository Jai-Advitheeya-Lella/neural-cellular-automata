"""VGG-16 Gram-matrix style loss for texture synthesis (Gatys et al., 2015; Niklasson et al., 2021)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

#: Indices of relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 in ``torchvision.models.vgg16().features``.
VGG_LAYERS = (1, 6, 11, 18, 25)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gram_matrix(feat: Tensor) -> Tensor:
    """Channel Gram matrix ``(B, C, C)`` of features ``(B, C, H, W)``, normalised by the number of positions."""
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (h * w)


class StyleLoss(nn.Module):
    r"""Relative Gram-matrix distance between an image batch and a target texture.

    For each of five VGG-16 ReLU layers :math:`l` the loss is
    :math:`\lVert G_l(x) - G_l(t) \rVert_F^2 / \lVert G_l(t) \rVert_F^2` and the layers are averaged.
    The relative form puts the five layers on a common scale, which differs by orders of magnitude otherwise.

    Args:
        target: Target texture ``(3, H, W)`` in [0, 1].
        pretrained: Load ImageNet weights (downloads ~528 MB on first use); tests pass ``False``.
    """

    def __init__(self, target: Tensor, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features[: max(VGG_LAYERS) + 1]
        for p in net.parameters():
            p.requires_grad_(False)
        # In-place ReLUs would modify the cached activations we keep for the Gram matrices.
        for m in net:
            if isinstance(m, nn.ReLU):
                m.inplace = False
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        with torch.no_grad():
            grams = [g.flatten(1) for g in self._grams(target[None])]
        self._sizes = [g.shape[1] for g in grams]
        self.register_buffer("target_grams", torch.cat(grams, 1))

    def _grams(self, x: Tensor) -> list[Tensor]:
        x = (x.to(self.mean) - self.mean) / self.std
        grams = []
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i in VGG_LAYERS:
                grams.append(gram_matrix(x))
        return grams

    def forward(self, rgb: Tensor) -> Tensor:
        """Per-sample style loss for images ``(B, 3, H, W)`` in [0, 1]; returns ``(B,)``."""
        grams = self._grams(rgb)
        target = torch.split(self.target_grams, self._sizes, dim=1)
        loss = 0.0
        for g, t in zip(grams, target):
            g = g.flatten(1)
            loss = loss + (g - t).pow(2).sum(1) / (t.pow(2).sum(1) + 1e-8)
        return loss / len(grams)
