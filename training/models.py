"""Trainable segmentation architectures that share one browser export contract."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TinyUNet(nn.Module):
    """Small U-Net suitable for binary or multiclass medical-image exercises."""

    def __init__(self, num_classes: int, base_channels: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bridge = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bridge = self.bridge(self.pool(enc3))
        dec3 = self.dec3(torch.cat([self.up3(bridge), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))
        return self.head(dec1)


class SegmentationOutput(nn.Module):
    """Normalize TorchVision's dictionary output to a single logits tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs)
        if isinstance(output, dict):
            return output["out"]
        return output


def build_model(
    architecture: str,
    num_classes: int,
    pretrained_voc: bool = False,
) -> nn.Module:
    if architecture == "lraspp":
        if pretrained_voc:
            if num_classes != 21:
                raise ValueError("VOC pretrained weights require exactly 21 classes.")
            return lraspp_mobilenet_v3_large(
                weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            )
        return lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=num_classes,
        )
    if architecture == "unet":
        return TinyUNet(num_classes=num_classes)
    raise ValueError(f"Unsupported architecture: {architecture}")
