"""
Shared Backbone (ResNet-18 Feature Extractor).
Both Baseline and CoRes models share the same backbone architecture.
"""

import torch
import torch.nn as nn
from torchvision import models


class SharedBackbone(nn.Module):
    """ResNet-18 based shared feature extractor.

    Removes the final FC layer and global average pools features
    to produce a fixed-size feature vector.
    """

    def __init__(self, pretrained=False):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        # Remove final FC layer
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512  # ResNet-18 final feature dimension

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]

        Returns:
            features: [B, 512]
        """
        h = self.features(x)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        return h
