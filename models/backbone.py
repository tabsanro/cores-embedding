"""
Shared Backbone (ResNet-18 Feature Extractor).
Both Baseline and CoRes models share the same backbone architecture.
"""

import torch
import torch.nn as nn
from torchvision import models


class SharedBackbone(nn.Module):
    """ResNet-50 based shared feature extractor.

    Removes the final FC layer and global average pools features
    to produce a fixed-size feature vector.

    v3 추가: forward_spatial() — 글로벌 풀링 전 공간 특징맵(C×H×W) 반환.
    Top-down Attention에서 GRU 은닉 상태가 이미지의 어느 부분을
    볼지 결정하기 위해 공간 정보를 유지합니다.
    """

    def __init__(self, pretrained=False):
        super().__init__()

        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
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
        self.feature_dim = 2048  # ResNet-50 final feature dimension

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

    def forward_spatial(self, x):
        """공간 정보가 보존된 특징맵 반환 (Top-down Attention용).

        글로벌 풀링을 적용하지 않고, ResNet layer4 출력의
        C×H×W 텐서를 그대로 반환합니다.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            spatial_features: [B, 512, H', W']  (H'=H/32, W'=W/32)
            pooled_features:  [B, 512]          (글로벌 풀링된 벡터)
        """
        spatial = self.features(x)               # [B, 512, H', W']
        pooled = self.avgpool(spatial)            # [B, 512, 1, 1]
        pooled = torch.flatten(pooled, 1)         # [B, 512]
        return spatial, pooled
