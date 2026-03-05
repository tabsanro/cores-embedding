"""
Shared Backbone — 다중 아키텍처 지원 특징 추출기.
ResNet, DenseNet, EfficientNet 계열을 지원하며
모든 아키텍처의 출력을 512 차원으로 투영합니다.
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# 아키텍처별 raw feature 차원 (글로벌 풀링 직전 채널 수)
# ---------------------------------------------------------------------------
_RAW_DIMS: dict[str, int] = {
    "resnet18": 512, "resnet34": 512, "resnet50": 2048,
    "resnet101": 2048, "resnet152": 2048,
    "densenet121": 1024, "densenet169": 1664,
    "densenet201": 1920, "densenet161": 2208,
    "efficientnet_b0": 1280, "efficientnet_b1": 1280,
    "efficientnet_b2": 1408, "efficientnet_b3": 1536,
    "efficientnet_b4": 1792, "efficientnet_b5": 2048,
    "efficientnet_b6": 2304, "efficientnet_b7": 2560,
}

_OUT_DIM = 512


def _get_family(arch: str) -> str:
    return re.match(r"(resnet|densenet|efficientnet)", arch).group(1)


def _build_backbone(arch: str, pretrained: bool):
    """torchvision 모델에서 공간 특징맵 추출기를 분리하여 반환."""
    # 모델 생성자 & weights 클래스를 이름으로 동적으로 가져옴
    constructor = getattr(models, arch)
    weights_name = arch.replace("resnet", "ResNet").replace("densenet", "DenseNet") \
                       .replace("efficientnet_b", "EfficientNet_B") + "_Weights"
    weights_cls = getattr(models, weights_name, None)
    m = constructor(weights=weights_cls.DEFAULT if pretrained and weights_cls else None)

    family = _get_family(arch)
    if family == "resnet":
        extractor = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
    else:  # densenet / efficientnet
        extractor = m.features

    return extractor, family


class SharedBackbone(nn.Module):
    """다중 아키텍처 공유 특징 추출기. 출력 차원은 항상 512."""

    def __init__(self, arch: str = "resnet18", pretrained: bool = False):
        super().__init__()
        if arch not in _RAW_DIMS:
            raise ValueError(f"지원하지 않는 backbone: '{arch}'.\n"
                             f"지원 목록: {', '.join(sorted(_RAW_DIMS))}")

        raw_dim = _RAW_DIMS[arch]
        self.features, self._family = _build_backbone(arch, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        need_proj = raw_dim != _OUT_DIM
        self.fc_proj = nn.Linear(raw_dim, _OUT_DIM) if need_proj else nn.Identity()
        self.spatial_proj = nn.Conv2d(raw_dim, _OUT_DIM, 1, bias=False) if need_proj else nn.Identity()
        self.feature_dim = _OUT_DIM

    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return F.relu(h, inplace=True) if self._family == "densenet" else h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] → [B, 512]"""
        return self.fc_proj(torch.flatten(self.avgpool(self._extract_spatial(x)), 1))

    def forward_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 3, H, W] → (spatial [B, 512, H', W'], pooled [B, 512])"""
        raw = self._extract_spatial(x)
        spatial = self.spatial_proj(raw)
        pooled = self.fc_proj(torch.flatten(self.avgpool(raw), 1))
        return spatial, pooled
