"""ResNet / DenseNet / EfficientNet / ViT 공유 특징 추출기."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_RAW_DIMS: dict[str, int] = {
    "resnet18": 512, "resnet34": 512, "resnet50": 2048,
    "resnet101": 2048, "resnet152": 2048,
    "densenet121": 1024, "densenet169": 1664,
    "densenet201": 1920, "densenet161": 2208,
    "efficientnet_b0": 1280, "efficientnet_b1": 1280,
    "efficientnet_b2": 1408, "efficientnet_b3": 1536,
    "efficientnet_b4": 1792, "efficientnet_b5": 2048,
    "efficientnet_b6": 2304, "efficientnet_b7": 2560,
    "vit_b_16": 768, "vit_b_32": 768,
    "vit_l_16": 1024, "vit_l_32": 1024,
    "vit_h_14": 1280,
}

def _get_family(arch: str) -> str:
    for f in ("resnet", "densenet", "efficientnet", "vit"):
        if arch.startswith(f):
            return f


def _weights_name(arch: str, family: str) -> str:
    if family == "vit":
        return "ViT_" + "_".join(p.upper() for p in arch.split("_")[1:]) + "_Weights"
    return (arch.replace("resnet", "ResNet")
                .replace("densenet", "DenseNet")
                .replace("efficientnet_b", "EfficientNet_B") + "_Weights")


def _build_backbone(arch: str, pretrained: bool):
    family = _get_family(arch)
    weights_cls = getattr(models, _weights_name(arch, family), None)
    m = getattr(models, arch)(weights=weights_cls.DEFAULT if pretrained and weights_cls else None)

    if family == "resnet":
        extractor = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                  m.layer1, m.layer2, m.layer3, m.layer4)
    elif family == "vit":
        extractor = _ViTSpatialWrapper(m)
    else:
        extractor = m.features

    return extractor, family


class _ViTSpatialWrapper(nn.Module):
    """ViT → (spatial [B,D,H',W'], cls [B,D])"""

    def __init__(self, vit):
        super().__init__()
        self._vit = vit

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        p = self._vit.patch_size
        tokens = self._vit._process_input(x)
        cls = self._vit.class_token.expand(B, -1, -1)
        tokens = self._vit.encoder(torch.cat([cls, tokens], dim=1))
        spatial = tokens[:, 1:].transpose(1, 2).reshape(B, -1, H // p, W // p)
        return spatial, tokens[:, 0]


class SharedBackbone(nn.Module):
    """다중 아키텍처 공유 특징 추출기.

    Args:
        arch:       backbone 아키텍처 이름.
        pretrained: ImageNet 사전학습 가중치 사용 여부.
        out_dim:    투영 차원 (None이면 raw 차원 그대로 출력).
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = False,
                 out_dim: int | None = None):
        super().__init__()
        if arch not in _RAW_DIMS:
            raise ValueError(f"지원하지 않는 backbone: '{arch}'.\n"
                             f"지원 목록: {', '.join(sorted(_RAW_DIMS))}")

        raw_dim = _RAW_DIMS[arch]
        self.features, self._family = _build_backbone(arch, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.raw_feature_dim = raw_dim
        self.feature_dim = out_dim or raw_dim

        self.fc_proj = nn.Sequential(
            nn.LayerNorm(raw_dim), nn.Linear(raw_dim, out_dim),
            nn.GELU(), nn.Linear(out_dim, out_dim),
        ) if out_dim else nn.Identity()
        self.spatial_proj = nn.Conv2d(raw_dim, out_dim, 1, bias=False) if out_dim else nn.Identity()

    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return F.relu(h, inplace=True) if self._family == "densenet" else h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] → [B,feature_dim]"""
        if self._family == "vit":
            _, cls = self.features(x)
            return self.fc_proj(cls)
        return self.fc_proj(torch.flatten(self.avgpool(self._extract_spatial(x)), 1))

    def forward_spatial(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B,3,H,W] → (spatial [B,feature_dim,H',W'], pooled [B,feature_dim])"""
        if self._family == "vit":
            spatial, cls = self.features(x)
            return self.spatial_proj(spatial), self.fc_proj(cls)
        raw = self._extract_spatial(x)
        return self.spatial_proj(raw), self.fc_proj(torch.flatten(self.avgpool(raw), 1))
