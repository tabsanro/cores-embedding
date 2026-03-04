"""
Shared Backbone — 다중 아키텍처 지원 특징 추출기.
ResNet, DenseNet, EfficientNet 계열을 지원하며
모든 아키텍처의 출력을 512 차원으로 투영합니다.

configs/default.yaml의 model.backbone 필드로 아키텍처를 선택합니다.
  예) backbone: "resnet18"  / "resnet50" / "densenet121" / "efficientnet_b3"

모든 모델에서 feature_dim = 512 (고정)이 보장되므로,
다운스트림 모듈(CoRes, SeqCoRes 등)은 백본 종류에 무관하게 동일한
차원을 사용할 수 있습니다.

v3 유지: forward_spatial() — 글로벌 풀링 전 공간 특징맵(C×H×W) 반환.
  spatial_features도 1×1 Conv로 512 채널에 투영합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# 아키텍처별 raw feature 차원 등록 테이블
# (아키텍처명) → raw 채널 수 (글로벌 풀링 직전)
# ---------------------------------------------------------------------------
_BACKBONE_RAW_DIMS: dict[str, int] = {
    # ResNet family
    "resnet18":  512,
    "resnet34":  512,
    "resnet50":  2048,
    "resnet101": 2048,
    "resnet152": 2048,
    # DenseNet family
    "densenet121": 1024,
    "densenet169": 1664,
    "densenet201": 1920,
    "densenet161": 2208,
    # EfficientNet family
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_b3": 1536,
    "efficientnet_b4": 1792,
    "efficientnet_b5": 2048,
    "efficientnet_b6": 2304,
    "efficientnet_b7": 2560,
}

# 고정 출력 차원 (모든 백본 공통)
_OUT_DIM = 512


def _build_backbone(arch: str, pretrained: bool):
    """아키텍처 이름으로 torchvision 모델 생성.

    Returns:
        feature_extractor (nn.Module): 공간 특징맵을 출력하는 모듈.
            - ResNet:      [B, raw_dim, H', W'] (layer4 출력)
            - DenseNet:    [B, raw_dim, H', W'] (features + relu)
            - EfficientNet:[B, raw_dim, H', W'] (features 출력)
        arch_family (str): "resnet" | "densenet" | "efficientnet"
    """
    def _w(weights_cls):
        return weights_cls.DEFAULT if pretrained else None

    a = arch.lower()

    # ── ResNet ──────────────────────────────────────────────────────────────
    if a == "resnet18":
        m = models.resnet18(weights=_w(models.ResNet18_Weights))
    elif a == "resnet34":
        m = models.resnet34(weights=_w(models.ResNet34_Weights))
    elif a == "resnet50":
        m = models.resnet50(weights=_w(models.ResNet50_Weights))
    elif a == "resnet101":
        m = models.resnet101(weights=_w(models.ResNet101_Weights))
    elif a == "resnet152":
        m = models.resnet152(weights=_w(models.ResNet152_Weights))
    # ── DenseNet ────────────────────────────────────────────────────────────
    elif a == "densenet121":
        m = models.densenet121(weights=_w(models.DenseNet121_Weights))
    elif a == "densenet169":
        m = models.densenet169(weights=_w(models.DenseNet169_Weights))
    elif a == "densenet201":
        m = models.densenet201(weights=_w(models.DenseNet201_Weights))
    elif a == "densenet161":
        m = models.densenet161(weights=_w(models.DenseNet161_Weights))
    # ── EfficientNet ────────────────────────────────────────────────────────
    elif a == "efficientnet_b0":
        m = models.efficientnet_b0(weights=_w(models.EfficientNet_B0_Weights))
    elif a == "efficientnet_b1":
        m = models.efficientnet_b1(weights=_w(models.EfficientNet_B1_Weights))
    elif a == "efficientnet_b2":
        m = models.efficientnet_b2(weights=_w(models.EfficientNet_B2_Weights))
    elif a == "efficientnet_b3":
        m = models.efficientnet_b3(weights=_w(models.EfficientNet_B3_Weights))
    elif a == "efficientnet_b4":
        m = models.efficientnet_b4(weights=_w(models.EfficientNet_B4_Weights))
    elif a == "efficientnet_b5":
        m = models.efficientnet_b5(weights=_w(models.EfficientNet_B5_Weights))
    elif a == "efficientnet_b6":
        m = models.efficientnet_b6(weights=_w(models.EfficientNet_B6_Weights))
    elif a == "efficientnet_b7":
        m = models.efficientnet_b7(weights=_w(models.EfficientNet_B7_Weights))
    else:
        supported = ", ".join(sorted(_BACKBONE_RAW_DIMS.keys()))
        raise ValueError(
            f"지원하지 않는 backbone: '{arch}'.\n지원 목록: {supported}"
        )

    # ── 계열별 feature extractor 분리 ──────────────────────────────────────
    if a.startswith("resnet"):
        extractor = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3, m.layer4,
        )
        family = "resnet"
    elif a.startswith("densenet"):
        extractor = m.features   # DenseNet2d block (마지막 BN 포함)
        family = "densenet"
    else:  # efficientnet
        extractor = m.features   # Sequential of MBConv blocks
        family = "efficientnet"

    return extractor, family


# ---------------------------------------------------------------------------
# SharedBackbone
# ---------------------------------------------------------------------------

class SharedBackbone(nn.Module):
    """다중 아키텍처를 지원하는 공유 특징 추출기.

    아키텍처에 상관없이 항상 512 차원 벡터를 출력합니다.
    raw 채널 수가 512가 아닌 경우 마지막 단에 깊이-1 FC 레이어(투영 레이어)가
    자동으로 삽입됩니다. 투영 레이어는 BatchNorm 없이 선형 변환만 수행합니다.

    forward_spatial() 에서도 동일한 투영이 적용됩니다.
    공간 특징맵(spatial)은 1×1 Conv로, 풀링된 벡터는 FC로 각각 512에 매핑됩니다.

    Args:
        arch (str): 백본 아키텍처 이름 (default.yaml의 model.backbone).
        pretrained (bool): ImageNet 사전학습 가중치 사용 여부.
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if arch not in _BACKBONE_RAW_DIMS:
            supported = ", ".join(sorted(_BACKBONE_RAW_DIMS.keys()))
            raise ValueError(
                f"지원하지 않는 backbone: '{arch}'.\n지원 목록: {supported}"
            )

        self._arch = arch
        self._family = arch.split("_")[0] if arch.startswith("efficientnet") else arch.rstrip("0123456789")
        raw_dim = _BACKBONE_RAW_DIMS[arch]

        self.features, self._family = _build_backbone(arch, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 투영 레이어: raw_dim → 512  (raw_dim == 512이면 Identity)
        if raw_dim != _OUT_DIM:
            self.fc_proj = nn.Linear(raw_dim, _OUT_DIM)
            # forward_spatial용 1×1 Conv 공간 투영
            self.spatial_proj = nn.Conv2d(raw_dim, _OUT_DIM, kernel_size=1, bias=False)
        else:
            self.fc_proj = nn.Identity()
            self.spatial_proj = nn.Identity()

        self.feature_dim = _OUT_DIM  # 항상 512 — 다운스트림 모듈 호환 보장

    # ------------------------------------------------------------------
    # 내부: raw 공간 특징맵 추출 (풀링·투영 없음)
    # ------------------------------------------------------------------
    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """백본의 raw 공간 특징맵 [B, raw_dim, H', W'] 반환."""
        spatial = self.features(x)
        # DenseNet은 features block 끝에 BN만 있고 ReLU가 없으므로 수동 적용
        if self._family == "densenet":
            spatial = F.relu(spatial, inplace=True)
        return spatial

    # ------------------------------------------------------------------
    # forward: 투영된 풀링 벡터 [B, 512]
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 이미지 [B, 3, H, W]

        Returns:
            features: [B, 512]
        """
        h = self._extract_spatial(x)       # [B, raw_dim, H', W']
        h = self.avgpool(h)                 # [B, raw_dim, 1, 1]
        h = torch.flatten(h, 1)            # [B, raw_dim]
        h = self.fc_proj(h)                 # [B, 512]
        return h

    # ------------------------------------------------------------------
    # forward_spatial: 공간 특징맵 + 풀링 벡터 (모두 512 채널로 투영)
    # ------------------------------------------------------------------
    def forward_spatial(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """공간 정보가 보존된 특징맵 반환 (Top-down Attention용).

        공간 특징맵과 풀링 벡터 모두 512 채널/차원으로 투영됩니다.

        Args:
            x: 입력 이미지 [B, 3, H, W]

        Returns:
            spatial_features: [B, 512, H', W']  — 1×1 Conv 투영 완료
            pooled_features:  [B, 512]           — FC 투영 완료
        """
        raw_spatial = self._extract_spatial(x)          # [B, raw_dim, H', W']

        # 공간 투영 (1×1 Conv)
        spatial = self.spatial_proj(raw_spatial)         # [B, 512, H', W']

        # 풀링 투영 (FC)
        pooled = self.avgpool(raw_spatial)               # [B, raw_dim, 1, 1]
        pooled = torch.flatten(pooled, 1)                # [B, raw_dim]
        pooled = self.fc_proj(pooled)                    # [B, 512]

        return spatial, pooled
