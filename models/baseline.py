"""Baseline Model: Monolithic embedding without compositional structure."""

import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


class BaselineModel(nn.Module):
    """Backbone → FC → z ∈ R^D. Supports SimCLR / Supervised training."""

    def __init__(self, latent_dim=64, num_concepts=20,
                 method="simclr", temperature=0.5, arch="resnet18"):
        super().__init__()
        self.method = method
        self.temperature = temperature

        self.backbone = SharedBackbone(arch=arch, pretrained=False)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, num_concepts)

        if method == "simclr":
            self.simclr_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
            )

    def encode(self, x):
        """x [B,3,H,W] → z [B,D]"""
        return self.projector(self.backbone(x))

    def forward(self, x, x_aug=None):
        z = self.encode(x)
        output = {"z": z, "logits": self.classifier(z)}

        if self.method == "simclr" and x_aug is not None:
            z_aug = self.encode(x_aug)
            output["z_proj"] = F.normalize(self.simclr_head(z), dim=-1)
            output["z_proj_aug"] = F.normalize(self.simclr_head(z_aug), dim=-1)

        return output

    def get_embedding(self, x):
        """평가용: 정규화된 임베딩 반환."""
        return F.normalize(self.encode(x), dim=-1)
