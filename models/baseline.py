"""
Baseline Model (Monolithic Embedding).
Maps input to a single z ∈ R^D without compositional structure.
Supports SimCLR (contrastive) and Supervised training methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


class BaselineModel(nn.Module):
    """Monolithic baseline model.

    Architecture:
        ResNet-18 → FC → z ∈ R^D

    Training methods:
        - SimCLR: Contrastive learning with NT-Xent loss
        - Supervised: Multi-label classification (all attributes combined)
    """

    def __init__(self, latent_dim=64, num_concepts=20,
                 method="simclr", temperature=0.5):
        """
        Args:
            latent_dim: Total latent dimension D.
            num_concepts: Number of concepts (for supervised head).
            method: Training method - "simclr" or "supervised".
            temperature: SimCLR temperature parameter.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.method = method
        self.temperature = temperature

        # Shared backbone
        self.backbone = SharedBackbone(pretrained=False)

        # Projection head: backbone features → latent space
        self.projector = nn.Sequential(
            nn.Linear(self.backbone.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

        # SimCLR projection head (non-linear, for contrastive loss only)
        if method == "simclr":
            self.simclr_head = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
            )

        # Supervised classification head
        self.classifier = nn.Linear(latent_dim, num_concepts)

    def encode(self, x):
        """Extract latent embedding.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z: Latent embedding [B, D]
        """
        features = self.backbone(x)
        z = self.projector(features)
        return z

    def forward(self, x, x_aug=None):
        """
        Args:
            x: Input images [B, 3, H, W]
            x_aug: Augmented images for SimCLR [B, 3, H, W] (optional)

        Returns:
            dict with:
                - z: Latent embedding [B, D]
                - logits: Classification logits [B, num_concepts]
                - z_proj: SimCLR projection (if applicable) [B, 64]
                - z_proj_aug: SimCLR projection of augmented (if applicable)
        """
        z = self.encode(x)
        logits = self.classifier(z)

        output = {
            "z": z,
            "logits": logits,
        }

        if self.method == "simclr" and x_aug is not None:
            z_aug = self.encode(x_aug)
            output["z_proj"] = F.normalize(self.simclr_head(z), dim=-1)
            output["z_proj_aug"] = F.normalize(self.simclr_head(z_aug), dim=-1)

        return output

    def get_embedding(self, x):
        """Get the final embedding for evaluation (no grad).

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z: Normalized latent embedding [B, D]
        """
        z = self.encode(x)
        return F.normalize(z, dim=-1)
