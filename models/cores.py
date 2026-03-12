"""CoRes: Compositional Residual Embedding. z = W(z_concept) + z_residual"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


class ConceptBranch(nn.Module):
    """Predicts concept probs and builds z_concept = Σ_i p_i · E_i."""

    def __init__(self, input_dim, num_concepts, concept_dim, temperature=1.0, use_soft=True):
        super().__init__()
        self.temperature = temperature
        self.use_soft = use_soft

        self.concept_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_concepts),
        )

        self.embedding_bank = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.02)
        if concept_dim >= num_concepts:
            nn.init.orthogonal_(self.embedding_bank)
        else:
            nn.init.xavier_uniform_(self.embedding_bank)

    def forward(self, features):
        concept_logits = self.concept_classifier(features)
        concept_probs = torch.sigmoid(concept_logits / self.temperature)

        if self.use_soft:
            weights = concept_probs
        else:
            hard = (concept_probs > 0.5).float()
            weights = hard - concept_probs.detach() + concept_probs  # STE

        z_concept = torch.matmul(weights, self.embedding_bank)
        return z_concept, concept_probs, concept_logits


class CoResModel(nn.Module):
    """Compositional Residual Embedding (CoRes) Model.

    backbone → ConceptBranch → z_concept ┐
                                          → aggregate → z ∈ R^D
             → residual_encoder → z_res  ┘
    """

    def __init__(self, latent_dim=64, num_concepts=20, concept_dim=32, residual_dim=32,
                 use_soft_concepts=True, concept_temperature=1.0,
                 aggregation="sum", arch="resnet18",
                 residual_scale: float = 1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.aggregation_type = aggregation

        self.backbone = SharedBackbone(arch=arch, pretrained=False)
        feat_dim = self.backbone.feature_dim

        self.concept_branch = ConceptBranch(
            feat_dim, num_concepts, concept_dim, concept_temperature, use_soft_concepts
        )

        self.residual_encoder = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, residual_dim),
        )

        self.residual_scale = residual_scale

        if aggregation == "sum":
            self.concept_proj = nn.Linear(concept_dim, latent_dim)
            self.residual_proj = nn.Linear(residual_dim, latent_dim)
        elif aggregation == "projection":
            self.proj = nn.Sequential(
                nn.Linear(concept_dim + residual_dim, latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim),
            )
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        self.classifier = nn.Linear(latent_dim, num_concepts)

    def _aggregate(self, z_concept, z_residual):
        if self.aggregation_type == "sum":
            concept_part = self.concept_proj(z_concept)
            residual_part = self.residual_proj(z_residual)
            # Gram-Schmidt 직교 투영 (SeqCoRes 방식 차용)
            dot = (residual_part * concept_part).sum(dim=-1, keepdim=True)
            norm_sq = (concept_part * concept_part).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            residual_orth = residual_part - (dot / norm_sq) * concept_part
            return concept_part + residual_orth
        return self.proj(torch.cat([z_concept, z_residual], dim=-1))

    def encode(self, x):
        features = self.backbone(x)
        z_concept, concept_probs, concept_logits = self.concept_branch(features)
        z_residual = self.residual_encoder(features)
        z_total = self._aggregate(z_concept, z_residual)
        return z_total, z_concept, z_residual, concept_probs, concept_logits

    def forward(self, x, x_aug=None):
        z_total, z_concept, z_residual, concept_probs, concept_logits = self.encode(x)
        output = {
            "z": z_total,
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
            "concept_logits": concept_logits,
            "logits": self.classifier(z_total),
        }
        if x_aug is not None:
            z_aug, z_concept_aug, z_residual_aug, _, _ = self.encode(x_aug)
            output.update({"z_aug": z_aug, "z_concept_aug": z_concept_aug, "z_residual_aug": z_residual_aug})
        return output

    def get_embedding(self, x):
        z_total, *_ = self.encode(x)
        return F.normalize(z_total, dim=-1)

    def get_decomposed_embedding(self, x):
        z_total, z_concept, z_residual, concept_probs, _ = self.encode(x)
        return {
            "z_total": F.normalize(z_total, dim=-1),
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
        }
