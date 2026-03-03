"""
CoRes Model (Compositional Residual Embedding).
Decomposes representation into structured concepts + residual.

z_total = W(z_concept) + z_residual

Where:
    z_concept = Σ_i p(concept_i | x) · E_i  (concept embeddings)
    z_residual captures information not explained by concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone

class ConceptBranch(nn.Module):
    """Concept Branch: predicts concept probabilities and constructs
    compositional concept embedding via learned embedding bank.

    z_concept = Σ_i p(concept_i | x) · E_i

    Where:
        - p(concept_i | x) is the predicted probability of concept i
        - E_i is the learned embedding vector for concept i
    """

    def __init__(self, input_dim, num_concepts, concept_dim,
                 temperature=1.0, use_soft=True):
        """
        Args:
            input_dim: Dimension of backbone features (512 for ResNet-18).
            num_concepts: Number of binary concepts N.
            concept_dim: Dimension of each concept embedding d_concept.
            temperature: Softmax temperature for concept prediction.
            use_soft: If True, use soft (probabilistic) concept assignment.
                     If False, use hard (argmax) assignment with straight-through.
        """
        super().__init__()

        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.temperature = temperature
        self.use_soft = use_soft

        # Concept classifier: predicts probability for each concept
        self.concept_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_concepts),
        )

        # Learnable embedding bank: E ∈ R^{N × d_concept}
        self.embedding_bank = nn.Parameter(
            torch.randn(num_concepts, concept_dim) * 0.02
        )

        # Initialize embedding bank with orthogonal vectors if possible
        if concept_dim >= num_concepts:
            nn.init.orthogonal_(self.embedding_bank)
        else:
            nn.init.xavier_uniform_(self.embedding_bank)

    def forward(self, features):
        """
        Args:
            features: Backbone features [B, input_dim]

        Returns:
            z_concept: Compositional concept embedding [B, concept_dim]
            concept_probs: Concept probabilities [B, num_concepts]
            concept_logits: Raw logits [B, num_concepts]
        """
        # Predict concept probabilities
        concept_logits = self.concept_classifier(features)
        concept_probs = torch.sigmoid(concept_logits / self.temperature)

        if self.use_soft:
            # Soft assignment: weighted sum of embeddings
            weights = concept_probs  # [B, N]
        else:
            # Hard assignment with straight-through estimator
            hard = (concept_probs > 0.5).float()
            weights = hard - concept_probs.detach() + concept_probs  # STE

        # Compositional embedding: z_concept = Σ_i w_i · E_i
        # weights: [B, N], embedding_bank: [N, d_concept]
        z_concept = torch.matmul(weights, self.embedding_bank)  # [B, d_concept]

        return z_concept, concept_probs, concept_logits


class ResidualBranch(nn.Module):
    """Residual Branch: captures information not explained by concepts.

    This branch learns to encode the "residual" information -
    lighting, pose, texture details, background, etc.
    """

    def __init__(self, input_dim, residual_dim):
        """
        Args:
            input_dim: Dimension of backbone features.
            residual_dim: Dimension of residual embedding d_res.
        """
        super().__init__()

        self.residual_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, residual_dim),
        )

    def forward(self, features):
        """
        Args:
            features: Backbone features [B, input_dim]

        Returns:
            z_residual: Residual embedding [B, residual_dim]
        """
        z_residual = self.residual_encoder(features)
        return z_residual


class AggregationLayer(nn.Module):
    """Aggregation Layer: combines concept and residual embeddings.

    z_total = W(z_concept) + z_residual  (projected to total dim D)

    Or direct concatenation + projection.
    """

    def __init__(self, concept_dim, residual_dim, latent_dim,
                 aggregation="sum"):
        """
        Args:
            concept_dim: Dimension of concept embedding.
            residual_dim: Dimension of residual embedding.
            latent_dim: Total latent dimension budget D.
            aggregation: "sum" or "projection".
        """
        super().__init__()
        self.aggregation = aggregation
        self.latent_dim = latent_dim

        if aggregation == "sum":
            # Project both to same dimension, then sum
            self.concept_proj = nn.Linear(concept_dim, latent_dim)
            self.residual_proj = nn.Linear(residual_dim, latent_dim)
        elif aggregation == "projection":
            # Concatenate then project
            self.projection = nn.Sequential(
                nn.Linear(concept_dim + residual_dim, latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim),
            )
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def forward(self, z_concept, z_residual):
        """
        Args:
            z_concept: [B, concept_dim]
            z_residual: [B, residual_dim]

        Returns:
            z_total: [B, latent_dim]
        """
        if self.aggregation == "sum":
            z_total = self.concept_proj(z_concept) + self.residual_proj(z_residual)
        elif self.aggregation == "projection":
            z_cat = torch.cat([z_concept, z_residual], dim=-1)
            z_total = self.projection(z_cat)

        return z_total



class CoResModel(nn.Module):
    """Compositional Residual Embedding (CoRes) Model.

    Architecture:
        ResNet-18 (shared) →  ┌→ ConceptBranch → z_concept
                               └→ ResidualBranch → z_residual
                               → AggregationLayer → z_total ∈ R^D

    The concept branch acts as a discrete "anchor" that stabilizes the
    representation against noise, while the residual branch captures
    fine-grained details.
    """

    def __init__(self, latent_dim=64, num_concepts=20,
                 concept_dim=32, residual_dim=32,
                 use_soft_concepts=True, concept_temperature=1.0,
                 aggregation="sum"):
        """
        Args:
            latent_dim: Total latent dimension budget D.
            num_concepts: Number of binary concepts N.
            concept_dim: Dimension of concept embedding space.
            residual_dim: Dimension of residual embedding space.
            use_soft_concepts: Use soft (probabilistic) concept assignment.
            concept_temperature: Temperature for concept prediction.
            aggregation: Aggregation method ("sum" or "projection").
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim

        # Shared backbone
        self.backbone = SharedBackbone(pretrained=False)

        # Concept branch
        self.concept_branch = ConceptBranch(
            input_dim=self.backbone.feature_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            temperature=concept_temperature,
            use_soft=use_soft_concepts,
        )

        # Residual branch
        self.residual_branch = ResidualBranch(
            input_dim=self.backbone.feature_dim,
            residual_dim=residual_dim,
        )

        # Aggregation layer
        self.aggregation = AggregationLayer(
            concept_dim=concept_dim,
            residual_dim=residual_dim,
            latent_dim=latent_dim,
            aggregation=aggregation,
        )

        # Supervised classification head (for training signal)
        self.classifier = nn.Linear(latent_dim, num_concepts)

    def encode(self, x):
        """Extract decomposed latent embedding.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z_total: Combined embedding [B, D]
            z_concept: Concept embedding [B, concept_dim]
            z_residual: Residual embedding [B, residual_dim]
            concept_probs: Concept probabilities [B, num_concepts]
            concept_logits: Raw concept logits [B, num_concepts]
        """
        # Shared features
        features = self.backbone(x)

        # Concept branch
        z_concept, concept_probs, concept_logits = self.concept_branch(features)

        # Residual branch
        z_residual = self.residual_branch(features)

        # Aggregation
        z_total = self.aggregation(z_concept, z_residual)

        return z_total, z_concept, z_residual, concept_probs, concept_logits

    def forward(self, x, x_aug=None):
        """
        Args:
            x: Input images [B, 3, H, W]
            x_aug: Augmented images (optional, for contrastive learning)

        Returns:
            dict with all intermediate and final representations
        """
        z_total, z_concept, z_residual, concept_probs, concept_logits = self.encode(x)
        logits = self.classifier(z_total)

        output = {
            "z": z_total,
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
            "concept_logits": concept_logits,
            "logits": logits,
        }

        if x_aug is not None:
            z_total_aug, z_concept_aug, z_residual_aug, _, _ = self.encode(x_aug)
            output["z_aug"] = z_total_aug
            output["z_concept_aug"] = z_concept_aug
            output["z_residual_aug"] = z_residual_aug

        return output

    def get_embedding(self, x):
        """Get the final embedding for evaluation.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z: Normalized total embedding [B, D]
        """
        z_total, _, _, _, _ = self.encode(x)
        return F.normalize(z_total, dim=-1)

    def get_decomposed_embedding(self, x):
        """Get decomposed embeddings for analysis.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            dict with z_total, z_concept, z_residual, concept_probs
        """
        z_total, z_concept, z_residual, concept_probs, _ = self.encode(x)
        return {
            "z_total": F.normalize(z_total, dim=-1),
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
        }
