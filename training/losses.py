"""
Loss functions for CoRes-Embedding training.

1. CoResLoss: Combined loss for the CoRes model
   - Concept classification loss (BCE)
   - Residual regularization (L2)
   - Concept-Residual orthogonality loss
   - Optional contrastive loss

2. SimCLRLoss: NT-Xent contrastive loss for baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Used for the SimCLR baseline.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Normalized projections of view 1 [B, D]
            z_j: Normalized projections of view 2 [B, D]

        Returns:
            loss: NT-Xent loss
        """
        batch_size = z_i.shape[0]

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = 1
        pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = 1

        # NT-Xent loss
        log_prob = F.log_softmax(sim, dim=1)
        loss = -(log_prob * pos_mask).sum() / (2 * batch_size)

        return loss


class CoResLoss(nn.Module):
    """Combined loss for the CoRes model.

    L_total = L_concept + α·L_residual_reg + β·L_orthogonality + γ·L_supervised

    Where:
        L_concept: Binary cross-entropy for concept prediction
        L_residual_reg: L2 regularization on residual to prevent dominance
        L_orthogonality: Encourage concept and residual to capture different info
        L_supervised: Multi-label classification on total embedding
    """

    def __init__(self, concept_weight=1.0, residual_reg_weight=0.01,
                 orthogonality_weight=0.1):
        super().__init__()

        self.concept_weight = concept_weight
        self.residual_reg_weight = residual_reg_weight
        self.orthogonality_weight = orthogonality_weight

        self.concept_criterion = nn.BCEWithLogitsLoss()
        self.supervised_criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, concept_labels):
        """
        Args:
            output: Dict from CoResModel.forward() containing:
                - concept_logits: [B, N]
                - z_concept: [B, concept_dim]
                - z_residual: [B, residual_dim]
                - logits: [B, N] (supervised classifier)
            concept_labels: Ground truth concept labels [B, N]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        concept_logits = output["concept_logits"]
        z_concept = output["z_concept"]
        z_residual = output["z_residual"]
        logits = output["logits"]

        # 1. Concept classification loss
        loss_concept = self.concept_criterion(concept_logits, concept_labels)

        # 2. Residual regularization (prevent residual from dominating)
        loss_residual_reg = torch.mean(z_residual ** 2)

        # 3. Orthogonality loss: concept and residual should be independent
        # Minimize absolute cosine similarity between concept and residual
        z_c_norm = F.normalize(z_concept, dim=-1)
        z_r_norm = F.normalize(z_residual, dim=-1)

        # Since concept and residual may have different dims, project to common space
        # Use the minimum dimension for comparison
        min_dim = min(z_concept.shape[-1], z_residual.shape[-1])
        z_c_proj = z_c_norm[:, :min_dim]
        z_r_proj = z_r_norm[:, :min_dim]

        cos_sim = torch.sum(z_c_proj * z_r_proj, dim=-1)
        loss_orthogonality = torch.mean(cos_sim ** 2)

        # 4. Supervised classification loss on total embedding
        loss_supervised = self.supervised_criterion(logits, concept_labels)

        # Total loss
        total_loss = (
            self.concept_weight * loss_concept +
            self.residual_reg_weight * loss_residual_reg +
            self.orthogonality_weight * loss_orthogonality +
            loss_supervised
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_concept": loss_concept.item(),
            "loss_residual_reg": loss_residual_reg.item(),
            "loss_orthogonality": loss_orthogonality.item(),
            "loss_supervised": loss_supervised.item(),
        }

        return total_loss, loss_dict


class BaselineLoss(nn.Module):
    """Loss for the Baseline model.

    Supports both SimCLR (contrastive) and Supervised modes.
    """

    def __init__(self, method="supervised", temperature=0.5):
        super().__init__()
        self.method = method

        if method == "simclr":
            self.contrastive_loss = SimCLRLoss(temperature)

        self.supervised_criterion = nn.BCEWithLogitsLoss()

    def forward(self, output, concept_labels):
        """
        Args:
            output: Dict from BaselineModel.forward()
            concept_labels: Ground truth concept labels [B, N]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}

        # Supervised loss
        loss_supervised = self.supervised_criterion(output["logits"], concept_labels)
        total_loss = loss_supervised
        loss_dict["loss_supervised"] = loss_supervised.item()

        # Optional SimCLR loss
        if self.method == "simclr" and "z_proj" in output and "z_proj_aug" in output:
            loss_contrastive = self.contrastive_loss(
                output["z_proj"], output["z_proj_aug"]
            )
            total_loss = total_loss + loss_contrastive
            loss_dict["loss_contrastive"] = loss_contrastive.item()

        loss_dict["loss_total"] = total_loss.item()

        return total_loss, loss_dict
