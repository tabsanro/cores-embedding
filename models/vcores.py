"""
V-CoRes Model (Variational Compositional Residual Embedding).

Backbone → VariationalConceptBranch → z_concept (sampled)
         → ResidualBranch           → z_residual
         → Aggregation              → z_total ∈ R^D
         → Decoder                  → x̂ (reconstruction)

ELBO: L = L_recon + L_supervised - β·KL[q(z_k|x) || p(z_k|c_k)] + regularizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


def _mlp(*dims, dropout=0.0):
    """Build MLP: Linear-BN-ReLU-[Dropout] for hidden layers, Linear for last."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# =============================================================================
# Variational Concept Branch
# =============================================================================

class VariationalConceptBranch(nn.Module):
    """각 개념을 분포 p(z|c_k)=N(μ_k,σ_k²)로 표현하는 Variational Branch."""

    def __init__(self, input_dim, num_concepts, concept_dim,
                 temperature=1.0, use_soft=True):
        super().__init__()
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.temperature = temperature
        self.use_soft = use_soft

        # Concept classifier
        self.concept_classifier = _mlp(input_dim, 256, num_concepts, dropout=0.2)

        # Memory Bank (learnable prior)
        self.prior_mu = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.02)
        self.prior_log_var = nn.Parameter(torch.zeros(num_concepts, concept_dim))
        (nn.init.orthogonal_ if concept_dim >= num_concepts
         else nn.init.xavier_uniform_)(self.prior_mu)

        # Posterior network
        self.posterior_shared = _mlp(input_dim, 512, 256, dropout=0.1)
        self.posterior_mu_head = nn.Linear(256, num_concepts * concept_dim)
        self.posterior_log_var_head = nn.Linear(256, num_concepts * concept_dim)

        # Transformation f_n
        self.transform_fn = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(concept_dim * 2, concept_dim),
        )

    def forward(self, features):
        """Returns: (z_concept, concept_probs, concept_logits, kl_loss, var_info)"""
        B, N, d = features.shape[0], self.num_concepts, self.concept_dim

        # Concept activation
        concept_logits = self.concept_classifier(features)
        concept_probs = torch.sigmoid(concept_logits / self.temperature)
        if self.use_soft:
            weights = concept_probs
        else:
            hard = (concept_probs > 0.5).float()
            weights = hard - concept_probs.detach() + concept_probs  # STE

        # Posterior q(z_k|x)
        h = self.posterior_shared(features)
        post_mu = self.posterior_mu_head(h).view(B, N, d)
        post_lv = self.posterior_log_var_head(h).view(B, N, d).clamp(-10, 2)

        # Reparameterize
        if self.training:
            z = post_mu + torch.exp(0.5 * post_lv) * torch.randn_like(post_mu)
        else:
            z = post_mu

        # Transform & compose: z_concept = Σ_k w_k · f_n(z̃_k)
        z_t = self.transform_fn(z.reshape(B * N, d)).view(B, N, d)
        z_concept = (weights.unsqueeze(-1) * z_t).sum(dim=1)

        # KL divergence
        p_mu = self.prior_mu.unsqueeze(0)
        p_lv = self.prior_log_var.unsqueeze(0)
        kl = 0.5 * (p_lv - post_lv
                     + (post_lv.exp() + (post_mu - p_mu) ** 2) / (p_lv.exp() + 1e-8)
                     - 1)
        kl_loss = (weights * kl.sum(-1)).sum(-1).mean()

        var_info = {
            "posterior_mu": post_mu, "posterior_log_var": post_lv,
            "prior_mu": self.prior_mu, "prior_log_var": self.prior_log_var,
            "z_samples": z, "z_transformed": z_t,
        }
        return z_concept, concept_probs, concept_logits, kl_loss, var_info


# =============================================================================
# Decoder
# =============================================================================

class ResidualConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.leaky_relu(x + self.block(x), 0.2)


class LatentDecoder(nn.Module):
    """z_total → x̂ (image reconstruction)."""

    def __init__(self, latent_dim, image_channels=3, image_size=64):
        super().__init__()
        self.image_size = image_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, True),
            nn.Linear(512, 512 * 8 * 8), nn.LeakyReLU(0.2, True),
        )

        def _up(ic, oc):
            return [nn.ConvTranspose2d(ic, oc, 4, 2, 1),
                    nn.BatchNorm2d(oc), nn.LeakyReLU(0.2, True),
                    ResidualConvBlock(oc)]

        layers = _up(512, 256) + _up(256, 128) + _up(128, 64)
        if image_size >= 128:
            layers += _up(64, 32)
            final_ch = 32
        else:
            final_ch = 64

        layers += [nn.Conv2d(final_ch, 32, 3, 1, 1), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(32, image_channels, 3, 1, 1), nn.Sigmoid()]
        self.conv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 8, 8)
        out = self.conv(h)
        if out.shape[-1] != self.image_size or out.shape[-2] != self.image_size:
            out = F.interpolate(out, self.image_size, mode="bilinear", align_corners=False)
        return out


# =============================================================================
# V-CoRes Model
# =============================================================================

class VCoResModel(nn.Module):
    """Variational Compositional Residual Embedding (V-CoRes)."""

    def __init__(self, latent_dim=64, num_concepts=20,
                 concept_dim=32, residual_dim=32,
                 use_soft_concepts=True, concept_temperature=1.0,
                 aggregation="sum", image_size=64, use_decoder=True,
                 arch="resnet18"):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.use_decoder = use_decoder

        self.backbone = SharedBackbone(arch=arch, pretrained=False)
        feat = self.backbone.feature_dim

        self.concept_branch = VariationalConceptBranch(
            feat, num_concepts, concept_dim, concept_temperature, use_soft_concepts)
        self.residual_branch = _mlp(feat, 256, 128, residual_dim, dropout=0.2)

        # Aggregation
        if aggregation == "projection":
            self.agg = nn.Sequential(
                nn.Linear(concept_dim + residual_dim, latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim))
            self._agg_mode = "projection"
        else:
            self.concept_proj = nn.Linear(concept_dim, latent_dim)
            self.residual_proj = nn.Linear(residual_dim, latent_dim)
            self._agg_mode = "sum"

        self.classifier = nn.Linear(latent_dim, num_concepts)
        if use_decoder:
            self.decoder = LatentDecoder(latent_dim, 3, image_size)

    def _aggregate(self, z_concept, z_residual):
        if self._agg_mode == "sum":
            return self.concept_proj(z_concept) + self.residual_proj(z_residual)
        return self.agg(torch.cat([z_concept, z_residual], dim=-1))

    def encode(self, x):
        """Returns: (z_total, z_concept, z_residual, concept_probs, concept_logits, kl_loss, var_info)"""
        features = self.backbone(x)
        z_concept, c_probs, c_logits, kl, vi = self.concept_branch(features)
        z_residual = self.residual_branch(features)
        z_total = self._aggregate(z_concept, z_residual)
        return z_total, z_concept, z_residual, c_probs, c_logits, kl, vi

    def decode(self, z_total):
        if not self.use_decoder:
            raise RuntimeError("Decoder is disabled")
        return self.decoder(z_total)

    def forward(self, x, x_aug=None):
        z, z_c, z_r, c_probs, c_logits, kl, vi = self.encode(x)
        out = {
            "z": z, "z_concept": z_c, "z_residual": z_r,
            "concept_probs": c_probs, "concept_logits": c_logits,
            "logits": self.classifier(z), "kl_loss": kl,
            "variational_info": vi,
        }
        if self.use_decoder:
            out["x_recon"] = self.decode(z)
        if x_aug is not None:
            za, zca, zra, _, _, kla, _ = self.encode(x_aug)
            out.update(z_aug=za, z_concept_aug=zca,
                       z_residual_aug=zra, kl_loss_aug=kla)
        return out

    def get_embedding(self, x):
        z, *_ = self.encode(x)
        return F.normalize(z, dim=-1)

    def get_decomposed_embedding(self, x):
        z, z_c, z_r, c_probs, _, _, vi = self.encode(x)
        return {
            "z_total": F.normalize(z, dim=-1),
            "z_concept": z_c, "z_residual": z_r,
            "concept_probs": c_probs,
            "posterior_mu": vi["posterior_mu"],
            "posterior_log_var": vi["posterior_log_var"],
            "prior_mu": vi["prior_mu"],
            "prior_log_var": vi["prior_log_var"],
        }

    def get_concept_distributions(self):
        mu = self.concept_branch.prior_mu.detach()
        return mu, torch.exp(0.5 * self.concept_branch.prior_log_var.detach())

    def sample_from_concept(self, concept_idx, num_samples=1):
        mu = self.concept_branch.prior_mu[concept_idx]
        std = torch.exp(0.5 * self.concept_branch.prior_log_var[concept_idx])
        return mu + std * torch.randn(num_samples, self.concept_dim, device=mu.device)


# =============================================================================
# Loss
# =============================================================================

class VCoResLoss(nn.Module):
    """Combined ELBO loss for V-CoRes."""

    def __init__(self, concept_weight=1.0, recon_weight=1.0,
                 kl_weight=1.0, residual_reg_weight=0.01,
                 orthogonality_weight=0.1, kl_annealing_epochs=0,
                 perceptual_weight=0.0, use_perceptual=False):
        super().__init__()
        self.concept_weight = concept_weight
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.residual_reg_weight = residual_reg_weight
        self.orthogonality_weight = orthogonality_weight
        self.kl_annealing_epochs = kl_annealing_epochs
        self.perceptual_weight = perceptual_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.perceptual = PerceptualLoss() if use_perceptual and perceptual_weight > 0 else None

    def forward(self, output, concept_labels, x_input=None, epoch=None):
        dev = output["z_concept"].device
        zero = torch.tensor(0.0, device=dev)
        kl_raw = output["kl_loss"]
        if kl_raw.dim() > 0:
            kl_raw = kl_raw.mean()

        z_c, z_r = output["z_concept"], output["z_residual"]
        has_recon = "x_recon" in output and x_input is not None

        # Individual losses
        l_concept = self.bce(output["concept_logits"], concept_labels)
        l_supervised = self.bce(output["logits"], concept_labels)
        l_recon = self.mse(output["x_recon"], x_input) if has_recon else zero
        l_percep = self.perceptual(output["x_recon"], x_input) if self.perceptual and has_recon else zero

        # KL with annealing
        anneal = (min(1.0, epoch / self.kl_annealing_epochs)
                  if epoch is not None and self.kl_annealing_epochs > 0 else 1.0)
        kl_w = self.kl_weight * anneal
        l_kl = kl_w * kl_raw

        # Regularizers
        l_res_reg = z_r.pow(2).mean()
        d = min(z_c.shape[-1], z_r.shape[-1])
        l_ortho = (F.normalize(z_c[:, :d], dim=-1)
                   * F.normalize(z_r[:, :d], dim=-1)).sum(-1).pow(2).mean()

        total = (l_supervised
                 + self.concept_weight * l_concept
                 + self.recon_weight * l_recon
                 + self.perceptual_weight * l_percep
                 + l_kl
                 + self.residual_reg_weight * l_res_reg
                 + self.orthogonality_weight * l_ortho)

        loss_dict = {
            "loss_total": total.item(), "loss_supervised": l_supervised.item(),
            "loss_concept": l_concept.item(), "loss_recon": l_recon.item(),
            "loss_perceptual": l_percep.item(), "loss_kl": l_kl.item(),
            "loss_kl_raw": kl_raw.item(), "kl_weight": kl_w,
            "loss_residual_reg": l_res_reg.item(), "loss_orthogonality": l_ortho.item(),
        }
        return total, loss_dict


class PerceptualLoss(nn.Module):
    """VGG-16 기반 perceptual loss."""

    def __init__(self, layers=None):
        super().__init__()
        import torchvision.models as models
        layers = layers or [3, 8, 15]
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList()
        prev = 0
        for idx in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:idx + 1]))
            prev = idx + 1
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred, target = (pred - self.mean) / self.std, (target - self.mean) / self.std
        return sum(F.l1_loss(blk(pred), blk(target)) for blk in self.blocks)
