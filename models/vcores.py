"""
V-CoRes Model (Variational Compositional Residual Embedding).

점(Point) 임베딩에서 분포(Distribution) 임베딩으로 전환하여,
각 개념의 '의미' 뿐만 아니라 '범위(Variance)'와 '불확실성(Uncertainty)'까지 학습.

Architecture:
    ResNet-18 (shared) → ┌→ VariationalConceptBranch → z_concept (sampled)
                          └→ ResidualBranch            → z_residual
                          → AggregationLayer           → z_total ∈ R^D
                          → Decoder                    → x̂ (reconstruction)

Key Differences from Deterministic CoRes:
    - Memory Bank (Global Prior): p(z|c_k) = N(μ_k, σ_k²) (learnable per concept)
    - Posterior Network:          q(z_k|x) = N(μ_k(x), σ_k(x)²) (inferred from input)
    - Reparameterization Trick:   z̃_k = μ_k(x) + σ_k(x) ⊙ ε,  ε ~ N(0, I)
    - Transformation Function:    z_concept = Σ_k w_k · f_n(z̃_k)
    - ELBO Objective:             L = L_recon - β Σ_k KL[q(z_k|x) || p(z_k|c_k)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone

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


# =============================================================================
# Variational Concept Branch
# =============================================================================

class VariationalConceptBranch(nn.Module):
    """Variational Concept Branch with learnable prior Memory Bank.

    기존 ConceptBranch가 고정된 벡터 집합 M = {e_1, ..., e_N}으로
    개념을 표현했다면, VariationalConceptBranch는 각 개념을
    **분포** p(z|c_k) = N(μ_k, σ_k²)로 표현합니다.

    Components:
        1. Concept Classifier: 입력 x에서 활성화된 개념 선택
        2. Memory Bank (Global Prior): 학습 가능한 개념 분포 {(μ_k, σ_k²)}
        3. Posterior Network: q(z_k|x) = N(μ_k(x), σ_k(x)²)
        4. Reparameterization: z̃_k = μ(x) + σ(x) ⊙ ε
        5. Transformation f_n: 분포 공간 → Feature 공간 (Linear-ReLU-Linear MLP)
    """

    def __init__(self, input_dim, num_concepts, concept_dim,
                 temperature=1.0, use_soft=True):
        """
        Args:
            input_dim: Backbone feature dimension (512 for ResNet-18).
            num_concepts: Number of concepts N.
            concept_dim: Dimension of each concept's latent distribution.
            temperature: Temperature for concept activation sigmoid.
            use_soft: If True, soft (probabilistic) concept activation.
        """
        super().__init__()

        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.temperature = temperature
        self.use_soft = use_soft

        # ------------------------------------------------------------------
        # 1. Concept Classifier: predicts activation w_k = σ(logit_k / τ)
        # ------------------------------------------------------------------
        self.concept_classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_concepts),
        )

        # ------------------------------------------------------------------
        # 2. Memory Bank (Learnable Global Prior)
        #    p(z | c_k) = N(μ_k, diag(σ_k²))   for k = 1..N
        # ------------------------------------------------------------------
        self.prior_mu = nn.Parameter(
            torch.randn(num_concepts, concept_dim) * 0.02
        )
        self.prior_log_var = nn.Parameter(
            torch.zeros(num_concepts, concept_dim)
        )

        # Initialize prior means for diversity
        if concept_dim >= num_concepts:
            nn.init.orthogonal_(self.prior_mu)
        else:
            nn.init.xavier_uniform_(self.prior_mu)

        # ------------------------------------------------------------------
        # 3. Posterior Network: q(z_k | x) = N(μ_k(x), diag(σ_k(x)²))
        #    Shared trunk → per-concept μ and log_var heads
        # ------------------------------------------------------------------
        self.posterior_shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.posterior_mu_head = nn.Linear(256, num_concepts * concept_dim)
        self.posterior_log_var_head = nn.Linear(256, num_concepts * concept_dim)

        # ------------------------------------------------------------------
        # 4. Transformation Function f_n
        #    잠재 공간(Latent)에서 표현 공간(Manifestation)으로의 비선형 투영
        #    분포 상에서는 가우시안을 유지하면서, 실제 합산 시 비선형 표현력 확보
        # ------------------------------------------------------------------
        self.transform_fn = nn.Sequential(
            nn.Linear(concept_dim, concept_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(concept_dim * 2, concept_dim),
        )

    def reparameterize(self, mu, log_var):
        """Reparameterization Trick: z = μ + σ ⊙ ε,  ε ~ N(0, I).

        Training 시에는 stochastic sampling, Evaluation 시에는 mean 사용.
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, features):
        """
        Args:
            features: Backbone features [B, input_dim]

        Returns:
            z_concept: Transformed compositional concept embedding [B, concept_dim]
            concept_probs: Concept activation probabilities [B, N]
            concept_logits: Raw concept logits [B, N]
            kl_loss: Weighted KL divergence KL[q(z_k|x) || p(z_k|c_k)] [scalar]
            variational_info: Dict with posterior/prior params for analysis
        """
        B = features.shape[0]

        # --- Step 1: Concept activation (which concepts are present?) ---
        concept_logits = self.concept_classifier(features)
        concept_probs = torch.sigmoid(concept_logits / self.temperature)

        if self.use_soft:
            weights = concept_probs                                     # [B, N]
        else:
            hard = (concept_probs > 0.5).float()
            weights = hard - concept_probs.detach() + concept_probs     # STE

        # --- Step 2: Posterior q(z_k | x) ---
        h = self.posterior_shared(features)                             # [B, 256]
        post_mu = self.posterior_mu_head(h).view(
            B, self.num_concepts, self.concept_dim)                     # [B, N, d]
        post_log_var = self.posterior_log_var_head(h).view(
            B, self.num_concepts, self.concept_dim)                     # [B, N, d]
        post_log_var = torch.clamp(post_log_var, min=-10.0, max=2.0)   # stability

        # --- Step 3: Reparameterization ---
        z_samples = self.reparameterize(post_mu, post_log_var)          # [B, N, d]

        # --- Step 4: Transformation f_n ---
        z_flat = z_samples.reshape(B * self.num_concepts, self.concept_dim)
        z_transformed = self.transform_fn(z_flat).view(
            B, self.num_concepts, self.concept_dim)                     # [B, N, d]

        # --- Step 5: Weighted composition ---
        # z_concept = Σ_k w_k · f_n(z̃_k)
        z_concept = torch.sum(
            weights.unsqueeze(-1) * z_transformed, dim=1)               # [B, d]

        # --- Step 6: KL divergence ---
        # KL[N(μ_q, σ_q²) || N(μ_p, σ_p²)] per dimension
        prior_mu = self.prior_mu.unsqueeze(0)                           # [1, N, d]
        prior_log_var = self.prior_log_var.unsqueeze(0)                 # [1, N, d]

        kl_per_dim = 0.5 * (
            prior_log_var - post_log_var
            + (torch.exp(post_log_var) + (post_mu - prior_mu) ** 2)
              / (torch.exp(prior_log_var) + 1e-8)
            - 1.0
        )                                                               # [B, N, d]

        kl_per_concept = kl_per_dim.sum(dim=-1)                         # [B, N]
        # Weight KL by concept activation (only active concepts contribute)
        weighted_kl = (weights * kl_per_concept).sum(dim=-1)            # [B]
        kl_loss = weighted_kl.mean()                                    # scalar

        variational_info = {
            "posterior_mu": post_mu,         # [B, N, d]
            "posterior_log_var": post_log_var,# [B, N, d]
            "prior_mu": self.prior_mu,       # [N, d]
            "prior_log_var": self.prior_log_var,  # [N, d]
            "z_samples": z_samples,          # [B, N, d] (before transform)
            "z_transformed": z_transformed,  # [B, N, d] (after transform)
        }

        return z_concept, concept_probs, concept_logits, kl_loss, variational_info


# =============================================================================
# Decoder for Reconstruction
# =============================================================================

class LatentDecoder(nn.Module):
    """Decoder: z_total → x̂ (image reconstruction)."""

    def __init__(self, latent_dim, image_channels=3, image_size=64):
        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels

        self.init_size = 8
        self.init_channels = 512

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, self.init_channels * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []

        # 8×8 → 16×16
        layers += [
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(256),
        ]

        # 16×16 → 32×32
        layers += [
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(128),
        ]

        # 32×32 → 64×64
        layers += [
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualConvBlock(64),
        ]

        # 64×64 → 128×128 (image_size >= 128인 경우에만)
        if image_size >= 128:
            layers += [
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2, inplace=True),
                ResidualConvBlock(32),
            ]
            final_ch = 32
        else:
            final_ch = 64

        # Final refinement
        layers += [
            nn.Conv2d(final_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, image_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        ]

        self.decoder_conv = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_size, self.init_size)
        x_recon = self.decoder_conv(h)
        return x_recon


class ResidualConvBlock(nn.Module):
    """디코더 내 Residual Block — 디테일 복원력 향상."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# =============================================================================
# V-CoRes Model
# =============================================================================

class VCoResModel(nn.Module):
    """Variational Compositional Residual Embedding (V-CoRes) Model.

    Hierarchical VAE 구조로, 각 개념을 분포로 표현하여
    의미(Mean)와 불확실성(Variance)을 동시에 학습합니다.

    Architecture:
        ResNet-18 (shared) →  ┌→ VariationalConceptBranch → z_concept (sampled)
                               └→ ResidualBranch           → z_residual
                               → AggregationLayer → z_total ∈ R^D
                               → LatentDecoder    → x̂ (reconstruction)

    Training Objective (ELBO):
        L = L_recon(x, x̂) + L_supervised
            - β · KL[q(z_k|x) || p(z_k|c_k)]
            + α · L_residual_reg
            + γ · L_orthogonality

    At test time, sampling is replaced by the posterior mean for
    deterministic embeddings.
    """

    def __init__(self, latent_dim=64, num_concepts=20,
                 concept_dim=32, residual_dim=32,
                 use_soft_concepts=True, concept_temperature=1.0,
                 aggregation="sum", image_size=64, use_decoder=True,
                 arch="resnet18"):
        """
        Args:
            latent_dim: Total latent dimension budget D.
            num_concepts: Number of binary concepts N.
            concept_dim: Dimension of concept distribution space.
            residual_dim: Dimension of residual embedding space.
            use_soft_concepts: Use soft (probabilistic) concept activation.
            concept_temperature: Temperature for concept activation.
            aggregation: Aggregation method ("sum" or "projection").
            image_size: Input/output image spatial size (for decoder).
            use_decoder: Whether to include decoder for reconstruction.
            arch: Backbone architecture (e.g. "resnet18", "efficientnet_b3").
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_concepts = num_concepts
        self.concept_dim = concept_dim
        self.residual_dim = residual_dim
        self.use_decoder = use_decoder

        # Shared backbone
        self.backbone = SharedBackbone(arch=arch, pretrained=False)

        # Variational concept branch (replaces deterministic ConceptBranch)
        self.concept_branch = VariationalConceptBranch(
            input_dim=self.backbone.feature_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            temperature=concept_temperature,
            use_soft=use_soft_concepts,
        )

        # Residual branch (deterministic — captures fine-grained details)
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

        # Supervised classification head (training signal)
        self.classifier = nn.Linear(latent_dim, num_concepts)

        # Decoder for reconstruction (ELBO L_recon term)
        if use_decoder:
            self.decoder = LatentDecoder(
                latent_dim=latent_dim,
                image_channels=3,
                image_size=image_size,
            )

    def encode(self, x):
        """Extract variational decomposed latent embedding.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z_total: Combined embedding [B, D]
            z_concept: Sampled concept embedding [B, concept_dim]
            z_residual: Residual embedding [B, residual_dim]
            concept_probs: Concept activation probs [B, N]
            concept_logits: Raw concept logits [B, N]
            kl_loss: KL divergence loss [scalar]
            variational_info: Dict with posterior/prior parameters
        """
        # Shared features
        features = self.backbone(x)

        # Variational concept branch
        z_concept, concept_probs, concept_logits, kl_loss, var_info = \
            self.concept_branch(features)

        # Residual branch
        z_residual = self.residual_branch(features)

        # Aggregation: z_total = Σ f_n(z̃_k) + z_res
        z_total = self.aggregation(z_concept, z_residual)

        return (z_total, z_concept, z_residual,
                concept_probs, concept_logits, kl_loss, var_info)

    def decode(self, z_total):
        """Decode latent embedding to reconstructed image.

        Args:
            z_total: Combined embedding [B, D]

        Returns:
            x_recon: Reconstructed image [B, 3, H, W]
        """
        if not self.use_decoder:
            raise RuntimeError("Decoder is disabled (use_decoder=False)")
        return self.decoder(z_total)

    def forward(self, x, x_aug=None):
        """
        Args:
            x: Input images [B, 3, H, W]
            x_aug: Augmented images (optional, for contrastive learning)

        Returns:
            dict with all intermediate and final representations
        """
        (z_total, z_concept, z_residual,
         concept_probs, concept_logits, kl_loss, var_info) = self.encode(x)

        logits = self.classifier(z_total)

        output = {
            "z": z_total,
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
            "concept_logits": concept_logits,
            "logits": logits,
            "kl_loss": kl_loss,
            "variational_info": var_info,
        }

        # Reconstruction
        if self.use_decoder:
            x_recon = self.decode(z_total)
            output["x_recon"] = x_recon

        # Augmented view (for contrastive learning)
        if x_aug is not None:
            (z_total_aug, z_concept_aug, z_residual_aug,
             _, _, kl_loss_aug, _) = self.encode(x_aug)
            output["z_aug"] = z_total_aug
            output["z_concept_aug"] = z_concept_aug
            output["z_residual_aug"] = z_residual_aug
            output["kl_loss_aug"] = kl_loss_aug

        return output

    def get_embedding(self, x):
        """Get the final embedding for evaluation (deterministic).

        Evaluation 시에는 posterior mean을 사용하여 결정론적 embedding 반환.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            z: Normalized total embedding [B, D]
        """
        z_total, _, _, _, _, _, _ = self.encode(x)
        return F.normalize(z_total, dim=-1)

    def get_decomposed_embedding(self, x):
        """Get decomposed embeddings for analysis.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            dict with z_total, z_concept, z_residual, concept_probs,
                 and variational parameters (posterior/prior)
        """
        (z_total, z_concept, z_residual,
         concept_probs, _, _, var_info) = self.encode(x)

        return {
            "z_total": F.normalize(z_total, dim=-1),
            "z_concept": z_concept,
            "z_residual": z_residual,
            "concept_probs": concept_probs,
            "posterior_mu": var_info["posterior_mu"],
            "posterior_log_var": var_info["posterior_log_var"],
            "prior_mu": var_info["prior_mu"],
            "prior_log_var": var_info["prior_log_var"],
        }

    def get_concept_distributions(self):
        """Return the learned prior distributions from the Memory Bank.

        Returns:
            prior_mu: Concept prior means [N, concept_dim]
            prior_std: Concept prior stds  [N, concept_dim]
        """
        mu = self.concept_branch.prior_mu.detach()
        std = torch.exp(0.5 * self.concept_branch.prior_log_var.detach())
        return mu, std

    def sample_from_concept(self, concept_idx, num_samples=1):
        """Sample embeddings from a specific concept's prior distribution.

        Useful for generative analysis and concept interpolation.

        Args:
            concept_idx: Index of the concept to sample from.
            num_samples: Number of samples to draw.

        Returns:
            z_samples: Sampled embeddings [num_samples, concept_dim]
        """
        mu = self.concept_branch.prior_mu[concept_idx]          # [d]
        log_var = self.concept_branch.prior_log_var[concept_idx] # [d]
        std = torch.exp(0.5 * log_var)

        eps = torch.randn(num_samples, self.concept_dim, device=mu.device)
        z_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps    # [S, d]
        return z_samples


# =============================================================================
# V-CoRes Loss
# =============================================================================

class VCoResLoss(nn.Module):
    """Combined loss for the V-CoRes model.

    L_total = L_supervised
              + λ_concept · L_concept
              + λ_recon   · L_recon(x, x̂)          (if decoder enabled)
              + λ_percep  · L_perceptual(x, x̂)     (VGG feature matching)
              + β         · KL[q(z_k|x) || p(z_k|c_k)]
              + α         · L_residual_reg
              + γ         · L_orthogonality
    """

    def __init__(self, concept_weight=1.0, recon_weight=1.0,
                 kl_weight=1.0, residual_reg_weight=0.01,
                 orthogonality_weight=0.1, kl_annealing_epochs=0,
                 perceptual_weight=0.0, use_perceptual=False):
        """
        Args:
            concept_weight: Weight for concept classification loss.
            recon_weight: Weight for reconstruction loss (MSE).
            kl_weight: β — weight for KL divergence (β-VAE style).
            residual_reg_weight: α — weight for residual L2 regularization.
            orthogonality_weight: γ — weight for orthogonality loss.
            kl_annealing_epochs: Number of epochs for KL warm-up.
            perceptual_weight: Weight for VGG perceptual loss.
            use_perceptual: Whether to enable perceptual loss.
        """
        super().__init__()

        self.concept_weight = concept_weight
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.residual_reg_weight = residual_reg_weight
        self.orthogonality_weight = orthogonality_weight
        self.kl_annealing_epochs = kl_annealing_epochs
        self.perceptual_weight = perceptual_weight

        self.concept_criterion = nn.BCEWithLogitsLoss()
        self.supervised_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss(reduction="mean")

        # Perceptual loss (VGG feature matching)
        self.perceptual_loss = None
        if use_perceptual and perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()

    def _get_kl_weight(self, epoch=None):
        """Compute KL weight with optional linear annealing."""
        if epoch is None or self.kl_annealing_epochs <= 0:
            return self.kl_weight
        anneal_factor = min(1.0, epoch / self.kl_annealing_epochs)
        return self.kl_weight * anneal_factor

    def forward(self, output, concept_labels, x_input=None, epoch=None):
        """
        Args:
            output: Dict from VCoResModel.forward()
            concept_labels: Ground truth concept labels [B, N]
            x_input: Original input images [B, 3, H, W] in [0, 1]
            epoch: Current epoch (for KL annealing)

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        concept_logits = output["concept_logits"]
        z_concept = output["z_concept"]
        z_residual = output["z_residual"]
        logits = output["logits"]
        kl_loss = output["kl_loss"]

        # 1. Concept classification loss
        loss_concept = self.concept_criterion(concept_logits, concept_labels)

        # 2. Supervised classification loss on z_total
        loss_supervised = self.supervised_criterion(logits, concept_labels)

        # 3. Reconstruction loss (MSE)
        loss_recon = torch.tensor(0.0, device=z_concept.device)
        if "x_recon" in output and x_input is not None:
            loss_recon = self.recon_criterion(output["x_recon"], x_input)

        # 4. Perceptual loss (VGG feature matching)
        loss_perceptual = torch.tensor(0.0, device=z_concept.device)
        if (self.perceptual_loss is not None
                and "x_recon" in output and x_input is not None):
            loss_perceptual = self.perceptual_loss(output["x_recon"], x_input)

        # 5. KL divergence (with optional annealing)
        effective_kl_weight = self._get_kl_weight(epoch)
        loss_kl = effective_kl_weight * kl_loss

        # 6. Residual regularization
        loss_residual_reg = torch.mean(z_residual ** 2)

        # 7. Orthogonality loss
        z_c_norm = F.normalize(z_concept, dim=-1)
        z_r_norm = F.normalize(z_residual, dim=-1)
        min_dim = min(z_concept.shape[-1], z_residual.shape[-1])
        cos_sim = torch.sum(z_c_norm[:, :min_dim] * z_r_norm[:, :min_dim], dim=-1)
        loss_orthogonality = torch.mean(cos_sim ** 2)

        # Total loss
        total_loss = (
            loss_supervised
            + self.concept_weight * loss_concept
            + self.recon_weight * loss_recon
            + self.perceptual_weight * loss_perceptual
            + loss_kl
            + self.residual_reg_weight * loss_residual_reg
            + self.orthogonality_weight * loss_orthogonality
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_supervised": loss_supervised.item(),
            "loss_concept": loss_concept.item(),
            "loss_recon": loss_recon.item(),
            "loss_perceptual": loss_perceptual.item(),
            "loss_kl": loss_kl.item(),
            "loss_kl_raw": kl_loss.item(),
            "kl_weight": effective_kl_weight,
            "loss_residual_reg": loss_residual_reg.item(),
            "loss_orthogonality": loss_orthogonality.item(),
        }

        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """VGG-based Perceptual Loss — 구조적 선명도 향상.

    VGG-16의 중간 feature map 간 L1 거리를 측정하여,
    픽셀 단위 MSE가 놓치는 고주파 디테일(엣지, 텍스처)을 보존합니다.
    """

    def __init__(self, layers=None, normalize_input=True):
        super().__init__()
        import torchvision.models as models

        if layers is None:
            layers = [3, 8, 15]  # relu1_2, relu2_2, relu3_3

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList()
        prev = 0
        for layer_idx in layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer_idx + 1]))
            prev = layer_idx + 1

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        """[0,1] 범위 입력을 ImageNet 정규화."""
        if self.normalize_input:
            return (x - self.mean) / self.std
        return x

    def forward(self, pred, target):
        """
        Args:
            pred: 재구성 이미지 [B, 3, H, W] in [0, 1]
            target: 원본 이미지 [B, 3, H, W] in [0, 1]
        Returns:
            perceptual_loss: scalar
        """
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)

        return loss
