"""
Seq-CoRes Model (Sequential Compositional Residual Embedding).

자기회귀적 조합 모델 — 이미지를 "자생적 개념 사전(Codebook)에서 단어를
순서대로 꺼내 문장을 만들고, 마지막에 잔차로 마무리"하는 방식으로 임베딩합니다.

Architecture:
    ResNet-18 (shared) → F (visual context)
                         ↓
    AutoRegressiveConceptGenerator (GRU):
        h_0 = 0 (zero vector)       ← 정보 누수 차단 (F는 h_0에 주입되지 않음)
        h_t = GRU(ẽ_{t-1}, h_{t-1}) ← 매 스텝 입력은 이전 코드만 사용
        π_t = Linear([h_t ∥ proj(F)])← F는 코드 선택 확률 계산에만 conditioning
        ẽ_t = Σ_k GumbelSoftmax(π_t)_k · e_k
                         ↓
    Commitment Loss:     ||proj(h_t) - sg(ẽ_t)||^2  (연속공간 ↔ 코드북 정렬)
                         ↓
    ResidualTerminator:  z_res = MLP([F ∥ h_T])
                         ↓
    Final Embedding:     z_total = W · [h_T ∥ z_res]
                         ↓
    Classifier:          ŷ = Linear(z_total)

Key Differences from V-CoRes:
    - Discrete Codebook:    E ∈ R^{K×D} (learnable, VQ-VAE style)
    - Sequential Selection: GRU로 개념을 순차적으로 선택 (인과적 조합)
    - Gumbel-Softmax:       이산 선택의 미분 가능성 확보
    - Zero-Init h_0:        h_0 = 0벡터로 GRU 초기 상태 고정,
                            F는 logit conditioning에서만 사용하여 정보 누수 차단
    - Commitment Loss:      인코더 연속 출력 ↔ 코드북 벡터 정렬 (VQ-VAE β)
    - Batch-wise Entropy Maximization: 배치 전체에서 코드 선택 확률의
                                평균 분포 엔트로피를 최대화하여 코드북 붕괴 방지
    - Dead Code Revival:    사용 빈도 낮은 코드를 인코더 연속 투영값으로 재초기화
    - Residual Terminator:  시퀀스 후 남은 정보만 잔차로 추출, L2 패널티로 억압
    - Two-Phase Training:   Phase 1 (VQ warmup) → Phase 2 (Task + Residual squeezing)

Training Objective:
    L = L_task + λ_commit · L_commitment + λ_res · ||z_res||_2
    + λ_batch_entropy · (log K − H(p̄))   ← 배치 평균 분포 엔트로피 최대화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


# =============================================================================
# Discrete Codebook (자생적 개념 사전)
# =============================================================================

class DiscreteCodebook(nn.Module):
    """Discrete Codebook — 모델이 스스로 학습하는 K개의 개념 임베딩 사전.

    VQ-VAE 스타일의 학습 가능한 이산 코드북으로, 각 슬롯 e_k는
    모델이 데이터를 압축하기 위해 자생적으로 정의한 시각적/의미적 기저(Basis)입니다.

    E ∈ R^{K × D}  (K: 사전 크기, D: 개념 벡터 차원)
    """

    def __init__(self, num_codes: int, code_dim: int,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.0):
        """
        Args:
            num_codes: 코드북 크기 K.
            code_dim:  각 코드 벡터의 차원 D.
            commitment_cost: Commitment loss 가중치 (VQ-VAE β).
            ema_decay: EMA 업데이트 decay (0이면 EMA 비활성화, gradient 사용).
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # 코드북 임베딩 행렬
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_codes, 1.0 / num_codes)

        # Dead Code Revival: 코드별 사용 빈도 추적
        self.register_buffer("_usage_count", torch.zeros(num_codes))
        self.register_buffer("_total_batches", torch.tensor(0, dtype=torch.long))

        # EMA 추적 변수 (ema_decay > 0일 때만 사용)
        if ema_decay > 0:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("_ema_w", self.embedding.weight.data.clone())
        else:
            self._ema_cluster_size = None
            self._ema_w = None

    @property
    def codebook(self) -> torch.Tensor:
        """코드북 가중치 반환 [K, D]."""
        return self.embedding.weight

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """인덱스로 코드북 벡터 조회.

        Args:
            indices: 정수 인덱스 [...].

        Returns:
            vectors: 코드북 벡터 [..., D].
        """
        return self.embedding(indices)

    def compute_distances(self, z: torch.Tensor) -> torch.Tensor:
        """연속 벡터와 코드북 간 L2 거리 계산.

        Args:
            z: 연속 벡터 [*, D].

        Returns:
            distances: [*, K].
        """
        flat_z = z.reshape(-1, self.code_dim)  # [N, D]
        # ||z - e||^2 = ||z||^2 - 2⟨z, e⟩ + ||e||^2
        d = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)    # [N, 1]
            - 2 * flat_z @ self.codebook.t()                # [N, K]
            + torch.sum(self.codebook ** 2, dim=1)          # [K]
        )
        return d.view(*z.shape[:-1], self.num_codes)

    def quantize(self, z: torch.Tensor):
        """Nearest-neighbor 양자화 (Hard).

        Args:
            z: 연속 벡터 [*, D].

        Returns:
            z_q: 양자화된 벡터 [*, D] (STE 통과).
            indices: 선택된 인덱스 [*].
            vq_loss: Commitment + codebook loss (scalar).
        """
        distances = self.compute_distances(z)  # [*, K]
        indices = distances.argmin(dim=-1)      # [*]
        z_q = self.lookup(indices)              # [*, D]

        # VQ Loss
        if self.training:
            codebook_loss = F.mse_loss(z_q.detach(), z)  # commitment
            embedding_loss = F.mse_loss(z_q, z.detach())  # codebook update
            vq_loss = embedding_loss + self.commitment_cost * codebook_loss

            # EMA update (optional)
            if self.ema_decay > 0:
                self._ema_update(z, indices)

            # Straight-Through Estimator
            z_q = z + (z_q - z).detach()
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
            z_q = z_q  # no STE at eval

        return z_q, indices, vq_loss

    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """EMA 방식으로 코드북 업데이트."""
        flat_z = z.reshape(-1, self.code_dim)
        flat_idx = indices.reshape(-1)

        # One-hot encoding
        encodings = F.one_hot(flat_idx, self.num_codes).float()  # [N, K]

        # EMA cluster size
        self._ema_cluster_size.mul_(self.ema_decay).add_(
            encodings.sum(0), alpha=1 - self.ema_decay
        )
        # Laplace smoothing
        n = self._ema_cluster_size.sum()
        self._ema_cluster_size = (
            (self._ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
        )

        # EMA weights
        dw = encodings.t() @ flat_z  # [K, D]
        self._ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

        # Normalize
        self.embedding.weight.data.copy_(
            self._ema_w / self._ema_cluster_size.unsqueeze(1)
        )

    def update_usage(self, indices: torch.Tensor):
        """배치에서 사용된 코드 인덱스의 사용 빈도를 업데이트.

        Args:
            indices: 선택된 코드 인덱스 텐서 (임의 shape).
        """
        if not self.training:
            return
        flat = indices.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_codes).float()
        self._usage_count.add_(counts)
        self._total_batches.add_(1)

    def restart_dead_codes(self, features: torch.Tensor,
                           dead_threshold: float = 1.0):
        """Dead Code Revival — 사용 빈도가 낮은 코드를 재초기화.

        추적된 사용 빈도에서 dead_threshold 미만인 코드를 '죽은 코드'로
        판별하고, 현재 배치의 인코더 출력(features)에서 임의 샘플로
        코드북 값을 덮어씌웁니다.

        Args:
            features: 현재 배치의 인코더 출력 [B, D] (코드북과 같은 차원).
            dead_threshold: 이 값 미만 사용된 코드를 죽은 코드로 판별.

        Returns:
            num_restarted: 재초기화된 코드 수.
        """
        if self._total_batches.item() == 0:
            return 0

        dead_mask = self._usage_count < dead_threshold  # [K]
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return 0

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]  # 죽은 코드 인덱스

        # features에서 랜덤 샘플링하여 죽은 코드를 덮어씌움
        flat_features = features.detach().reshape(-1, self.code_dim)
        num_available = flat_features.shape[0]

        if num_available == 0:
            return 0

        # 랜덤 인덱스 선택 (복원 추출)
        rand_indices = torch.randint(0, num_available, (num_dead,),
                                     device=features.device)
        new_codes = flat_features[rand_indices]

        # 약간의 노이즈 추가 (동일 코드 방지)
        noise = torch.randn_like(new_codes) * 0.01
        new_codes = new_codes + noise

        # 코드북 업데이트
        with torch.no_grad():
            self.embedding.weight.data[dead_indices] = new_codes

        return num_dead

    def reset_usage_stats(self):
        """사용 빈도 통계 초기화 (에포크 단위)."""
        self._usage_count.zero_()
        self._total_batches.zero_()

    def get_usage_stats(self):
        """코드북 사용 통계 반환."""
        total = self._total_batches.item()
        usage = self._usage_count.clone()
        alive = (usage > 0).sum().item()
        return {
            "usage_count": usage,
            "total_batches": total,
            "alive_codes": alive,
            "dead_codes": self.num_codes - alive,
            "utilization": alive / self.num_codes,
        }

    def forward(self, z: torch.Tensor):
        """Forward pass: 양자화 수행.

        Args:
            z: 연속 벡터 [*, D].

        Returns:
            z_q, indices, vq_loss
        """
        return self.quantize(z)


# =============================================================================
# Auto-Regressive Concept Generator (자기회귀 개념 생성기)
# =============================================================================

class AutoRegressiveConceptGenerator(nn.Module):
    """GRU 기반 자기회귀 개념 생성기.

    정보 누수 차단 설계:
      h_0 = 0 (zero vector) — F의 정보가 h_T로 직접 흘러가는 것을 원천 차단.
      h_T는 오직 "선택된 코드들의 조합"만으로 구성됩니다.
      F는 logit conditioning 한 경로로만 사용됩니다.

    h_0 = 0                       ← 영벡터로 고정 (정보 누수 차단)
    h_t = GRU(ẽ_{t-1}, h_{t-1})   ← 입력은 이전 코드만
    π_t = Linear([h_t ∥ proj(F)]) ∈ R^K
    ẽ_t = Σ_k GumbelSoftmax(π_t)_k · e_k

    Commitment Loss:
      z_cont_t = Linear(h_t)  →  code space R^D
      L_commit = Σ_t ||z_cont_t - sg(ẽ_t)||^2
    """

    def __init__(self, visual_dim: int, code_dim: int, num_codes: int,
                 hidden_dim: int = 256, max_steps: int = 8,
                 gumbel_tau_init: float = 1.0, gumbel_tau_min: float = 0.1,
                 num_gru_layers: int = 1):
        """
        Args:
            visual_dim:  시각 특징 차원 (ResNet-18: 512).
            code_dim:    코드북 벡터 차원 D.
            num_codes:   코드북 크기 K.
            hidden_dim:  GRU 은닉 상태 차원.
            max_steps:   최대 시퀀스 길이 T.
            gumbel_tau_init: Gumbel-Softmax 초기 temperature.
            gumbel_tau_min:  Gumbel-Softmax 최소 temperature.
            num_gru_layers:  GRU 레이어 수.
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.num_gru_layers = num_gru_layers

        # Gumbel-Softmax temperature (외부에서 annealing)
        self.register_buffer("gumbel_tau", torch.tensor(gumbel_tau_init))
        self.gumbel_tau_min = gumbel_tau_min

        # <BOS> 토큰 (학습 가능한 시작 벡터)
        self.bos_token = nn.Parameter(torch.randn(code_dim) * 0.02)

        # GRU: input = ẽ_{t-1} only (F는 절대 GRU에 유입되지 않음)
        self.gru = nn.GRU(
            input_size=code_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1 if num_gru_layers > 1 else 0.0,
        )

        # 초기 은닉 상태: h_0 = 0벡터 (정보 누수 차단)
        # F의 정보가 h_T로 직접 흘러가지 않도록 영벡터로 고정.
        # 모델은 매 스텝 f_cond를 보고 유의미한 코드를 선택하여
        # GRU에 넣어야만 h_T에 이미지 정보를 담을 수 있습니다.
        # (init_proj 제거됨)

        # 시각 특징 F → conditioning 프로젝션 (logit 계산용)
        self.visual_cond_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # [h_t ∥ proj(F)] → 코드북 logits (F는 여기서만 conditioning)
        self.logit_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_codes),
        )

        # Commitment: h_t → code space 프로젝션 (z_continuous)
        self.commitment_proj = nn.Sequential(
            nn.Linear(hidden_dim, code_dim),
        )

    def set_gumbel_tau(self, tau: float):
        """Gumbel-Softmax temperature 설정 (외부 annealing)."""
        tau = max(tau, self.gumbel_tau_min)
        self.gumbel_tau.fill_(tau)

    def forward(self, features: torch.Tensor, codebook: torch.Tensor):
        """
        Args:
            features: 시각 특징 F [B, visual_dim].
            codebook: 코드북 임베딩 E [K, D].

        Returns:
            h_final:          GRU 최종 은닉 상태 [B, hidden_dim].
            selected_codes:   매 스텝 선택된 코드북 벡터 [B, T, D].
            all_logits:       매 스텝 코드북 logits [B, T, K].
            all_indices:      매 스텝 선택된 인덱스 (hard) [B, T].
            all_soft_weights: 매 스텝 Gumbel-Softmax 가중치 [B, T, K].
            commitment_loss:  Commitment Loss (scalar).
        """
        B = features.shape[0]

        # 초기 은닉 상태: h_0 = 0벡터 (정보 누수 차단)
        # F의 정보가 h_T로 직접 흘러가지 않도록 영벡터로 고정
        h = torch.zeros(self.num_gru_layers, B, self.hidden_dim,
                        device=features.device,
                        dtype=features.dtype)  # [num_layers, B, hidden_dim]

        # F → conditioning 벡터 (logit 계산용)
        f_cond = self.visual_cond_proj(features)  # [B, hidden_dim]

        # <BOS> 토큰으로 시작
        prev_code = self.bos_token.unsqueeze(0).expand(B, -1)  # [B, D]

        all_selected_codes = []
        all_logits = []
        all_indices = []
        all_soft_weights = []
        all_z_continuous = []  # Commitment Loss용

        for t in range(self.max_steps):
            # GRU 입력: ẽ_{t-1} only (F는 h_0과 logit에서만 사용)
            gru_input = prev_code.unsqueeze(1)  # [B, 1, D]

            # GRU step
            gru_out, h = self.gru(gru_input, h)  # gru_out: [B, 1, hidden_dim]
            h_t = gru_out.squeeze(1)              # [B, hidden_dim]

            # 코드북 logits: [h_t ∥ f_cond] → F는 여기서만 conditioning
            logit_input = torch.cat([h_t, f_cond], dim=-1)  # [B, 2*hidden_dim]
            logits = self.logit_head(logit_input)  # [B, K]

            all_logits.append(logits)

            # Commitment: h_t → code space 연속 프로젝션
            z_cont = self.commitment_proj(h_t)  # [B, code_dim]
            all_z_continuous.append(z_cont)

            # Gumbel-Softmax (미분 가능한 이산 선택)
            if self.training:
                soft_weights = F.gumbel_softmax(
                    logits, tau=self.gumbel_tau.item(), hard=False, dim=-1
                )  # [B, K]
                # Straight-Through: hard argmax으로 forward, soft로 backward
                hard_indices = logits.argmax(dim=-1)  # [B]
                hard_onehot = F.one_hot(hard_indices, self.num_codes).float()
                # STE: forward는 hard, backward는 soft gradient
                ste_weights = hard_onehot - soft_weights.detach() + soft_weights
                selected_code = ste_weights @ codebook  # [B, D]
            else:
                # 평가 시 hard selection
                hard_indices = logits.argmax(dim=-1)
                soft_weights = F.one_hot(hard_indices, self.num_codes).float()
                selected_code = codebook[hard_indices]  # [B, D]

            all_indices.append(hard_indices if not self.training else logits.argmax(dim=-1))
            all_soft_weights.append(soft_weights)
            all_selected_codes.append(selected_code)

            # 다음 스텝 입력
            prev_code = selected_code

        # Stack results: [B, T, ...]
        selected_codes = torch.stack(all_selected_codes, dim=1)   # [B, T, D]
        all_logits = torch.stack(all_logits, dim=1)               # [B, T, K]
        all_indices = torch.stack(all_indices, dim=1)             # [B, T]
        all_soft_weights = torch.stack(all_soft_weights, dim=1)   # [B, T, K]
        all_z_continuous = torch.stack(all_z_continuous, dim=1)   # [B, T, D]

        # Commitment Loss: ||z_continuous - sg(selected_codes)||^2
        # 인코더의 연속 출력이 선택된 코드북 벡터에서 멀어지지 않도록
        if self.training:
            commitment_loss = F.mse_loss(
                all_z_continuous, selected_codes.detach()
            )
        else:
            commitment_loss = torch.tensor(0.0, device=features.device)

        # 최종 은닉 상태 (마지막 레이어)
        h_final = h[-1]  # [B, hidden_dim]

        return (h_final, selected_codes, all_logits, all_indices,
                all_soft_weights, commitment_loss, all_z_continuous)


# =============================================================================
# Residual Terminator (잔차 터미네이터)
# =============================================================================

class ResidualTerminator(nn.Module):
    """잔차 터미네이터 — 시퀀스 생성 후 남은 정보를 추출.

    "이때까지 네가 뱉어낸 T개의 개념들(h_T)만으로는 설명되지 않는
     남은 정보가 무엇인가?" 를 계산하여 연속적인 벡터 z_res로 출력.

    z_res = MLP([F ∥ h_T])

    L2 패널티로 z_res의 크기를 극단적으로 억압 →
    모델은 코드북 조합으로 최대한 설명하도록 강제됨.
    """

    def __init__(self, visual_dim: int, hidden_dim: int, residual_dim: int):
        """
        Args:
            visual_dim:   시각 특징 차원.
            hidden_dim:   GRU 은닉 상태 차원.
            residual_dim: 잔차 벡터 차원.
        """
        super().__init__()

        self.residual_mlp = nn.Sequential(
            nn.Linear(visual_dim + hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, residual_dim),
        )

    def forward(self, features: torch.Tensor, h_final: torch.Tensor):
        """
        Args:
            features: 시각 특징 F [B, visual_dim].
            h_final:  GRU 최종 은닉 상태 h_T [B, hidden_dim].

        Returns:
            z_res: 잔차 임베딩 [B, residual_dim].
        """
        concat = torch.cat([features, h_final], dim=-1)
        z_res = self.residual_mlp(concat)
        return z_res


# =============================================================================
# Reconstruction Decoder (Phase 1 VQ 워밍업용)
# =============================================================================

class SeqCoResDecoder(nn.Module):
    """Phase 1 VQ 워밍업을 위한 디코더.

    코드북이 의미 있는 시각적 기저로 초기화되도록
    이미지 재구성 태스크를 수행합니다.
    """

    def __init__(self, latent_dim: int, image_channels: int = 3,
                 image_size: int = 64):
        super().__init__()

        self.image_size = image_size
        self.init_size = 8
        self.init_channels = 256

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.init_channels * self.init_size * self.init_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        # 8→16
        layers += [
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # 16→32
        layers += [
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        # 32→64
        layers += [
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if image_size >= 128:
            layers += [
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            final_ch = 16
        else:
            final_ch = 32

        layers += [
            nn.Conv2d(final_ch, image_channels, 3, padding=1),
            nn.Sigmoid(),
        ]

        self.decoder_conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        h = self.fc(z)
        h = h.view(-1, self.init_channels, self.init_size, self.init_size)
        return self.decoder_conv(h)


# =============================================================================
# Seq-CoRes Model (통합 모델)
# =============================================================================

class SeqCoResModel(nn.Module):
    """Sequential Compositional Residual Embedding (Seq-CoRes) Model.

    자기회귀적 조합 모델 — 자생적 코드북에서 개념을 순서대로 선택하고,
    남은 정보만 잔차로 추출하여 임베딩을 구성합니다.

    Architecture:
        Visual Encoder (ResNet-18):
            X → F ∈ R^{512}

        Auto-Regressive Concept Generator (GRU):
            (F, <BOS>) → c_1, c_2, ..., c_T  (코드북 인덱스)
            h_T (최종 은닉 상태: 개념의 인과적 문맥 요약)

        Residual Terminator:
            z_res = MLP([F ∥ h_T])

        Final Embedding:
            z_total = W · [h_T ∥ z_res]

    Training (Two-Phase):
        Phase 1: VQ Warmup (코드북 초기화, 재구성 학습)
        Phase 2: Task + Residual Squeezing (분류 + 잔차 억압)
    """

    def __init__(self, latent_dim: int = 64, num_codes: int = 128,
                 code_dim: int = 32, hidden_dim: int = 256,
                 residual_dim: int = 32, max_steps: int = 8,
                 num_concepts: int = 20,
                 gumbel_tau_init: float = 1.0, gumbel_tau_min: float = 0.1,
                 commitment_cost: float = 0.25,
                 image_size: int = 64, use_decoder: bool = True,
                 num_gru_layers: int = 1):
        """
        Args:
            latent_dim:       최종 임베딩 차원 D.
            num_codes:        코드북 크기 K (자생적 개념 수).
            code_dim:         각 코드 벡터 차원.
            hidden_dim:       GRU 은닉 상태 차원.
            residual_dim:     잔차 벡터 차원.
            max_steps:        최대 시퀀스 길이 T.
            num_concepts:     다운스트림 태스크 개념 수 (분류 헤드).
            gumbel_tau_init:  Gumbel-Softmax 초기 temperature.
            gumbel_tau_min:   Gumbel-Softmax 최소 temperature.
            commitment_cost:  VQ Commitment loss 가중치.
            image_size:       입출력 이미지 공간 크기.
            use_decoder:      디코더 포함 여부 (Phase 1용).
            num_gru_layers:   GRU 레이어 수.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.residual_dim = residual_dim
        self.max_steps = max_steps
        self.num_concepts = num_concepts
        self.use_decoder = use_decoder

        # 1. 시각적 인코더 (Shared Backbone)
        self.backbone = SharedBackbone(pretrained=False)
        self.visual_dim = self.backbone.feature_dim  # 512

        # 2. 자생적 개념 사전 (Discrete Codebook)
        self.codebook = DiscreteCodebook(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_cost=commitment_cost,
        )

        # 3. 자기회귀 개념 생성기
        self.ar_generator = AutoRegressiveConceptGenerator(
            visual_dim=self.visual_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            gumbel_tau_init=gumbel_tau_init,
            gumbel_tau_min=gumbel_tau_min,
            num_gru_layers=num_gru_layers,
        )

        # 4. 잔차 터미네이터
        self.residual_terminator = ResidualTerminator(
            visual_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            residual_dim=residual_dim,
        )

        # 5. 최종 임베딩 프로젝션: [h_T ∥ z_res] → z_total
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim + residual_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # 6. 분류 헤드 (다운스트림 태스크)
        self.classifier = nn.Linear(latent_dim, num_concepts)

        # 7. 디코더 (Phase 1 VQ 워밍업용)
        if use_decoder:
            self.decoder = SeqCoResDecoder(
                latent_dim=latent_dim,
                image_channels=3,
                image_size=image_size,
            )
            # Phase 1 디코더 입력 프로젝션: T*code_dim → latent_dim
            # Phase 1에서는 h_T를 차단하고 시퀀스 flatten으로 재구성 강제
            # (Sum 대신 flatten으로 시퀀스 순서 정보 보존)
            self.code_seq_proj = nn.Sequential(
                nn.Linear(max_steps * code_dim, latent_dim * 2),
                nn.LayerNorm(latent_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(latent_dim * 2, latent_dim),
            )

    def set_gumbel_tau(self, tau: float):
        """Gumbel-Softmax temperature 외부 설정."""
        self.ar_generator.set_gumbel_tau(tau)

    def get_gumbel_tau(self) -> float:
        """현재 Gumbel-Softmax temperature 반환."""
        return self.ar_generator.gumbel_tau.item()

    def encode(self, x: torch.Tensor):
        """이미지를 분해된 임베딩으로 인코딩.

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            z_total:          최종 임베딩 [B, D].
            h_final:          GRU 최종 은닉 상태 [B, hidden_dim].
            z_res:            잔차 벡터 [B, residual_dim].
            selected_codes:   선택된 코드 시퀀스 [B, T, code_dim].
            all_logits:       코드북 logits [B, T, K].
            all_indices:      선택된 인덱스 [B, T].
            all_soft_weights: Gumbel-Softmax 가중치 [B, T, K].
            commitment_loss:  Commitment Loss (scalar).
        """
        # 시각 특징 추출
        features = self.backbone(x)  # [B, 512]

        # 자기회귀 개념 생성
        (h_final, selected_codes, all_logits, all_indices,
         all_soft_weights, commitment_loss, all_z_continuous) = \
            self.ar_generator(features, self.codebook.codebook)

        # 잔차 추출
        z_res = self.residual_terminator(features, h_final)

        # 최종 임베딩
        concat = torch.cat([h_final, z_res], dim=-1)  # [B, hidden+residual]
        z_total = self.final_projection(concat)        # [B, D]

        return (z_total, h_final, z_res, selected_codes,
                all_logits, all_indices, all_soft_weights,
                commitment_loss, all_z_continuous)

    def decode(self, z_total: torch.Tensor):
        """잠재 임베딩으로부터 이미지 재구성.

        Args:
            z_total: 최종 임베딩 [B, D].

        Returns:
            x_recon: 재구성된 이미지 [B, 3, H, W].
        """
        if not self.use_decoder:
            raise RuntimeError("Decoder disabled (use_decoder=False)")
        return self.decoder(z_total)

    def forward(self, x: torch.Tensor, x_aug: torch.Tensor = None,
                phase: int = 2):
        """
        Args:
            x:     입력 이미지 [B, 3, H, W].
            x_aug: 증강된 이미지 (대조 학습용, optional).
            phase: 학습 페이즈 (1: VQ Warmup, 2: Task).

        Returns:
            output dict.
        """
        (z_total, h_final, z_res, selected_codes,
         all_logits, all_indices, all_soft_weights,
         commitment_loss, all_z_continuous) = self.encode(x)

        logits = self.classifier(z_total)

        output = {
            "z": z_total,                       # [B, D]
            "h_final": h_final,                 # [B, hidden_dim]
            "z_residual": z_res,                # [B, residual_dim]
            "selected_codes": selected_codes,   # [B, T, code_dim]
            "concept_logits": all_logits,       # [B, T, K]
            "concept_indices": all_indices,     # [B, T]
            "soft_weights": all_soft_weights,   # [B, T, K]
            "logits": logits,                   # [B, num_concepts]
            "commitment_loss": commitment_loss, # scalar
            "z_continuous": all_z_continuous,   # [B, T, code_dim] (Dead Code Revival용)
        }

        # 재구성
        if self.use_decoder:
            if phase == 1:
                # Phase 1: h_T 차단 — 코드 시퀀스를 flatten하여 시퀀스 정보 보존
                # Sum을 사용하면 시퀀스 의미가 사라지므로, flatten [B, T*code_dim]으로 전달
                B_dec = selected_codes.shape[0]
                z_decoder_input_flat = selected_codes.view(B_dec, -1)  # [B, T*code_dim]
                z_decoder_input = self.code_seq_proj(z_decoder_input_flat)  # [B, latent_dim]
                output["x_recon"] = self.decode(z_decoder_input)
                output["z_decoder_input"] = z_decoder_input_flat
            else:
                # Phase 2: z_total (h_T + z_res 포함) 사용
                output["x_recon"] = self.decode(z_total)

        # 증강 뷰 (대조 학습)
        if x_aug is not None:
            (z_aug, h_aug, z_res_aug, _, _, _, _, _, _) = self.encode(x_aug)
            output["z_aug"] = z_aug
            output["z_residual_aug"] = z_res_aug

        return output

    def get_embedding(self, x: torch.Tensor):
        """평가용 결정론적 임베딩 반환.

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            z: 정규화된 최종 임베딩 [B, D].
        """
        z_total, _, _, _, _, _, _, _, _ = self.encode(x)
        return F.normalize(z_total, dim=-1)

    def get_decomposed_embedding(self, x: torch.Tensor):
        """분해된 임베딩 반환 (분석용).

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            dict with z_total, h_final, z_res, concept_indices, selected_codes.
        """
        (z_total, h_final, z_res, selected_codes,
         all_logits, all_indices, all_soft_weights, _, _) = self.encode(x)

        return {
            "z_total": F.normalize(z_total, dim=-1),
            "h_final": h_final,
            "z_residual": z_res,
            "concept_indices": all_indices,
            "selected_codes": selected_codes,
            "soft_weights": all_soft_weights,
            "residual_norm": torch.norm(z_res, dim=-1),
        }

    def get_concept_sequence(self, x: torch.Tensor):
        """이미지의 개념 시퀀스 추출 (해석 가능성 분석용).

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            indices:  코드북 인덱스 시퀀스 [B, T].
            codes:    코드북 벡터 시퀀스 [B, T, D].
            h_final:  GRU 최종 은닉 상태 [B, hidden_dim].
        """
        (_, h_final, _, selected_codes, _, all_indices, _, _, _) = self.encode(x)
        return all_indices, selected_codes, h_final

    def get_codebook_utilization(self):
        """코드북 사용률 통계 반환 (collapse 감지)."""
        return {
            "codebook_weights": self.codebook.codebook.detach().clone(),
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
        }


# =============================================================================
# Seq-CoRes Loss
# =============================================================================

class SeqCoResLoss(nn.Module):
    """Seq-CoRes 통합 손실 함수.

    L = L_task + λ_commit · L_commitment + λ_res · ||z_res||_2
        + λ_recon · L_recon
        + λ_batch_entropy · (log K − H(p̄))   ← 배치 평균 분포 엔트로피 최대화

    Phase 1 (VQ Warmup):
        L = L_recon + λ_commit · L_commitment
        (코드북 시각적 기저 초기화 + 연속공간↔코드북 정렬)

    Phase 2 (Task + Residual Squeezing):
        L = L_task + λ_commit · L_commitment + λ_res(epoch) · ||z_res||_2
        + λ_phase2_recon · L_recon + λ_batch_entropy · L_batch_entropy
        (재구성 가중치를 낮춰 Regularization 역할로 유지,
         잔차 패널티 점진적 강화, 배치 단위 코드 사용 균등화)
    """

    def __init__(self, task_weight: float = 1.0,
                 vq_weight: float = 1.0,
                 recon_weight: float = 1.0,
                 phase2_recon_weight: float = 0.2,
                 residual_penalty_weight: float = 0.1,
                 batch_entropy_weight: float = 0.1,
                 phase1_batch_entropy_weight: float = 5.0,
                 commitment_weight: float = 0.25,
                 residual_annealing_start: int = 0,
                 residual_annealing_end: int = 50,
                 phase: int = 2):
        """
        Args:
            task_weight:              L_task 가중치.
            vq_weight:                L_vq (Commitment + Codebook) 가중치.
            recon_weight:             L_recon 가중치 (Phase 1).
            phase2_recon_weight:      L_recon 가중치 (Phase 2, Regularization 역할, 낮게).
            residual_penalty_weight:  λ_res 최대치 (Phase 2에서 annealing).
            batch_entropy_weight:     배치 엔트로피 최대화 가중치 (Phase 2, collapse 방지).
            phase1_batch_entropy_weight: Phase 1 배치 엔트로피 가중치 (높게).
            commitment_weight:        Commitment loss 가중치 (β).
            residual_annealing_start: 잔차 패널티 시작 에포크.
            residual_annealing_end:   잔차 패널티 최대 도달 에포크.
            phase:                    학습 페이즈 (1 또는 2).
        """
        super().__init__()

        self.task_weight = task_weight
        self.vq_weight = vq_weight
        self.recon_weight = recon_weight
        self.phase2_recon_weight = phase2_recon_weight
        self.residual_penalty_weight = residual_penalty_weight
        self.batch_entropy_weight = batch_entropy_weight
        self.phase1_batch_entropy_weight = phase1_batch_entropy_weight
        self.commitment_weight = commitment_weight
        self.residual_annealing_start = residual_annealing_start
        self.residual_annealing_end = residual_annealing_end
        self.phase = phase

        self.task_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss(reduction="mean")

    def set_phase(self, phase: int):
        """학습 페이즈 전환."""
        self.phase = phase

    def _get_residual_weight(self, epoch: int = None):
        """잔차 패널티 가중치 (점진적 강화 annealing)."""
        if epoch is None or self.residual_annealing_end <= self.residual_annealing_start:
            return self.residual_penalty_weight
        if epoch < self.residual_annealing_start:
            return 0.0
        progress = min(1.0,
                       (epoch - self.residual_annealing_start)
                       / max(1, self.residual_annealing_end - self.residual_annealing_start))
        return self.residual_penalty_weight * progress

    def _batch_entropy_maximization(self, soft_weights: torch.Tensor):
        """배치 단위 소프트 엔트로피 최대화 (Batch-wise Entropy Maximization).

        배치 전체에서 코드 선택 확률을 배치×스텝 차원으로 평균 내어
        K개 코드에 대한 평균 사용 분포 p̄ ∈ R^K를 구합니다.
        이 분포의 엔트로피 H(p̄)를 최대화하여 (≈ log K)
        모든 코드가 균등하게 사용되도록 유도합니다.

        개별 샘플은 특정 코드에 집중(샤프 선택)하되,
        배치 전체로 보면 모든 코드가 골고루 쓰이는 효과를 냅니다.

        Loss = log(K) - H(p̄)  → 0이면 완벽한 균일 분포.

        Args:
            soft_weights: Gumbel-Softmax 가중치 [B, T, K].

        Returns:
            batch_entropy_loss: scalar (낮을수록 균일 분포에 가까움).
        """
        # 배치 × 스텝 차원 평균 → 코드별 평균 사용 확률 [K]
        avg_probs = soft_weights.mean(dim=(0, 1))  # [K]

        # 엔트로피: H(p̄) = -Σ p̄_k log(p̄_k)
        log_probs = (avg_probs + 1e-10).log()
        entropy = -(avg_probs * log_probs).sum()

        # 최대 엔트로피 (uniform): log(K)
        K = avg_probs.shape[0]
        max_entropy = torch.log(
            torch.tensor(K, device=avg_probs.device, dtype=avg_probs.dtype)
        )

        # Loss = log(K) - H(p̄)  (0이면 완벽한 균일 분포)
        return max_entropy - entropy

    def forward(self, output: dict, concept_labels: torch.Tensor,
                x_input: torch.Tensor = None, epoch: int = None):
        """
        Args:
            output:         SeqCoResModel.forward()의 출력 dict.
            concept_labels: 정답 개념 레이블 [B, N].
            x_input:        원본 이미지 [B, 3, H, W] (재구성용).
            epoch:          현재 에포크 (annealing용).

        Returns:
            total_loss: 총 손실.
            loss_dict:  개별 손실 딕셔너리.
        """
        z_res = output["z_residual"]
        logits = output["logits"]
        soft_weights = output["soft_weights"]
        device = z_res.device

        loss_dict = {}

        # =====================================================================
        # Phase 1: VQ Warmup (코드북 초기화)
        # =====================================================================
        if self.phase == 1:
            # 재구성 손실
            loss_recon = torch.tensor(0.0, device=device)
            if "x_recon" in output and x_input is not None:
                loss_recon = self.recon_criterion(output["x_recon"], x_input)

            # Commitment Loss (Phase 1에서도 활성화 — 코드북 정렬)
            loss_commit = output.get("commitment_loss", torch.tensor(0.0, device=device))
            if isinstance(loss_commit, (int, float)):
                loss_commit = torch.tensor(loss_commit, device=device)

            # 배치 엔트로피 최대화 (Phase 1에서도 활성화 — 코드북 붕괴 방지)
            # 초기에 코드를 골고루 쓰도록 높은 가중치 적용
            loss_batch_entropy = self._batch_entropy_maximization(soft_weights)

            total_loss = (self.recon_weight * loss_recon
                          + self.commitment_weight * loss_commit
                          + self.phase1_batch_entropy_weight * loss_batch_entropy)
            loss_dict = {
                "loss_total": total_loss.item(),
                "loss_recon": loss_recon.item(),
                "loss_commitment": loss_commit.item(),
                "loss_batch_entropy": loss_batch_entropy.item(),
                "phase": 1,
            }
            return total_loss, loss_dict

        # =====================================================================
        # Phase 2: Task + Residual Squeezing
        # =====================================================================

        # 1. Task Loss (분류)
        loss_task = self.task_criterion(logits, concept_labels)

        # 2. 잔차 L2 패널티 (핵심! 점진적 강화)
        effective_res_weight = self._get_residual_weight(epoch)
        loss_res_penalty = torch.norm(z_res, p=2, dim=-1).mean()

        # 3. 배치 엔트로피 최대화 (codebook collapse 방지)
        loss_batch_entropy = self._batch_entropy_maximization(soft_weights)

        # 4. Commitment Loss (연속공간 ↔ 코드북 정렬)
        loss_commit = output.get("commitment_loss", torch.tensor(0.0, device=device))
        if isinstance(loss_commit, (int, float)):
            loss_commit = torch.tensor(loss_commit, device=device)

        # 5. 재구성 (Phase 2에서도 낮은 가중치로 유지 — Regularization 역할)
        #    코드북이 시각적 디테일을 유지하도록 강제하여 collapse 방지
        loss_recon = torch.tensor(0.0, device=device)
        if "x_recon" in output and x_input is not None:
            loss_recon = self.recon_criterion(output["x_recon"], x_input)

        # 총 손실
        total_loss = (
            self.task_weight * loss_task
            + effective_res_weight * loss_res_penalty
            + self.batch_entropy_weight * loss_batch_entropy
            + self.commitment_weight * loss_commit
            + self.phase2_recon_weight * loss_recon
        )

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_task": loss_task.item(),
            "loss_res_penalty": loss_res_penalty.item(),
            "loss_batch_entropy": loss_batch_entropy.item(),
            "loss_commitment": loss_commit.item(),
            "loss_recon": loss_recon.item(),
            "effective_res_weight": effective_res_weight,
            "residual_norm": torch.norm(z_res, p=2, dim=-1).mean().item(),
            "phase": 2,
        }

        return total_loss, loss_dict
