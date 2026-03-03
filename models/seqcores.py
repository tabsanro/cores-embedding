"""
Seq-CoRes Model v3 (Sequential Compositional Residual Embedding).

v2 → v3 주요 변경:
    1. 하향식 공간 주의 (Top-down Spatial Attention)
       - 고정된 1D 벡터(F) 대신 공간 특징맵(C×H×W)을 유지
       - GRU의 h_t가 Query로 이미지의 어디를 볼지 Cross-Attention으로 결정
       - 매 스텝마다 이미지를 '능동적으로 재관찰' → 궤도 발산 방지
    2. 개념 레이블 직접 지도학습 (v2 유지)
    3. 직교 투영 잔차 제약 (v2 유지)
    4. VQ-VAE STE + EMA (v2 유지)
    5. Phase 1/2 표현 통일 (v2 유지)

Architecture:
    ResNet-18 (shared, 글로벌 풀링 제거) → F_spatial (C×H'×W')
                                          ↓
    AutoRegressiveConceptGenerator (GRU + Cross-Attention + VQ):
        h_0 = 0 (zero vector)               ← 정보 누수 차단
        h_t = GRU(z_q_{t-1}, h_{t-1})       ← 이전 코드만 입력
        c_t = CrossAttn(h_t, F_spatial)      ← 하향식 공간 주의
        z_cont_t = probe([h_t ∥ c_t]) ∈ R^D  ← 어텐션 컨텍스트 기반
        z_q_t = VQ(z_cont_t)                ← STE + EMA
                         ↓
    SlotConceptPredictor (앞 N개 슬롯):
        ĉ_t = σ(Linear(z_q_t))             ← 지도학습된 개념 예측
                         ↓
    ResidualTerminator:
        z_res = MLP([F_pooled ∥ h_T])       ← 풀링된 특징 사용
                         ↓
    Orthogonal Projection (잠재 공간에서):
        concept_emb = proj_concept(h_T)
        residual_emb = proj_residual(z_res)
        residual_emb_orth = residual_emb - Proj_{concept}(residual_emb)
        z_total = concept_emb + residual_emb_orth
                         ↓
    Classifier:          ŷ = Linear(z_total)

Training Objective:
    L = L_task + λ_vq · L_vq + λ_concept · L_concept_supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SharedBackbone


# =============================================================================
# Discrete Codebook (EMA 기본 활성화)
# =============================================================================

class DiscreteCodebook(nn.Module):
    """Discrete Codebook — VQ-VAE 스타일 + EMA 업데이트.

    v2 변경: ema_decay 기본값 0.99 (EMA 활성화).
    EMA 활성화 시 embedding.weight의 gradient를 비활성화하고,
    코드북 갱신은 지수 이동 평균(EMA)으로만 수행합니다.

    E ∈ R^{K × D}  (K: 사전 크기, D: 개념 벡터 차원)
    """

    def __init__(self, num_codes: int, code_dim: int,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99):
        """
        Args:
            num_codes: 코드북 크기 K.
            code_dim:  각 코드 벡터의 차원 D.
            commitment_cost: Commitment loss 가중치 (VQ-VAE β).
            ema_decay: EMA 업데이트 decay (0이면 gradient 사용, >0이면 EMA).
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

        # EMA 추적 변수
        if ema_decay > 0:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("_ema_w", self.embedding.weight.data.clone())
            # EMA가 코드북을 갱신하므로 gradient 비활성화
            self.embedding.weight.requires_grad = False
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
        """VQ 양자화 — STE + EMA.

        Args:
            z: 연속 벡터 [*, D].

        Returns:
            z_q: 양자화된 벡터 [*, D] (STE 통과).
            indices: 선택된 인덱스 [*].
            vq_loss: Commitment loss (scalar).
        """
        distances = self.compute_distances(z)  # [*, K]
        indices = distances.argmin(dim=-1)      # [*]
        z_q = self.lookup(indices)              # [*, D]

        if self.training:
            if self.ema_decay > 0:
                # EMA가 코드북 갱신을 대체 → commitment loss만 사용
                self._ema_update(z, indices)
                vq_loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
            else:
                # Gradient 기반: codebook loss + commitment loss
                codebook_loss = F.mse_loss(z_q, z.detach())
                commitment_loss = F.mse_loss(z, z_q.detach())
                vq_loss = codebook_loss + self.commitment_cost * commitment_loss

            # Straight-Through Estimator
            z_q = z + (z_q - z).detach()
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        return z_q, indices, vq_loss

    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """EMA 방식으로 코드북 업데이트."""
        flat_z = z.detach().reshape(-1, self.code_dim)
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

        Args:
            features: 현재 배치의 인코더 출력 [B, D].
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

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]

        flat_features = features.detach().reshape(-1, self.code_dim)
        num_available = flat_features.shape[0]

        if num_available == 0:
            return 0

        # 랜덤 인덱스 선택 (복원 추출)
        rand_indices = torch.randint(0, num_available, (num_dead,),
                                     device=features.device)
        new_codes = flat_features[rand_indices]

        # 약간의 노이즈 추가
        noise = torch.randn_like(new_codes) * 0.01
        new_codes = new_codes + noise

        # 코드북 업데이트
        with torch.no_grad():
            self.embedding.weight.data[dead_indices] = new_codes
            # EMA 추적 변수도 갱신
            if self.ema_decay > 0:
                self._ema_w[dead_indices] = new_codes
                self._ema_cluster_size[dead_indices] = 1.0

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
        """Forward pass: 양자화 수행."""
        return self.quantize(z)


# =============================================================================
# Perspective Modulator (동적 관점 변환기 — FiLM 기반)
# =============================================================================

class PerspectiveModulator(nn.Module):
    """동적 관점 변환기 (Dynamic Perspective Modulator).

    철학: "이전 개념(h_t)에 따라 이미지를 보는 관점(필터) 자체가 변한다."
    구현: h_t를 기반으로 공간 특징맵(F_spatial)의 채널별 Scale(γ)과 Shift(β)를
          생성하여 이미지의 특징 공간(Feature Space) 자체를 왜곡/변환시킵니다.

    설계 의도:
      h_1 (아무것도 모름) → Identity 변환 → 원본 특징 그대로 읽기
      h_2 ("바퀴" 인지)  → 바퀴 관련 채널 증폭, 배경 억제
      h_3 ("바퀴+궤도")  → 상부 구조 채널 활성화
      → 좌표가 아닌 "뇌의 필터"가 변하므로 공간 오버피팅 없음
    """

    def __init__(self, query_dim: int, spatial_dim: int):
        """
        Args:
            query_dim:   GRU 은닉 상태 차원 (hidden_dim).
            spatial_dim: 공간 특징맵 채널 차원 C (ResNet-18: 512).
        """
        super().__init__()

        # h_t를 받아 특징맵 채널수(C)만큼의 Scale과 Shift를 생성
        self.gamma_proj = nn.Linear(query_dim, spatial_dim)
        self.beta_proj = nn.Linear(query_dim, spatial_dim)

        # 초기에는 관점 변환이 없도록(Identity) 가중치 0으로 초기화
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, h_t: torch.Tensor, spatial_features: torch.Tensor):
        """
        Args:
            h_t:              현재의 생각/개념 [B, query_dim].
            spatial_features: 원본 공간 특징맵 [B, C, H, W].

        Returns:
            context: 새로운 관점이 적용된 요약 벡터 [B, C].
        """
        # 1. 관점 변환 파라미터 생성
        gamma = self.gamma_proj(h_t).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_proj(h_t).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]

        # 2. 이미지 특징맵 자체를 변환 (Modulation — 관점의 변화)
        # (1 + gamma)를 사용하여 초기에는 원본을 유지하다가 학습됨
        modulated_features = (1.0 + gamma) * spatial_features + beta

        # 3. 새로운 관점으로 렌더링된 이미지를 전체적으로 다시 읽음 (GAP)
        # 좌표(공간) 오버피팅을 막기 위해 Global Average Pooling 사용
        new_perspective_context = modulated_features.mean(dim=(2, 3))  # [B, C]

        return new_perspective_context


# =============================================================================
# Auto-Regressive Concept Generator (VQ-STE, Gumbel 제거)
# =============================================================================

class AutoRegressiveConceptGenerator(nn.Module):
    """GRU 기반 자기회귀 개념 생성기 (v4: Dynamic Perspective Modulation).

    v3 → v4 핵심 변경:
      Cross-Attention 대신 PerspectiveModulator(FiLM)를 사용.
      좌표(공간) 기반 주의 대신 채널별 관점 변환으로 오버피팅 방지.

    정보 흐름 (동적 관점 변환):
      h_0 = 0 (zero vector)           ← 정보 누수 차단 (유지)
      h_t = GRU(z_q_{t-1}, h_{t-1})   ← 이전 양자화 코드만 입력
      c_t = Modulate(h_t, F_spatial)   ← h_t가 채널별 γ/β 생성 → GAP
      z_cont_t = probe([h_t ∥ c_t])   ← 관점 컨텍스트로 코드 선택
      z_q_t = VQ(z_cont_t)            ← STE + EMA

    작동 원리:
      h_1 (아무것도 모름) → Identity 변환 → 전체 특징 읽기 → "바퀴" 발견
      h_2 ("바퀴" 인지)  → 바퀴 채널 증폭 → "궤도" 추출
      h_3 ("바퀴+궤도")  → 상부 구조 채널 활성화 → "포탑" 발견
      → 뇌의 필터가 변하므로 초기 노이즈가 자연스럽게 보정됨
    """

    def __init__(self, visual_dim: int, code_dim: int, num_codes: int,
                 hidden_dim: int = 256, max_steps: int = 8,
                 num_gru_layers: int = 1):
        """
        Args:
            visual_dim:      공간 특징맵 채널 차원 C (ResNet-18: 512).
            code_dim:        코드북 벡터 차원 D.
            num_codes:       코드북 크기 K.
            hidden_dim:      GRU 은닉 상태 차원.
            max_steps:       최대 시퀀스 길이 T.
            num_gru_layers:  GRU 레이어 수.
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.num_gru_layers = num_gru_layers

        # <BOS> 토큰 (학습 가능한 시작 벡터)
        self.bos_token = nn.Parameter(torch.randn(code_dim) * 0.02)

        # GRU: input = z_q_{t-1} only (특징맵은 절대 GRU에 유입되지 않음)
        self.gru = nn.GRU(
            input_size=code_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=0.1 if num_gru_layers > 1 else 0.0,
        )

        # === v4: Dynamic Perspective Modulator ===
        # h_t를 기반으로 채널별 Scale/Shift 생성 → 관점 변환
        self.perspective_modulator = PerspectiveModulator(
            query_dim=hidden_dim,
            spatial_dim=visual_dim,
        )

        # Probe projection: [h_t ∥ c_t] → code_dim
        # c_t는 관점 변환 후 GAP 결과 (매 스텝 달라짐, 차원 = visual_dim)
        self.probe_proj = nn.Sequential(
            nn.Linear(hidden_dim + visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, code_dim),
        )

    def forward(self, spatial_features: torch.Tensor,
                codebook_module: 'DiscreteCodebook'):
        """
        Args:
            spatial_features: 공간 특징맵 [B, C, H', W'].
            codebook_module:  DiscreteCodebook 인스턴스.

        Returns:
            h_final:          GRU 최종 은닉 상태 [B, hidden_dim].
            selected_codes:   매 스텝 양자화된 코드 [B, T, code_dim].
            all_indices:      매 스텝 선택된 코드 인덱스 [B, T].
            all_z_continuous: 매 스텝 연속 프로브 벡터 [B, T, code_dim].
            vq_loss:          평균 VQ Loss (scalar).
        """
        B = spatial_features.shape[0]

        # 초기 은닉 상태: h_0 = 0벡터 (정보 누수 차단)
        h = torch.zeros(self.num_gru_layers, B, self.hidden_dim,
                        device=spatial_features.device,
                        dtype=spatial_features.dtype)

        # <BOS> 토큰으로 시작
        prev_code = self.bos_token.unsqueeze(0).expand(B, -1)  # [B, code_dim]

        all_selected_codes = []
        all_indices = []
        all_z_continuous = []
        total_vq_loss = torch.tensor(0.0, device=spatial_features.device)

        for t in range(self.max_steps):
            # GRU 입력: z_q_{t-1} only
            gru_input = prev_code.unsqueeze(1)  # [B, 1, code_dim]

            # GRU step
            gru_out, h = self.gru(gru_input, h)
            h_t = gru_out.squeeze(1)  # [B, hidden_dim]

            # === v4: 핵심 — 관점(Lens)의 변환 ===
            # 이미지를 보는 좌표를 바꾸는 게 아니라, 뇌의 필터를 바꾼다.
            c_t = self.perspective_modulator(
                h_t, spatial_features
            )  # [B, visual_dim]

            # 새로운 관점(c_t)과 현재 생각(h_t)을 합쳐 다음 코드를 결정
            probe_input = torch.cat([h_t, c_t], dim=-1)  # [B, hidden_dim + visual_dim]
            z_cont = self.probe_proj(probe_input)  # [B, code_dim]
            all_z_continuous.append(z_cont)

            # VQ 양자화 (STE + EMA)
            z_q, indices, vq_loss = codebook_module.quantize(z_cont)
            total_vq_loss = total_vq_loss + vq_loss

            all_indices.append(indices)
            all_selected_codes.append(z_q)

            # 다음 스텝 입력: 양자화된 코드
            prev_code = z_q

        # Stack results: [B, T, ...]
        selected_codes = torch.stack(all_selected_codes, dim=1)   # [B, T, code_dim]
        all_indices = torch.stack(all_indices, dim=1)             # [B, T]
        all_z_continuous = torch.stack(all_z_continuous, dim=1)   # [B, T, code_dim]

        # 평균 VQ Loss (스텝 수로 나눔)
        avg_vq_loss = total_vq_loss / self.max_steps

        # 최종 은닉 상태 (마지막 레이어)
        h_final = h[-1]  # [B, hidden_dim]

        return h_final, selected_codes, all_indices, all_z_continuous, avg_vq_loss


# =============================================================================
# Residual Terminator (잔차 터미네이터)
# =============================================================================

class ResidualTerminator(nn.Module):
    """잔차 터미네이터 — 시퀀스 생성 후 남은 정보를 추출.

    z_res = MLP([F ∥ h_T])

    v2: 직교 투영은 SeqCoResModel 레벨에서 잠재 공간(latent space)에서 수행.
    ResidualTerminator 자체는 단순 MLP로 잔차 추출만 담당합니다.
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
# Slot Concept Predictor (지도학습 개념 예측기)
# =============================================================================

class SlotConceptPredictor(nn.Module):
    """슬롯별 개념 예측기 — 앞 N개 슬롯이 지도학습된 개념을 예측.

    각 슬롯 t (t < num_supervised)의 양자화된 코드 벡터 z_q_t로부터
    해당 개념 레이블을 예측합니다. 이를 통해:
      - 앞 N개 슬롯: 인간이 해석 가능한 명시적 개념 (CBM의 장점)
      - 뒤 T-N개 슬롯: 모델이 자생적으로 발견한 숨겨진 패턴 (VQ의 장점)

    각 개념별 독립 예측 헤드를 사용하여 슬롯↔개념 1:1 매핑을 강제합니다.
    """

    def __init__(self, code_dim: int, num_supervised_concepts: int):
        """
        Args:
            code_dim:               코드북 벡터 차원 D.
            num_supervised_concepts: 지도학습할 개념 수 N (≤ max_steps).
        """
        super().__init__()
        self.num_supervised = num_supervised_concepts

        # 각 개념별 독립 예측 헤드
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(code_dim, code_dim),
                nn.ReLU(inplace=True),
                nn.Linear(code_dim, 1),
            )
            for _ in range(num_supervised_concepts)
        ])

    def forward(self, selected_codes: torch.Tensor):
        """
        Args:
            selected_codes: 양자화된 코드 시퀀스 [B, T, code_dim].

        Returns:
            concept_preds: 개념 예측 logits [B, N].
        """
        preds = []
        for i in range(self.num_supervised):
            code_i = selected_codes[:, i, :]  # [B, code_dim]
            pred_i = self.concept_heads[i](code_i).squeeze(-1)  # [B]
            preds.append(pred_i)
        return torch.stack(preds, dim=-1)  # [B, N]


# =============================================================================
# Reconstruction Decoder (Phase 1/2 공통)
# =============================================================================

class SeqCoResDecoder(nn.Module):
    """Phase 1/2 공통 디코더.

    v2: 두 Phase 모두 z_total (최종 조합된 임베딩)을 입력받아
    이미지를 재구성합니다. 이를 통해 Phase 1에서 워밍업된 표현 공간이
    Phase 2의 분류기가 사용할 공간과 일치합니다.
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
# Seq-CoRes Model v2 (통합 모델)
# =============================================================================

class SeqCoResModel(nn.Module):
    """Sequential Compositional Residual Embedding (Seq-CoRes) v4.

    v4 핵심 변경:
      1. 동적 관점 변환 (Dynamic Perspective Modulation)
         - Cross-Attention 대신 FiLM 기반 채널별 Scale/Shift
         - h_t가 "뇌의 필터"를 바꿔 이미지를 다른 관점으로 재해석
         - GAP로 좌표(공간) 오버피팅 방지
      2. VQ-STE + EMA (유지)
      3. 직교 투영: concept_emb ⊥ residual_emb_orth (유지)
      4. 슬롯 개념 지도학습 (유지)
      5. Phase 1/2 디코더 입력 통일 (유지)

    Architecture:
        Visual Encoder (ResNet-18, 글로벌 풀링 제거):
            X → F_spatial ∈ R^{C × H' × W'}  (공간 정보 보존)
            X → F_pooled  ∈ R^{C}             (잔차용 풀링)

        Auto-Regressive Concept Generator (GRU + PerspectiveModulator + VQ):
            h_0 = 0                                      ← 정보 누수 차단
            h_t = GRU(z_q_{t-1}, h_{t-1})                ← 이전 코드만 입력
            c_t = Modulate(h_t, F_spatial)                ← 동적 관점 변환
            z_cont_t = probe([h_t ∥ c_t])                 ← 관점 컨텍스트
            z_q_t = VQ(z_cont_t)                          ← STE + EMA

            작동 원리:
              h_1 (아무것도 모름) → Identity 변환 → 전체 특징 읽기
              h_2 ("바퀴" 인지)  → 바퀴 채널 증폭 → "궤도" 추출
              h_3 ("바퀴+궤도")  → 상부 구조 채널 활성화 → "포탑" 발견
              → 뇌의 필터가 변하므로 초기 노이즈가 자연스럽게 보정됨

        Residual Terminator:
            z_res = MLP([F_pooled ∥ h_T])

        Orthogonal Projection (잠재 공간):
            concept_emb = proj_concept(h_T)        ∈ R^{latent_dim}
            residual_emb = proj_residual(z_res)    ∈ R^{latent_dim}
            residual_emb_orth = Gram-Schmidt(residual_emb, concept_emb)
            z_total = concept_emb + residual_emb_orth
    """

    def __init__(self, latent_dim: int = 64, num_codes: int = 128,
                 code_dim: int = 32, hidden_dim: int = 256,
                 residual_dim: int = 32, max_steps: int = 8,
                 num_concepts: int = 20,
                 commitment_cost: float = 0.25,
                 ema_decay: float = 0.99,
                 image_size: int = 64, use_decoder: bool = True,
                 num_gru_layers: int = 1,
                 num_supervised_slots: int = -1):
        """
        Args:
            latent_dim:          최종 임베딩 차원 D.
            num_codes:           코드북 크기 K (자생적 개념 수).
            code_dim:            각 코드 벡터 차원.
            hidden_dim:          GRU 은닉 상태 차원.
            residual_dim:        잔차 벡터 차원.
            max_steps:           최대 시퀀스 길이 T.
            num_concepts:        다운스트림 태스크 개념 수.
            commitment_cost:     VQ Commitment loss 가중치.
            ema_decay:           EMA 코드북 업데이트 decay (0이면 gradient).
            image_size:          입출력 이미지 공간 크기.
            use_decoder:         디코더 포함 여부 (Phase 1용).
            num_gru_layers:      GRU 레이어 수.
            num_supervised_slots: 지도학습 슬롯 수 (-1이면 min(num_concepts, max_steps)).
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

        # 지도학습 슬롯 수 결정
        if num_supervised_slots < 0:
            self.num_supervised_slots = min(num_concepts, max_steps)
        else:
            self.num_supervised_slots = min(num_supervised_slots, max_steps)

        # 1. 시각적 인코더 (Shared Backbone — 공간 특징맵 + 풀링)
        self.backbone = SharedBackbone(pretrained=True)
        self.visual_dim = self.backbone.feature_dim  # 512

        # 2. 자생적 개념 사전 (Discrete Codebook + EMA)
        self.codebook = DiscreteCodebook(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )

        # 3. 자기회귀 개념 생성기 (v4: Dynamic Perspective Modulation + VQ-STE)
        self.ar_generator = AutoRegressiveConceptGenerator(
            visual_dim=self.visual_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            hidden_dim=hidden_dim,
            max_steps=max_steps,
            num_gru_layers=num_gru_layers,
        )

        # 4. 잔차 터미네이터 (풀링된 특징 사용)
        self.residual_terminator = ResidualTerminator(
            visual_dim=self.visual_dim,
            hidden_dim=hidden_dim,
            residual_dim=residual_dim,
        )

        # 5. 직교 투영을 위한 분리된 프로젝션
        #    concept_emb ⊥ residual_emb_orth 을 잠재 공간에서 강제
        self.concept_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.residual_projection = nn.Sequential(
            nn.Linear(residual_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 6. 분류 헤드 (다운스트림 태스크)
        self.classifier = nn.Linear(latent_dim, num_concepts)

        # 7. 슬롯 개념 예측기 (지도학습)
        if self.num_supervised_slots > 0:
            self.slot_concept_predictor = SlotConceptPredictor(
                code_dim=code_dim,
                num_supervised_concepts=self.num_supervised_slots,
            )
        else:
            self.slot_concept_predictor = None

        # 8. 디코더 (Phase 1 VQ 워밍업용, Phase 1/2 공통 z_total 입력)
        if use_decoder:
            self.decoder = SeqCoResDecoder(
                latent_dim=latent_dim,
                image_channels=3,
                image_size=image_size,
            )

    def _orthogonal_projection(self, concept_emb: torch.Tensor,
                                residual_emb: torch.Tensor):
        """Gram-Schmidt 직교 투영 — 잔차에서 개념 성분 제거.

        residual_emb에서 concept_emb 방향 성분을 차감하여
        잔차가 개념 공간과 직교하도록 강제합니다.

        Args:
            concept_emb:  개념 임베딩 [B, latent_dim].
            residual_emb: 잔차 임베딩 [B, latent_dim].

        Returns:
            residual_emb_orth: 직교화된 잔차 임베딩 [B, latent_dim].
        """
        # proj = (residual · concept) / (concept · concept) * concept
        dot = (residual_emb * concept_emb).sum(dim=-1, keepdim=True)
        norm_sq = (concept_emb * concept_emb).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        projection = (dot / norm_sq) * concept_emb

        # 직교 성분만 남김
        residual_emb_orth = residual_emb - projection
        return residual_emb_orth

    def encode(self, x: torch.Tensor):
        """이미지를 분해된 임베딩으로 인코딩.

        v4: 공간 특징맵에 동적 관점 변환(PerspectiveModulator) 적용.

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            dict with:
              z_total, concept_emb, residual_emb_orth,
              h_final, z_res, selected_codes, all_indices,
              all_z_continuous, vq_loss, slot_concept_preds,
              features (pooled), spatial_features
        """
        # 시각 특징 추출 (공간 특징맵 + 풀링)
        spatial_features, pooled_features = self.backbone.forward_spatial(x)
        # spatial_features: [B, 512, H', W']  (공간 정보 보존)
        # pooled_features:  [B, 512]           (잔차 터미네이터용)

        # 자기회귀 개념 생성 (v4: Dynamic Perspective Modulation + VQ-STE)
        h_final, selected_codes, all_indices, all_z_continuous, vq_loss = \
            self.ar_generator(spatial_features, self.codebook)

        # 잔차 추출 (풀링된 특징 사용)
        z_res = self.residual_terminator(pooled_features, h_final)

        # === 직교 투영 (잠재 공간에서) ===
        concept_emb = self.concept_projection(h_final)      # [B, latent_dim]
        residual_emb = self.residual_projection(z_res)       # [B, latent_dim]
        residual_emb_orth = self._orthogonal_projection(concept_emb, residual_emb)

        # 최종 임베딩: 개념 + 직교 잔차
        z_total = concept_emb + residual_emb_orth  # [B, latent_dim]

        # 슬롯 개념 예측 (지도학습)
        slot_concept_preds = None
        if self.slot_concept_predictor is not None:
            slot_concept_preds = self.slot_concept_predictor(selected_codes)

        return {
            "z_total": z_total,
            "concept_emb": concept_emb,
            "residual_emb": residual_emb,
            "residual_emb_orth": residual_emb_orth,
            "h_final": h_final,
            "z_res": z_res,
            "selected_codes": selected_codes,
            "all_indices": all_indices,
            "all_z_continuous": all_z_continuous,
            "vq_loss": vq_loss,
            "slot_concept_preds": slot_concept_preds,
            "features": pooled_features,
            "spatial_features": spatial_features,
        }

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
        enc = self.encode(x)

        logits = self.classifier(enc["z_total"])

        output = {
            "z": enc["z_total"],                         # [B, D]
            "h_final": enc["h_final"],                   # [B, hidden_dim]
            "z_residual": enc["z_res"],                  # [B, residual_dim]
            "selected_codes": enc["selected_codes"],     # [B, T, code_dim]
            "concept_indices": enc["all_indices"],       # [B, T]
            "logits": logits,                            # [B, num_concepts]
            "vq_loss": enc["vq_loss"],                   # scalar
            "z_continuous": enc["all_z_continuous"],      # [B, T, code_dim]
            "concept_emb": enc["concept_emb"],           # [B, latent_dim]
            "residual_emb": enc["residual_emb"],         # [B, latent_dim]
            "residual_emb_orth": enc["residual_emb_orth"], # [B, latent_dim]
            "slot_concept_preds": enc["slot_concept_preds"],  # [B, N_sup] or None
        }

        # 재구성 (Phase 1/2 공통: z_total 사용)
        if self.use_decoder:
            output["x_recon"] = self.decode(enc["z_total"])

        # 증강 뷰 (대조 학습)
        if x_aug is not None:
            enc_aug = self.encode(x_aug)
            output["z_aug"] = enc_aug["z_total"]
            output["z_residual_aug"] = enc_aug["z_res"]

        return output

    def get_embedding(self, x: torch.Tensor):
        """평가용 결정론적 임베딩 반환.

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            z: 정규화된 최종 임베딩 [B, D].
        """
        enc = self.encode(x)
        return F.normalize(enc["z_total"], dim=-1)

    def get_decomposed_embedding(self, x: torch.Tensor):
        """분해된 임베딩 반환 (분석용).

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            dict with z_total, concept_emb, residual_emb_orth, concept_indices, etc.
        """
        enc = self.encode(x)
        return {
            "z_total": F.normalize(enc["z_total"], dim=-1),
            "concept_emb": enc["concept_emb"],
            "residual_emb_orth": enc["residual_emb_orth"],
            "h_final": enc["h_final"],
            "z_residual": enc["z_res"],
            "concept_indices": enc["all_indices"],
            "selected_codes": enc["selected_codes"],
            "residual_norm": torch.norm(enc["residual_emb_orth"], dim=-1),
            "concept_norm": torch.norm(enc["concept_emb"], dim=-1),
            "slot_concept_preds": enc["slot_concept_preds"],
        }

    def get_concept_sequence(self, x: torch.Tensor):
        """이미지의 개념 시퀀스 추출 (해석 가능성 분석용).

        Args:
            x: 입력 이미지 [B, 3, H, W].

        Returns:
            indices:      코드북 인덱스 시퀀스 [B, T].
            codes:        코드북 벡터 시퀀스 [B, T, D].
            h_final:      GRU 최종 은닉 상태 [B, hidden_dim].
        """
        enc = self.encode(x)
        return enc["all_indices"], enc["selected_codes"], enc["h_final"]

    def get_codebook_utilization(self):
        """코드북 사용률 통계 반환 (collapse 감지)."""
        return {
            "codebook_weights": self.codebook.codebook.detach().clone(),
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
        }

    # =========================================================================
    # v1 호환성을 위한 no-op 메서드
    # =========================================================================

    def set_gumbel_tau(self, tau: float):
        """[v1 호환] Gumbel-Softmax 제거됨 — no-op."""
        pass

    def get_gumbel_tau(self) -> float:
        """[v1 호환] Gumbel-Softmax 제거됨 — 항상 0.0 반환."""
        return 0.0


# =============================================================================
# Seq-CoRes Loss v2
# =============================================================================

class SeqCoResLoss(nn.Module):
    """Seq-CoRes v2 통합 손실 함수.

    v2 변경:
      - 배치 엔트로피 최대화 제거 (Dead Code Revival로 대체)
      - 개념 지도학습 손실 추가 (슬롯별 개념 예측)
      - L2 잔차 패널티 제거 → 직교 투영이 정보 누수 차단
      - VQ Loss 단일화 (Gumbel 제거에 따라)

    Phase 1 (VQ Warmup):
        L = L_recon + λ_vq · L_vq
        (코드북 시각적 기저 초기화)

    Phase 2 (Task + Concept Supervision):
        L = L_task + λ_vq · L_vq + λ_concept · L_concept_supervision
        (분류 + 슬롯 개념 지도학습)
    """

    def __init__(self, task_weight: float = 1.0,
                 vq_weight: float = 1.0,
                 recon_weight: float = 1.0,
                 phase2_recon_weight: float = 0.1,
                 concept_supervision_weight: float = 1.0,
                 residual_penalty_weight: float = 0.0,
                 phase: int = 2):
        """
        Args:
            task_weight:                L_task 가중치.
            vq_weight:                  L_vq (VQ commitment) 가중치.
            recon_weight:               L_recon 가중치 (Phase 1).
            phase2_recon_weight:        L_recon 가중치 (Phase 2, 낮게).
            concept_supervision_weight: L_concept 가중치 (슬롯 개념 지도학습).
            residual_penalty_weight:    잔차 L2 패널티 가중치 (기본 0, 직교 투영이 대체).
            phase:                      학습 페이즈 (1 또는 2).
        """
        super().__init__()

        self.task_weight = task_weight
        self.vq_weight = vq_weight
        self.recon_weight = recon_weight
        self.phase2_recon_weight = phase2_recon_weight
        self.concept_supervision_weight = concept_supervision_weight
        self.residual_penalty_weight = residual_penalty_weight
        self.phase = phase

        self.task_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.MSELoss(reduction="mean")
        self.concept_criterion = nn.BCEWithLogitsLoss()

    def set_phase(self, phase: int):
        """학습 페이즈 전환."""
        self.phase = phase

    def forward(self, output: dict, concept_labels: torch.Tensor,
                x_input: torch.Tensor = None, epoch: int = None):
        """
        Args:
            output:         SeqCoResModel.forward()의 출력 dict.
            concept_labels: 정답 개념 레이블 [B, N].
            x_input:        원본 이미지 [B, 3, H, W] (재구성용).
            epoch:          현재 에포크 (미사용, 호환성 유지).

        Returns:
            total_loss: 총 손실.
            loss_dict:  개별 손실 딕셔너리.
        """
        device = concept_labels.device
        loss_dict = {}

        # VQ Loss (Phase 1/2 공통)
        vq_loss = output.get("vq_loss", torch.tensor(0.0, device=device))
        if isinstance(vq_loss, (int, float)):
            vq_loss = torch.tensor(vq_loss, device=device)

        # =====================================================================
        # Phase 1: VQ Warmup (코드북 초기화)
        # =====================================================================
        if self.phase == 1:
            # 재구성 손실
            loss_recon = torch.tensor(0.0, device=device)
            if "x_recon" in output and x_input is not None:
                if output["x_recon"].shape[-2:] != x_input.shape[-2:]:
                    x_recon_resized = F.interpolate(
                        output["x_recon"], size=x_input.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                    loss_recon = self.recon_criterion(x_recon_resized, x_input)
                else:
                    loss_recon = self.recon_criterion(output["x_recon"], x_input)

            # 개념 지도학습 (Phase 1에서도 슬롯 정렬 시작)
            loss_concept = torch.tensor(0.0, device=device)
            if output.get("slot_concept_preds") is not None:
                slot_preds = output["slot_concept_preds"]  # [B, N_sup]
                n_sup = slot_preds.shape[1]
                target = concept_labels[:, :n_sup]
                loss_concept = self.concept_criterion(slot_preds, target)

            total_loss = (
                self.recon_weight * loss_recon
                + self.vq_weight * vq_loss
                + self.concept_supervision_weight * 0.5 * loss_concept  # Phase 1은 절반 가중
            )

            loss_dict = {
                "loss_total": total_loss.item(),
                "loss_recon": loss_recon.item(),
                "loss_vq": vq_loss.item(),
                "loss_concept_supervision": loss_concept.item(),
                "phase": 1,
            }
            return total_loss, loss_dict

        # =====================================================================
        # Phase 2: Task + Concept Supervision
        # =====================================================================

        # 1. Task Loss (분류)
        logits = output["logits"]
        loss_task = self.task_criterion(logits, concept_labels)

        # 2. 개념 지도학습 손실 (슬롯별 개념 예측)
        loss_concept = torch.tensor(0.0, device=device)
        if output.get("slot_concept_preds") is not None:
            slot_preds = output["slot_concept_preds"]  # [B, N_sup]
            n_sup = slot_preds.shape[1]
            target = concept_labels[:, :n_sup]
            loss_concept = self.concept_criterion(slot_preds, target)

        # 3. 재구성 (Phase 2: 낮은 가중치로 regularization)
        loss_recon = torch.tensor(0.0, device=device)
        if "x_recon" in output and x_input is not None:
            if output["x_recon"].shape[-2:] != x_input.shape[-2:]:
                x_recon_resized = F.interpolate(
                    output["x_recon"], size=x_input.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
                loss_recon = self.recon_criterion(x_recon_resized, x_input)
            else:
                loss_recon = self.recon_criterion(output["x_recon"], x_input)

        # 4. 잔차 L2 패널티 (선택적, 기본 0 — 직교 투영이 대체)
        residual_emb_orth = output.get("residual_emb_orth")
        loss_res_penalty = torch.tensor(0.0, device=device)
        if residual_emb_orth is not None and self.residual_penalty_weight > 0:
            loss_res_penalty = torch.norm(residual_emb_orth, p=2, dim=-1).mean()

        # 총 손실
        total_loss = (
            self.task_weight * loss_task
            + self.vq_weight * vq_loss
            + self.concept_supervision_weight * loss_concept
            + self.phase2_recon_weight * loss_recon
            + self.residual_penalty_weight * loss_res_penalty
        )

        # 직교성 지표 (모니터링용)
        concept_emb = output.get("concept_emb")
        residual_emb = output.get("residual_emb")
        orthogonality = 0.0
        if concept_emb is not None and residual_emb is not None:
            cos_sim = F.cosine_similarity(concept_emb, residual_emb, dim=-1)
            orthogonality = cos_sim.abs().mean().item()

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_task": loss_task.item(),
            "loss_vq": vq_loss.item(),
            "loss_concept_supervision": loss_concept.item(),
            "loss_recon": loss_recon.item(),
            "loss_res_penalty": loss_res_penalty.item(),
            "orthogonality_violation": orthogonality,
            "residual_norm": torch.norm(
                residual_emb_orth, p=2, dim=-1
            ).mean().item() if residual_emb_orth is not None else 0.0,
            "concept_norm": torch.norm(
                concept_emb, p=2, dim=-1
            ).mean().item() if concept_emb is not None else 0.0,
            "phase": 2,
        }

        return total_loss, loss_dict
