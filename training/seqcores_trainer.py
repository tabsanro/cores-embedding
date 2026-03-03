"""
Seq-CoRes v2 전용 트레이너 — Two-Phase Training Strategy.

Phase 1: VQ Warmup (코드북 워밍업)
    - 재구성(Reconstruction) 학습 + 슬롯 개념 지도학습 (절반 가중)
    - 코드북이 의미 있는 시각적 기저(Visual Basis)로 초기화
    - VQ-STE + EMA 코드북 업데이트

Phase 2: End-to-End Task + Concept Supervision
    - Task Loss + VQ Loss + Concept Supervision Loss
    - 직교 투영으로 잔차 정보 누수 차단 (L2 패널티 제거)
    - Dead Code Revival로 코드북 붕괴 방지
"""

import os
import time
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class SeqCoResTrainer:
    """Seq-CoRes v2 Two-Phase Trainer.

    Phase 1과 Phase 2를 통합 관리합니다.
    v2: Gumbel-Softmax 제거, VQ-STE + EMA, 직교 투영, 개념 지도학습.
    """

    def __init__(self, model, criterion, config, output_dir=None):
        """
        Args:
            model:     SeqCoResModel 인스턴스.
            criterion: SeqCoResLoss 인스턴스.
            config:    설정 딕셔너리.
            output_dir: 출력 경로 (None이면 config에서 추출).
        """
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = torch.device(config["experiment"].get("device", "cuda"))

        # 모델을 디바이스로 이동
        self.model = self.model.to(self.device)

        # 출력 디렉토리
        if output_dir is None:
            self.output_dir = os.path.join(
                config["experiment"]["output_dir"],
                config["experiment"]["name"],
                "seqcores",
            )
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_dir, "logs"))

        # 학습 설정 추출
        self.tc = config["training"]
        self.sc = config["training"].get("seqcores", {})

        # Phase 1 설정
        self.phase1_epochs = self.sc.get("phase1_epochs", 20)
        self.phase1_lr = self.sc.get("phase1_lr", 1e-3)

        # Phase 2 설정
        self.phase2_epochs = self.sc.get("phase2_epochs", 80)
        self.phase2_lr = self.sc.get("phase2_lr", 3e-4)

        # 학습 상태
        self.current_epoch = 0
        self.current_phase = 1
        self.best_loss = float("inf")
        self.global_step = 0

        # Optimizer / Scheduler 는 phase 전환 시 재생성
        self.optimizer = None
        self.scheduler = None

    def _setup_optimizer(self, phase):
        """Phase별 optimizer 및 scheduler 설정."""
        if phase == 1:
            lr = self.sc.get("phase1_lr", 1e-3)
        else:
            lr = self.sc.get("phase2_lr", 3e-4)

        weight_decay = self.tc.get("weight_decay", 1e-4)

        # YAML에서 문자열로 파싱될 수 있으므로 float 변환
        lr = float(lr)
        weight_decay = float(weight_decay)

        optimizer_name = self.tc.get("optimizer", "adam")

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

        # Scheduler
        scheduler_name = self.tc.get("scheduler", "cosine")
        total_epochs = self.phase1_epochs if phase == 1 else self.phase2_epochs
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=1e-6,
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1,
            )
        else:
            self.scheduler = None

    # =========================================================================
    # Phase 1: VQ Warmup
    # =========================================================================

    def train_phase1(self, train_loader, val_loader=None):
        """Phase 1: 코드북 워밍업 (재구성 + 개념 지도학습).

        재구성 학습으로 코드북을 시각적 기저로 초기화하면서,
        슬롯 개념 예측으로 코드↔개념 정렬을 시작합니다.

        Args:
            train_loader: 학습 데이터 로더.
            val_loader:   검증 데이터 로더 (optional).

        Returns:
            history: Phase 1 학습 기록.
        """
        print(f"\n{'='*60}")
        print("Phase 1: VQ Codebook Warmup (Reconstruction + Concept Align)")
        print(f"  Epochs: {self.phase1_epochs}")
        print(f"  LR: {self.phase1_lr}")
        print(f"  VQ: STE + EMA")
        print(f"{'='*60}\n")

        self.current_phase = 1
        self.criterion.set_phase(1)
        self._setup_optimizer(phase=1)

        history = {"train": [], "val": []}
        save_every = self.tc.get("save_every", 10)

        for epoch in range(self.phase1_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # --- Train ---
            train_loss, train_losses = self._train_epoch(train_loader)
            history["train"].append(train_losses)

            # TensorBoard logging
            for key, val in train_losses.items():
                self.writer.add_scalar(f"phase1/train/{key}", val, epoch)
            if "codebook_alive" in train_losses:
                self.writer.add_scalar("phase1/codebook_alive",
                                       train_losses["codebook_alive"], epoch)
                self.writer.add_scalar("phase1/codebook_utilization",
                                       train_losses["codebook_utilization"], epoch)
            if "loss_vq" in train_losses:
                self.writer.add_scalar("phase1/vq_loss",
                                       train_losses["loss_vq"], epoch)
            if "loss_concept_supervision" in train_losses:
                self.writer.add_scalar("phase1/concept_supervision",
                                       train_losses["loss_concept_supervision"], epoch)

            # --- Validate ---
            val_info = ""
            if val_loader is not None:
                val_loss, val_metrics = self._validate(val_loader)
                history["val"].append(val_metrics)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"phase1/val/{key}", val, epoch)
                val_info = f" | Val Loss: {val_loss:.4f}"

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("phase1_best.pt")

            elapsed = time.time() - start_time
            print(f"  P1 Epoch {epoch:3d}/{self.phase1_epochs} | "
                  f"Train Loss: {train_loss:.4f}{val_info} | "
                  f"Time: {elapsed:.1f}s")

            # Save periodic
            if epoch % save_every == 0:
                self.save_checkpoint(f"phase1_epoch_{epoch}.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        # Phase 1 완료 → 전이 준비
        self.save_checkpoint("phase1_final.pt")
        self.best_loss = float("inf")  # Phase 2에서 재설정
        print("\n✅ Phase 1 Complete. Codebook initialized.\n")

        return history

    # =========================================================================
    # Phase 2: Task + Concept Supervision
    # =========================================================================

    def train_phase2(self, train_loader, val_loader=None):
        """Phase 2: End-to-End Task 학습 + 개념 지도학습.

        Task Loss + VQ Loss + Concept Supervision Loss
        직교 투영이 잔차 정보 누수를 차단합니다.

        Args:
            train_loader: 학습 데이터 로더.
            val_loader:   검증 데이터 로더 (optional).

        Returns:
            history: Phase 2 학습 기록.
        """
        print(f"\n{'='*60}")
        print("Phase 2: Task Learning + Concept Supervision")
        print(f"  Epochs: {self.phase2_epochs}")
        print(f"  LR: {self.phase2_lr}")
        print(f"  VQ: STE + EMA")
        print(f"  Orthogonal Projection: ON")
        print(f"{'='*60}\n")

        self.current_phase = 2
        self.criterion.set_phase(2)
        self._setup_optimizer(phase=2)

        history = {"train": [], "val": []}
        save_every = self.tc.get("save_every", 10)
        eval_every = self.tc.get("eval_every", 5)

        for epoch in range(self.phase2_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # --- Train ---
            train_loss, train_losses = self._train_epoch(
                train_loader, epoch=epoch
            )
            history["train"].append(train_losses)

            # TensorBoard logging
            for key, val in train_losses.items():
                self.writer.add_scalar(f"phase2/train/{key}", val, epoch)
            if "residual_norm" in train_losses:
                self.writer.add_scalar("phase2/residual_norm",
                                       train_losses["residual_norm"], epoch)
            if "concept_norm" in train_losses:
                self.writer.add_scalar("phase2/concept_norm",
                                       train_losses["concept_norm"], epoch)
            if "orthogonality_violation" in train_losses:
                self.writer.add_scalar("phase2/orthogonality_violation",
                                       train_losses["orthogonality_violation"], epoch)
            if "codebook_alive" in train_losses:
                self.writer.add_scalar("phase2/codebook_alive",
                                       train_losses["codebook_alive"], epoch)
                self.writer.add_scalar("phase2/codebook_utilization",
                                       train_losses["codebook_utilization"], epoch)
            if "loss_vq" in train_losses:
                self.writer.add_scalar("phase2/vq_loss",
                                       train_losses["loss_vq"], epoch)
            if "loss_concept_supervision" in train_losses:
                self.writer.add_scalar("phase2/concept_supervision",
                                       train_losses["loss_concept_supervision"], epoch)

            # --- Validate ---
            val_info = ""
            if val_loader is not None and (epoch % eval_every == 0 or epoch == self.phase2_epochs - 1):
                val_loss, val_metrics = self._validate(val_loader, epoch=epoch)
                history["val"].append(val_metrics)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"phase2/val/{key}", val, epoch)
                val_info = (f" | Val Loss: {val_loss:.4f}"
                            f" | Val Acc: {val_metrics.get('accuracy', 0):.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best.pt")

            elapsed = time.time() - start_time
            res_norm = train_losses.get("residual_norm", 0)
            orth_viol = train_losses.get("orthogonality_violation", 0)
            print(f"  P2 Epoch {epoch:3d}/{self.phase2_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"||z_res||: {res_norm:.4f} | "
                  f"orth: {orth_viol:.4f}{val_info} | "
                  f"Time: {elapsed:.1f}s")

            # Save periodic
            if epoch % save_every == 0:
                self.save_checkpoint(f"phase2_epoch_{epoch}.pt")

            if self.scheduler is not None:
                self.scheduler.step()

        self.save_checkpoint("final.pt")
        print("\n✅ Phase 2 Complete. Training finished.\n")

        return history

    # =========================================================================
    # Full Two-Phase Training
    # =========================================================================

    def train(self, train_loader, val_loader=None):
        """두 단계 학습 전체 실행.

        Args:
            train_loader: 학습 데이터 로더.
            val_loader:   검증 데이터 로더.

        Returns:
            history: 전체 학습 기록.
        """
        print(f"\n{'='*60}")
        print("Seq-CoRes v2 Two-Phase Training")
        print(f"  Device: {self.device}")
        print(f"  Phase 1 (VQ Warmup):        {self.phase1_epochs} epochs")
        print(f"  Phase 2 (Task + Concepts):  {self.phase2_epochs} epochs")
        print(f"  Total:                      {self.phase1_epochs + self.phase2_epochs} epochs")
        print(f"  VQ: STE + EMA  |  Orthogonal Projection: ON")
        print(f"{'='*60}\n")

        history = {}

        # Phase 1: VQ Warmup
        history["phase1"] = self.train_phase1(train_loader, val_loader)

        # Phase 2: Task + Concept Supervision
        history["phase2"] = self.train_phase2(train_loader, val_loader)

        self.writer.close()

        # 학습 기록 저장
        history_path = os.path.join(self.output_dir, "training_history.json")
        serializable = self._make_serializable(history)
        with open(history_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"📁 Training history saved to: {history_path}")

        return history

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _train_epoch(self, train_loader, epoch=None):
        """1 에포크 학습."""
        self.model.train()
        total_losses = {}
        num_batches = 0
        last_features_for_revival = None  # Dead Code Revival용

        # 에포크 시작 시 사용 통계 초기화
        self.model.codebook.reset_usage_stats()

        phase_name = f"Phase {self.current_phase}"
        pbar = tqdm(train_loader,
                    desc=f"{phase_name} | Epoch {self.current_epoch}",
                    leave=False)

        for batch_idx, (images, concept_labels, factor_labels) in enumerate(pbar):
            images = images.to(self.device)
            concept_labels = concept_labels.to(self.device)

            # Forward (phase를 전달)
            output = self.model(images, phase=self.current_phase)

            # 코드북 사용 빈도 추적 (Dead Code Revival)
            if "concept_indices" in output:
                self.model.codebook.update_usage(output["concept_indices"])
                # 재초기화용 연속 공간 투영값 사용
                last_features_for_revival = output["z_continuous"].detach().reshape(-1, self.model.code_dim)

            # Loss
            loss, loss_dict = self.criterion(
                output, concept_labels,
                x_input=images, epoch=epoch,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Accumulate
            for key, val in loss_dict.items():
                if isinstance(val, (int, float)):
                    total_losses[key] = total_losses.get(key, 0) + val
            num_batches += 1
            self.global_step += 1

            # Progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })

        # === Dead Code Revival: 에포크 끝에 죽은 코드 재초기화 ===
        usage_stats = self.model.codebook.get_usage_stats()
        num_dead = usage_stats["dead_codes"]
        if num_dead > 0 and last_features_for_revival is not None:
            num_restarted = self.model.codebook.restart_dead_codes(
                last_features_for_revival, dead_threshold=1.0
            )
            if num_restarted > 0:
                tqdm.write(
                    f"  🔄 Dead Code Revival: {num_restarted}/{self.model.num_codes} "
                    f"codes restarted (alive: {usage_stats['alive_codes']} → "
                    f"{usage_stats['alive_codes'] + num_restarted})"
                )

        # 코드북 통계 로깅
        total_losses["codebook_alive"] = usage_stats["alive_codes"]
        total_losses["codebook_utilization"] = usage_stats["utilization"]

        avg_losses = {k: v / num_batches if k not in ("codebook_alive", "codebook_utilization") else v
                      for k, v in total_losses.items()}
        return avg_losses.get("loss_total", 0), avg_losses

    def _validate(self, val_loader, epoch=None):
        """검증."""
        self.model.eval()
        total_losses = {}
        all_preds = []
        all_labels = []
        num_batches = 0

        with torch.no_grad():
            for images, concept_labels, factor_labels in val_loader:
                images = images.to(self.device)
                concept_labels = concept_labels.to(self.device)

                output = self.model(images, phase=self.current_phase)
                loss, loss_dict = self.criterion(
                    output, concept_labels,
                    x_input=images, epoch=epoch,
                )

                for key, val in loss_dict.items():
                    if isinstance(val, (int, float)):
                        total_losses[key] = total_losses.get(key, 0) + val
                num_batches += 1

                # 분류 정확도 (Phase 2)
                if self.current_phase == 2:
                    preds = (torch.sigmoid(output["logits"]) > 0.5).float()
                    all_preds.append(preds.cpu())
                    all_labels.append(concept_labels.cpu())

        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        # 정확도
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            accuracy = (all_preds == all_labels).float().mean().item()
            avg_losses["accuracy"] = accuracy

        return avg_losses.get("loss_total", 0), avg_losses

    def save_checkpoint(self, filename: str):
        """체크포인트 저장."""
        path = os.path.join(self.output_dir, "checkpoints", filename)
        torch.save({
            "epoch": self.current_epoch,
            "phase": self.current_phase,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "global_step": self.global_step,
            "config": self.config,
        }, path)

    def load_checkpoint(self, filename: str):
        """체크포인트 로드."""
        path = os.path.join(self.output_dir, "checkpoints", filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.current_phase = checkpoint["phase"]
        self.best_loss = checkpoint["best_loss"]
        self.global_step = checkpoint.get("global_step", 0)
        if self.optimizer is not None and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint: phase={self.current_phase}, "
              f"epoch={self.current_epoch}")

    @staticmethod
    def _make_serializable(obj):
        """JSON 직렬화 가능한 형태로 변환."""
        if isinstance(obj, dict):
            return {k: SeqCoResTrainer._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [SeqCoResTrainer._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return obj
