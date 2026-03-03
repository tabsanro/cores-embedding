"""
Seq-CoRes 학습 스크립트 — Two-Phase Training.

Usage:
    # 전체 Two-Phase 학습 (Phase 1 + Phase 2)
    python scripts/train_seqcores.py --config configs/seqcores.yaml

    # Phase 1만 (코드북 워밍업)
    python scripts/train_seqcores.py --config configs/seqcores.yaml --phase 1

    # Phase 2만 (Phase 1 체크포인트로부터)
    python scripts/train_seqcores.py --config configs/seqcores.yaml --phase 2 \\
        --resume outputs/cores_embedding_default/seqcores/checkpoints/phase1_final.pt

    # CUB-200 데이터셋으로 학습
    python scripts/train_seqcores.py --config configs/seqcores.yaml --dataset cub200

    # 에포크/디바이스 오버라이드
    python scripts/train_seqcores.py --config configs/seqcores.yaml \\
        --phase1-epochs 10 --phase2-epochs 50 --device cuda
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders
from models.seqcores import SeqCoResModel, SeqCoResLoss
from training.seqcores_trainer import SeqCoResTrainer


def set_seed(seed):
    """재현성을 위한 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Train Seq-CoRes (Sequential Compositional Residual) Model"
    )
    parser.add_argument("--config", type=str, default="configs/seqcores.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2],
                        help="특정 phase만 실행 (1: VQ Warmup, 2: Task+Squeeze)")
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트에서 이어서 학습")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["celeba", "cub200", "mpi3d", "clevr"],
                        help="데이터셋 오버라이드")
    parser.add_argument("--phase1-epochs", type=int, default=None,
                        help="Phase 1 에포크 오버라이드")
    parser.add_argument("--phase2-epochs", type=int, default=None,
                        help="Phase 2 에포크 오버라이드")
    parser.add_argument("--device", type=str, default=None,
                        help="디바이스 오버라이드 (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="배치 크기 오버라이드")
    parser.add_argument("--num-codes", type=int, default=None,
                        help="코드북 크기 K 오버라이드")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="최대 시퀀스 길이 T 오버라이드")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="잠재 차원 D 오버라이드")
    args = parser.parse_args()

    # =========================================================================
    # Config 로드 및 오버라이드
    # =========================================================================
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 오버라이드 적용
    if args.dataset is not None:
        config["dataset"]["name"] = args.dataset
    if args.device is not None:
        config["experiment"]["device"] = args.device
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.latent_dim is not None:
        config["model"]["latent_dim"] = args.latent_dim

    sc = config["training"].setdefault("seqcores", {})
    if args.phase1_epochs is not None:
        sc["phase1_epochs"] = args.phase1_epochs
    if args.phase2_epochs is not None:
        sc["phase2_epochs"] = args.phase2_epochs

    mc = config["model"].setdefault("seqcores", {})
    if args.num_codes is not None:
        mc["num_codes"] = args.num_codes
    if args.max_steps is not None:
        mc["max_steps"] = args.max_steps

    # CUDA 가용성 체크
    if config["experiment"]["device"] == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        config["experiment"]["device"] = "cpu"

    # 시드 고정
    set_seed(config["experiment"]["seed"])

    # =========================================================================
    # 정보 출력
    # =========================================================================
    print("=" * 60)
    print("Seq-CoRes Training")
    print("=" * 60)
    print(f"Dataset:       {config['dataset']['name']}")
    print(f"Device:        {config['experiment']['device']}")
    print(f"Batch Size:    {config['training']['batch_size']}")

    mc = config["model"].get("seqcores", {})
    print(f"Latent Dim:    {config['model']['latent_dim']}")
    print(f"Codebook K:    {mc.get('num_codes', 128)}")
    print(f"Code Dim:      {mc.get('code_dim', 32)}")
    print(f"Max Steps T:   {mc.get('max_steps', 8)}")
    print(f"Hidden Dim:    {mc.get('hidden_dim', 256)}")
    print(f"Residual Dim:  {mc.get('residual_dim', 32)}")
    print(f"EMA Decay:     {mc.get('ema_decay', 0.99)}")
    print(f"Sup. Slots:    {mc.get('num_supervised_slots', -1)} (-1=auto)")

    sc = config["training"].get("seqcores", {})
    print(f"Phase 1 Epochs: {sc.get('phase1_epochs', 20)}")
    print(f"Phase 2 Epochs: {sc.get('phase2_epochs', 80)}")
    if args.phase is not None:
        print(f"Running Phase: {args.phase} only")
    print("=" * 60)

    # =========================================================================
    # 데이터 로드
    # =========================================================================
    print("\n📂 Loading dataset...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)
    print(f"  Num concepts: {num_concepts}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # =========================================================================
    # 모델 생성
    # =========================================================================
    print("\n🏗️  Building Seq-CoRes model...")

    mc = config["model"].get("seqcores", {})
    image_size = config["dataset"].get("image_size", 64)

    model = SeqCoResModel(
        latent_dim=config["model"]["latent_dim"],
        num_codes=mc.get("num_codes", 128),
        code_dim=mc.get("code_dim", 32),
        hidden_dim=mc.get("hidden_dim", 256),
        residual_dim=mc.get("residual_dim", 32),
        max_steps=mc.get("max_steps", 8),
        num_concepts=num_concepts,
        commitment_cost=mc.get("commitment_cost", 0.25),
        ema_decay=mc.get("ema_decay", 0.99),
        image_size=image_size,
        use_decoder=mc.get("use_decoder", True),
        num_gru_layers=mc.get("num_gru_layers", 1),
        num_supervised_slots=mc.get("num_supervised_slots", -1),
    )

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Codebook parameters:  {model.num_codes * model.code_dim:,}")

    # =========================================================================
    # Loss 생성
    # =========================================================================
    sc = config["training"].get("seqcores", {})
    criterion = SeqCoResLoss(
        task_weight=sc.get("task_weight", 1.0),
        vq_weight=sc.get("vq_weight", 1.0),
        recon_weight=sc.get("recon_weight", 1.0),
        phase2_recon_weight=sc.get("phase2_recon_weight", 0.1),
        concept_supervision_weight=sc.get("concept_supervision_weight", 1.0),
        residual_penalty_weight=sc.get("residual_penalty_weight", 0.0),
    )

    # =========================================================================
    # Trainer 생성
    # =========================================================================
    trainer = SeqCoResTrainer(model, criterion, config)

    # 체크포인트 이어서 학습
    if args.resume is not None:
        print(f"\n📥 Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # =========================================================================
    # 학습 실행
    # =========================================================================
    if args.phase is None:
        # 전체 Two-Phase 학습
        history = trainer.train(train_loader, test_loader)
    elif args.phase == 1:
        history = trainer.train_phase1(train_loader, test_loader)
    elif args.phase == 2:
        history = trainer.train_phase2(train_loader, test_loader)

    print("\n✅ Seq-CoRes training complete!")
    print(f"📁 Outputs saved to: {trainer.output_dir}")

    return history


if __name__ == "__main__":
    main()
