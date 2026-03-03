"""
Training script for CoRes-Embedding experiments.

Usage:
    python scripts/train.py --config configs/default.yaml --model baseline
    python scripts/train.py --config configs/default.yaml --model cores
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders, get_num_concepts
from models import build_model
from training.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train CoRes-Embedding models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default="cores",
                        choices=["baseline", "cores", "vcores", "seqcores"],
                        help="Model type to train")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Override latent dimension")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.latent_dim is not None:
        config["model"]["latent_dim"] = args.latent_dim
    if args.device is not None:
        config["experiment"]["device"] = args.device

    # Auto-detect device
    if config["experiment"]["device"] == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        config["experiment"]["device"] = "cpu"

    # Set model type in config
    config["_model_type"] = args.model

    # Set seed
    set_seed(config["experiment"]["seed"])

    print("=" * 60)
    print("CoRes-Embedding Training")
    print("=" * 60)
    print(f"Model:         {args.model}")
    print(f"Dataset:       {config['dataset']['name']}")
    print(f"Latent Dim:    {config['model']['latent_dim']}")
    print(f"Device:        {config['experiment']['device']}")
    print(f"Epochs:        {config['training']['epochs']}")
    print(f"Batch Size:    {config['training']['batch_size']}")
    print("=" * 60)

    # Get data
    print("\n📂 Loading dataset...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)
    print(f"  Num concepts: {num_concepts}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Build model
    print("\n🏗️  Building model...")
    if args.model == "cores":
        config["model"]["cores"]["num_concepts"] = num_concepts
    elif args.model == "vcores":
        vcores_cfg = config["model"].get("vcores", config["model"]["cores"])
        vcores_cfg["num_concepts"] = num_concepts
        config["model"]["vcores"] = vcores_cfg
    model = build_model(config, num_concepts)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Train
    if args.model == "seqcores":
        # Seq-CoRes는 전용 Two-Phase 트레이너 사용
        from models.seqcores import SeqCoResLoss
        from training.seqcores_trainer import SeqCoResTrainer

        sc = config["training"].get("seqcores", {})
        criterion = SeqCoResLoss(
            task_weight=sc.get("task_weight", 1.0),
            vq_weight=sc.get("vq_weight", 1.0),
            recon_weight=sc.get("recon_weight", 1.0),
            residual_penalty_weight=sc.get("residual_penalty_weight", 0.1),
            batch_entropy_weight=sc.get("batch_entropy_weight", 0.1),
            residual_annealing_start=sc.get("residual_annealing_start", 0),
            residual_annealing_end=sc.get("residual_annealing_end", 50),
        )
        trainer = SeqCoResTrainer(model, criterion, config)
        history = trainer.train(train_loader, test_loader)
    else:
        trainer = Trainer(model, config, model_type=args.model)
        history = trainer.train(train_loader, test_loader)

    print("\n✅ Training complete!")
    print(f"📁 Outputs saved to: {trainer.output_dir}")

    return history


if __name__ == "__main__":
    main()
