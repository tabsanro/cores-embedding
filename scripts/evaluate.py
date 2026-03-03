"""
Evaluation script for CoRes-Embedding experiments.

Runs all three evaluation protocols on trained models:
1. Perturbation Stability
2. Few-shot Generalization
3. Manifold Smoothness

Usage:
    python scripts/evaluate.py --config configs/default.yaml
    python scripts/evaluate.py --config configs/default.yaml --model cores --checkpoint best.pt
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders
from models import build_model
from evaluation.perturbation import PerturbationEvaluator
from evaluation.fewshot import FewShotEvaluator
from evaluation.manifold import ManifoldEvaluator
from visualization.plots import ResultsPlotter


def load_model(config, model_type, checkpoint_path=None):
    """Load a trained model from checkpoint.

    Args:
        config: Configuration dict.
        model_type: "baseline" or "cores".
        checkpoint_path: Path to checkpoint file.

    Returns:
        model: Loaded model.
    """
    config["_model_type"] = model_type

    # Get num_concepts from config
    _, test_loader, num_concepts = get_dataloaders(config)

    if model_type == "cores":
        config["model"]["cores"]["num_concepts"] = num_concepts

    model = build_model(config, num_concepts)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}, using random weights")

    return model, num_concepts


def evaluate_model(model, config, model_type, train_loader, test_loader):
    """Run all evaluations on a single model.

    Args:
        model: Trained model.
        config: Configuration dict.
        model_type: "baseline" or "cores".
        train_loader: Training data loader.
        test_loader: Test data loader.

    Returns:
        results: Dict with all evaluation results.
    """
    device = config["experiment"].get("device", "cpu")
    results = {}

    # Experiment 1: Perturbation Stability
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}: Perturbation Stability")
    print(f"{'='*60}")
    perturb_eval = PerturbationEvaluator(model, device=device)
    results["perturbation"] = perturb_eval.run_full_evaluation(test_loader, config)

    # Experiment 2: Few-shot Generalization
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}: Few-shot Generalization")
    print(f"{'='*60}")
    fewshot_eval = FewShotEvaluator(model, device=device)
    results["fewshot"] = fewshot_eval.run_full_evaluation(
        train_loader, test_loader, config
    )

    # Experiment 3: Manifold Smoothness
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type.upper()}: Manifold Smoothness")
    print(f"{'='*60}")
    manifold_eval = ManifoldEvaluator(model, device=device)
    results["manifold"] = manifold_eval.run_full_evaluation(test_loader, config)

    return results


def convert_results_for_json(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_results_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_results_for_json(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoRes-Embedding models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--model", type=str, default=None,
                        choices=["baseline", "cores", "both"],
                        help="Model(s) to evaluate (default: both)")
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint filename to load")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides
    if args.device is not None:
        config["experiment"]["device"] = args.device

    if config["experiment"]["device"] == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        config["experiment"]["device"] = "cpu"

    # Determine which models to evaluate
    models_to_eval = []
    if args.model is None or args.model == "both":
        models_to_eval = ["baseline", "cores"]
    else:
        models_to_eval = [args.model]

    # Output directory
    output_dir = args.output_dir or os.path.join(
        config["experiment"]["output_dir"],
        config["experiment"]["name"],
        "evaluation",
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CoRes-Embedding Evaluation")
    print("=" * 60)

    # Get data loaders
    print("\n📂 Loading dataset...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)

    # Evaluate each model
    all_results = {}

    for model_type in models_to_eval:
        # Build checkpoint path
        checkpoint_dir = os.path.join(
            config["experiment"]["output_dir"],
            config["experiment"]["name"],
            model_type,
            "checkpoints",
        )
        checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

        # Load model
        model, _ = load_model(config, model_type, checkpoint_path)

        # Evaluate
        results = evaluate_model(
            model, config, model_type, train_loader, test_loader
        )
        all_results[model_type] = results

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(convert_results_for_json(all_results), f, indent=2)
    print(f"\n📁 Results saved to: {results_path}")

    # Generate visualizations
    print("\n📊 Generating figures...")
    figures_dir = os.path.join(output_dir, "figures")
    plotter = ResultsPlotter(
        output_dir=figures_dir,
        save_format=config["visualization"]["save_format"],
        dpi=config["visualization"]["dpi"],
        figsize=tuple(config["visualization"]["figsize"]),
    )
    plotter.plot_all_results(all_results)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
