"""
Run All Experiments: End-to-end pipeline for CoRes-Embedding.

This script:
1. Trains both Baseline and CoRes models
2. Runs all three evaluation protocols
3. Runs latent dimension ablation study
4. Generates all publication-quality figures

Usage:
    python scripts/run_all_experiments.py --config configs/default.yaml
    python scripts/run_all_experiments.py --config configs/default.yaml --quick
"""

import os
import sys
import json
import argparse
import yaml
import torch
import random
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders
from models import build_model
from training.trainer import Trainer
from evaluation.perturbation import PerturbationEvaluator
from evaluation.fewshot import FewShotEvaluator
from evaluation.manifold import ManifoldEvaluator
from visualization.plots import ResultsPlotter


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def train_model(config, model_type, train_loader, test_loader, num_concepts,
                epochs=None):
    """Train a model and return the trained model.

    Args:
        config: Configuration dict.
        model_type: "baseline" or "cores".
        train_loader: Training data loader.
        test_loader: Test data loader.
        num_concepts: Number of concepts.
        epochs: Override number of epochs.

    Returns:
        model: Trained model.
        history: Training history.
    """
    config["_model_type"] = model_type

    if model_type == "cores":
        config["model"]["cores"]["num_concepts"] = num_concepts

    model = build_model(config, num_concepts)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  📐 {model_type.upper()} Parameters: {total_params:,}")

    trainer = Trainer(model, config, model_type=model_type)

    if epochs is not None:
        history = trainer.train(train_loader, test_loader, epochs=epochs)
    else:
        history = trainer.train(train_loader, test_loader)

    # Load best checkpoint
    try:
        trainer.load_checkpoint("best.pt")
    except FileNotFoundError:
        pass

    return trainer.model, history


def evaluate_model(model, config, model_type, train_loader, test_loader):
    """Run all evaluations on a model."""
    device = config["experiment"].get("device", "cpu")
    results = {}

    # Experiment 1: Perturbation Stability
    perturb_eval = PerturbationEvaluator(model, device=device)
    results["perturbation"] = perturb_eval.run_full_evaluation(test_loader, config)

    # Experiment 2: Few-shot Generalization
    fewshot_eval = FewShotEvaluator(model, device=device)
    results["fewshot"] = fewshot_eval.run_full_evaluation(
        train_loader, test_loader, config
    )

    # Experiment 3: Manifold Smoothness
    manifold_eval = ManifoldEvaluator(model, device=device)
    results["manifold"] = manifold_eval.run_full_evaluation(test_loader, config)

    return results


def run_latent_dim_ablation(config, train_loader, test_loader, num_concepts,
                             dims=None, epochs=None):
    """Run latent dimension ablation study.

    Trains models with different latent dimensions and compares
    few-shot performance.

    Args:
        config: Configuration dict.
        train_loader, test_loader: Data loaders.
        num_concepts: Number of concepts.
        dims: List of latent dimensions to test.
        epochs: Training epochs per model.

    Returns:
        results: Dict with baseline and cores accuracy per dimension.
    """
    if dims is None:
        dims = config["evaluation"]["ablation"]["latent_dims"]

    if epochs is None:
        epochs = config["training"]["epochs"]

    device = config["experiment"].get("device", "cpu")

    baseline_results = {}
    cores_results = {}

    for D in dims:
        print(f"\n{'='*60}")
        print(f"Ablation: D = {D}")
        print(f"{'='*60}")

        config["model"]["latent_dim"] = D

        # Adjust concept/residual dims for CoRes
        config["model"]["cores"]["concept_dim"] = max(D // 2, 4)
        config["model"]["cores"]["residual_dim"] = max(D // 2, 4)

        for model_type in ["baseline", "cores"]:
            print(f"\n  Training {model_type} with D={D}...")
            model, _ = train_model(
                config, model_type, train_loader, test_loader,
                num_concepts, epochs=epochs,
            )

            # Quick few-shot evaluation (10-shot only)
            fewshot_eval = FewShotEvaluator(model, device=device)
            fewshot_results = fewshot_eval.evaluate_fewshot(
                train_loader, test_loader,
                shots_list=[10], num_trials=5,
            )

            acc = fewshot_results[10]["accuracy_mean"]

            if model_type == "baseline":
                baseline_results[D] = acc
            else:
                cores_results[D] = acc

            print(f"  {model_type} D={D}: 10-shot accuracy = {acc:.4f}")

    return baseline_results, cores_results


def main():
    parser = argparse.ArgumentParser(
        description="Run all CoRes-Embedding experiments"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs, smaller eval")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip latent dimension ablation")
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

    # Quick mode adjustments
    if args.quick:
        config["training"]["epochs"] = 10
        config["evaluation"]["perturbation"]["num_samples"] = 200
        config["evaluation"]["perturbation"]["noise_levels"] = [0.01, 0.1, 0.5, 1.0]
        config["evaluation"]["perturbation"]["adversarial_epsilon"] = [0.01, 0.1]
        config["evaluation"]["fewshot"]["shots"] = [10, 50]
        config["evaluation"]["fewshot"]["num_trials"] = 3
        config["evaluation"]["manifold"]["num_pairs"] = 100
        config["evaluation"]["ablation"]["latent_dims"] = [16, 64]
        print("⚡ Quick mode enabled: reduced epochs and evaluation size")

    # Set seed
    set_seed(config["experiment"]["seed"])

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        config["experiment"]["output_dir"],
        f"{config['experiment']['name']}_{timestamp}",
    )
    config["experiment"]["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("=" * 60)
    print("🚀 CoRes-Embedding: Full Experiment Pipeline")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Device: {config['experiment']['device']}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Time: {timestamp}")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Phase 1: Data Loading
    # ═══════════════════════════════════════════════════════
    print("\n📂 Phase 1: Loading dataset...")
    train_loader, test_loader, num_concepts = get_dataloaders(config)
    print(f"  Concepts: {num_concepts}")

    # ═══════════════════════════════════════════════════════
    # Phase 2: Training
    # ═══════════════════════════════════════════════════════
    print("\n🏋️ Phase 2: Training models...")

    baseline_model, baseline_history = train_model(
        config, "baseline", train_loader, test_loader, num_concepts,
    )

    cores_model, cores_history = train_model(
        config, "cores", train_loader, test_loader, num_concepts,
    )

    # ═══════════════════════════════════════════════════════
    # Phase 3: Evaluation
    # ═══════════════════════════════════════════════════════
    print("\n📊 Phase 3: Running evaluations...")

    all_results = {}

    print("\n--- Evaluating Baseline ---")
    all_results["baseline"] = evaluate_model(
        baseline_model, config, "baseline", train_loader, test_loader,
    )

    print("\n--- Evaluating CoRes ---")
    all_results["cores"] = evaluate_model(
        cores_model, config, "cores", train_loader, test_loader,
    )

    # ═══════════════════════════════════════════════════════
    # Phase 4: Latent Dimension Ablation
    # ═══════════════════════════════════════════════════════
    if not args.skip_ablation:
        print("\n🔬 Phase 4: Latent Dimension Ablation...")
        ablation_epochs = config["training"]["epochs"] // 2
        if args.quick:
            ablation_epochs = 5

        baseline_ablation, cores_ablation = run_latent_dim_ablation(
            config, train_loader, test_loader, num_concepts,
            epochs=ablation_epochs,
        )
        all_results["ablation"] = {
            "baseline": baseline_ablation,
            "cores": cores_ablation,
        }

    # ═══════════════════════════════════════════════════════
    # Phase 5: Visualization
    # ═══════════════════════════════════════════════════════
    print("\n🎨 Phase 5: Generating figures...")

    figures_dir = os.path.join(output_dir, "figures")
    plotter = ResultsPlotter(
        output_dir=figures_dir,
        save_format=config["visualization"]["save_format"],
        dpi=config["visualization"]["dpi"],
        figsize=tuple(config["visualization"]["figsize"]),
    )

    plotter.plot_all_results(all_results)

    # Ablation plot
    if "ablation" in all_results:
        plotter.plot_latent_dim_ablation(
            all_results["ablation"]["baseline"],
            all_results["ablation"]["cores"],
        )

    # ═══════════════════════════════════════════════════════
    # Phase 6: Save Results
    # ═══════════════════════════════════════════════════════
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(convert_results_for_json(all_results), f, indent=2)

    # ═══════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("📋 EXPERIMENT SUMMARY")
    print("=" * 60)

    # Perturbation stability summary
    if "perturbation" in all_results.get("baseline", {}) and \
       "perturbation" in all_results.get("cores", {}):
        gaussian_b = all_results["baseline"]["perturbation"].get("gaussian", {})
        gaussian_c = all_results["cores"]["perturbation"].get("gaussian", {})
        if gaussian_b and gaussian_c:
            max_sigma = max(gaussian_b.keys())
            b_sim = gaussian_b[max_sigma]["cosine_similarity_mean"]
            c_sim = gaussian_c[max_sigma]["cosine_similarity_mean"]
            print(f"\n🛡️  Perturbation Stability (σ={max_sigma}):")
            print(f"   Baseline:  {b_sim:.4f}")
            print(f"   CoRes:     {c_sim:.4f}")
            print(f"   Δ = {c_sim - b_sim:+.4f} {'✅' if c_sim > b_sim else '❌'}")

    # Few-shot summary
    if "fewshot" in all_results.get("baseline", {}) and \
       "fewshot" in all_results.get("cores", {}):
        b_fs = all_results["baseline"]["fewshot"]
        c_fs = all_results["cores"]["fewshot"]
        min_shots = min(b_fs.keys())
        b_acc = b_fs[min_shots]["accuracy_mean"]
        c_acc = c_fs[min_shots]["accuracy_mean"]
        print(f"\n🎯 Few-shot ({min_shots}-shot):")
        print(f"   Baseline:  {b_acc:.4f}")
        print(f"   CoRes:     {c_acc:.4f}")
        print(f"   Δ = {c_acc - b_acc:+.4f} {'✅' if c_acc > b_acc else '❌'}")

    # Manifold summary
    if "manifold" in all_results.get("baseline", {}) and \
       "manifold" in all_results.get("cores", {}):
        b_ppl = all_results["baseline"]["manifold"]["smoothness"]["ppl_mean"]
        c_ppl = all_results["cores"]["manifold"]["smoothness"]["ppl_mean"]
        print(f"\n🌊 Manifold Smoothness (PPL ↓):")
        print(f"   Baseline:  {b_ppl:.4f}")
        print(f"   CoRes:     {c_ppl:.4f}")
        print(f"   Δ = {c_ppl - b_ppl:+.4f} {'✅' if c_ppl < b_ppl else '❌'}")

    print(f"\n📁 All results saved to: {output_dir}")
    print("=" * 60)
    print("✅ All experiments complete!")


if __name__ == "__main__":
    main()
