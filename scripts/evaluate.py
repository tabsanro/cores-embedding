"""Evaluation script for CoRes-Embedding experiments."""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders
from models import build_model
from evaluation.perturbation import PerturbationEvaluator
from evaluation.fewshot import FewShotEvaluator
from evaluation.manifold import ManifoldEvaluator
from visualization.plots import ResultsPlotter

EVALUATORS = [
    ("perturbation", PerturbationEvaluator, lambda e, trl, tel, c: e.run_full_evaluation(tel, c)),
    ("fewshot",      FewShotEvaluator,      lambda e, trl, tel, c: e.run_full_evaluation(trl, tel, c)),
    ("manifold",     ManifoldEvaluator,      lambda e, trl, tel, c: e.run_full_evaluation(tel, c)),
]


def load_model(config, model_type, num_concepts, checkpoint_path):
    """Load a trained model from checkpoint."""
    config["_model_type"] = model_type
    if model_type == "cores":
        config["model"]["cores"]["num_concepts"] = num_concepts

    model = build_model(config, num_concepts)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint at {checkpoint_path}, using random weights")

    return model


def evaluate_model(model, config, model_type, train_loader, test_loader):
    """Run all evaluations on a single model."""
    device = config["experiment"].get("device", "cpu")
    results = {}
    for name, cls, run_fn in EVALUATORS:
        print(f"\n[{model_type.upper()}] {name}")
        evaluator = cls(model, device=device)
        results[name] = run_fn(evaluator, train_loader, test_loader, config)
    return results


def convert_for_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoRes-Embedding models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, choices=["baseline", "cores", "both"])
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device:
        config["experiment"]["device"] = args.device
    if config["experiment"]["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config["experiment"]["device"] = "cpu"

    models_to_eval = (
        [args.model] if args.model and args.model != "both"
        else ["baseline", "cores"]
    )

    exp = config["experiment"]
    output_dir = args.output_dir or os.path.join(exp["output_dir"], exp["name"], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    train_loader, test_loader, num_concepts = get_dataloaders(config)

    all_results = {}
    for model_type in models_to_eval:
        ckpt_path = os.path.join(exp["output_dir"], exp["name"], model_type, "checkpoints", args.checkpoint)
        model = load_model(config, model_type, num_concepts, ckpt_path)
        all_results[model_type] = evaluate_model(model, config, model_type, train_loader, test_loader)

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
    print(f"\nResults saved to: {results_path}")

    vis = config["visualization"]
    plotter = ResultsPlotter(
        output_dir=os.path.join(output_dir, "figures"),
        save_format=vis["save_format"], dpi=vis["dpi"],
        figsize=tuple(vis["figsize"]),
    )
    plotter.plot_all_results(all_results)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
