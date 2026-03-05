"""
Visualization module for CoRes-Embedding experiment results.

Generates publication-quality figures for the three experiments:
1. Perturbation Stability (Noise σ vs Cosine Similarity)
2. Few-shot Generalization (Shots vs Accuracy)
3. Manifold Smoothness (Interpolation Visualization)
4. Latent Dimension Ablation (D vs Accuracy)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsPlotter:
    """Generate publication-quality figures for CoRes experiments."""

    def __init__(self, output_dir, save_format="pdf", dpi=300,
                 figsize=(8, 6)):
        """
        Args:
            output_dir: Directory to save figures.
            save_format: Figure format (pdf, png, svg).
            dpi: Figure DPI.
            figsize: Default figure size.
        """
        self.output_dir = output_dir
        self.save_format = save_format
        self.dpi = dpi
        self.figsize = figsize

        os.makedirs(output_dir, exist_ok=True)

        # Set style
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            plt.style.use("seaborn-whitegrid")

        # Colors
        self.colors = {
            "baseline": "#E74C3C",       # Red
            "cores": "#2ECC71",          # Green
            "cores_concept": "#3498DB",  # Blue
            "cores_residual": "#F39C12", # Orange
        }

        self.labels = {
            "baseline": "Baseline (Monolithic)",
            "cores": "CoRes (Proposed)",
            "cores_concept": "CoRes (Concept Only)",
            "cores_residual": "CoRes (Residual Only)",
        }

    def plot_perturbation_stability(self, baseline_results, cores_results,
                                     noise_type="gaussian",
                                     filename=None):
        """Plot Experiment 1: Perturbation Stability.

        X-axis: Noise Intensity (σ or ε)
        Y-axis: Embedding Cosine Similarity

        Args:
            baseline_results: Dict mapping noise_level → metrics.
            cores_results: Dict mapping noise_level → metrics.
            noise_type: "gaussian" or "adversarial".
            filename: Output filename.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Extract data
        noise_levels = sorted(baseline_results.keys())

        baseline_means = [baseline_results[s]["cosine_similarity_mean"] for s in noise_levels]
        baseline_stds = [baseline_results[s]["cosine_similarity_std"] for s in noise_levels]

        cores_means = [cores_results[s]["cosine_similarity_mean"] for s in noise_levels]
        cores_stds = [cores_results[s]["cosine_similarity_std"] for s in noise_levels]

        # Plot with error bands
        ax.plot(noise_levels, baseline_means, "o-",
                color=self.colors["baseline"], label=self.labels["baseline"],
                linewidth=2, markersize=8)
        ax.fill_between(noise_levels,
                        np.array(baseline_means) - np.array(baseline_stds),
                        np.array(baseline_means) + np.array(baseline_stds),
                        alpha=0.15, color=self.colors["baseline"])

        ax.plot(noise_levels, cores_means, "s-",
                color=self.colors["cores"], label=self.labels["cores"],
                linewidth=2, markersize=8)
        ax.fill_between(noise_levels,
                        np.array(cores_means) - np.array(cores_stds),
                        np.array(cores_means) + np.array(cores_stds),
                        alpha=0.15, color=self.colors["cores"])

        # Plot concept and residual stability for CoRes
        if "concept_stability_mean" in cores_results[noise_levels[0]]:
            concept_means = [cores_results[s]["concept_stability_mean"] for s in noise_levels]
            residual_means = [cores_results[s]["residual_stability_mean"] for s in noise_levels]

            ax.plot(noise_levels, concept_means, "^--",
                    color=self.colors["cores_concept"],
                    label=self.labels["cores_concept"],
                    linewidth=1.5, markersize=6, alpha=0.7)
            ax.plot(noise_levels, residual_means, "v--",
                    color=self.colors["cores_residual"],
                    label=self.labels["cores_residual"],
                    linewidth=1.5, markersize=6, alpha=0.7)

        # Labels and formatting
        xlabel = "Noise Intensity (σ)" if noise_type == "gaussian" else "Adversarial ε"
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Embedding Cosine Similarity", fontsize=14)
        ax.set_title("Perturbation Stability", fontsize=16, fontweight="bold")
        ax.legend(fontsize=11, loc="lower left")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=12)

        # Add annotation
        ax.annotate(
            "Concept acts as anchor",
            xy=(noise_levels[len(noise_levels)//2],
                cores_means[len(noise_levels)//2]),
            xytext=(noise_levels[-2], 0.3),
            arrowprops=dict(arrowstyle="->", color=self.colors["cores"]),
            fontsize=10, color=self.colors["cores"],
        )

        plt.tight_layout()

        if filename is None:
            filename = f"perturbation_stability_{noise_type}"
        self._save_figure(fig, filename)

    def plot_pgd_stability(self, baseline_results, cores_results,
                           epsilon=8/255, filename=None):
        """Plot PGD adversarial stability.

        X-axis: Number of PGD steps
        Y-axis: Embedding Cosine Similarity
        Epsilon is fixed at 8/255.

        Args:
            baseline_results: Dict mapping num_steps → metrics.
            cores_results: Dict mapping num_steps → metrics.
            epsilon: Fixed L∞ budget (for display only).
            filename: Output filename.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        steps = sorted(baseline_results.keys())

        baseline_means = [baseline_results[s]["cosine_similarity_mean"] for s in steps]
        baseline_stds  = [baseline_results[s]["cosine_similarity_std"]  for s in steps]

        cores_means = [cores_results[s]["cosine_similarity_mean"] for s in steps]
        cores_stds  = [cores_results[s]["cosine_similarity_std"]  for s in steps]

        ax.plot(steps, baseline_means, "o-",
                color=self.colors["baseline"], label=self.labels["baseline"],
                linewidth=2, markersize=8)
        ax.fill_between(steps,
                        np.array(baseline_means) - np.array(baseline_stds),
                        np.array(baseline_means) + np.array(baseline_stds),
                        alpha=0.15, color=self.colors["baseline"])

        ax.plot(steps, cores_means, "s-",
                color=self.colors["cores"], label=self.labels["cores"],
                linewidth=2, markersize=8)
        ax.fill_between(steps,
                        np.array(cores_means) - np.array(cores_stds),
                        np.array(cores_means) + np.array(cores_stds),
                        alpha=0.15, color=self.colors["cores"])

        # Plot concept / residual decomposition if available
        if "concept_stability_mean" in cores_results[steps[0]]:
            concept_means  = [cores_results[s]["concept_stability_mean"]  for s in steps]
            residual_means = [cores_results[s]["residual_stability_mean"] for s in steps]

            ax.plot(steps, concept_means, "^--",
                    color=self.colors["cores_concept"],
                    label=self.labels["cores_concept"],
                    linewidth=1.5, markersize=6, alpha=0.7)
            ax.plot(steps, residual_means, "v--",
                    color=self.colors["cores_residual"],
                    label=self.labels["cores_residual"],
                    linewidth=1.5, markersize=6, alpha=0.7)

        ax.set_xlabel("PGD Steps", fontsize=14)
        ax.set_ylabel("Embedding Cosine Similarity", fontsize=14)
        ax.set_title(
            f"PGD Adversarial Stability  (ε = 8/255 ≈ {epsilon:.4f})",
            fontsize=15, fontweight="bold"
        )
        ax.legend(fontsize=11, loc="lower left")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(steps)
        ax.tick_params(labelsize=12)

        plt.tight_layout()

        if filename is None:
            filename = "perturbation_stability_pgd"
        self._save_figure(fig, filename)

    def plot_fewshot_generalization(self, baseline_results, cores_results,
                                    filename=None):
        """Plot Experiment 2: Few-shot Generalization.

        X-axis: Number of shots
        Y-axis: Accuracy

        Args:
            baseline_results: Dict mapping shots → metrics.
            cores_results: Dict mapping shots → metrics.
            filename: Output filename.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        shots = sorted(baseline_results.keys())

        baseline_means = [baseline_results[s]["accuracy_mean"] for s in shots]
        baseline_stds = [baseline_results[s]["accuracy_std"] for s in shots]

        cores_means = [cores_results[s]["accuracy_mean"] for s in shots]
        cores_stds = [cores_results[s]["accuracy_std"] for s in shots]

        # Plot
        ax.errorbar(shots, baseline_means, yerr=baseline_stds,
                    fmt="o-", color=self.colors["baseline"],
                    label=self.labels["baseline"],
                    linewidth=2, markersize=8, capsize=5)

        ax.errorbar(shots, cores_means, yerr=cores_stds,
                    fmt="s-", color=self.colors["cores"],
                    label=self.labels["cores"],
                    linewidth=2, markersize=8, capsize=5)

        ax.set_xlabel("Number of Shots (Training Samples)", fontsize=14)
        ax.set_ylabel("Linear Probe Accuracy", fontsize=14)
        ax.set_title("Few-shot Generalization", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12, loc="lower right")
        ax.set_xscale("log")
        ax.tick_params(labelsize=12)

        # Highlight the gap at low shots
        if len(shots) >= 2:
            gap = cores_means[0] - baseline_means[0]
            if gap > 0:
                ax.annotate(
                    f"Δ = {gap:.3f}",
                    xy=(shots[0], (baseline_means[0] + cores_means[0]) / 2),
                    fontsize=11, fontweight="bold",
                    color=self.colors["cores"],
                    ha="right",
                )

        plt.tight_layout()

        if filename is None:
            filename = "fewshot_generalization"
        self._save_figure(fig, filename)

    def plot_latent_dim_ablation(self, baseline_results, cores_results,
                                  filename=None):
        """Plot Latent Dimension Ablation.

        X-axis: Latent Dimension Budget (D)
        Y-axis: Accuracy (Few-shot)

        Shows that CoRes degrades gracefully with smaller D.

        Args:
            baseline_results: Dict mapping D → accuracy.
            cores_results: Dict mapping D → accuracy.
            filename: Output filename.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        dims = sorted(baseline_results.keys())

        baseline_accs = [baseline_results[d] for d in dims]
        cores_accs = [cores_results[d] for d in dims]

        ax.plot(dims, baseline_accs, "o-",
                color=self.colors["baseline"], label=self.labels["baseline"],
                linewidth=2.5, markersize=10)
        ax.plot(dims, cores_accs, "s-",
                color=self.colors["cores"], label=self.labels["cores"],
                linewidth=2.5, markersize=10)

        ax.set_xlabel("Latent Dimension Budget (D)", fontsize=14)
        ax.set_ylabel("Few-shot Accuracy (10-shot)", fontsize=14)
        ax.set_title("Compression Efficiency", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(d) for d in dims])
        ax.tick_params(labelsize=12)

        # Shade the "critical region" where D is small
        ax.axvspan(min(dims), 32, alpha=0.05, color="red",
                   label="Low-dim region")

        # Add message
        ax.text(
            0.02, 0.02,
            '"Structured representations have\n superior compression efficiency"',
            transform=ax.transAxes, fontsize=10, style="italic",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8),
        )

        plt.tight_layout()

        if filename is None:
            filename = "latent_dim_ablation"
        self._save_figure(fig, filename)

    def plot_manifold_comparison(self, baseline_results, cores_results,
                                  filename=None):
        """Plot Experiment 3: Manifold Smoothness comparison.

        Bar chart comparing PPL and neighbor consistency.

        Args:
            baseline_results: Dict with manifold metrics.
            cores_results: Dict with manifold metrics.
            filename: Output filename.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PPL comparison
        ax1 = axes[0]
        models = ["Baseline", "CoRes"]
        ppls = [
            baseline_results["smoothness"]["ppl_mean"],
            cores_results["smoothness"]["ppl_mean"],
        ]
        ppl_stds = [
            baseline_results["smoothness"]["ppl_std"],
            cores_results["smoothness"]["ppl_std"],
        ]

        bars1 = ax1.bar(models, ppls, yerr=ppl_stds,
                        color=[self.colors["baseline"], self.colors["cores"]],
                        capsize=10, edgecolor="black", linewidth=0.5)
        ax1.set_ylabel("Perceptual Path Length (PPL) ↓", fontsize=13)
        ax1.set_title("Interpolation Smoothness", fontsize=14, fontweight="bold")
        ax1.tick_params(labelsize=12)

        # Neighbor consistency comparison
        ax2 = axes[1]
        consistencies = [
            baseline_results["smoothness"]["neighbor_consistency_mean"],
            cores_results["smoothness"]["neighbor_consistency_mean"],
        ]
        cons_stds = [
            baseline_results["smoothness"]["neighbor_consistency_std"],
            cores_results["smoothness"]["neighbor_consistency_std"],
        ]

        bars2 = ax2.bar(models, consistencies, yerr=cons_stds,
                        color=[self.colors["baseline"], self.colors["cores"]],
                        capsize=10, edgecolor="black", linewidth=0.5)
        ax2.set_ylabel("Neighbor Consistency ↑", fontsize=13)
        ax2.set_title("Path Coherence", fontsize=14, fontweight="bold")
        ax2.tick_params(labelsize=12)

        plt.tight_layout()

        if filename is None:
            filename = "manifold_smoothness"
        self._save_figure(fig, filename)

    def plot_concept_residual_analysis(self, cores_results, filename=None):
        """Plot concept vs residual contribution analysis.

        Shows the relative norms and stability of concept vs residual parts.

        Args:
            cores_results: Dict with decomposed metrics.
            filename: Output filename.
        """
        if "gaussian" not in cores_results:
            return

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        noise_levels = sorted(cores_results["gaussian"].keys())

        total_sims = [cores_results["gaussian"][s]["cosine_similarity_mean"]
                      for s in noise_levels]
        concept_sims = [cores_results["gaussian"][s].get("concept_stability_mean", 0)
                        for s in noise_levels]
        residual_sims = [cores_results["gaussian"][s].get("residual_stability_mean", 0)
                         for s in noise_levels]

        ax.plot(noise_levels, total_sims, "o-",
                color=self.colors["cores"], label="Total Embedding",
                linewidth=2.5, markersize=8)
        ax.plot(noise_levels, concept_sims, "^--",
                color=self.colors["cores_concept"], label="Concept Part",
                linewidth=2, markersize=7)
        ax.plot(noise_levels, residual_sims, "v--",
                color=self.colors["cores_residual"], label="Residual Part",
                linewidth=2, markersize=7)

        ax.set_xlabel("Gaussian Noise σ", fontsize=14)
        ax.set_ylabel("Cosine Similarity", fontsize=14)
        ax.set_title("Concept vs Residual Stability", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=12)

        # Add annotation about the "anchor" effect
        ax.annotate(
            '"Concepts act as anchors\nthat stabilize memory"',
            xy=(0.6, 0.15), xycoords="axes fraction",
            fontsize=10, style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                      edgecolor="gray", alpha=0.8),
        )

        plt.tight_layout()

        if filename is None:
            filename = "concept_residual_analysis"
        self._save_figure(fig, filename)

    def plot_all_results(self, all_results):
        """Generate all figures from complete experiment results.

        Args:
            all_results: Dict with structure:
                {
                    'baseline': {
                        'perturbation': {...},
                        'fewshot': {...},
                        'manifold': {...},
                    },
                    'cores': {
                        'perturbation': {...},
                        'fewshot': {...},
                        'manifold': {...},
                    }
                }
        """
        baseline = all_results.get("baseline", {})
        cores = all_results.get("cores", {})

        # 1. Perturbation Stability
        if "perturbation" in baseline and "perturbation" in cores:
            if "gaussian" in baseline["perturbation"]:
                self.plot_perturbation_stability(
                    baseline["perturbation"]["gaussian"],
                    cores["perturbation"]["gaussian"],
                    noise_type="gaussian",
                )
            if "adversarial" in baseline["perturbation"]:
                self.plot_perturbation_stability(
                    baseline["perturbation"]["adversarial"],
                    cores["perturbation"]["adversarial"],
                    noise_type="adversarial",
                )
            if ("pgd" in baseline["perturbation"]
                    and "pgd" in cores["perturbation"]):
                self.plot_pgd_stability(
                    baseline["perturbation"]["pgd"],
                    cores["perturbation"]["pgd"],
                )

        # 2. Few-shot Generalization
        if "fewshot" in baseline and "fewshot" in cores:
            self.plot_fewshot_generalization(
                baseline["fewshot"], cores["fewshot"],
            )

        # 3. Manifold Smoothness
        if "manifold" in baseline and "manifold" in cores:
            self.plot_manifold_comparison(
                baseline["manifold"], cores["manifold"],
            )

        # 4. Concept-Residual Analysis (CoRes only)
        if "perturbation" in cores:
            self.plot_concept_residual_analysis(cores["perturbation"])

        print(f"\n📊 All figures saved to: {self.output_dir}/")

    def _save_figure(self, fig, filename):
        """Save figure to file."""
        path = os.path.join(self.output_dir, f"{filename}.{self.save_format}")
        fig.savefig(path, format=self.save_format, dpi=self.dpi,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  💾 Saved: {path}")
