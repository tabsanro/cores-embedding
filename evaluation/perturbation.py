"""
Experiment 1: Perturbation Stability Evaluation.

Measures how well embeddings resist noise perturbations.
Score = cos(z(x), z(x + ε))
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class PerturbationEvaluator:
    """Evaluate embedding stability under input perturbations."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._has_decomposed = hasattr(model, "get_decomposed_embedding")

    def _compute_similarities(self, images, perturbed):
        """Compute cosine similarities between clean and perturbed embeddings."""
        z_clean = self.model.get_embedding(images)
        z_pert = self.model.get_embedding(perturbed)
        sims = {"overall": F.cosine_similarity(z_clean, z_pert, dim=-1).detach().cpu().numpy()}

        if self._has_decomposed:
            dec_c = self.model.get_decomposed_embedding(images)
            dec_p = self.model.get_decomposed_embedding(perturbed)
            for key in ("z_concept", "z_residual"):
                sims[key] = F.cosine_similarity(dec_c[key], dec_p[key], dim=-1).detach().cpu().numpy()

        return sims

    @staticmethod
    def _aggregate(collected):
        """Aggregate collected similarity arrays into mean/std result dict."""
        result = {
            "cosine_similarity_mean": np.mean(collected["overall"]),
            "cosine_similarity_std": np.std(collected["overall"]),
        }
        for key, name in [("z_concept", "concept"), ("z_residual", "residual")]:
            if key in collected:
                result[f"{name}_stability_mean"] = np.mean(collected[key])
                result[f"{name}_stability_std"] = np.std(collected[key])
        return result

    def _collect_over_batches(self, dataloader, perturb_fn, num_samples, desc):
        """Iterate batches, apply perturb_fn, and collect similarities."""
        collected = {}
        count = 0
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if count >= num_samples:
                break
            perturbed, images = perturb_fn(batch)
            sims = self._compute_similarities(images, perturbed)
            for k, v in sims.items():
                collected.setdefault(k, []).append(v)
            count += images.shape[0]
        return self._aggregate({k: np.concatenate(v) for k, v in collected.items()})

    @torch.no_grad()
    def evaluate_gaussian_noise(self, dataloader, noise_levels, num_samples=1000):
        """Evaluate stability under Gaussian noise."""
        results = {}
        for sigma in noise_levels:
            def perturb(batch, s=sigma):
                images = batch[0].to(self.device)
                noisy = torch.clamp(images + torch.randn_like(images) * s, -3, 3)
                return noisy, images
            results[sigma] = self._collect_over_batches(
                dataloader, perturb, num_samples, f"Gaussian σ={sigma:.3f}")
        return results

    def evaluate_adversarial(self, dataloader, epsilons, num_samples=1000):
        """Evaluate stability under FGSM adversarial attack."""
        results = {}
        for epsilon in epsilons:
            def perturb(batch, eps=epsilon):
                images = batch[0].to(self.device).requires_grad_(True)
                labels = batch[1].to(self.device)
                self.model.eval()
                loss = F.binary_cross_entropy_with_logits(
                    self.model(images)["logits"], labels)
                self.model.zero_grad()
                loss.backward()
                with torch.no_grad():
                    adv = torch.clamp(images + eps * images.grad.sign(), -3, 3)
                return adv, images.detach()
            results[epsilon] = self._collect_over_batches(
                dataloader, perturb, num_samples, f"FGSM ε={epsilon:.3f}")
        return results

    def evaluate_pgd(self, dataloader, step_counts, num_samples=1000):
        """Evaluate stability under PGD adversarial attack.

        epsilon is fixed at 8/255; step_counts controls the number of PGD
        iterations so the caller can observe how robustness degrades as the
        attack becomes stronger.

        Args:
            dataloader: DataLoader providing (image, label) batches.
            step_counts: Iterable of integers, e.g. [1, 3, 5, 10, 20, 50].
            epsilon: L∞ perturbation budget (default 8/255).
            num_samples: Maximum number of images to evaluate.

        Returns:
            dict mapping each step count to an aggregated similarity result.
        """
        results = {}
        epsilon=8/255
        for num_steps in step_counts:
            # Common PGD heuristic: α = 2.5 * ε / T
            alpha = 2.5 * epsilon / num_steps

            def perturb(batch, n=num_steps, a=alpha, eps=epsilon):
                images = batch[0].to(self.device)

                with torch.no_grad():
                    z_clean = self.model.get_embedding(images)

                # Random start within L∞ ball (standard PGD)
                x_adv = images.clone().detach()
                x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
                delta = torch.clamp(x_adv - images, -eps, eps)
                x_adv = torch.clamp(images + delta, -3, 3).detach()

                for _ in range(n):
                    with torch.enable_grad():
                        x_adv.requires_grad_(True)
                        z_adv = self.model.get_embedding(x_adv)
                        # Minimise cosine similarity → maximise embedding disruption
                        loss = -F.cosine_similarity(z_clean, z_adv, dim=-1).mean()
                        self.model.zero_grad()
                        loss.backward()
                        grad_sign = x_adv.grad.sign()

                    with torch.no_grad():
                        x_adv = x_adv.detach() + a * grad_sign
                        delta = torch.clamp(x_adv - images, -eps, eps)
                        x_adv = torch.clamp(images + delta, -3, 3)

                return x_adv.detach(), images

            results[num_steps] = self._collect_over_batches(
                dataloader, perturb, num_samples, f"PGD steps={num_steps}")
        return results

    @staticmethod
    def _print_results(label, results):
        print(f"\n  {label}:")
        for key, res in results.items():
            line = f"    {key:.3f}: cos_sim = {res['cosine_similarity_mean']:.4f} ± {res['cosine_similarity_std']:.4f}"
            if "concept_stability_mean" in res:
                line += f"  concept: {res['concept_stability_mean']:.4f}  residual: {res['residual_stability_mean']:.4f}"
            print(line)

    @staticmethod
    def _print_pgd_results(results):
        print(f"\n  PGD (ε=8/255):")
        for steps, res in results.items():
            line = (f"    steps={steps:>3d}: cos_sim = {res['cosine_similarity_mean']:.4f}"
                    f" ± {res['cosine_similarity_std']:.4f}")
            if "concept_stability_mean" in res:
                line += (f"  concept: {res['concept_stability_mean']:.4f}"
                         f"  residual: {res['residual_stability_mean']:.4f}")
            print(line)

    def run_full_evaluation(self, dataloader, config):
        """Run complete perturbation evaluation."""
        cfg = config["evaluation"]["perturbation"]
        n = cfg["num_samples"]
        print("\n Experiment 1: Perturbation Stability\n" + "=" * 50)

        gaussian = self.evaluate_gaussian_noise(dataloader, cfg["noise_levels"], n)
        adversarial = self.evaluate_adversarial(dataloader, cfg["adversarial_epsilon"], n)
        pgd = self.evaluate_pgd(dataloader, cfg["pgd_steps"], num_samples=n)

        self._print_results("Gaussian Noise", gaussian)
        self._print_results("Adversarial (FGSM)", adversarial)
        self._print_pgd_results(pgd)

        return {"gaussian": gaussian, "adversarial": adversarial, "pgd": pgd}
