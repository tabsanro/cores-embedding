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
        sims = {"overall": F.cosine_similarity(z_clean, z_pert, dim=-1).cpu().numpy()}

        if self._has_decomposed:
            dec_c = self.model.get_decomposed_embedding(images)
            dec_p = self.model.get_decomposed_embedding(perturbed)
            for key in ("z_concept", "z_residual"):
                sims[key] = F.cosine_similarity(dec_c[key], dec_p[key], dim=-1).cpu().numpy()

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

    @staticmethod
    def _print_results(label, results):
        print(f"\n  {label}:")
        for key, res in results.items():
            line = f"    {key:.3f}: cos_sim = {res['cosine_similarity_mean']:.4f} ± {res['cosine_similarity_std']:.4f}"
            if "concept_stability_mean" in res:
                line += f"  concept: {res['concept_stability_mean']:.4f}  residual: {res['residual_stability_mean']:.4f}"
            print(line)

    def run_full_evaluation(self, dataloader, config):
        """Run complete perturbation evaluation."""
        cfg = config["evaluation"]["perturbation"]
        n = cfg["num_samples"]
        print("\n Experiment 1: Perturbation Stability\n" + "=" * 50)

        gaussian = self.evaluate_gaussian_noise(dataloader, cfg["noise_levels"], n)
        adversarial = self.evaluate_adversarial(dataloader, cfg["adversarial_epsilon"], n)

        self._print_results("Gaussian Noise", gaussian)
        self._print_results("Adversarial (FGSM)", adversarial)

        return {"gaussian": gaussian, "adversarial": adversarial}
