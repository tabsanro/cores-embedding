"""
Experiment 1: Perturbation Stability Evaluation.

Measures how well embeddings resist noise perturbations.
Score = cos(z(x), z(x + ε))
"""

import gc
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def _empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PerturbationEvaluator:
    """Evaluate embedding stability under input perturbations."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._has_decomposed = hasattr(model, "get_decomposed_embedding")

    def _compute_similarities(self, images, perturbed):
        def cos(a, b):
            return F.cosine_similarity(a, b, dim=-1).detach().cpu().numpy()

        sims = {"overall": cos(self.model.get_embedding(images), self.model.get_embedding(perturbed))}
        if self._has_decomposed:
            dc = self.model.get_decomposed_embedding(images)
            dp = self.model.get_decomposed_embedding(perturbed)
            for key in ("z_concept", "z_residual"):
                sims[key] = cos(dc[key], dp[key])
        return sims

    @staticmethod
    def _aggregate(collected):
        result = {
            "cosine_similarity_mean": np.mean(collected["overall"]),
            "cosine_similarity_std":  np.std(collected["overall"]),
        }
        for key, name in [("z_concept", "concept"), ("z_residual", "residual")]:
            if key in collected:
                result[f"{name}_stability_mean"] = np.mean(collected[key])
                result[f"{name}_stability_std"]  = np.std(collected[key])
        return result

    def _collect_over_batches(self, dataloader, perturb_fn, num_samples, desc):
        collected, count = {}, 0
        for batch in tqdm(dataloader, desc=desc, leave=False):
            if count >= num_samples:
                break
            perturbed, images = perturb_fn(batch)
            with torch.no_grad():
                for k, v in self._compute_similarities(images, perturbed).items():
                    collected.setdefault(k, []).append(v)
            count += images.shape[0]
            _empty_cache()
        return self._aggregate({k: np.concatenate(v) for k, v in collected.items()})

    @torch.no_grad()
    def evaluate_gaussian_noise(self, dataloader, noise_levels, num_samples=1000):
        return {
            s: self._collect_over_batches(
                dataloader,
                lambda batch, s=s: (torch.clamp(batch[0].to(self.device) + torch.randn_like(batch[0].to(self.device)) * s, -3, 3), batch[0].to(self.device)),
                num_samples, f"Gaussian σ={s:.3f}"
            )
            for s in noise_levels
        }

    def evaluate_adversarial(self, dataloader, epsilons, num_samples=1000):
        results = {}
        for eps in epsilons:
            def perturb(batch, eps=eps):
                images = batch[0].to(self.device).clone().detach().requires_grad_(True)
                labels = batch[1].to(self.device)
                F.binary_cross_entropy_with_logits(self.model(images)["logits"], labels).backward()
                with torch.no_grad():
                    adv = torch.clamp(images.detach() + eps * images.grad.sign(), -3, 3)
                images.grad = None
                return adv, images.detach()

            results[eps] = self._collect_over_batches(dataloader, perturb, num_samples, f"FGSM ε={eps:.3f}")
            _empty_cache()
        return results

    def evaluate_pgd(self, dataloader, step_counts, num_samples=1000):
        results = {}
        eps = 8 / 255
        for n in step_counts:
            alpha = 2.5 * eps / n

            def perturb(batch, n=n, a=alpha, eps=eps):
                images = batch[0].to(self.device)
                with torch.no_grad():
                    z_clean = self.model.get_embedding(images).detach()

                x_adv = torch.clamp(images + torch.empty_like(images).uniform_(-eps, eps), -3, 3).detach()
                for _ in range(n):
                    x_adv = x_adv.detach().requires_grad_(True)
                    loss = -F.cosine_similarity(z_clean, self.model.get_embedding(x_adv), dim=-1).mean()
                    self.model.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        x_adv = torch.clamp(images + torch.clamp(x_adv.detach() + a * x_adv.grad.sign() - images, -eps, eps), -3, 3).detach()
                    del loss

                del z_clean
                return x_adv, images.detach()

            results[n] = self._collect_over_batches(dataloader, perturb, num_samples, f"PGD steps={n}")
            _empty_cache()
            gc.collect()
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
            line = f"    steps={steps:>3d}: cos_sim = {res['cosine_similarity_mean']:.4f} ± {res['cosine_similarity_std']:.4f}"
            if "concept_stability_mean" in res:
                line += f"  concept: {res['concept_stability_mean']:.4f}  residual: {res['residual_stability_mean']:.4f}"
            print(line)

    def run_full_evaluation(self, dataloader, config):
        cfg = config["evaluation"]["perturbation"]
        n = cfg["num_samples"]
        print("\n Experiment 1: Perturbation Stability\n" + "=" * 50)

        gaussian    = self.evaluate_gaussian_noise(dataloader, cfg["noise_levels"], n)
        adversarial = self.evaluate_adversarial(dataloader, cfg["adversarial_epsilon"], n)
        pgd         = self.evaluate_pgd(dataloader, cfg["pgd_steps"], num_samples=n)

        self._print_results("Gaussian Noise", gaussian)
        self._print_results("Adversarial (FGSM)", adversarial)
        self._print_pgd_results(pgd)

        return {"gaussian": gaussian, "adversarial": adversarial, "pgd": pgd}
