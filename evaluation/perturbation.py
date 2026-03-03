"""
Experiment 1: Perturbation Stability Evaluation.

Measures how well embeddings resist noise perturbations.
Score = cos(z(x), z(x + ε))

Hypothesis: CoRes model's z_concept acts as a stable anchor,
so overall embedding similarity remains higher than Baseline.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class PerturbationEvaluator:
    """Evaluate embedding stability under input perturbations."""

    def __init__(self, model, device="cuda"):
        """
        Args:
            model: Trained model (BaselineModel or CoResModel).
            device: Device to use.
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_gaussian_noise(self, dataloader, noise_levels,
                                 num_samples=1000):
        """Evaluate stability under Gaussian noise.

        Args:
            dataloader: Test data loader.
            noise_levels: List of noise standard deviations σ.
            num_samples: Maximum number of samples to evaluate.

        Returns:
            results: Dict mapping σ → {
                'cosine_similarity_mean': float,
                'cosine_similarity_std': float,
                'concept_stability': float (CoRes only),
                'residual_stability': float (CoRes only),
            }
        """
        results = {}

        for sigma in noise_levels:
            similarities = []
            concept_sims = []
            residual_sims = []
            count = 0

            for images, _, _ in tqdm(dataloader,
                                      desc=f"Gaussian σ={sigma:.3f}",
                                      leave=False):
                if count >= num_samples:
                    break

                images = images.to(self.device)
                batch_size = images.shape[0]

                # Add Gaussian noise
                noise = torch.randn_like(images) * sigma
                noisy_images = images + noise

                # Clamp to valid range (after normalization, this is approximate)
                noisy_images = torch.clamp(noisy_images, -3, 3)

                # Get embeddings
                z_clean = self.model.get_embedding(images)
                z_noisy = self.model.get_embedding(noisy_images)

                # Cosine similarity
                cos_sim = F.cosine_similarity(z_clean, z_noisy, dim=-1)
                similarities.extend(cos_sim.cpu().numpy().tolist())

                # Decomposed analysis for CoRes model
                if hasattr(self.model, "get_decomposed_embedding"):
                    dec_clean = self.model.get_decomposed_embedding(images)
                    dec_noisy = self.model.get_decomposed_embedding(noisy_images)

                    # Concept stability
                    c_sim = F.cosine_similarity(
                        dec_clean["z_concept"], dec_noisy["z_concept"], dim=-1
                    )
                    concept_sims.extend(c_sim.cpu().numpy().tolist())

                    # Residual stability
                    r_sim = F.cosine_similarity(
                        dec_clean["z_residual"], dec_noisy["z_residual"], dim=-1
                    )
                    residual_sims.extend(r_sim.cpu().numpy().tolist())

                count += batch_size

            result = {
                "cosine_similarity_mean": np.mean(similarities),
                "cosine_similarity_std": np.std(similarities),
            }

            if concept_sims:
                result["concept_stability_mean"] = np.mean(concept_sims)
                result["concept_stability_std"] = np.std(concept_sims)
                result["residual_stability_mean"] = np.mean(residual_sims)
                result["residual_stability_std"] = np.std(residual_sims)

            results[sigma] = result

        return results

    def evaluate_adversarial(self, dataloader, epsilons, num_samples=1000):
        """Evaluate stability under FGSM adversarial attack.

        Uses the model's own classification loss to generate adversarial examples.

        Args:
            dataloader: Test data loader.
            epsilons: List of adversarial perturbation magnitudes.
            num_samples: Maximum number of samples.

        Returns:
            results: Dict mapping ε → similarity metrics.
        """
        results = {}

        for epsilon in epsilons:
            similarities = []
            concept_sims = []
            residual_sims = []
            count = 0

            for images, concept_labels, _ in tqdm(
                dataloader, desc=f"FGSM ε={epsilon:.3f}", leave=False
            ):
                if count >= num_samples:
                    break

                images = images.to(self.device).requires_grad_(True)
                concept_labels = concept_labels.to(self.device)

                # Forward pass (need gradients for FGSM)
                self.model.eval()
                output = self.model(images)
                loss = F.binary_cross_entropy_with_logits(
                    output["logits"], concept_labels
                )

                # Backward to get gradients
                self.model.zero_grad()
                if images.grad is not None:
                    images.grad.zero_()
                loss.backward()

                # FGSM attack
                with torch.no_grad():
                    sign_grad = images.grad.data.sign()
                    adv_images = images + epsilon * sign_grad
                    adv_images = torch.clamp(adv_images, -3, 3)

                    # Get embeddings
                    z_clean = self.model.get_embedding(images)
                    z_adv = self.model.get_embedding(adv_images)

                    # Cosine similarity
                    cos_sim = F.cosine_similarity(z_clean, z_adv, dim=-1)
                    similarities.extend(cos_sim.cpu().numpy().tolist())

                    # Decomposed analysis
                    if hasattr(self.model, "get_decomposed_embedding"):
                        dec_clean = self.model.get_decomposed_embedding(images)
                        dec_adv = self.model.get_decomposed_embedding(adv_images)

                        c_sim = F.cosine_similarity(
                            dec_clean["z_concept"], dec_adv["z_concept"], dim=-1
                        )
                        concept_sims.extend(c_sim.cpu().numpy().tolist())

                        r_sim = F.cosine_similarity(
                            dec_clean["z_residual"], dec_adv["z_residual"], dim=-1
                        )
                        residual_sims.extend(r_sim.cpu().numpy().tolist())

                count += images.shape[0]

            result = {
                "cosine_similarity_mean": np.mean(similarities),
                "cosine_similarity_std": np.std(similarities),
            }

            if concept_sims:
                result["concept_stability_mean"] = np.mean(concept_sims)
                result["concept_stability_std"] = np.std(concept_sims)
                result["residual_stability_mean"] = np.mean(residual_sims)
                result["residual_stability_std"] = np.std(residual_sims)

            results[epsilon] = result

        return results

    def run_full_evaluation(self, dataloader, config):
        """Run complete perturbation evaluation.

        Args:
            dataloader: Test data loader.
            config: Evaluation config dict.

        Returns:
            all_results: Dict with gaussian and adversarial results.
        """
        eval_config = config["evaluation"]["perturbation"]

        print("\n📊 Experiment 1: Perturbation Stability")
        print("=" * 50)

        # Gaussian noise
        print("\n🔸 Gaussian Noise Evaluation:")
        gaussian_results = self.evaluate_gaussian_noise(
            dataloader,
            noise_levels=eval_config["noise_levels"],
            num_samples=eval_config["num_samples"],
        )

        for sigma, res in gaussian_results.items():
            print(f"  σ={sigma:.3f}: cos_sim = {res['cosine_similarity_mean']:.4f}"
                  f" ± {res['cosine_similarity_std']:.4f}")
            if "concept_stability_mean" in res:
                print(f"    concept: {res['concept_stability_mean']:.4f}"
                      f"  residual: {res['residual_stability_mean']:.4f}")

        # Adversarial attack
        print("\n🔸 Adversarial (FGSM) Evaluation:")
        adversarial_results = self.evaluate_adversarial(
            dataloader,
            epsilons=eval_config["adversarial_epsilon"],
            num_samples=eval_config["num_samples"],
        )

        for eps, res in adversarial_results.items():
            print(f"  ε={eps:.3f}: cos_sim = {res['cosine_similarity_mean']:.4f}"
                  f" ± {res['cosine_similarity_std']:.4f}")
            if "concept_stability_mean" in res:
                print(f"    concept: {res['concept_stability_mean']:.4f}"
                      f"  residual: {res['residual_stability_mean']:.4f}")

        return {
            "gaussian": gaussian_results,
            "adversarial": adversarial_results,
        }
