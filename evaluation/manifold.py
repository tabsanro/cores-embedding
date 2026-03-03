"""
Experiment 3: Manifold Smoothness Evaluation.

Measures how smoothly the latent space interpolates between two points.
Uses a Perceptual Path Length (PPL)-like metric based on nearest neighbors.

Hypothesis: CoRes model's compositional structure produces logical
transitions (e.g., "red circle" → "blue square") rather than
"monster regions" in the interpolation path.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


class ManifoldEvaluator:
    """Evaluate manifold smoothness via latent space interpolation."""

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
    def extract_embeddings_and_labels(self, dataloader, max_samples=5000):
        """Extract embeddings with their factor labels.

        Args:
            dataloader: Data loader.
            max_samples: Maximum samples to extract.

        Returns:
            embeddings: [N, D]
            factor_labels: [N, F]
        """
        all_emb = []
        all_factors = []
        count = 0

        for images, _, factor_labels in dataloader:
            if count >= max_samples:
                break
            images = images.to(self.device)
            z = self.model.get_embedding(images)
            all_emb.append(z.cpu().numpy())
            all_factors.append(factor_labels.numpy())
            count += images.shape[0]

        embeddings = np.concatenate(all_emb, axis=0)[:max_samples]
        factors = np.concatenate(all_factors, axis=0)[:max_samples]
        return embeddings, factors

    def compute_interpolation_path(self, z_a, z_b, num_steps=20):
        """Compute linear interpolation path in latent space.

        Args:
            z_a: Start embedding [D]
            z_b: End embedding [D]
            num_steps: Number of interpolation steps.

        Returns:
            path: [num_steps+1, D] interpolated embeddings.
        """
        alphas = np.linspace(0, 1, num_steps + 1)
        path = np.array([
            (1 - alpha) * z_a + alpha * z_b
            for alpha in alphas
        ])
        return path

    def perceptual_path_length(self, path, all_embeddings, k=5):
        """Compute PPL-like metric for an interpolation path.

        For each point on the path, find its k nearest neighbors in the
        actual embedding space and measure how much the neighbors change
        between consecutive steps.

        Args:
            path: [T, D] interpolation path.
            all_embeddings: [N, D] all dataset embeddings.
            k: Number of nearest neighbors.

        Returns:
            ppl: Perceptual path length.
            neighbor_consistency: Average neighbor overlap between steps.
        """
        T = len(path)

        # Find k-NN for each interpolation point
        nn_indices = []
        for t in range(T):
            dists = cdist(path[t:t+1], all_embeddings, metric="cosine")[0]
            indices = np.argsort(dists)[:k]
            nn_indices.append(set(indices.tolist()))

        # Measure neighbor consistency between consecutive steps
        consistencies = []
        for t in range(T - 1):
            overlap = len(nn_indices[t] & nn_indices[t + 1]) / k
            consistencies.append(overlap)

        neighbor_consistency = np.mean(consistencies)

        # Measure step distances in embedding space
        step_distances = []
        for t in range(T - 1):
            dist = np.linalg.norm(path[t + 1] - path[t])
            step_distances.append(dist)

        # PPL: sum of squared step distances
        ppl = np.sum(np.array(step_distances) ** 2)

        return ppl, neighbor_consistency

    def evaluate_manifold_smoothness(self, dataloader, num_pairs=500,
                                      num_steps=20, k=5):
        """Evaluate manifold smoothness over random interpolation pairs.

        Args:
            dataloader: Test data loader.
            num_pairs: Number of random pairs to interpolate.
            num_steps: Interpolation steps per pair.
            k: Nearest neighbors for PPL computation.

        Returns:
            results: Dict with PPL and consistency metrics.
        """
        # Extract embeddings
        print("  Extracting embeddings for manifold evaluation...")
        embeddings, factors = self.extract_embeddings_and_labels(dataloader)

        ppls = []
        consistencies = []

        # Random pairs
        rng = np.random.RandomState(42)
        indices = rng.choice(len(embeddings), size=(num_pairs, 2), replace=True)

        for i in tqdm(range(num_pairs), desc="  Interpolation pairs", leave=False):
            idx_a, idx_b = indices[i]

            # Skip if same point
            if idx_a == idx_b:
                continue

            z_a = embeddings[idx_a]
            z_b = embeddings[idx_b]

            # Compute interpolation path
            path = self.compute_interpolation_path(z_a, z_b, num_steps)

            # Compute PPL
            ppl, consistency = self.perceptual_path_length(
                path, embeddings, k=k
            )

            ppls.append(ppl)
            consistencies.append(consistency)

        results = {
            "ppl_mean": np.mean(ppls),
            "ppl_std": np.std(ppls),
            "ppl_median": np.median(ppls),
            "neighbor_consistency_mean": np.mean(consistencies),
            "neighbor_consistency_std": np.std(consistencies),
        }

        return results

    def evaluate_factor_interpolation(self, dataloader, num_pairs=100,
                                       num_steps=20):
        """Evaluate if interpolation produces semantically meaningful transitions.

        For each pair, check if the interpolation path passes through
        embeddings with intermediate factor values.

        Args:
            dataloader: Test data loader.
            num_pairs: Number of pairs.
            num_steps: Interpolation steps.

        Returns:
            results: Dict with factor transition smoothness.
        """
        print("  Extracting embeddings for factor interpolation...")
        embeddings, factors = self.extract_embeddings_and_labels(dataloader)

        transition_scores = []
        rng = np.random.RandomState(42)

        for _ in tqdm(range(num_pairs), desc="  Factor interpolation", leave=False):
            # Pick two random points
            idx_a, idx_b = rng.choice(len(embeddings), size=2, replace=False)

            z_a = embeddings[idx_a]
            z_b = embeddings[idx_b]
            f_a = factors[idx_a]
            f_b = factors[idx_b]

            # Interpolation path
            path = self.compute_interpolation_path(z_a, z_b, num_steps)

            # For each point on path, find nearest neighbor's factors
            path_factors = []
            for t in range(len(path)):
                dists = cdist(path[t:t+1], embeddings, metric="cosine")[0]
                nn_idx = np.argmin(dists)
                path_factors.append(factors[nn_idx])
            path_factors = np.array(path_factors)

            # Measure monotonicity of factor transitions
            # A smooth interpolation should show monotonic changes
            monotonicity_scores = []
            for f in range(path_factors.shape[1]):
                factor_seq = path_factors[:, f].astype(float)

                # Check if factor changes monotonically
                if f_a[f] != f_b[f]:
                    # Direction should be consistent
                    diffs = np.diff(factor_seq)
                    nonzero_diffs = diffs[diffs != 0]
                    if len(nonzero_diffs) > 0:
                        direction = np.sign(f_b[f] - f_a[f])
                        consistent = np.mean(np.sign(nonzero_diffs) == direction)
                        monotonicity_scores.append(consistent)

            if monotonicity_scores:
                transition_scores.append(np.mean(monotonicity_scores))

        results = {
            "factor_transition_smoothness_mean": np.mean(transition_scores) if transition_scores else 0,
            "factor_transition_smoothness_std": np.std(transition_scores) if transition_scores else 0,
        }

        return results

    def run_full_evaluation(self, dataloader, config):
        """Run complete manifold smoothness evaluation.

        Args:
            dataloader: Test data loader.
            config: Evaluation config.

        Returns:
            all_results: Dict with all manifold metrics.
        """
        eval_config = config["evaluation"]["manifold"]

        print("\n📊 Experiment 3: Manifold Smoothness")
        print("=" * 50)

        # PPL evaluation
        print("\n🔸 Perceptual Path Length:")
        smoothness_results = self.evaluate_manifold_smoothness(
            dataloader,
            num_pairs=eval_config["num_pairs"],
            num_steps=eval_config["num_interpolation_steps"],
            k=eval_config["k_neighbors"],
        )
        print(f"  PPL = {smoothness_results['ppl_mean']:.4f} ± {smoothness_results['ppl_std']:.4f}")
        print(f"  Neighbor Consistency = {smoothness_results['neighbor_consistency_mean']:.4f}")

        # Factor interpolation
        print("\n🔸 Factor Transition Smoothness:")
        factor_results = self.evaluate_factor_interpolation(
            dataloader,
            num_pairs=min(eval_config["num_pairs"], 100),
            num_steps=eval_config["num_interpolation_steps"],
        )
        print(f"  Smoothness = {factor_results['factor_transition_smoothness_mean']:.4f}")

        return {
            "smoothness": smoothness_results,
            "factor_transition": factor_results,
        }
