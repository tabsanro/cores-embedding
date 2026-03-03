"""
Experiment 2: Few-shot Generalization Evaluation.

Trains a linear probe on top of frozen embeddings with limited data.
Measures how well the representation space supports new task adaptation.

Hypothesis: CoRes representations are more "disentangled" and thus
require fewer examples to find good decision boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class FewShotEvaluator:
    """Evaluate few-shot generalization via linear probing."""

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
    def extract_embeddings(self, dataloader, max_samples=None):
        """Extract embeddings and labels from a dataloader.

        Args:
            dataloader: Data loader.
            max_samples: Max number of samples to extract.

        Returns:
            embeddings: numpy array [N, D]
            concept_labels: numpy array [N, C]
            factor_labels: numpy array [N, F]
        """
        all_embeddings = []
        all_concept_labels = []
        all_factor_labels = []
        count = 0

        for images, concept_labels, factor_labels in dataloader:
            if max_samples is not None and count >= max_samples:
                break

            images = images.to(self.device)
            z = self.model.get_embedding(images)

            all_embeddings.append(z.cpu().numpy())
            all_concept_labels.append(concept_labels.numpy())
            all_factor_labels.append(factor_labels.numpy())

            count += images.shape[0]

        embeddings = np.concatenate(all_embeddings, axis=0)
        concept_labels = np.concatenate(all_concept_labels, axis=0)
        factor_labels = np.concatenate(all_factor_labels, axis=0)

        if max_samples is not None:
            embeddings = embeddings[:max_samples]
            concept_labels = concept_labels[:max_samples]
            factor_labels = factor_labels[:max_samples]

        return embeddings, concept_labels, factor_labels

    def linear_probe_sklearn(self, train_embeddings, train_labels,
                              test_embeddings, test_labels):
        """Run linear probe using sklearn LogisticRegression.

        Args:
            train_embeddings: [N_train, D]
            train_labels: [N_train] (single task) or [N_train, C] (multi-label)
            test_embeddings: [N_test, D]
            test_labels: [N_test] or [N_test, C]

        Returns:
            accuracy: Classification accuracy.
            f1: F1 score.
        """
        if train_labels.ndim == 1:
            # Single-label classification
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs",
                multi_class="multinomial", C=1.0,
            )
            clf.fit(train_embeddings, train_labels)
            preds = clf.predict(test_embeddings)
            acc = accuracy_score(test_labels, preds)
            f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
        else:
            # Multi-label classification (per-attribute)
            accs = []
            f1s = []
            for i in range(train_labels.shape[1]):
                y_train = train_labels[:, i]
                y_test = test_labels[:, i]

                # Skip if only one class present
                if len(np.unique(y_train)) < 2:
                    continue

                clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
                clf.fit(train_embeddings, y_train)
                preds = clf.predict(test_embeddings)
                accs.append(accuracy_score(y_test, preds))
                f1s.append(f1_score(y_test, preds, average="binary", zero_division=0))

            acc = np.mean(accs) if accs else 0
            f1 = np.mean(f1s) if f1s else 0

        return acc, f1

    def linear_probe_pytorch(self, train_embeddings, train_labels,
                              test_embeddings, test_labels,
                              epochs=100, lr=0.01):
        """Run linear probe using PyTorch (for GPU acceleration).

        Args:
            train_embeddings: [N_train, D]
            train_labels: [N_train, C]
            test_embeddings: [N_test, D]
            test_labels: [N_test, C]
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            accuracy: Classification accuracy.
            f1: F1 score.
        """
        input_dim = train_embeddings.shape[1]
        num_classes = train_labels.shape[1] if train_labels.ndim > 1 else int(train_labels.max()) + 1

        # Build linear classifier
        if train_labels.ndim > 1:
            # Multi-label
            linear = nn.Linear(input_dim, num_classes).to(self.device)
            criterion = nn.BCEWithLogitsLoss()
        else:
            linear = nn.Linear(input_dim, num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(linear.parameters(), lr=lr)

        # Convert to tensors
        X_train = torch.FloatTensor(train_embeddings).to(self.device)
        y_train = torch.FloatTensor(train_labels).to(self.device)
        X_test = torch.FloatTensor(test_embeddings).to(self.device)
        y_test = torch.FloatTensor(test_labels).to(self.device)

        # Train
        linear.train()
        for _ in range(epochs):
            logits = linear(X_train)
            loss = criterion(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        linear.eval()
        with torch.no_grad():
            logits = linear(X_test)

            if train_labels.ndim > 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == y_test).float().mean().item()
                # Per-attribute F1
                f1s = []
                for i in range(num_classes):
                    tp = ((preds[:, i] == 1) & (y_test[:, i] == 1)).sum().float()
                    fp = ((preds[:, i] == 1) & (y_test[:, i] == 0)).sum().float()
                    fn = ((preds[:, i] == 0) & (y_test[:, i] == 1)).sum().float()
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    f1s.append(f1.item())
                f1_mean = np.mean(f1s)
            else:
                preds = logits.argmax(dim=-1)
                acc = (preds == y_test.long()).float().mean().item()
                f1_mean = acc  # Simplified

        return acc, f1_mean

    def evaluate_fewshot(self, train_loader, test_loader,
                          shots_list, num_trials=10,
                          method="sklearn"):
        """Run few-shot evaluation with varying number of shots.

        Args:
            train_loader: Training data loader (full).
            test_loader: Test data loader.
            shots_list: List of shot counts [5, 10, 20, 50, 100].
            num_trials: Number of random trials per shot count.
            method: "sklearn" or "pytorch".

        Returns:
            results: Dict mapping shots → {
                'accuracy_mean': float,
                'accuracy_std': float,
                'f1_mean': float,
                'f1_std': float,
            }
        """
        # Extract all embeddings
        print("  Extracting embeddings...")
        train_emb, train_labels, train_factors = self.extract_embeddings(train_loader)
        test_emb, test_labels, test_factors = self.extract_embeddings(test_loader)

        results = {}

        for n_shots in shots_list:
            accs = []
            f1s = []

            for trial in range(num_trials):
                # Randomly sample n_shots per concept/class
                rng = np.random.RandomState(trial * 100 + n_shots)
                indices = rng.choice(len(train_emb), size=min(n_shots, len(train_emb)),
                                     replace=False)

                train_sub_emb = train_emb[indices]
                train_sub_labels = train_labels[indices]

                if method == "sklearn":
                    acc, f1 = self.linear_probe_sklearn(
                        train_sub_emb, train_sub_labels,
                        test_emb, test_labels,
                    )
                else:
                    acc, f1 = self.linear_probe_pytorch(
                        train_sub_emb, train_sub_labels,
                        test_emb, test_labels,
                    )

                accs.append(acc)
                f1s.append(f1)

            results[n_shots] = {
                "accuracy_mean": np.mean(accs),
                "accuracy_std": np.std(accs),
                "f1_mean": np.mean(f1s),
                "f1_std": np.std(f1s),
            }

        return results

    def run_full_evaluation(self, train_loader, test_loader, config):
        """Run complete few-shot evaluation.

        Args:
            train_loader: Training data loader.
            test_loader: Test data loader.
            config: Evaluation config.

        Returns:
            results: Few-shot evaluation results.
        """
        eval_config = config["evaluation"]["fewshot"]

        print("\n📊 Experiment 2: Few-shot Generalization")
        print("=" * 50)

        results = self.evaluate_fewshot(
            train_loader, test_loader,
            shots_list=eval_config["shots"],
            num_trials=eval_config["num_trials"],
        )

        for shots, res in results.items():
            print(f"  {shots:3d}-shot: "
                  f"Acc = {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f} | "
                  f"F1 = {res['f1_mean']:.4f} ± {res['f1_std']:.4f}")

        return results
