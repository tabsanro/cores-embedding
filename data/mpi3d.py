"""
MPI3D Dataset Loader.
MPI3D contains images of objects with 7 factors of variation:
  - object_color (6), object_shape (6), object_size (2),
  - camera_height (3), background_color (3),
  - horizontal_axis (40), vertical_axis (40)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MPI3DDataset(Dataset):
    """MPI3D dataset for compositional representation learning.

    The dataset stores images as a flat array indexed by factor combinations.
    Factors are decomposed into binary concept vectors for the CoRes model.
    """

    # Factor sizes for MPI3D
    FACTOR_SIZES = {
        "object_color": 6,
        "object_shape": 6,
        "object_size": 2,
        "camera_height": 3,
        "background_color": 3,
        "horizontal_axis": 40,
        "vertical_axis": 40,
    }

    def __init__(self, root, variant="toy", split="train", transform=None,
                 concept_factors=None, train_ratio=0.8, seed=42):
        """
        Args:
            root: Path to directory containing mpi3d .npz files.
            variant: 'toy', 'realistic', or 'real'.
            split: 'train' or 'test'.
            transform: Image transforms.
            concept_factors: List of factor names to use as concepts.
                            Remaining factors contribute to residual.
            train_ratio: Fraction of data for training.
            seed: Random seed for train/test split.
        """
        super().__init__()
        self.root = root
        self.variant = variant
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio

        # Default concept factors (discrete, interpretable ones)
        if concept_factors is None:
            self.concept_factors = [
                "object_color", "object_shape", "object_size",
                "camera_height", "background_color"
            ]
        else:
            self.concept_factors = concept_factors

        # Load data
        self._load_data(seed)

    def _load_data(self, seed):
        """Load MPI3D data from .npz file or generate synthetic version."""
        filepath = os.path.join(self.root, f"mpi3d_{self.variant}.npz")

        if os.path.exists(filepath):
            data = np.load(filepath)
            self.images = data["images"]
        else:
            # Generate synthetic data for testing/development
            print(f"[MPI3D] File not found: {filepath}")
            print("[MPI3D] Generating synthetic data for development...")
            self._generate_synthetic_data(seed)
            return

        # Compute total number of images
        factor_sizes = list(self.FACTOR_SIZES.values())
        total = int(np.prod(factor_sizes))

        # Compute factor indices for each image
        self.factor_values = np.zeros((total, len(factor_sizes)), dtype=np.int64)
        for i in range(total):
            idx = i
            for j in range(len(factor_sizes) - 1, -1, -1):
                self.factor_values[i, j] = idx % factor_sizes[j]
                idx //= factor_sizes[j]

        # Train/test split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(total)
        split_idx = int(total * self.train_ratio)

        if self.split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Build concept label vectors
        self._build_concept_labels()

    def _generate_synthetic_data(self, seed):
        """Generate synthetic MPI3D-like data for development."""
        rng = np.random.RandomState(seed)

        # Use smaller factor sizes for synthetic data
        synthetic_factor_sizes = {
            "object_color": 6,
            "object_shape": 6,
            "object_size": 2,
            "camera_height": 3,
            "background_color": 3,
            "horizontal_axis": 5,  # reduced from 40
            "vertical_axis": 5,    # reduced from 40
        }
        factor_sizes = list(synthetic_factor_sizes.values())
        total = int(np.prod(factor_sizes))

        # Generate random images (64x64x3)
        self.images = rng.randint(0, 256, (total, 64, 64, 3), dtype=np.uint8)

        # Generate factor values
        self.factor_values = np.zeros((total, len(factor_sizes)), dtype=np.int64)
        for i in range(total):
            idx = i
            for j in range(len(factor_sizes) - 1, -1, -1):
                self.factor_values[i, j] = idx % factor_sizes[j]
                idx //= factor_sizes[j]

        # Update FACTOR_SIZES for synthetic data
        self._synthetic_factor_sizes = synthetic_factor_sizes

        # Train/test split
        indices = rng.permutation(total)
        split_idx = int(total * self.train_ratio)
        if self.split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        self._build_concept_labels()

    def _build_concept_labels(self):
        """Build one-hot concept label vectors."""
        factor_names = list(self.FACTOR_SIZES.keys())
        factor_sizes = (
            list(self._synthetic_factor_sizes.values())
            if hasattr(self, "_synthetic_factor_sizes")
            else list(self.FACTOR_SIZES.values())
        )

        # Compute total concept dimension (sum of one-hot sizes for concept factors)
        self.concept_dim = 0
        self.concept_factor_indices = []
        self.concept_factor_sizes = []

        for name in self.concept_factors:
            if name in factor_names:
                idx = factor_names.index(name)
                if hasattr(self, "_synthetic_factor_sizes"):
                    size = list(self._synthetic_factor_sizes.values())[idx]
                else:
                    size = factor_sizes[idx]
                self.concept_factor_indices.append(idx)
                self.concept_factor_sizes.append(size)
                self.concept_dim += size

        self.num_concepts = self.concept_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image = self.images[real_idx]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Build concept label (one-hot concatenation)
        factors = self.factor_values[real_idx]
        concept_label = []
        for fi, fs in zip(self.concept_factor_indices, self.concept_factor_sizes):
            one_hot = np.zeros(fs, dtype=np.float32)
            one_hot[factors[fi]] = 1.0
            concept_label.append(one_hot)
        concept_label = np.concatenate(concept_label)

        # Full factor values for evaluation
        all_factors = factors.astype(np.int64)

        return image, torch.FloatTensor(concept_label), torch.LongTensor(all_factors)


def get_mpi3d_loaders(config):
    """Get MPI3D train/test data loaders."""
    img_size = config["dataset"].get("image_size", 64)
    variant = config["dataset"].get("mpi3d_variant", "toy")
    root = config["dataset"].get("root", "data/raw")
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    seed = config["experiment"].get("seed", 42)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MPI3DDataset(
        root=root, variant=variant, split="train",
        transform=train_transform, seed=seed,
    )
    test_dataset = MPI3DDataset(
        root=root, variant=variant, split="test",
        transform=test_transform, seed=seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, train_dataset.num_concepts
