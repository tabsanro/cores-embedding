"""
CUB-200-2011 Dataset Loader.
Caltech-UCSD Birds-200-2011 provides 11,788 bird images with
312 binary attribute annotations and 200 species labels.

Download: https://data.caltech.edu/records/65de6-vp158
Expected directory structure:
    data/raw/cub200/
        CUB_200_2011/
            images/
            images.txt
            image_class_labels.txt
            train_test_split.txt
            attributes/
                image_attribute_labels.txt
                attributes.txt
            parts/
            bounding_boxes.txt
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CUB200Dataset(Dataset):
    """CUB-200-2011 dataset for compositional representation learning.

    Binary attribute annotations (312 attributes) serve as concepts.
    These capture fine-grained bird characteristics such as:
    - Bill shape, wing color/pattern, breast color
    - Tail shape, upper/under parts color, head pattern, etc.

    Attributes are grouped by body part and visual property,
    making them ideal for concept-based models.
    """

    def __init__(self, root, split="train", transform=None,
                 concept_attrs=None, num_concepts=None, seed=42):
        """
        Args:
            root: Path to dataset root (e.g., 'data/raw').
            split: 'train' or 'test'.
            transform: Image transforms.
            concept_attrs: Optional list of attribute indices to use as concepts.
                           If None, uses top `num_concepts` most balanced attributes.
            num_concepts: Number of concepts to select if concept_attrs is None.
                          Defaults to 20.
            seed: Random seed for synthetic fallback.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.num_concepts_requested = num_concepts or 20

        self.cub_dir = os.path.join(root, "cub200", "CUB_200_2011")

        if os.path.exists(self.cub_dir):
            self._load_real_data(concept_attrs)
        else:
            print(f"[CUB-200] Data not found at {self.cub_dir}")
            print("[CUB-200] Please download CUB-200-2011 from:")
            print("  https://data.caltech.edu/records/65de6-vp158")
            print("[CUB-200] Generating synthetic data for development...")
            self._generate_synthetic_data(seed)

    def _load_real_data(self, concept_attrs=None):
        """Load real CUB-200-2011 data."""
        self.synthetic = False

        # --- Read image paths ---
        images_txt = os.path.join(self.cub_dir, "images.txt")
        with open(images_txt, "r") as f:
            lines = f.readlines()
        self.all_image_ids = []
        self.all_image_paths = []
        for line in lines:
            img_id, img_path = line.strip().split()
            self.all_image_ids.append(int(img_id))
            self.all_image_paths.append(img_path)

        # --- Read train/test split ---
        split_txt = os.path.join(self.cub_dir, "train_test_split.txt")
        with open(split_txt, "r") as f:
            lines = f.readlines()
        is_train = {}
        for line in lines:
            img_id, is_tr = line.strip().split()
            is_train[int(img_id)] = int(is_tr)

        # --- Read class labels ---
        labels_txt = os.path.join(self.cub_dir, "image_class_labels.txt")
        with open(labels_txt, "r") as f:
            lines = f.readlines()
        class_labels = {}
        for line in lines:
            img_id, label = line.strip().split()
            class_labels[int(img_id)] = int(label) - 1  # 0-indexed

        # --- Read attribute names ---
        attrs_txt = os.path.join(self.cub_dir, "attributes", "attributes.txt")
        self.attribute_names = []
        if os.path.exists(attrs_txt):
            with open(attrs_txt, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split(maxsplit=1)
                    self.attribute_names.append(parts[1] if len(parts) > 1 else parts[0])

        # --- Read image attributes ---
        # Format: <image_id> <attribute_id> <is_present> <certainty_id> <time>
        attr_labels_txt = os.path.join(
            self.cub_dir, "attributes", "image_attribute_labels.txt"
        )
        num_images = len(self.all_image_ids)
        num_attrs = 312
        all_attrs = np.zeros((num_images, num_attrs), dtype=np.float32)

        if os.path.exists(attr_labels_txt):
            with open(attr_labels_txt, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    img_id = int(parts[0])
                    attr_id = int(parts[1])
                    is_present = int(parts[2])
                    # img_id and attr_id are 1-indexed
                    all_attrs[img_id - 1, attr_id - 1] = float(is_present)

        # --- Select concept attributes ---
        if concept_attrs is not None:
            self.concept_indices = concept_attrs
        else:
            # Select the most balanced attributes (closest to 50% positive rate)
            positive_rates = all_attrs.mean(axis=0)
            balance_score = np.abs(positive_rates - 0.5)
            # Pick top-k most balanced (lowest score = most balanced)
            self.concept_indices = np.argsort(balance_score)[
                :self.num_concepts_requested
            ].tolist()

        self.num_concepts = len(self.concept_indices)

        # --- Filter by split ---
        split_is_train = 1 if self.split == "train" else 0
        indices = [
            i for i, img_id in enumerate(self.all_image_ids)
            if is_train[img_id] == split_is_train
        ]

        self.image_paths = [self.all_image_paths[i] for i in indices]
        self.class_labels = np.array(
            [class_labels[self.all_image_ids[i]] for i in indices],
            dtype=np.int64,
        )
        self.attrs = all_attrs[indices]
        self.image_dir = os.path.join(self.cub_dir, "images")

        print(f"[CUB-200] Loaded {len(self)} {self.split} images, "
              f"{self.num_concepts} concepts, 200 classes")

    def _generate_synthetic_data(self, seed):
        """Generate synthetic CUB-200-like data for development."""
        self.synthetic = True
        rng = np.random.RandomState(seed + (0 if self.split == "train" else 1))
        n = 3000 if self.split == "train" else 600

        self.syn_images = rng.randint(0, 256, (n, 64, 64, 3), dtype=np.uint8)
        self.attrs = rng.randint(0, 2, (n, 312)).astype(np.float32)
        self.class_labels = rng.randint(0, 200, n).astype(np.int64)
        self.concept_indices = list(range(self.num_concepts_requested))
        self.num_concepts = self.num_concepts_requested
        self.attribute_names = [f"attr_{i}" for i in range(312)]

    def __len__(self):
        if self.synthetic:
            return len(self.syn_images)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.synthetic:
            image = Image.fromarray(self.syn_images[idx])
        else:
            img_path = os.path.join(self.image_dir, self.image_paths[idx])
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Concept labels (selected attributes)
        concept_label = self.attrs[idx, self.concept_indices]

        # Class label (species)
        class_label = self.class_labels[idx]

        return image, torch.FloatTensor(concept_label), torch.tensor(class_label)

    def get_attribute_names(self):
        """Return names of selected concept attributes."""
        if hasattr(self, "attribute_names") and self.attribute_names:
            return [self.attribute_names[i] for i in self.concept_indices]
        return [f"concept_{i}" for i in range(self.num_concepts)]


def get_cub200_loaders(config):
    """Get CUB-200 train/test data loaders.

    Args:
        config: Configuration dict with dataset/training settings.

    Returns:
        train_loader, test_loader, num_concepts
    """
    img_size = config["dataset"].get("image_size", 64)
    root = config["dataset"].get("root", "data/raw")
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    seed = config["experiment"].get("seed", 42)
    num_concepts = config["dataset"].get("cub200_num_concepts", 20)

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 8, img_size + 8)),
        transforms.RandomCrop((img_size, img_size)),
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

    train_dataset = CUB200Dataset(
        root=root, split="train", transform=train_transform,
        num_concepts=num_concepts, seed=seed,
    )
    test_dataset = CUB200Dataset(
        root=root, split="test", transform=test_transform,
        num_concepts=num_concepts, seed=seed,
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
