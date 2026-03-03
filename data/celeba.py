"""
CelebA Dataset Loader.
CelebA provides face images with 40 binary attribute annotations.
Selected attributes serve as concepts, remainder captured by residual.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd


class CelebADataset(Dataset):
    """CelebA dataset for compositional representation learning.

    Concept attributes are selected from the 40 binary annotations.
    Residual captures lighting, pose, background, expression nuances, etc.
    """

    ALL_ATTRS = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
        "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
        "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
        "Wearing_Necklace", "Wearing_Necktie", "Young",
    ]

    def __init__(self, root, split="train", transform=None,
                 concept_attrs=None, seed=42):
        """
        Args:
            root: Path to CelebA directory.
            split: 'train', 'valid', or 'test'.
            transform: Image transforms.
            concept_attrs: List of attribute names to use as concepts.
            seed: Random seed.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        if concept_attrs is None:
            self.concept_attrs = [
                "Male", "Eyeglasses", "Smiling", "Young",
                "Blond_Hair", "Black_Hair", "Brown_Hair",
                "Bangs", "Heavy_Makeup", "Wearing_Hat",
            ]
        else:
            self.concept_attrs = concept_attrs

        self.concept_indices = [
            self.ALL_ATTRS.index(a) for a in self.concept_attrs
            if a in self.ALL_ATTRS
        ]
        self.num_concepts = len(self.concept_indices)

        self._load_data(seed)

    def _load_data(self, seed):
        """Load CelebA data or generate synthetic version."""
        celeba_dir = os.path.join(self.root, "celeba")
        attr_file = os.path.join(celeba_dir, "list_attr_celeba.csv")

        if os.path.exists(attr_file):
            self._load_real_data(celeba_dir, attr_file)
        else:
            print(f"[CelebA] Data not found at {celeba_dir}")
            print("[CelebA] Generating synthetic data for development...")
            self._generate_synthetic_data(seed)

    def _load_real_data(self, celeba_dir, attr_file):
        """Load real CelebA data."""
        self.image_dir = os.path.join(celeba_dir, "img_align_celeba")

        # Read attributes (CSV format: image_id, attr1, attr2, ...)
        attrs_df = pd.read_csv(attr_file)
        # Set image_id as index
        attrs_df = attrs_df.set_index("image_id")
        # Convert -1/1 to 0/1
        attrs_df = (attrs_df + 1) // 2

        # Read partition file
        partition_file = os.path.join(celeba_dir, "list_eval_partition.csv")
        partition_file_txt = os.path.join(celeba_dir, "list_eval_partition.txt")
        if os.path.exists(partition_file):
            partitions = pd.read_csv(partition_file)
            partitions.columns = [c.strip() for c in partitions.columns]
            if "image_id" in partitions.columns and "partition" in partitions.columns:
                pass
            else:
                partitions.columns = ["image_id", "partition"]
            split_map = {"train": 0, "valid": 1, "test": 2}
            partition_val = split_map.get(self.split, 0)
            mask = partitions["partition"] == partition_val
            self.filenames = partitions[mask]["image_id"].values
            self.attrs = attrs_df.loc[self.filenames].values.astype(np.float32)
        elif os.path.exists(partition_file_txt):
            partitions = pd.read_csv(
                partition_file_txt, sep=r"\s+", header=None,
                names=["filename", "partition"]
            )
            split_map = {"train": 0, "valid": 1, "test": 2}
            partition_val = split_map.get(self.split, 0)
            mask = partitions["partition"] == partition_val
            self.filenames = partitions[mask]["filename"].values
            self.attrs = attrs_df.loc[self.filenames].values.astype(np.float32)
        else:
            self.filenames = attrs_df.index.values
            n = len(self.filenames)
            if self.split == "train":
                self.filenames = self.filenames[:int(0.8 * n)]
                self.attrs = attrs_df.values[:int(0.8 * n)].astype(np.float32)
            else:
                self.filenames = self.filenames[int(0.8 * n):]
                self.attrs = attrs_df.values[int(0.8 * n):].astype(np.float32)

        self.synthetic = False

    def _generate_synthetic_data(self, seed):
        """Generate synthetic CelebA-like data."""
        rng = np.random.RandomState(seed)
        n = 5000 if self.split == "train" else 1000

        self.syn_images = rng.randint(0, 256, (n, 64, 64, 3), dtype=np.uint8)
        # Binary attributes
        self.attrs = rng.randint(0, 2, (n, 40)).astype(np.float32)
        self.synthetic = True

    def __len__(self):
        if hasattr(self, "synthetic") and self.synthetic:
            return len(self.syn_images)
        return len(self.filenames)

    def __getitem__(self, idx):
        if hasattr(self, "synthetic") and self.synthetic:
            image = Image.fromarray(self.syn_images[idx])
        else:
            img_path = os.path.join(self.image_dir, self.filenames[idx])
            image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Concept labels (selected attributes)
        concept_label = self.attrs[idx, self.concept_indices]

        # All attributes as factor label
        all_attrs = self.attrs[idx].astype(np.int64)

        return image, torch.FloatTensor(concept_label), torch.LongTensor(all_attrs)


def get_celeba_loaders(config):
    """Get CelebA train/test data loaders."""
    img_size = config["dataset"].get("image_size", 64)
    root = config["dataset"].get("root", "data/raw")
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    seed = config["experiment"].get("seed", 42)
    concept_attrs = config["dataset"].get("celeba_attrs", None)

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

    train_dataset = CelebADataset(
        root=root, split="train", transform=train_transform,
        concept_attrs=concept_attrs, seed=seed,
    )
    test_dataset = CelebADataset(
        root=root, split="test", transform=test_transform,
        concept_attrs=concept_attrs, seed=seed,
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
