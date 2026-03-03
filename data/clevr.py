"""
CLEVR Dataset Loader.
CLEVR (Compositional Language and Elementary Visual Reasoning) provides
synthetic 3D scenes with objects of varying shape, color, size, and material.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CLEVRDataset(Dataset):
    """CLEVR dataset for compositional representation learning.

    Concepts are defined per-object: shape, color, size, material.
    We encode the primary object's properties as the concept vector.
    """

    SHAPES = ["cube", "sphere", "cylinder"]
    COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
    SIZES = ["small", "large"]
    MATERIALS = ["rubber", "metal"]

    def __init__(self, root, split="train", transform=None, max_objects=1, seed=42):
        """
        Args:
            root: Path to CLEVR directory.
            split: 'train' or 'val'.
            transform: Image transforms.
            max_objects: Number of objects to encode concepts for.
            seed: Random seed.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.max_objects = max_objects

        self._load_data(seed)

    def _load_data(self, seed):
        """Load CLEVR data or generate synthetic version."""
        scene_file = os.path.join(
            self.root, "CLEVR_v1.0", "scenes",
            f"CLEVR_{self.split}_scenes.json"
        )
        image_dir = os.path.join(
            self.root, "CLEVR_v1.0", "images", self.split
        )

        if os.path.exists(scene_file) and os.path.exists(image_dir):
            with open(scene_file, "r") as f:
                scenes_data = json.load(f)
            self.scenes = scenes_data["scenes"]
            self.image_dir = image_dir
            self.synthetic = False
        else:
            print(f"[CLEVR] Data not found at {self.root}")
            print("[CLEVR] Generating synthetic data for development...")
            self._generate_synthetic_data(seed)
            self.synthetic = True

        # Compute concept dimensions
        # Per object: shape(3) + color(8) + size(2) + material(2) = 15
        self.concepts_per_object = (
            len(self.SHAPES) + len(self.COLORS) +
            len(self.SIZES) + len(self.MATERIALS)
        )
        self.num_concepts = self.concepts_per_object * self.max_objects

    def _generate_synthetic_data(self, seed):
        """Generate synthetic CLEVR-like data."""
        rng = np.random.RandomState(seed)
        n_samples = 5000 if self.split == "train" else 1000

        self.syn_images = rng.randint(0, 256, (n_samples, 64, 64, 3), dtype=np.uint8)
        self.syn_objects = []

        for _ in range(n_samples):
            n_obj = rng.randint(1, self.max_objects + 1)
            objects = []
            for _ in range(n_obj):
                obj = {
                    "shape": rng.choice(self.SHAPES),
                    "color": rng.choice(self.COLORS),
                    "size": rng.choice(self.SIZES),
                    "material": rng.choice(self.MATERIALS),
                    "3d_coords": rng.uniform(-3, 3, 3).tolist(),
                }
                objects.append(obj)
            self.syn_objects.append(objects)

    def _encode_object_concepts(self, obj):
        """Encode a single object's properties as a one-hot vector."""
        concept = np.zeros(self.concepts_per_object, dtype=np.float32)
        offset = 0

        # Shape
        if obj["shape"] in self.SHAPES:
            concept[offset + self.SHAPES.index(obj["shape"])] = 1.0
        offset += len(self.SHAPES)

        # Color
        if obj["color"] in self.COLORS:
            concept[offset + self.COLORS.index(obj["color"])] = 1.0
        offset += len(self.COLORS)

        # Size
        if obj["size"] in self.SIZES:
            concept[offset + self.SIZES.index(obj["size"])] = 1.0
        offset += len(self.SIZES)

        # Material
        if obj["material"] in self.MATERIALS:
            concept[offset + self.MATERIALS.index(obj["material"])] = 1.0

        return concept

    def __len__(self):
        if self.synthetic:
            return len(self.syn_images)
        return len(self.scenes)

    def __getitem__(self, idx):
        if self.synthetic:
            image = Image.fromarray(self.syn_images[idx])
            objects = self.syn_objects[idx]
        else:
            scene = self.scenes[idx]
            img_path = os.path.join(self.image_dir, scene["image_filename"])
            image = Image.open(img_path).convert("RGB")
            objects = scene["objects"]

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Encode concepts for up to max_objects
        concept_label = np.zeros(self.num_concepts, dtype=np.float32)
        for i in range(min(len(objects), self.max_objects)):
            start = i * self.concepts_per_object
            end = start + self.concepts_per_object
            concept_label[start:end] = self._encode_object_concepts(objects[i])

        # Factor label (simplified: first object's properties as indices)
        factor_label = np.zeros(4, dtype=np.int64)
        if len(objects) > 0:
            obj = objects[0]
            factor_label[0] = self.SHAPES.index(obj["shape"]) if obj["shape"] in self.SHAPES else 0
            factor_label[1] = self.COLORS.index(obj["color"]) if obj["color"] in self.COLORS else 0
            factor_label[2] = self.SIZES.index(obj["size"]) if obj["size"] in self.SIZES else 0
            factor_label[3] = self.MATERIALS.index(obj["material"]) if obj["material"] in self.MATERIALS else 0

        return image, torch.FloatTensor(concept_label), torch.LongTensor(factor_label)


def get_clevr_loaders(config):
    """Get CLEVR train/test data loaders."""
    img_size = config["dataset"].get("image_size", 64)
    root = config["dataset"].get("root", "data/raw")
    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 4)
    seed = config["experiment"].get("seed", 42)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
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

    train_dataset = CLEVRDataset(
        root=root, split="train", transform=train_transform, seed=seed,
    )
    test_dataset = CLEVRDataset(
        root=root, split="val", transform=test_transform, seed=seed,
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
