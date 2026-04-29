from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS, TRAIN_DIR, VALID_DIR, TEST_DIR, CLASS_NAMES


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class RealFakeDataset(ImageFolder):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label, path


def build_dataloader(
    data_folder: Path,
    batch_size: int = BATCH_SIZE,
    train: bool = False,
    shuffle: bool = False,
) -> DataLoader:
    dataset = RealFakeDataset(str(data_folder), transform=get_transforms(train=train))
    
    # Only use pin_memory if we have a GPU accelerator
    pin_memory = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )


def load_data() -> Dict[str, DataLoader]:
    return {
        "train": build_dataloader(TRAIN_DIR, train=True, shuffle=True),
        "valid": build_dataloader(VALID_DIR, train=False, shuffle=False),
        "test": build_dataloader(TEST_DIR, train=False, shuffle=False),
    }


def dataset_stats(data_folder: Path) -> Dict[str, int]:
    dataset = RealFakeDataset(str(data_folder), transform=get_transforms(train=False))
    counts = {label: 0 for label in CLASS_NAMES}
    for _, label, _ in dataset:
        counts[CLASS_NAMES[label]] += 1
    return counts


def count_images_per_split() -> Dict[str, Dict[str, int]]:
    return {
        "train": dataset_stats(TRAIN_DIR),
        "valid": dataset_stats(VALID_DIR),
        "test": dataset_stats(TEST_DIR),
    }
