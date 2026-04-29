from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import argparse
import random
import shutil


def get_project_root():
    """Get the project root directory (one level up from src/)."""
    return root


def parse_args():
    project_root = get_project_root()
    parser = argparse.ArgumentParser(description="Create a validation split from an existing dataset folder.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=project_root / "data" / "train",
        help="Source folder containing class subfolders (default: data/train)",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=project_root / "data" / "valid",
        help="Target validation folder to populate (default: data/valid)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.1,
        help="Fraction of source images to allocate to validation per class (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files to validation instead of moving them from the source set",
    )
    return parser.parse_args()


def gather_class_folders(base_dir: Path):
    return sorted([path for path in base_dir.iterdir() if path.is_dir()])


def make_validation_split(source_dir: Path, target_dir: Path, split: float, copy: bool, seed: int):
    random.seed(seed)
    source_dirs = gather_class_folders(source_dir)
    if not source_dirs:
        raise ValueError(f"Source directory '{source_dir}' contains no class subfolders.")

    target_dir.mkdir(parents=True, exist_ok=True)
    for class_dir in source_dirs:
        class_name = class_dir.name
        target_class_dir = target_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file()])
        if not image_paths:
            print(f"Warning: class '{class_name}' has no images in {class_dir}")
            continue

        num_val = max(1, int(len(image_paths) * split))
        validation_paths = random.sample(image_paths, num_val)

        for source_path in validation_paths:
            destination_path = target_class_dir / source_path.name
            if copy:
                shutil.copy2(source_path, destination_path)
            else:
                shutil.move(str(source_path), str(destination_path))

        print(
            f"Class '{class_name}': moved {len(validation_paths)} images to {target_class_dir} "
            f"({len(image_paths) - len(validation_paths)} remain in source)."
        )

    print("Validation split completed.")


if __name__ == "__main__":
    args = parse_args()
    make_validation_split(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        split=args.split,
        copy=args.copy,
        seed=args.seed,
    )
