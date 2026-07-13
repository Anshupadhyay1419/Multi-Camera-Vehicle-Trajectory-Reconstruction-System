"""
Create data.yaml for YOLO training and split training data into train/val.

Splits training data into 80% train and 20% validation if needed.
Generates dataset/data.yaml with correct paths and class names.
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List
import logging

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_validation_split(
    dataset_root: str, train_ratio: float = 0.8, seed: int = 42
) -> None:
    """
    Create validation split from training data if val split doesn't exist.

    Uses symbolic links for efficiency instead of copying files.

    Args:
        dataset_root: Path to dataset root
        train_ratio: Ratio of data to keep in train split
        seed: Random seed for reproducibility
    """
    dataset_path = Path(dataset_root)
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"

    # If validation split already exists, don't create it
    if val_path.exists() and len(list((val_path / "images").glob("*"))) > 0:
        logger.info("Validation split already exists.")
        return

    logger.info(f"Creating validation split from training data ({train_ratio*100:.0f}% train)")

    # Create validation directories
    val_images_dir = val_path / "images"
    val_labels_dir = val_path / "labels"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    train_images_dir = train_path / "images"
    train_labels_dir = train_path / "labels"

    image_files = sorted([f.stem for f in train_images_dir.glob("*.*")])
    random.seed(seed)
    random.shuffle(image_files)

    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)

    val_stems = set(image_files[split_idx:])
    train_stems = set(image_files[:split_idx])

    logger.info(f"Creating symbolic links for {len(val_stems)} validation images...")

    # Create symbolic links for validation (much faster than copying)
    for idx, stem in enumerate(val_stems):
        if (idx + 1) % 1000 == 0:
            logger.info(f"  Processed {idx + 1}/{len(val_stems)}")

        # Find image file and create symlink
        for img_file in train_images_dir.glob(f"{stem}.*"):
            val_img_link = val_images_dir / img_file.name
            if not val_img_link.exists():
                try:
                    os.symlink(
                        img_file.absolute(),
                        val_img_link,
                    )
                except FileExistsError:
                    pass

            # Create label symlink
            label_file = train_labels_dir / f"{stem}.txt"
            if label_file.exists():
                val_label_link = val_labels_dir / label_file.name
                if not val_label_link.exists():
                    try:
                        os.symlink(
                            label_file.absolute(),
                            val_label_link,
                        )
                    except FileExistsError:
                        pass

    logger.info(f"Kept {len(train_stems)} images in training split")
    logger.info(f"Created {len(val_stems)} validation image links")


def create_data_yaml(
    dataset_root: str, class_names: Dict[int, str] = None, output_file: str = "data.yaml"
) -> None:
    """
    Create data.yaml for YOLO training.

    Args:
        dataset_root: Path to dataset root
        class_names: Dictionary mapping class IDs to names
        output_file: Output YAML file path
    """
    dataset_path = Path(dataset_root)

    # Default class name
    if class_names is None:
        class_names = {0: "license_plate"}

    # Get absolute paths
    train_path = str((dataset_path / "train" / "images").absolute())
    val_path = str((dataset_path / "val" / "images").absolute())
    test_path = str((dataset_path / "test" / "images").absolute())

    # Create data configuration
    data_yaml = {
        "path": str(dataset_path.absolute()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(class_names),
        "names": class_names,
    }

    # Save to YAML
    output_path = dataset_path / output_file
    with open(output_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Created {output_path}")
    logger.info(f"Train: {train_path}")
    logger.info(f"Val: {val_path}")
    logger.info(f"Test: {test_path}")
    logger.info(f"Classes: {class_names}")


def verify_dataset_structure(dataset_root: str) -> bool:
    """
    Verify dataset structure is correct.

    Args:
        dataset_root: Path to dataset root

    Returns:
        True if valid, False otherwise
    """
    dataset_path = Path(dataset_root)

    required_dirs = [
        "train/images",
        "train/labels",
        "val/images",
        "val/labels",
        "test/images",
        "test/labels",
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        logger.info(f"{status} {dir_name}: {dir_path}")
        if not exists:
            all_exist = False

    return all_exist


def main():
    """Main function."""
    dataset_root = "dataset"

    logger.info("=" * 80)
    logger.info("DATASET PREPARATION")
    logger.info("=" * 80)

    # Create validation split if needed
    create_validation_split(dataset_root)

    # Create data.yaml
    create_data_yaml(
        dataset_root,
        class_names={0: "license_plate"},
        output_file="data.yaml",
    )

    # Verify structure
    logger.info("\nVerifying dataset structure...")
    if verify_dataset_structure(dataset_root):
        logger.info("✓ Dataset structure is valid")
    else:
        logger.error("✗ Dataset structure has issues")
        return False

    logger.info("\nDataset preparation complete!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
