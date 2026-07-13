"""
Dataset Validation for YOLO License Plate Detection.

Validates YOLO-format datasets for completeness, correctness, and quality.
Checks for missing files, corrupted images, invalid annotations, and class IDs.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates YOLO format datasets for training readiness."""

    def __init__(self, dataset_root: str, dataset_type: str = "yolo"):
        """
        Initialize dataset validator.

        Args:
            dataset_root: Path to dataset root directory
            dataset_type: Type of dataset ("yolo" for YOLO format)
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_type = dataset_type
        self.splits = ["train", "val", "test"]
        self.report = {}

    def validate(self) -> Dict:
        """
        Run complete dataset validation.

        Returns:
            Validation report with statistics and issues found
        """
        logger.info("=" * 80)
        logger.info("DATASET VALIDATION STARTED")
        logger.info("=" * 80)

        self.report = {
            "validation_status": "passed",
            "splits": {},
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        for split in self.splits:
            split_path = self.dataset_root / split
            if not split_path.exists():
                logger.warning(f"Split '{split}' not found at {split_path}")
                self.report["warnings"].append(f"Split '{split}' not found")
                continue

            logger.info(f"\nValidating {split.upper()} split...")
            split_report = self._validate_split(split_path, split)
            self.report["splits"][split] = split_report

        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        self._print_summary()

        return self.report

    def _validate_split(self, split_path: Path, split_name: str) -> Dict:
        """
        Validate a single dataset split.

        Args:
            split_path: Path to split directory
            split_name: Name of split (train/val/test)

        Returns:
            Validation report for this split
        """
        report = {
            "image_count": 0,
            "label_count": 0,
            "missing_labels": [],
            "missing_images": [],
            "empty_labels": [],
            "corrupted_images": [],
            "invalid_boxes": [],
            "class_ids": set(),
            "image_dimensions": [],
            "annotations_per_image": [],
        }

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.exists():
            self.report["issues"].append(
                f"Images directory not found in {split_name}: {images_dir}"
            )
            self.report["validation_status"] = "failed"
            return report

        if not labels_dir.exists():
            self.report["issues"].append(
                f"Labels directory not found in {split_name}: {labels_dir}"
            )
            self.report["validation_status"] = "failed"
            return report

        # Get all image and label files
        image_files = set()
        label_files = set()

        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
            image_files.update(f.stem for f in images_dir.glob(f"*{ext}"))

        for label_file in labels_dir.glob("*.txt"):
            label_files.add(label_file.stem)

        report["image_count"] = len(image_files)
        report["label_count"] = len(label_files)

        logger.info(f"  Images: {report['image_count']}")
        logger.info(f"  Labels: {report['label_count']}")

        # Check for missing labels
        missing_labels = image_files - label_files
        if missing_labels:
            report["missing_labels"] = sorted(list(missing_labels))
            msg = f"Found {len(missing_labels)} images without labels in {split_name}"
            logger.warning(f"  ⚠️  {msg}")
            self.report["warnings"].append(msg)

        # Check for missing images
        missing_images = label_files - image_files
        if missing_images:
            report["missing_images"] = sorted(list(missing_images))
            msg = f"Found {len(missing_images)} labels without images in {split_name}"
            logger.error(f"  ❌ {msg}")
            self.report["issues"].append(msg)
            self.report["validation_status"] = "failed"

        # Validate images and labels
        logger.info(f"  Validating {len(image_files)} image-label pairs...")
        for img_stem in tqdm(sorted(image_files), desc=f"Validating {split_name}"):
            # Find actual image file
            img_file = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
                candidate = images_dir / f"{img_stem}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break

            if not img_file:
                continue

            # Check image validity
            img_valid, img_info = self._validate_image(img_file)
            if not img_valid:
                report["corrupted_images"].append(img_stem)
                self.report["issues"].append(
                    f"Corrupted image in {split_name}: {img_stem}"
                )
                self.report["validation_status"] = "failed"
                continue

            report["image_dimensions"].append(img_info["shape"])

            # Validate labels
            label_file = labels_dir / f"{img_stem}.txt"
            if label_file.exists():
                boxes, valid = self._validate_labels(
                    label_file, img_info, img_stem, split_name
                )
                if not valid:
                    report["invalid_boxes"].append(img_stem)
                    self.report["validation_status"] = "failed"
                else:
                    report["annotations_per_image"].append(len(boxes))
                    for box in boxes:
                        report["class_ids"].add(int(box[0]))
            else:
                report["empty_labels"].append(img_stem)

        # Generate statistics
        report = self._compute_statistics(report)
        return report

    @staticmethod
    def _validate_image(img_path: Path) -> Tuple[bool, Dict]:
        """
        Validate image file integrity.

        Args:
            img_path: Path to image file

        Returns:
            Tuple of (is_valid, image_info)
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return False, {}
            h, w, c = img.shape
            return True, {"shape": (h, w, c), "path": str(img_path)}
        except Exception as e:
            logger.debug(f"Error reading image {img_path}: {e}")
            return False, {}

    def _validate_labels(
        self, label_file: Path, img_info: Dict, img_stem: str, split_name: str
    ) -> Tuple[List, bool]:
        """
        Validate YOLO format labels.

        Args:
            label_file: Path to label file
            img_info: Image information dict
            img_stem: Image stem for error reporting
            split_name: Dataset split name

        Returns:
            Tuple of (boxes_list, is_valid)
        """
        boxes = []
        valid = True

        try:
            with open(label_file, "r") as f:
                lines = f.readlines()

            if not lines:
                return boxes, True  # Empty labels are okay

            img_h, img_w = img_info["shape"][:2]

            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    logger.error(
                        f"Invalid bbox in {split_name}/{img_stem}: "
                        f"expected 5+ values, got {len(parts)}"
                    )
                    valid = False
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Validate class ID
                    if class_id < 0 or class_id > 999:
                        logger.error(
                            f"Invalid class ID {class_id} in {split_name}/{img_stem}"
                        )
                        valid = False
                        continue

                    # Validate coordinates are normalized (0-1)
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        logger.error(
                            f"Invalid center coordinates in {split_name}/{img_stem}: "
                            f"({x_center}, {y_center})"
                        )
                        valid = False
                        continue

                    if not (0 < width <= 1 and 0 < height <= 1):
                        logger.error(
                            f"Invalid dimensions in {split_name}/{img_stem}: "
                            f"({width}, {height})"
                        )
                        valid = False
                        continue

                    # Convert to pixel coordinates for actual validation
                    x1 = (x_center - width / 2) * img_w
                    y1 = (y_center - height / 2) * img_h
                    x2 = (x_center + width / 2) * img_w
                    y2 = (y_center + height / 2) * img_h

                    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                        logger.warning(
                            f"Box outside image boundaries in {split_name}/{img_stem}"
                        )

                    boxes.append([class_id, x_center, y_center, width, height])

                except (ValueError, IndexError) as e:
                    logger.error(
                        f"Error parsing bbox in {split_name}/{img_stem} "
                        f"line {line_idx}: {e}"
                    )
                    valid = False

        except Exception as e:
            logger.error(f"Error reading label file {label_file}: {e}")
            valid = False

        return boxes, valid

    @staticmethod
    def _compute_statistics(report: Dict) -> Dict:
        """Compute statistics from validation data."""
        report["statistics"] = {}

        if report["image_dimensions"]:
            dims = np.array(report["image_dimensions"])
            report["statistics"]["avg_height"] = float(np.mean(dims[:, 0]))
            report["statistics"]["avg_width"] = float(np.mean(dims[:, 1]))
            report["statistics"]["avg_channels"] = float(np.mean(dims[:, 2]))
            report["statistics"]["min_height"] = int(np.min(dims[:, 0]))
            report["statistics"]["max_height"] = int(np.max(dims[:, 0]))
            report["statistics"]["min_width"] = int(np.min(dims[:, 1]))
            report["statistics"]["max_width"] = int(np.max(dims[:, 1]))

        if report["annotations_per_image"]:
            annots = np.array(report["annotations_per_image"])
            report["statistics"]["avg_annotations_per_image"] = float(np.mean(annots))
            report["statistics"]["max_annotations_per_image"] = int(np.max(annots))
            report["statistics"]["min_annotations_per_image"] = int(np.min(annots))
            report["statistics"]["total_annotations"] = int(np.sum(annots))

        report["statistics"]["class_ids"] = sorted(list(report["class_ids"]))
        report["class_ids"] = sorted(list(report["class_ids"]))

        return report

    def _print_summary(self) -> None:
        """Print validation summary."""
        status_symbol = "✓" if self.report["validation_status"] == "passed" else "✗"
        logger.info(f"\nStatus: {status_symbol} {self.report['validation_status'].upper()}")

        for split, data in self.report["splits"].items():
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Images: {data['image_count']}")
            logger.info(f"  Labels: {data['label_count']}")

            if data["missing_labels"]:
                logger.warning(
                    f"  Missing labels: {len(data['missing_labels'])} images"
                )
            if data["missing_images"]:
                logger.error(
                    f"  Missing images: {len(data['missing_images'])} labels"
                )
            if data["corrupted_images"]:
                logger.error(
                    f"  Corrupted images: {len(data['corrupted_images'])}"
                )
            if data["invalid_boxes"]:
                logger.error(f"  Invalid boxes: {len(data['invalid_boxes'])}")

            if data["statistics"]:
                stats = data["statistics"]
                logger.info(f"  Image dimensions: {stats.get('avg_height', 0):.0f}x{stats.get('avg_width', 0):.0f}")
                logger.info(
                    f"  Avg annotations per image: {stats.get('avg_annotations_per_image', 0):.2f}"
                )
                logger.info(f"  Total annotations: {stats.get('total_annotations', 0)}")
                logger.info(f"  Classes: {stats.get('class_ids', [])}")

        if self.report["issues"]:
            logger.error(f"\nCritical Issues ({len(self.report['issues'])}):")
            for issue in self.report["issues"]:
                logger.error(f"  - {issue}")

        if self.report["warnings"]:
            logger.warning(f"\nWarnings ({len(self.report['warnings'])}):")
            for warning in self.report["warnings"]:
                logger.warning(f"  - {warning}")

    def save_report(self, output_path: str) -> None:
        """
        Save validation report to JSON file.

        Args:
            output_path: Path to save JSON report
        """
        # Convert sets to lists for JSON serialization
        report_copy = self.report.copy()
        for split in report_copy.get("splits", {}).values():
            if "class_ids" in split:
                split["class_ids"] = sorted(list(split["class_ids"]))

        with open(output_path, "w") as f:
            json.dump(report_copy, f, indent=2)

        logger.info(f"\nReport saved to {output_path}")


def main():
    """Run dataset validation."""
    dataset_root = "dataset"

    validator = DatasetValidator(dataset_root)
    report = validator.validate()

    # Save report
    report_path = "dataset_validation_report.json"
    validator.save_report(report_path)

    return report["validation_status"] == "passed"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
